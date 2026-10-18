"""The `vrtx` command.

    vrtx up                                   start control plane + worker + gateway on this machine
    vrtx deploy ./todo-api [--replicas 2]     build a folder and run it at todo-api.vrtx.local
    vrtx models deploy Qwen/Qwen2.5-0.5B-Instruct --name qwen-chat [--quant int8]
    vrtx ps                                   deployments, replicas, status
    vrtx logs todo-api [-f]                   logs from every replica
    vrtx scale todo-api 3
    vrtx rm todo-api
    vrtx nodes                                worker nodes, capacity, GPU memory
    vrtx cluster status

Talks to the control plane at $VRTX_API (default http://127.0.0.1:7070).
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import os
import re
import sys
import tarfile
import time
from pathlib import Path

import httpx
import yaml

from cloud.common import DEFAULT_API, valid_name

API = os.getenv("VRTX_API", DEFAULT_API).rstrip("/")
ALWAYS_IGNORE = [".git", ".venv", "venv", "node_modules", "__pycache__", "*.pyc", ".DS_Store", ".pytest_cache"]


class CLIError(Exception):
    pass


# --------------------------------------------------------------------- http

def client() -> httpx.Client:
    return httpx.Client(base_url=API, timeout=30.0)


def call(method: str, path: str, **kw) -> dict | list:
    try:
        with client() as c:
            r = c.request(method, path, **kw)
    except httpx.ConnectError:
        raise CLIError(f"cannot reach the control plane at {API}. Start one with `vrtx up`.")
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise CLIError(str(detail))
    return r.json()


def stream_text(path: str, params: dict) -> None:
    try:
        with client() as c, c.stream("GET", path, params=params, timeout=None) as r:
            if r.status_code >= 400:
                r.read()
                raise CLIError(r.json().get("detail", r.text))
            for chunk in r.iter_text():
                sys.stdout.write(chunk)
                sys.stdout.flush()
    except httpx.ConnectError:
        raise CLIError(f"cannot reach the control plane at {API}")


# ---------------------------------------------------------------- printing

class Step:
    """`Building image ............. done (38s)`"""

    def __init__(self, label: str, width: int = 28):
        self.label, self.width, self.t0, self.dots = label, width, time.time(), 0
        sys.stdout.write(label + " ")
        sys.stdout.flush()

    def tick(self):
        if self.dots < self.width - len(self.label):
            self.dots += 1
            sys.stdout.write(".")
            sys.stdout.flush()

    def done(self, word: str = "done"):
        pad = "." * max(0, self.width - len(self.label) - self.dots)
        sys.stdout.write(f"{pad} {word} ({time.time() - self.t0:.0f}s)\n")


def table(rows: list[list], headers: list[str]) -> None:
    rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip())
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())


# ----------------------------------------------------------------- packing

def ignore_patterns(folder: Path) -> list[str]:
    pats = list(ALWAYS_IGNORE)
    for name in (".vrtxignore", ".dockerignore"):
        f = folder / name
        if f.is_file():
            pats += [ln.strip().rstrip("/") for ln in f.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return pats


def pack(folder: Path) -> bytes:
    pats = ignore_patterns(folder)

    def skip(rel: str) -> bool:
        parts = rel.split("/")
        return any(fnmatch.fnmatch(rel, p) or any(fnmatch.fnmatch(part, p) for part in parts) for p in pats)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(folder.rglob("*")):
            rel = path.relative_to(folder).as_posix()
            if path.is_file() and not path.is_symlink() and not skip(rel):
                tar.add(path, arcname=rel, recursive=False)
    return buf.getvalue()


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")
    return re.sub(r"-{2,}", "-", s)[:42].strip("-")


# ---------------------------------------------------------------- commands

def wait_build(build: dict, verbose: bool, label: str = "Building image") -> dict:
    if verbose:
        stream_text(f"/api/builds/{build['id']}/logs", {"follow": True})
        result = call("GET", f"/api/builds/{build['id']}")
    else:
        step = Step(label)
        while (result := call("GET", f"/api/builds/{build['id']}"))["status"] == "building":
            step.tick()
            time.sleep(0.5)
        step.done("cached" if result["status"] == "succeeded" and result["cached"] else
                  "done" if result["status"] == "succeeded" else "FAILED")
    if result["status"] == "failed":
        with client() as c:
            log = c.get(f"/api/builds/{build['id']}/logs").text
        tail = "\n".join(log.splitlines()[-25:])
        raise CLIError(f"build failed: {result['error']}\n\n--- last build output ---\n{tail}")
    return result


def wait_live(name: str, label: str, timeout: float) -> dict:
    step = Step(label)
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = call("GET", f"/api/deployments/{name}")
        if d["status"] in ("running", "stopped"):
            step.done()
            return d
        if d["status"] in ("crashloop", "build failed"):
            step.done("FAILED")
            raise CLIError(f"{name} is in {d['status']}:\n{d['last_error']}\n\nSee `vrtx logs {name}`.")
        if d["status"] == "pending" and time.time() > deadline - timeout + 5:
            step.done("PENDING")
            raise CLIError(f"{name} cannot be placed: {d['last_error']}")
        step.tick()
        time.sleep(0.5)
    step.done("TIMEOUT")
    d = call("GET", f"/api/deployments/{name}")
    raise CLIError(f"{name} is still {d['status']} after {timeout:.0f}s. Last error: {d['last_error']}")


def parse_env(pairs: list[str]) -> dict:
    env = {}
    for p in pairs or []:
        if "=" not in p:
            raise CLIError(f"--env needs KEY=VALUE, got {p!r}")
        k, v = p.split("=", 1)
        env[k] = v
    return env


def cmd_deploy(a):
    folder = Path(a.path).expanduser().resolve()
    if not folder.is_dir():
        raise CLIError(f"{folder} is not a folder")
    spec: dict = {}
    for fname in ("vrtx.yaml", "vrtx.yml"):
        if (folder / fname).is_file():
            spec = yaml.safe_load((folder / fname).read_text()) or {}
    spec["type"] = "app"
    spec["name"] = a.name or spec.get("name") or slug(folder.name)
    for key in ("replicas", "port", "health", "run"):
        if getattr(a, key) is not None:
            spec[key] = getattr(a, key)
    if a.env:
        spec["env"] = (spec.get("env") or {}) | parse_env(a.env)
    if not valid_name(spec["name"]):
        raise CLIError(f"{spec['name']!r} is not a valid name: use lowercase letters, digits and dashes (--name)")

    source = pack(folder)
    before = call("POST", "/api/deployments", json=spec)
    print(f"Deploying {spec['name']} from {folder} ({len(source) / 1024:.0f} KB)")
    build = call("POST", f"/api/deployments/{spec['name']}/builds", content=source,
                 headers={"content-type": "application/gzip"})
    wait_build(build, a.verbose)
    if a.no_wait:
        print(f"Submitted. Watch it with `vrtx ps` and `vrtx logs {spec['name']}`.")
        return
    d = call("GET", f"/api/deployments/{spec['name']}")
    n = d["replicas"]["desired"]
    if d["version"] == before["version"] and d["status"] == "running":
        print(f"No changes since v{d['version']}; live at {d['url']}")
        return
    verb = "Rolling out" if before["version"] else "Starting"
    d = wait_live(spec["name"], f"{verb} {n} replica{'s' if n != 1 else ''}", d["spec"]["health_timeout_s"] + 30)
    print(f"Live at {d['url']}")


def cmd_models_deploy(a):
    spec = {
        "type": "model", "model": a.model, "name": a.name or slug(Path(a.model).name),
        "replicas": a.replicas, "quantization": a.quant, "kv_cache_mb": a.kv_cache_mb, "max_seqs": a.max_seqs,
        "resources": {"gpu_memory_gb": a.gpu_mem or 0},
    }
    if a.draft:
        spec["speculative"] = {"draft": a.draft}
    if a.env:
        spec["env"] = parse_env(a.env)
    d = call("POST", "/api/deployments", json=spec)
    print(f"Deploying model {a.model} as {d['name']}")
    if d["build"]:
        wait_build(d["build"], a.verbose, label="Preparing engine")
    d = wait_live(d["name"], "Loading weights", d["spec"]["health_timeout_s"] + 30)
    print(f"Healthy at {d['url']}/v1")


def cmd_ps(a):
    rows = []
    for d in call("GET", "/api/deployments"):
        r = d["replicas"]
        rows.append([d["name"], d["type"], f"{r['running']}/{r['desired']}", f"v{d['version']}", d["status"], d["url"]])
    if not rows:
        print("No deployments. Try `vrtx deploy ./examples/todo-api`.")
        return
    table(rows, ["NAME", "TYPE", "REPLICAS", "VERSION", "STATUS", "URL"])
    if a.all:
        for d in call("GET", "/api/deployments"):
            for i in d["instances"]:
                print(f"  {i['id']:<28} v{i['version']:<3} {i['state']:<10} {i['node']:<14} {i['url'] or ''}")
            if d["last_error"]:
                print(f"  {d['name']} last error: {d['last_error'].splitlines()[0]}")


def cmd_logs(a):
    stream_text(f"/api/deployments/{a.name}/logs", {"follow": a.follow, "tail": a.tail})


def cmd_scale(a):
    d = call("PATCH", f"/api/deployments/{a.name}", json={"replicas": a.replicas})
    print(f"{a.name}: {d['replicas']['running']} running, scaling to {a.replicas}")
    if a.replicas and not a.no_wait:
        wait_live(a.name, f"Scaling to {a.replicas}", d["spec"]["health_timeout_s"] + 30)


def cmd_rm(a):
    call("DELETE", f"/api/deployments/{a.name}")
    step = Step(f"Removing {a.name}")
    for _ in range(120):
        names = {d["name"] for d in call("GET", "/api/deployments")}
        if a.name not in names:
            step.done()
            return
        step.tick()
        time.sleep(0.5)
    step.done("still stopping")


def cmd_nodes(a):
    rows = []
    for n in call("GET", "/api/nodes"):
        cap = n["capacity"]
        gpus = cap.get("gpus", [])
        gpu = ", ".join(f"{g['name']} {g['memory_used_mb'] / 1024:.1f}/{g['memory_total_mb'] / 1024:.1f}G"
                        for g in n["usage"].get("gpus", gpus)) or "none"
        rows.append([n["id"], "up" if n["alive"] else f"DOWN ({n['last_seen_s']:.0f}s)", n["runtime"], cap.get("cpus"),
                     f"{cap.get('memory_mb', 0) / 1024:.0f}G", gpu, n["replicas"]])
    table(rows, ["NODE", "STATUS", "RUNTIME", "CPUS", "RAM", "GPU (used/total)", "REPLICAS"])


def cmd_cluster(a):
    c = call("GET", "/api/cluster")
    for k, v in c.items():
        print(f"{k:>13}: {v}")


def cmd_up(a):
    from cloud.common import vrtx_home
    from cloud.up import Stack

    stack = Stack(home=vrtx_home(), runtime=a.runtime, host=a.host, api_port=a.api_port,
                  agent_port=a.agent_port, gateway_port=a.gateway_port, log_level="info" if a.verbose else "warning")
    print(f"VRTX Cloud on this machine (state in {stack.home})")
    print(f"  control plane  {stack.api_url}")
    print(f"  worker agent   http://{a.host}:{a.agent_port}  runtime: {a.runtime}")
    print(f"  gateway        http://<name>.localhost:{a.gateway_port}  or  http://<name>.vrtx.local:{a.gateway_port}")
    print("Ctrl-C stops everything (replicas included).")
    stack.run_forever()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vrtx", description="VRTX Cloud command line")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("up", help="run control plane, worker agent and gateway on this machine")
    s.add_argument("--runtime", default="auto", choices=["auto", "docker", "process"])
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--api-port", type=int, default=7070)
    s.add_argument("--agent-port", type=int, default=7071)
    s.add_argument("--gateway-port", type=int, default=8080)
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_up)

    s = sub.add_parser("deploy", help="build a folder and run it")
    s.add_argument("path")
    s.add_argument("--name")
    s.add_argument("--replicas", type=int)
    s.add_argument("--port", type=int, help="port the app listens on (it also gets $PORT)")
    s.add_argument("--health", help="health check path, default /")
    s.add_argument("--run", help="start command, overrides detection")
    s.add_argument("--env", action="append", metavar="KEY=VALUE")
    s.add_argument("--no-wait", action="store_true")
    s.add_argument("-v", "--verbose", action="store_true", help="stream the full build output")
    s.set_defaults(fn=cmd_deploy)

    models = sub.add_parser("models", help="model deployments").add_subparsers(dest="models_cmd", required=True)
    s = models.add_parser("deploy", help="serve a model with VRTX Engine")
    s.add_argument("model", help="Hugging Face id or local folder")
    s.add_argument("--name")
    s.add_argument("--replicas", type=int, default=1)
    s.add_argument("--quant", choices=["int8", "int4"])
    s.add_argument("--draft", help="draft model for speculative decoding")
    s.add_argument("--gpu-mem", type=float, help="GPU memory to reserve, GB")
    s.add_argument("--kv-cache-mb", type=int)
    s.add_argument("--max-seqs", type=int)
    s.add_argument("--env", action="append", metavar="KEY=VALUE")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_models_deploy)

    s = sub.add_parser("ps", help="list deployments")
    s.add_argument("-a", "--all", action="store_true", help="show every replica too")
    s.set_defaults(fn=cmd_ps)

    s = sub.add_parser("logs", help="logs from every replica of a deployment")
    s.add_argument("name")
    s.add_argument("-f", "--follow", action="store_true")
    s.add_argument("--tail", type=int, default=100)
    s.set_defaults(fn=cmd_logs)

    s = sub.add_parser("scale", help="change the replica count")
    s.add_argument("name")
    s.add_argument("replicas", type=int)
    s.add_argument("--no-wait", action="store_true")
    s.set_defaults(fn=cmd_scale)

    s = sub.add_parser("rm", help="delete a deployment and stop its replicas")
    s.add_argument("name")
    s.set_defaults(fn=cmd_rm)

    s = sub.add_parser("nodes", help="list worker nodes")
    s.set_defaults(fn=cmd_nodes)

    s = sub.add_parser("cluster", help="cluster state")
    s.add_argument("what", nargs="?", default="status", choices=["status"])
    s.set_defaults(fn=cmd_cluster)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except CLIError as e:
        sys.stdout.flush()
        print(f"\nerror: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
