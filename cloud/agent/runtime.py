"""How a worker actually runs a replica: as a Docker container or as a local process.

Docker is the real thing: namespaces + cgroups isolation, CPU/memory limits, `--gpus all`
for NVIDIA nodes. The process runtime runs the same command directly on the machine,
which is what CI uses (no Docker there) and what a Mac uses for model replicas, since
Docker on macOS cannot reach the GPU.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from cloud.common import REPO_ROOT
from cloud.spec import engine_env

ENGINE_RUN = "python -m uvicorn gateway.server:app --host 0.0.0.0 --port $PORT"


def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


class ProcessRuntime:
    name = "process"

    def __init__(self, home: Path, node_id: str):
        self.logs_dir = home / "replica-logs" / node_id
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.procs: dict[str, subprocess.Popen] = {}

    def log_file(self, rid: str) -> Path:
        return self.logs_dir / f"{rid}.log"

    def start(self, rid: str, artifact: dict, spec: dict, port: int, env: dict) -> None:
        python_bin = os.path.dirname(sys.executable)
        if artifact["kind"] == "engine":
            cwd, run, env = REPO_ROOT, ENGINE_RUN, engine_env(spec) | env
        elif artifact["kind"] == "dir":
            cwd, run = artifact["path"], artifact["run"]
            if artifact.get("venv"):
                python_bin = os.path.join(artifact["venv"], "bin")
        else:
            raise ValueError(f"process runtime cannot run a {artifact['kind']} artifact")
        full_env = os.environ.copy() | {"PYTHONUNBUFFERED": "1"} | spec["env"] | env | {"PORT": str(port)}
        full_env["PATH"] = python_bin + os.pathsep + full_env.get("PATH", "")
        with open(self.log_file(rid), "ab") as log:
            self.procs[rid] = subprocess.Popen(
                ["/bin/sh", "-c", run], cwd=cwd, env=full_env, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,  # own process group, so stop() kills children too
            )

    def container_port(self, spec: dict, host_port: int) -> int:
        return host_port

    def exit_code(self, rid: str) -> int | None:
        proc = self.procs.get(rid)
        return None if proc is None else proc.poll()

    def stop(self, rid: str, remove: bool = True) -> None:
        proc = self.procs.get(rid)
        if proc and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)
            except (ProcessLookupError, PermissionError):
                pass
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        if remove:
            self.procs.pop(rid, None)

    def tail(self, rid: str, n: int) -> list[str]:
        path = self.log_file(rid)
        if not path.exists():
            return []
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 256 * 1024))
            lines = f.read().decode(errors="replace").splitlines()
        return lines[-n:] if n else []

    async def follow(self, rid: str, alive):
        """Yields new log lines until the replica is gone."""
        path = self.log_file(rid)
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            buf = b""
            while True:
                chunk = f.read()
                if chunk:
                    buf += chunk
                    *lines, buf = buf.split(b"\n")
                    for line in lines:
                        yield line.decode(errors="replace")
                elif not alive():
                    return
                else:
                    await asyncio.sleep(0.3)

    def cleanup_orphans(self) -> None:
        pass  # child processes die with their session when the agent is stopped


class DockerRuntime:
    name = "docker"

    def __init__(self, home: Path, node_id: str, gpu: bool):
        self.node_id, self.gpu = node_id, gpu

    @staticmethod
    def cname(rid: str) -> str:
        return f"vrtx-{rid}"

    def container_port(self, spec: dict, host_port: int) -> int:
        return spec["port"]

    def start(self, rid: str, artifact: dict, spec: dict, port: int, env: dict) -> None:
        if artifact["kind"] != "image":
            raise ValueError(f"docker runtime cannot run a {artifact['kind']} artifact")
        inner = spec["port"]
        env = dict(spec["env"]) | env | {"PORT": str(inner)}
        cmd = [
            "docker", "run", "-d", "--name", self.cname(rid),
            "--label", "vrtx.managed=1", "--label", f"vrtx.node={self.node_id}",
            "--label", f"vrtx.replica={rid}", "--label", f"vrtx.deployment={spec['name']}",
            "-p", f"{port}:{inner}",
        ]
        res = spec["resources"]
        if res["cpu"]:
            cmd += ["--cpus", str(res["cpu"])]
        if res["memory_mb"]:
            cmd += ["--memory", f"{res['memory_mb']}m"]
        if spec["type"] == "model":
            env = engine_env(spec) | env
            model = Path(spec["model"])
            if model.is_dir():  # local weights: mount read-only
                cmd += ["-v", f"{model}:/models/{model.name}:ro"]
                env["VRTX_MODEL"] = f"/models/{model.name}"
            hf_cache = Path.home() / ".cache" / "huggingface"
            hf_cache.mkdir(parents=True, exist_ok=True)
            cmd += ["-v", f"{hf_cache}:/root/.cache/huggingface"]
            if self.gpu:
                cmd += ["--gpus", "all"]
        elif self.gpu and res["gpu_memory_gb"] > 0:
            cmd += ["--gpus", "all"]
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        cmd.append(artifact["image"])
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip() or "docker run failed")

    def exit_code(self, rid: str) -> int | None:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}} {{.State.ExitCode}}", self.cname(rid)],
            capture_output=True, text=True,
        )
        if out.returncode != 0:
            return -1  # container is gone
        running, code = out.stdout.split()
        return None if running == "true" else int(code)

    def stop(self, rid: str, remove: bool = True) -> None:
        # SIGTERM first so the app can finish in-flight requests, SIGKILL after 5 s
        subprocess.run(["docker", "stop", "-t", "5", self.cname(rid)], capture_output=True)
        if remove:
            subprocess.run(["docker", "rm", "-f", self.cname(rid)], capture_output=True)

    def tail(self, rid: str, n: int) -> list[str]:
        out = subprocess.run(
            ["docker", "logs", "--tail", str(n), self.cname(rid)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
        )
        return out.stdout.splitlines() if out.returncode == 0 else []

    async def follow(self, rid: str, alive):
        proc = await asyncio.create_subprocess_exec(
            "docker", "logs", "-f", "--tail", "0", self.cname(rid),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            async for line in proc.stdout:
                yield line.decode(errors="replace").rstrip("\n")
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    def cleanup_orphans(self) -> None:
        """Remove containers a previous run of this agent left behind; the control plane reschedules."""
        out = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "label=vrtx.managed=1", "--filter", f"label=vrtx.node={self.node_id}"],
            capture_output=True, text=True,
        )
        ids = out.stdout.split()
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)
