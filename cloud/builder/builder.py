"""Builder: a folder of code -> something a worker can run.

`detect` works out how to run a folder, like a tiny buildpack:

    Dockerfile present       -> use it as is
    Procfile `web: ...`      -> that command
    package.json             -> Node: npm install, npm start
    FastAPI/Starlette app    -> uvicorn module:app --port $PORT
    Flask app                -> flask --app module run --port $PORT
    other Python script      -> python script.py
    index.html only          -> static files via python -m http.server

A `run:` line in vrtx.yaml beats all of these. Apps should listen on $PORT.

Two artifact kinds, one per worker runtime:
    {"kind": "image", "image": "vrtx/todo-api:3f9c..."}             docker runtime
    {"kind": "dir", "path": ..., "run": ..., "venv": ...}            process runtime (one machine)
    {"kind": "engine"}                                               process runtime, model deployments

Builds are keyed by a hash of the files plus the run plan, so deploying the same code twice
reuses the first build (the image already exists, or the folder is already prepared).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from cloud.common import REPO_ROOT

ASGI = re.compile(r"^(\w+)\s*(?::[^=]+)?=\s*(?:fastapi\.)?(FastAPI|Starlette)\(", re.M)
FLASK = re.compile(r"^(\w+)\s*(?::[^=]+)?=\s*(?:flask\.)?Flask\(", re.M)
PY_ENTRIES = ("main.py", "app.py", "server.py", "api.py", "wsgi.py")
ENGINE_FILES = ("requirements.txt", "engine", "gateway")
MAX_SOURCE_BYTES = 200 * 1024 * 1024


class BuildError(RuntimeError):
    pass


@dataclass
class Plan:
    kind: str                 # dockerfile | python | node | static
    run: str | None           # shell command, $PORT set; None = the Dockerfile's CMD
    dockerfile: str | None    # generated Dockerfile text; None = use the folder's own
    requirements: bool = False

    def describe(self) -> str:
        how = "own Dockerfile" if self.kind == "dockerfile" else f"detected {self.kind}"
        return f"{how}; run: {self.run or '(Dockerfile CMD)'}"


# ------------------------------------------------------------------ detection

def _cmd(run: str) -> str:
    """Dockerfile CMD for a shell command. `exec` makes the app PID 1, so `docker stop`'s
    SIGTERM reaches it instead of stopping at /bin/sh (which ignores it). Only for simple commands."""
    simple = not any(op in run for op in ("&&", "||", ";", "|", "\n"))
    return json.dumps(["/bin/sh", "-c", ("exec " if simple else "") + run])


def _python_run(src: Path) -> str | None:
    files = [src / f for f in PY_ENTRIES if (src / f).is_file()]
    files += sorted(p for p in src.glob("*.py") if p not in files)
    for path in files:
        text = path.read_text(errors="ignore")
        mod = path.stem
        if m := ASGI.search(text):
            return f"python -m uvicorn {mod}:{m.group(1)} --host 0.0.0.0 --port $PORT"
        if FLASK.search(text):
            return f"python -m flask --app {mod} run --host 0.0.0.0 --port $PORT"
    for path in files:
        if "__main__" in path.read_text(errors="ignore"):
            return f"python {path.name}"
    return f"python {files[0].name}" if files else None


def _procfile(src: Path) -> str | None:
    path = src / "Procfile"
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith("web:"):
                return line[4:].strip()
    return None


def detect(src: Path, spec: dict) -> Plan:
    port = spec["port"]
    run = spec.get("run") or _procfile(src)
    if (src / "Dockerfile").is_file():
        return Plan("dockerfile", run, None)

    if (src / "package.json").is_file():
        run = run or "npm start"
        dockerfile = (
            "FROM node:20-slim\nWORKDIR /app\nCOPY package*.json ./\nRUN npm install --omit=dev\nCOPY . .\n"
            f"ENV PORT={port}\nEXPOSE {port}\nCMD {_cmd(run)}\n"
        )
        return Plan("node", run, dockerfile)

    has_reqs = (src / "requirements.txt").is_file() and (src / "requirements.txt").read_text().strip() != ""
    py_run = _python_run(src)
    if py_run or has_reqs:
        run = run or py_run
        if not run:
            raise BuildError("found requirements.txt but no .py file to run; add `run:` to vrtx.yaml")
        reqs = (src / "requirements.txt").read_text().lower() if has_reqs else ""
        extra = []
        if "uvicorn" in run and "uvicorn" not in reqs:
            extra.append("uvicorn")
        if "flask" in run and "flask" not in reqs:
            extra.append("flask")
        lines = [
            "FROM python:3.11-slim",
            "WORKDIR /app",
            "ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1",
        ]
        if has_reqs:
            lines += ["COPY requirements.txt ./", "RUN pip install -r requirements.txt"]
        if extra:
            lines.append(f"RUN pip install {' '.join(extra)}")
        lines += ["COPY . .", f"ENV PORT={port}", f"EXPOSE {port}", f"CMD {_cmd(run)}"]
        return Plan("python", run, "\n".join(lines) + "\n", requirements=has_reqs)

    if (src / "index.html").is_file():
        run = run or "python -m http.server $PORT --bind 0.0.0.0"
        dockerfile = (
            "FROM python:3.11-slim\nWORKDIR /app\nCOPY . .\n"
            f"ENV PORT={port}\nEXPOSE {port}\nCMD {_cmd(run)}\n"
        )
        return Plan("static", run, dockerfile)

    raise BuildError(
        "could not tell how to run this folder: add a Dockerfile, a Procfile with `web: <command>`, "
        "or `run: <command>` in vrtx.yaml"
    )


# ------------------------------------------------------------------- helpers

def extract(tar_bytes: bytes, dest: Path) -> None:
    if len(tar_bytes) > MAX_SOURCE_BYTES:
        raise BuildError(f"source is {len(tar_bytes) >> 20} MB, the limit is {MAX_SOURCE_BYTES >> 20} MB")
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
            tar.extractall(dest, filter="data")  # refuses absolute paths, .. and device files
    except (tarfile.TarError, OSError) as e:
        raise BuildError(f"could not unpack the uploaded source: {e}") from e


def tree_hash(root: Path, extra: str = "") -> str:
    h = hashlib.sha256(extra.encode())
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(root)).encode() + b"\0")
        h.update(path.read_bytes())
    return h.hexdigest()


def docker_image_exists(tag: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0


def run_logged(cmd: list[str], log, cwd: Path | None = None) -> None:
    log.write(f"$ {' '.join(cmd)}\n")
    log.flush()
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        log.write(line)
        log.flush()
    if proc.wait() != 0:
        raise BuildError(f"`{' '.join(cmd[:3])} ...` exited with code {proc.returncode}")


# -------------------------------------------------------------------- builder

class Builder:
    """Runs builds on the control plane machine. `runtime` is the workers' runtime."""

    def __init__(self, home: Path, runtime: str):
        self.home = home
        self.runtime = runtime
        (home / "builds").mkdir(parents=True, exist_ok=True)
        (home / "build-logs").mkdir(parents=True, exist_ok=True)

    def log_path(self, build_id: str) -> Path:
        return self.home / "build-logs" / f"{build_id}.log"

    def build_app(self, build_id: str, spec: dict, tar_bytes: bytes) -> tuple[dict, bool]:
        """Returns (artifact, cached). Raises BuildError with the reason."""
        with open(self.log_path(build_id), "a") as log, tempfile.TemporaryDirectory(prefix="vrtx-src-") as tmp:
            src = Path(tmp)
            extract(tar_bytes, src)
            plan = detect(src, spec)
            log.write(f"build {build_id}: {spec['name']} ({plan.describe()})\n")
            digest = tree_hash(src, json.dumps([plan.kind, plan.run, plan.dockerfile, self.runtime]))[:12]
            if self.runtime == "docker":
                return self._docker(src, spec["name"], digest, plan, log)
            return self._process(src, spec["name"], digest, plan, log)

    def _docker(self, src: Path, name: str, digest: str, plan: Plan, log) -> tuple[dict, bool]:
        tag = f"vrtx/{name}:{digest}"
        if docker_image_exists(tag):
            log.write(f"image {tag} already built, reusing it\n")
            return {"kind": "image", "image": tag}, True
        dockerfile = src / "Dockerfile"
        if plan.dockerfile:
            dockerfile = src / ".vrtx.Dockerfile"
            dockerfile.write_text(plan.dockerfile)
            log.write("generated Dockerfile:\n" + "".join(f"  | {line}\n" for line in plan.dockerfile.splitlines()))
        run_logged(["docker", "build", "--progress=plain", "-t", tag, "-f", str(dockerfile), str(src)], log)
        log.write(f"built image {tag}\n")
        return {"kind": "image", "image": tag}, False

    def _process(self, src: Path, name: str, digest: str, plan: Plan, log) -> tuple[dict, bool]:
        if not plan.run:
            raise BuildError("the process runtime cannot use a Dockerfile's CMD; add `run:` to vrtx.yaml")
        out = self.home / "builds" / name / digest
        venv = out / "venv" if plan.requirements else None
        artifact = {"kind": "dir", "path": str(out / "src"), "run": plan.run, "venv": str(venv) if venv else None}
        if (out / ".ok").exists():
            log.write(f"build {digest} already prepared, reusing it\n")
            return artifact, True
        shutil.rmtree(out, ignore_errors=True)
        shutil.copytree(src, out / "src")
        if venv:
            run_logged([sys.executable, "-m", "venv", str(venv)], log)
            run_logged([str(venv / "bin" / "pip"), "install", "-r", str(out / "src" / "requirements.txt")], log)
        (out / ".ok").touch()
        log.write(f"prepared {out}\n")
        return artifact, False

    def build_engine(self, build_id: str) -> tuple[dict, bool]:
        """The VRTX Engine itself, for model deployments."""
        with open(self.log_path(build_id), "a") as log:
            if self.runtime != "docker":
                log.write("process runtime: the engine runs from this checkout, nothing to build\n")
                return {"kind": "engine"}, True
            with tempfile.TemporaryDirectory(prefix="vrtx-engine-") as tmp:
                ctx = Path(tmp)
                for item in ENGINE_FILES:
                    path = REPO_ROOT / item
                    if path.is_dir():
                        shutil.copytree(path, ctx / item, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                    else:
                        shutil.copy(path, ctx / item)
                shutil.copy(Path(__file__).with_name("engine.Dockerfile"), ctx / "Dockerfile")
                torch_index = os.getenv("VRTX_TORCH_INDEX", "https://download.pytorch.org/whl/cpu")
                tag = f"vrtx/engine:{tree_hash(ctx, torch_index)[:12]}"
                if docker_image_exists(tag):
                    log.write(f"engine image {tag} already built, reusing it\n")
                    return {"kind": "image", "image": tag}, True
                log.write("building the VRTX Engine image (first time only, a few minutes)\n")
                run_logged([
                    "docker", "build", "--progress=plain", "--build-arg", f"TORCH_INDEX={torch_index}",
                    "-t", tag, str(ctx),
                ], log)
                return {"kind": "image", "image": tag}, False
