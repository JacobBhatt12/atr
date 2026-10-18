"""Worker agent: runs on every machine that hosts replicas.

    control plane --POST /replicas-->   agent   --docker run / spawn-->  replica
    control plane <--heartbeat (2 s)--  agent   --GET health path----->  replica

The agent starts and stops replicas when the control plane asks, health-checks them,
and reports node capacity (CPUs, RAM, GPU memory from nvidia-smi) plus every replica's
state in a heartbeat. The heartbeat answer lists replicas the control plane no longer
knows about, and the agent stops those, so a restarted control plane never leaks containers.
"""
from __future__ import annotations

import asyncio
import os
import platform
import shutil
import socket
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from cloud.agent.runtime import DockerRuntime, ProcessRuntime, docker_available
from cloud.common import HEARTBEAT_S, free_port

RUNNING_CHECK_S = 5.0
MAX_HEALTH_FAILURES = 3


def gpu_info() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    gpus = []
    for line in out.stdout.strip().splitlines() if out.returncode == 0 else []:
        idx, name, total, used = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(idx), "name": name, "memory_total_mb": int(total), "memory_used_mb": int(used)})
    return gpus


def total_memory_mb() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1 << 20)
    except (ValueError, OSError):
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        return int(out.stdout.strip() or 0) // (1 << 20)


def capacity(runtime: str) -> dict:
    gpus = gpu_info()
    return {
        "hostname": socket.gethostname(),
        "platform": f"{platform.system()} {platform.machine()}",
        "runtime": runtime,
        "cpus": os.cpu_count() or 1,
        "memory_mb": total_memory_mb(),
        "gpus": gpus,
        "gpu_memory_mb": sum(g["memory_total_mb"] for g in gpus),
    }


@dataclass
class Replica:
    id: str
    deployment: str
    spec: dict
    artifact: dict
    port: int
    url: str
    state: str = "starting"   # starting -> running -> failed | stopped
    error: str | None = None
    exit_code: int | None = None
    started_at: float = field(default_factory=time.time)
    last_check: float = 0.0
    health_failures: int = 0

    def report(self) -> dict:
        return {"id": self.id, "state": self.state, "port": self.port, "url": self.url,
                "error": self.error, "exit_code": self.exit_code}


class StartRequest(BaseModel):
    id: str
    deployment: str
    spec: dict
    artifact: dict
    env: dict = {}


class Agent:
    def __init__(self, control_plane: str, node_id: str, runtime: str, advertise_url: str, home: Path):
        self.cp = control_plane.rstrip("/")
        self.node_id = node_id
        self.advertise_url = advertise_url.rstrip("/")
        self.host = httpx.URL(advertise_url).host
        if runtime == "auto":
            runtime = "docker" if docker_available() else "process"
        self.capacity = capacity(runtime)
        if runtime == "docker":
            self.runtime = DockerRuntime(home, node_id, gpu=bool(self.capacity["gpus"]))
        else:
            self.runtime = ProcessRuntime(home, node_id)
        self.replicas: dict[str, Replica] = {}
        self.stopping: set[str] = set()
        self.registered = False
        self.client = httpx.AsyncClient(timeout=5.0)

    # ------------------------------------------------------------ lifecycle

    async def start_replica(self, req: StartRequest) -> Replica:
        if req.id in self.replicas:
            return self.replicas[req.id]
        port = free_port()
        rep = Replica(req.id, req.deployment, req.spec, req.artifact, port, f"http://{self.host}:{port}")
        env = {"VRTX_REPLICA": req.id, "VRTX_DEPLOYMENT": req.deployment, "VRTX_NODE": self.node_id} | req.env
        try:
            await asyncio.to_thread(self.runtime.start, req.id, req.artifact, req.spec, port, env)
        except Exception as e:
            rep.state, rep.error = "failed", f"could not start: {e}"
        self.replicas[req.id] = rep
        return rep

    async def stop_replica(self, rid: str, remove: bool = True) -> None:
        await asyncio.to_thread(self.runtime.stop, rid, remove)
        if remove:
            self.replicas.pop(rid, None)

    def stop_in_background(self, rid: str) -> None:
        """Graceful stops take seconds (SIGTERM, wait, SIGKILL); don't make the caller wait."""
        if rid in self.stopping or rid not in self.replicas:
            return
        self.stopping.add(rid)

        async def run():
            try:
                await self.stop_replica(rid)
            finally:
                self.stopping.discard(rid)

        asyncio.create_task(run())

    def fail(self, rep: Replica, reason: str) -> None:
        tail = self.runtime.tail(rep.id, 15)
        rep.state = "failed"
        rep.error = reason + ("\n" + "\n".join(tail) if tail else "")

    async def check(self, rep: Replica) -> None:
        if rep.state not in ("starting", "running") or rep.id in self.stopping:
            return
        code = await asyncio.to_thread(self.runtime.exit_code, rep.id)
        if code is not None:
            rep.exit_code = code
            self.fail(rep, f"exited with code {code}" if code >= 0 else "container disappeared")
            return
        now = time.time()
        if rep.state == "running" and now - rep.last_check < RUNNING_CHECK_S:
            return
        rep.last_check = now
        try:
            r = await self.client.get(f"http://127.0.0.1:{rep.port}{rep.spec['health']}", timeout=2.0)
            healthy = r.status_code < 500
        except httpx.HTTPError:
            healthy = False
        if rep.state == "starting":
            if healthy:
                rep.state = "running"
            elif now - rep.started_at > rep.spec["health_timeout_s"]:
                self.fail(rep, f"not healthy after {rep.spec['health_timeout_s']}s on {rep.spec['health']}")
                await self.stop_replica(rep.id, remove=False)
        elif healthy:
            rep.health_failures = 0
        else:
            rep.health_failures += 1
            if rep.health_failures >= MAX_HEALTH_FAILURES:
                self.fail(rep, f"health check {rep.spec['health']} failed {MAX_HEALTH_FAILURES} times")
                await self.stop_replica(rep.id, remove=False)

    async def health_loop(self) -> None:
        while True:
            await asyncio.gather(*(self.check(r) for r in list(self.replicas.values())), return_exceptions=True)
            await asyncio.sleep(0.5)

    # ---------------------------------------------------- control plane link

    async def register(self) -> None:
        body = {"id": self.node_id, "url": self.advertise_url, "runtime": self.runtime.name, "capacity": self.capacity}
        r = await self.client.post(f"{self.cp}/api/nodes", json=body)
        r.raise_for_status()
        self.registered = True

    def usage(self) -> dict:
        gpus = gpu_info() if self.capacity["gpus"] else []
        return {
            "load_1m": round(os.getloadavg()[0], 2),
            "replicas": len(self.replicas),
            "gpus": gpus,
            "gpu_memory_used_mb": sum(g["memory_used_mb"] for g in gpus),
        }

    async def heartbeat_loop(self) -> None:
        while True:
            try:
                if not self.registered:
                    await self.register()
                body = {"usage": await asyncio.to_thread(self.usage),
                        "replicas": [r.report() for r in self.replicas.values() if r.id not in self.stopping]}
                r = await self.client.post(f"{self.cp}/api/nodes/{self.node_id}/heartbeat", json=body)
                if r.status_code == 404:
                    self.registered = False  # control plane lost its state: register again
                else:
                    r.raise_for_status()
                    for rid in r.json().get("stop", []):
                        self.stop_in_background(rid)
            except httpx.HTTPError:
                pass  # control plane down; keep replicas running and retry
            await asyncio.sleep(HEARTBEAT_S)


def create_agent(control_plane: str, node_id: str | None = None, runtime: str = "auto",
                 advertise_url: str = "http://127.0.0.1:7071", home: Path | None = None) -> FastAPI:
    from cloud.common import vrtx_home

    agent = Agent(control_plane, node_id or socket.gethostname().split(".")[0].lower(), runtime,
                  advertise_url, home or vrtx_home())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await asyncio.to_thread(agent.runtime.cleanup_orphans)
        tasks = [asyncio.create_task(agent.health_loop()), asyncio.create_task(agent.heartbeat_loop())]
        yield
        for t in tasks:
            t.cancel()
        await asyncio.gather(*(agent.stop_replica(rid) for rid in list(agent.replicas)), return_exceptions=True)
        await agent.client.aclose()

    app = FastAPI(title="VRTX Agent", lifespan=lifespan)
    app.state.agent = agent

    @app.get("/status")
    def status():
        return {"node": agent.node_id, "runtime": agent.runtime.name, "capacity": agent.capacity,
                "usage": agent.usage(), "replicas": [r.report() for r in agent.replicas.values()]}

    @app.post("/replicas")
    async def start(req: StartRequest):
        return (await agent.start_replica(req)).report()

    @app.get("/replicas")
    def list_replicas():
        return [r.report() for r in agent.replicas.values()]

    @app.delete("/replicas/{rid}")
    async def stop(rid: str):
        agent.stop_in_background(rid)
        return {"stopping": rid}

    @app.get("/replicas/{rid}/logs")
    async def logs(rid: str, tail: int = 100, follow: bool = False):
        if rid not in agent.replicas:
            raise HTTPException(404, f"no replica {rid} on this node")

        async def lines():
            for line in await asyncio.to_thread(agent.runtime.tail, rid, tail):
                yield line + "\n"
            if follow:
                async for line in agent.runtime.follow(rid, lambda: rid in agent.replicas):
                    yield line + "\n"

        return StreamingResponse(lines(), media_type="text/plain")

    return app
