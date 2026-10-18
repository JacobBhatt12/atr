"""VRTX control plane: REST API + a reconcile loop (single node for Month 5).

The API only records what the user wants (a deployment spec, a new build). The reconcile
loop, every half second or right after a change, compares that with what the agents report
and closes the gap:

    failed / lost replica      -> remove it, count a failure, retry with exponential backoff
    fewer replicas than wanted -> place new ones (cloud/scheduler/placement.py), ask the agent to start them
    more than wanted           -> stop the extras
    new version released       -> rolling update: old replicas keep serving until enough
                                  new ones pass their health check, then the old ones drain
                                  (out of the routing table first, stopped 3 s later)
    deployment deleted         -> stop everything, then forget it
    node silent for 10 s       -> its replicas are lost and get rescheduled

All writes go through ClusterState.apply, so Month 6 can put them on the Raft log unchanged.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from cloud.agent.runtime import docker_available
from cloud.builder.builder import Builder, BuildError
from cloud.common import DOMAIN, new_id, now, vrtx_home
from cloud.scheduler.placement import node_alive, place, reserved
from cloud.spec import SpecError, normalize
from cloud.state import ClusterState

RECONCILE_S = 0.5
CRASHLOOP_AFTER = 3          # failures in a row before the status says crashloop
STABLE_AFTER_S = 60          # a replica up this long resets the failure count
SCHEDULED_GRACE_S = 15       # how long a replica may be missing from heartbeats right after start
DRAIN_S = 3.0                # a replica leaves the routing table this long before it is stopped
                             # (must be longer than gateway/proxy.py's REFRESH_S)
RELEASE_KEYS = ("port", "health", "health_timeout_s", "run", "env", "resources",
                "model", "quantization", "speculative", "kv_cache_mb", "max_seqs")


def backoff(failures: int) -> float:
    return 0.0 if failures <= 1 else min(60.0, 2.0 ** (failures - 1))


def fingerprint(artifact: dict, spec: dict) -> str:
    core = {k: v for k, v in artifact.items() if k != "fp"} | {"spec": {k: spec.get(k) for k in RELEASE_KEYS}}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]


class NodeIn(BaseModel):
    id: str
    url: str
    runtime: str
    capacity: dict


class Heartbeat(BaseModel):
    usage: dict = {}
    replicas: list[dict] = []


class Patch(BaseModel):
    replicas: int | None = None
    env: dict[str, str] | None = None


class ControlPlane:
    def __init__(self, home: Path, runtime: str = "auto", gateway_port: int = 8080):
        if runtime == "auto":
            runtime = "docker" if docker_available() else "process"
        self.home = home
        home.mkdir(parents=True, exist_ok=True)
        self.state = ClusterState(str(home / "state.db"))
        self.builder = Builder(home, runtime)
        self.gateway_port = gateway_port
        self.client = httpx.AsyncClient(timeout=10.0)
        self.kick_event: asyncio.Event | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------- helpers

    def kick(self) -> None:
        """Run the reconcile loop now instead of at the next tick (safe from any thread)."""
        if self.loop and self.kick_event:
            self.loop.call_soon_threadsafe(self.kick_event.set)

    def url_for(self, name: str) -> str:
        port = "" if self.gateway_port == 80 else f":{self.gateway_port}"
        return f"http://{name}.{DOMAIN}{port}"

    def release(self, name: str, artifact: dict) -> bool:
        """Make `artifact` + the current spec the deployment's live version. False if nothing changed."""
        d = self.state.deployment(name)
        fp = fingerprint(artifact, d["spec"])
        if d["artifact"] and d["artifact"].get("fp") == fp:
            return False
        self.state.apply({"op": "deployment.patch", "name": name, "ts": now(), "fields": {
            "artifact": artifact | {"fp": fp}, "version": d["version"] + 1,
            "failures": 0, "next_attempt_at": 0, "last_error": None,
        }})
        self.kick()
        return True

    def start_build(self, name: str, source: bytes | None) -> dict:
        build_id = new_id("build")
        self.state.apply({"op": "build.put", "ts": now(), "build": {
            "id": build_id, "deployment": name,
            "source_hash": hashlib.sha256(source).hexdigest()[:12] if source else None,
        }})
        self.state.apply({"op": "deployment.patch", "name": name, "ts": now(), "fields": {"build_id": build_id}})
        threading.Thread(target=self._build, args=(build_id, name, source), daemon=True).start()
        return self.state.build(build_id)

    def _build(self, build_id: str, name: str, source: bytes | None) -> None:
        spec = self.state.deployment(name)["spec"]
        try:
            if spec["type"] == "model":
                artifact, cached = self.builder.build_engine(build_id)
            else:
                artifact, cached = self.builder.build_app(build_id, spec, source)
        except (BuildError, OSError) as e:
            with open(self.builder.log_path(build_id), "a") as log:
                log.write(f"build failed: {e}\n")
            self.state.apply({"op": "build.patch", "id": build_id, "fields": {
                "status": "failed", "error": str(e), "finished_at": now()}})
            self.state.apply({"op": "deployment.patch", "name": name, "ts": now(),
                              "fields": {"last_error": f"build failed: {e}"}})
            return
        if self.state.deployment(name) is None:
            return  # deleted while building
        changed = self.release(name, artifact)
        with open(self.builder.log_path(build_id), "a") as log:
            log.write("released as a new version\n" if changed else "same code and config as the live version, nothing to roll out\n")
        self.state.apply({"op": "build.patch", "id": build_id, "fields": {
            "status": "succeeded", "artifact": artifact, "cached": int(cached), "finished_at": now()}})

    # --------------------------------------------------------- reconciling

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.kick_event = asyncio.Event()
        while True:
            try:
                await self.reconcile()
            except Exception as e:  # never let one bad tick kill the loop
                print(f"[control-plane] reconcile error: {e!r}")
            try:
                await asyncio.wait_for(self.kick_event.wait(), RECONCILE_S)
            except asyncio.TimeoutError:
                pass
            self.kick_event.clear()

    async def reconcile(self) -> None:
        ts = now()
        nodes = {n["id"]: n for n in self.state.nodes()}
        for r in self.state.replicas():
            node = nodes.get(r["node"])
            if r["state"] not in ("lost", "stopping") and (node is None or not node_alive(node, ts)):
                self.state.apply({"op": "replica.patch", "id": r["id"], "ts": ts, "fields": {
                    "state": "lost", "error": f"node {r['node']} stopped responding"}})
        for d in self.state.deployments():
            await self.reconcile_deployment(d, nodes)

    async def stop(self, r: dict, nodes: dict) -> None:
        node = nodes.get(r["node"])
        if node and node_alive(node, now()):
            try:
                resp = await self.client.delete(f"{node['url']}/replicas/{r['id']}")
                if resp.status_code >= 500:
                    raise httpx.HTTPError(resp.text)
            except httpx.HTTPError:
                if r["state"] != "stopping":
                    self.state.apply({"op": "replica.patch", "id": r["id"], "ts": now(), "fields": {"state": "stopping"}})
                return  # retry next tick
        self.state.apply({"op": "replica.remove", "id": r["id"]})

    async def drain(self, r: dict, nodes: dict) -> None:
        """Take a serving replica out of the routing table now and stop it DRAIN_S later, once
        every gateway has refreshed its routes and in-flight requests have finished."""
        if r["state"] == "running":
            self.state.apply({"op": "replica.patch", "id": r["id"], "ts": now(), "fields": {"state": "draining"}})
        else:
            await self.stop(r, nodes)

    async def reconcile_deployment(self, d: dict, nodes: dict) -> None:
        name, spec, ts = d["name"], d["spec"], now()
        mine = self.state.replicas(deployment=name)

        if d["deleting"]:
            for r in mine:
                await self.stop(r, nodes)
            if not self.state.replicas(deployment=name):
                self.state.apply({"op": "deployment.remove", "name": name})
            return

        for r in [r for r in mine if r["state"] in ("failed", "lost", "stopping")]:
            if r["state"] != "stopping" and r["version"] == d["version"]:
                failures = d["failures"] + 1
                d |= {"failures": failures, "next_attempt_at": ts + backoff(failures), "last_error": r["error"]}
                self.state.apply({"op": "deployment.patch", "name": name, "ts": ts, "fields": {
                    k: d[k] for k in ("failures", "next_attempt_at", "last_error")}})
            await self.stop(r, nodes)
        for r in [r for r in mine if r["state"] == "draining" and ts - r["updated_at"] >= DRAIN_S]:
            await self.stop(r, nodes)
        live = [r for r in self.state.replicas(deployment=name)
                if r["state"] not in ("failed", "lost", "stopping", "draining")]

        if d["artifact"] is None:
            return  # first build still running (or it failed)
        current = [r for r in live if r["version"] == d["version"]]
        old = [r for r in live if r["version"] != d["version"]]
        running = [r for r in current if r["state"] == "running"]
        desired = spec["replicas"]

        if d["failures"] and any(ts - (r["running_since"] or ts) >= STABLE_AFTER_S for r in running):
            self.state.apply({"op": "deployment.patch", "name": name, "ts": ts,
                              "fields": {"failures": 0, "last_error": None}})

        if len(current) > desired:
            extras = sorted(current, key=lambda r: (r["state"] == "running", r["created_at"]))
            for r in extras[: len(current) - desired]:
                await self.drain(r, nodes)
        elif len(current) < desired and ts >= d["next_attempt_at"]:
            await self.scale_up(d, desired - len(current), nodes)

        if old and (len(running) >= desired):
            for r in old:
                await self.drain(r, nodes)

    async def scale_up(self, d: dict, count: int, nodes: dict) -> None:
        all_replicas = [r for r in self.state.replicas() if r["state"] not in ("failed", "lost")]
        for _ in range(count):
            node, reason = place(d["spec"], d["artifact"], list(nodes.values()), all_replicas, now())
            if node is None:
                msg = f"unschedulable: {reason}"
                if d["last_error"] != msg:
                    self.state.apply({"op": "deployment.patch", "name": d["name"], "ts": now(), "fields": {"last_error": msg}})
                return
            rid = new_id(d["name"])
            replica = {"id": rid, "deployment": d["name"], "version": d["version"], "node": node["id"],
                       "resources": d["spec"]["resources"]}
            self.state.apply({"op": "replica.put", "ts": now(), "replica": replica})
            all_replicas.append(self.state.replica(rid))
            if d["last_error"] and d["last_error"].startswith("unschedulable"):
                self.state.apply({"op": "deployment.patch", "name": d["name"], "ts": now(), "fields": {"last_error": None}})
            artifact = {k: v for k, v in d["artifact"].items() if k != "fp"}
            body = {"id": rid, "deployment": d["name"], "spec": d["spec"], "artifact": artifact,
                    "env": {"VRTX_VERSION": str(d["version"])}}
            try:
                resp = await self.client.post(f"{node['url']}/replicas", json=body)
                resp.raise_for_status()
                rep = resp.json()
                fields = {"port": rep["port"], "url": rep["url"]}
                if rep["state"] == "failed":
                    fields |= {"state": "failed", "error": rep["error"]}
                elif self.state.replica(rid)["state"] == "scheduled":
                    fields["state"] = "starting"
            except httpx.HTTPError as e:
                fields = {"state": "failed", "error": f"agent on {node['id']} refused the replica: {e}"}
            self.state.apply({"op": "replica.patch", "id": rid, "ts": now(), "fields": fields})

    # ----------------------------------------------------------- reporting

    def heartbeat(self, node_id: str, hb: Heartbeat) -> dict:
        ts = now()
        self.state.apply({"op": "node.seen", "id": node_id, "usage": hb.usage, "ts": ts})
        known = {r["id"]: r for r in self.state.replicas(node=node_id)}
        stop, changed = [], False
        for rep in hb.replicas:
            r = known.pop(rep["id"], None)
            if r is None:
                stop.append(rep["id"])
                continue
            if r["state"] not in ("scheduled", "starting", "running"):
                continue  # failed / lost / stopping are final from the control plane's side
            fields = {k: rep[k] for k in ("state", "port", "url", "error", "exit_code") if rep.get(k) != r[k]}
            if fields.get("state") == "running":
                fields["running_since"] = ts
            if fields:
                changed = True
                self.state.apply({"op": "replica.patch", "id": r["id"], "ts": ts, "fields": fields})
        for r in known.values():  # the agent does not have these
            if r["state"] in ("starting", "running") or (r["state"] == "scheduled" and ts - r["created_at"] > SCHEDULED_GRACE_S):
                changed = True
                self.state.apply({"op": "replica.patch", "id": r["id"], "ts": ts, "fields": {
                    "state": "failed", "error": f"replica is gone from node {node_id} (agent restarted?)"}})
        if changed:
            self.kick()
        return {"stop": stop}

    def describe(self, d: dict) -> dict:
        reps = self.state.replicas(deployment=d["name"])
        build = self.state.build(d["build_id"]) if d["build_id"] else None
        current = [r for r in reps if r["version"] == d["version"]]
        running = sum(1 for r in current if r["state"] == "running")
        desired = d["spec"]["replicas"]
        err = d["last_error"] or ""
        if d["deleting"]:
            status = "deleting"
        elif build and build["status"] == "building":
            status = "building"
        elif d["artifact"] is None:
            status = "build failed" if build and build["status"] == "failed" else "waiting for build"
        elif desired == 0:
            status = "stopped"
        elif running >= desired and all(r["version"] == d["version"] for r in reps):
            status = "running"
        elif d["failures"] >= CRASHLOOP_AFTER:
            status = "crashloop"
        elif err.startswith("unschedulable"):
            status = "pending"
        else:
            status = "deploying"
        return {
            "name": d["name"], "type": d["type"], "status": status, "version": d["version"],
            "replicas": {"desired": desired, "running": running, "total": len(reps)},
            "url": self.url_for(d["name"]), "artifact": d["artifact"], "spec": d["spec"],
            "failures": d["failures"], "last_error": d["last_error"],
            "build": build and {k: build[k] for k in ("id", "status", "error", "cached")},
            "instances": [{k: r[k] for k in ("id", "version", "node", "state", "url", "error")} for r in reps],
        }


def create_control_plane(home: Path | None = None, runtime: str = "auto", gateway_port: int = 8080) -> FastAPI:
    cp = ControlPlane(home or vrtx_home(), runtime, gateway_port)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(cp.run())
        yield
        task.cancel()
        await cp.client.aclose()

    app = FastAPI(title="VRTX Control Plane", version="0.5.0", lifespan=lifespan)
    app.state.cp = cp

    def get(name: str) -> dict:
        d = cp.state.deployment(name)
        if d is None:
            raise HTTPException(404, f"no deployment named {name!r}")
        return d

    @app.get("/health")
    def health():
        return {"status": "ok", "builder_runtime": cp.builder.runtime}

    @app.get("/api/cluster")
    def cluster():
        # Month 6 fills this in from Raft: leader, term, commit index, peers
        return {"mode": "single node", "leader": "self", "term": 0, "commit_index": None,
                "nodes": len(cp.state.nodes()), "deployments": len(cp.state.deployments())}

    @app.post("/api/deployments", status_code=201)
    async def create(body: dict):
        try:
            spec = normalize(body)
        except SpecError as e:
            raise HTTPException(422, str(e))
        existing = cp.state.deployment(spec["name"])
        if existing and existing["type"] != spec["type"]:
            raise HTTPException(409, f"{spec['name']} is already a {existing['type']} deployment")
        cp.state.apply({"op": "deployment.put", "spec": spec, "ts": now()})
        if spec["type"] == "model":
            cp.start_build(spec["name"], None)  # engine image (cached after the first time)
        cp.kick()
        return cp.describe(cp.state.deployment(spec["name"]))

    @app.get("/api/deployments")
    def list_deployments():
        return [cp.describe(d) for d in cp.state.deployments()]

    @app.get("/api/deployments/{name}")
    def show(name: str):
        return cp.describe(get(name))

    @app.patch("/api/deployments/{name}")
    def patch(name: str, body: Patch):
        d = get(name)
        change = body.model_dump(exclude_none=True)
        try:
            normalize(d["spec"] | change)
        except SpecError as e:
            raise HTTPException(422, str(e))
        cp.state.apply({"op": "deployment.patch", "name": name, "ts": now(), "spec": change,
                        "fields": {"failures": 0, "next_attempt_at": 0}})
        if "env" in change and d["artifact"]:
            cp.release(name, {k: v for k, v in d["artifact"].items() if k != "fp"})
        cp.kick()
        return cp.describe(get(name))

    @app.delete("/api/deployments/{name}")
    def delete(name: str):
        get(name)
        cp.state.apply({"op": "deployment.patch", "name": name, "ts": now(), "fields": {"deleting": 1}})
        cp.kick()
        return {"deleting": name}

    @app.post("/api/deployments/{name}/builds", status_code=202)
    async def upload(name: str, request: Request):
        d = get(name)
        if d["type"] != "app":
            raise HTTPException(400, "model deployments are built from the engine, not from uploaded code")
        source = await request.body()
        if not source:
            raise HTTPException(400, "empty upload: send a .tar.gz of the app folder")
        return cp.start_build(name, source)

    @app.get("/api/builds/{build_id}")
    def build(build_id: str):
        b = cp.state.build(build_id)
        if b is None:
            raise HTTPException(404, "no such build")
        return b

    @app.get("/api/builds/{build_id}/logs")
    async def build_logs(build_id: str, follow: bool = False):
        if cp.state.build(build_id) is None:
            raise HTTPException(404, "no such build")
        path = cp.builder.log_path(build_id)

        async def lines():
            pos = 0
            while True:
                done = cp.state.build(build_id)["status"] != "building"
                if path.exists():
                    with open(path) as f:
                        f.seek(pos)
                        chunk = f.read()
                        pos = f.tell()
                    if chunk:
                        yield chunk
                if done or not follow:
                    return
                await asyncio.sleep(0.25)

        return StreamingResponse(lines(), media_type="text/plain")

    @app.get("/api/deployments/{name}/logs")
    async def logs(name: str, tail: int = 100, follow: bool = False):
        d = get(name)
        queue: asyncio.Queue = asyncio.Queue()
        attached: set[str] = set()

        async def pump(r: dict):
            node = cp.state.node(r["node"])
            prefix = f"[{r['id'][len(name) + 1:]}] "
            try:
                async with cp.client.stream("GET", f"{node['url']}/replicas/{r['id']}/logs",
                                            params={"tail": tail, "follow": follow}, timeout=None) as resp:
                    async for line in resp.aiter_lines():
                        await queue.put(prefix + line + "\n")
            except httpx.HTTPError:
                pass
            await queue.put(None)

        def attach_new() -> int:
            n = 0
            for r in cp.state.replicas(deployment=name):
                if r["id"] not in attached and r["state"] in ("starting", "running", "failed") and r["url"]:
                    attached.add(r["id"])
                    asyncio.create_task(pump(r))
                    n += 1
            return n

        async def lines():
            healthy = any(r["state"] == "running" for r in cp.state.replicas(deployment=name))
            if d["last_error"] and not healthy:
                yield "".join(f"[last error] {line}\n" for line in d["last_error"].splitlines())
            open_streams = attach_new()
            while open_streams or follow:
                try:
                    item = await asyncio.wait_for(queue.get(), 2.0)
                except asyncio.TimeoutError:
                    if follow:
                        open_streams += attach_new()  # pick up replicas started since (rollouts, restarts)
                    continue
                if item is None:
                    open_streams -= 1
                else:
                    yield item

        return StreamingResponse(lines(), media_type="text/plain")

    @app.post("/api/nodes")
    def register(node: NodeIn):
        cp.state.apply({"op": "node.register", "node": node.model_dump(), "ts": now()})
        cp.kick()
        return {"registered": node.id}

    @app.post("/api/nodes/{node_id}/heartbeat")
    def heartbeat(node_id: str, hb: Heartbeat):
        if cp.state.node(node_id) is None:
            raise HTTPException(404, "unknown node, register first")
        return cp.heartbeat(node_id, hb)

    @app.get("/api/nodes")
    def nodes():
        ts, replicas = now(), cp.state.replicas()
        return [{
            "id": n["id"], "url": n["url"], "runtime": n["runtime"], "alive": node_alive(n, ts),
            "last_seen_s": round(ts - n["last_seen"], 1), "capacity": n["capacity"], "usage": n["usage"],
            "replicas": sum(1 for r in replicas if r["node"] == n["id"]),
            "reserved": reserved(n["id"], replicas),
        } for n in cp.state.nodes()]

    @app.get("/api/routes")
    def routes():
        """What the gateway proxies to: every healthy replica of every deployment."""
        out = {}
        for d in cp.state.deployments():
            if d["deleting"]:
                continue
            backends = [r["url"] for r in cp.state.replicas(deployment=d["name"]) if r["state"] == "running"]
            out[d["name"]] = {"type": d["type"], "backends": backends}
        return {"domain": DOMAIN, "routes": out}

    return app
