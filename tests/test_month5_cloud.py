"""Month 5: state store, spec, builder detection, placement, gateway routing, and a full
deploy -> serve -> crash -> roll out -> scale -> delete run on the process runtime (no Docker needed)."""
import io
import tarfile
import textwrap
import time

import httpx
import pytest

from cloud.api import server as cp_server
from cloud.builder.builder import BuildError, _cmd, detect
from cloud.scheduler.placement import place
from cloud.spec import SpecError, engine_env, normalize
from cloud.state import ClusterState
from gateway.proxy import deployment_from_host


# ------------------------------------------------------------------ units

def test_state_replays_to_the_same_result():
    """Commands carry their own ids and timestamps: replaying them gives identical state (what Raft needs)."""
    spec = normalize({"name": "web", "replicas": 2})
    cmds = [
        {"op": "node.register", "ts": 1.0, "node": {"id": "n1", "url": "http://n1", "runtime": "process", "capacity": {"cpus": 4}}},
        {"op": "deployment.put", "ts": 2.0, "spec": spec},
        {"op": "deployment.patch", "ts": 3.0, "name": "web", "fields": {"artifact": {"kind": "dir"}, "version": 1}},
        {"op": "replica.put", "ts": 4.0, "replica": {"id": "web-1", "deployment": "web", "version": 1, "node": "n1"}},
        {"op": "replica.patch", "ts": 5.0, "id": "web-1", "fields": {"state": "running", "port": 9000}},
        {"op": "deployment.patch", "ts": 6.0, "name": "web", "spec": {"replicas": 3}},
    ]
    a, b = ClusterState(), ClusterState()
    for c in cmds:
        a.apply(c)
        b.apply(c)
    assert a.deployments() == b.deployments() and a.replicas() == b.replicas() and a.nodes() == b.nodes()
    assert a.deployment("web")["spec"]["replicas"] == 3
    assert a.replica("web-1")["state"] == "running"


def test_state_rejects_unknown_fields_and_rolls_back():
    s = ClusterState()
    s.apply({"op": "deployment.put", "ts": 1.0, "spec": normalize({"name": "web"})})
    with pytest.raises(ValueError):
        s.apply({"op": "deployment.patch", "ts": 2.0, "name": "web", "fields": {"name": "evil"}})
    assert s.deployment("web") is not None


def test_spec_defaults_and_errors():
    app = normalize({"name": "todo-api"})
    assert app["type"] == "app" and app["replicas"] == 1 and app["port"] == 8000 and app["health"] == "/"
    model = normalize({"name": "qwen", "type": "model", "model": "Qwen/Qwen2.5-0.5B-Instruct", "quantization": "int8",
                       "speculative": {"draft": "d"}})
    assert model["health"] == "/health"
    assert engine_env(model) == {"VRTX_MODEL": "Qwen/Qwen2.5-0.5B-Instruct", "VRTX_QUANT": "8", "VRTX_DRAFT_MODEL": "d"}
    for bad in ({"name": "Bad_Name"}, {"name": "x", "replicas": -1}, {"name": "x", "type": "vm"},
                {"name": "x", "type": "model"}, {"name": "x", "colour": "red"}, {"name": "x", "health": "health"}):
        with pytest.raises(SpecError):
            normalize(bad)


def write(root, files):
    for name, text in files.items():
        (root / name).write_text(textwrap.dedent(text))


@pytest.mark.parametrize("files, kind, run", [
    ({"main.py": "from fastapi import FastAPI\napi = FastAPI()\n"}, "python",
     "python -m uvicorn main:api --host 0.0.0.0 --port $PORT"),
    ({"app.py": "import flask\napp = flask.Flask(__name__)\n", "requirements.txt": "flask\n"}, "python",
     "python -m flask --app app run --host 0.0.0.0 --port $PORT"),
    ({"bot.py": "if __name__ == '__main__':\n    pass\n"}, "python", "python bot.py"),
    ({"package.json": "{}"}, "node", "npm start"),
    ({"index.html": "<h1>hi</h1>"}, "static", "python -m http.server $PORT --bind 0.0.0.0"),
    ({"Dockerfile": "FROM scratch\n"}, "dockerfile", None),
    ({"index.html": "x", "Procfile": "web: ./serve --port $PORT\n"}, "static", "./serve --port $PORT"),
])
def test_detect(tmp_path, files, kind, run):
    write(tmp_path, files)
    plan = detect(tmp_path, normalize({"name": "x"}))
    assert (plan.kind, plan.run) == (kind, run)
    if plan.dockerfile and "uvicorn" in run:
        assert "pip install uvicorn" in plan.dockerfile  # not in requirements, so the builder adds it


def test_detect_nothing_runnable(tmp_path):
    write(tmp_path, {"README.md": "hello"})
    with pytest.raises(BuildError):
        detect(tmp_path, normalize({"name": "x"}))


def test_exec_only_for_simple_commands():
    assert '"exec python main.py"' in _cmd("python main.py")
    assert "exec" not in _cmd("cd src && python main.py")


def node(nid, runtime="process", gpu_mb=0, seen=100.0):
    return {"id": nid, "url": f"http://{nid}", "runtime": runtime, "last_seen": seen,
            "capacity": {"cpus": 8, "memory_mb": 16384, "gpu_memory_mb": gpu_mb}}


def test_placement():
    gpu_spec = normalize({"name": "m", "type": "model", "model": "x", "resources": {"gpu_memory_gb": 10}})
    engine = {"kind": "engine"}
    nodes = [node("cpu"), node("small-gpu", gpu_mb=8 * 1024), node("big-gpu", gpu_mb=24 * 1024)]
    chosen, _ = place(gpu_spec, engine, nodes, [], now=101.0)
    assert chosen["id"] == "big-gpu"
    # two 10 GB replicas already there: no room left anywhere
    taken = [{"node": "big-gpu", "deployment": "m", "resources": {"gpu_memory_gb": 10}}] * 2
    chosen, why = place(gpu_spec, engine, nodes, taken, now=101.0)
    assert chosen is None and "no node has room" in why
    # replicas of one app spread out
    app = normalize({"name": "web"})
    one = [{"node": "a", "deployment": "web", "resources": app["resources"]}]
    chosen, _ = place(app, {"kind": "dir"}, [node("a"), node("b")], one, now=101.0)
    assert chosen["id"] == "b"
    # an image needs a docker node; a silent node is skipped
    assert place(app, {"kind": "image"}, [node("a")], [], now=101.0)[0] is None
    assert place(app, {"kind": "dir"}, [node("a", seen=0.0)], [], now=101.0)[0] is None


def test_backoff_grows_and_caps():
    assert [cp_server.backoff(n) for n in (1, 2, 3, 4, 10)] == [0.0, 2.0, 4.0, 8.0, 60.0]


def test_host_routing():
    assert deployment_from_host("todo-api.vrtx.local:8080") == "todo-api"
    assert deployment_from_host("todo-api.localhost") == "todo-api"
    assert deployment_from_host("v1.qwen.vrtx.local") == "qwen"
    assert deployment_from_host("localhost:8080") is None
    assert deployment_from_host("example.com") is None


def test_pack_skips_junk(tmp_path):
    from cli.vrtx import pack

    write(tmp_path, {"main.py": "x", ".vrtxignore": "secrets.txt\n", "secrets.txt": "pw"})
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "big.js").write_text("x")
    with tarfile.open(fileobj=io.BytesIO(pack(tmp_path))) as tar:
        assert sorted(tar.getnames()) == [".vrtxignore", "main.py"]


# ------------------------------------------------------------ end to end

APP = '''
import os
from fastapi import FastAPI

app = FastAPI()
GREETING = "hello v1"


@app.get("/")
def index():
    return {"greeting": GREETING, "replica": os.environ["VRTX_REPLICA"], "port": os.environ["PORT"]}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/crash")
def crash():
    print("crashing on purpose", flush=True)
    os._exit(3)
'''


@pytest.fixture
def cloud(tmp_path, monkeypatch):
    from cloud import common
    from cloud.agent import agent as agent_mod
    from cloud.scheduler import placement
    from cloud.up import Stack

    monkeypatch.setattr(agent_mod, "HEARTBEAT_S", 0.3)
    monkeypatch.setattr(placement, "NODE_TIMEOUT_S", 5.0)
    monkeypatch.setattr(cp_server, "DRAIN_S", 1.5)  # must stay above the gateway's 1 s route refresh
    ports = [common.free_port() for _ in range(3)]
    stack = Stack(home=tmp_path / "home", runtime="process", api_port=ports[0], agent_port=ports[1],
                  gateway_port=ports[2], node_id="test-node")
    running = stack.start_in_thread()
    api = httpx.Client(base_url=stack.api_url, timeout=10)
    gw = httpx.Client(base_url=f"http://127.0.0.1:{ports[2]}", timeout=10)
    wait_for(lambda: api.get("/api/nodes").json() and api.get("/api/nodes").json()[0]["alive"], "node registers")
    yield api, gw
    api.close()
    gw.close()
    running.stop()


def wait_for(cond, what, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if cond():
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for: {what}")


def tarball(folder) -> bytes:
    from cli.vrtx import pack

    return pack(folder)


def deploy(api, folder, name="web", replicas=2):
    before = api.post("/api/deployments", json={"name": name, "replicas": replicas, "health": "/health"}).json()
    build = api.post(f"/api/deployments/{name}/builds", content=tarball(folder)).json()
    wait_for(lambda: api.get(f"/api/builds/{build['id']}").json()["status"] != "building", "build")
    assert api.get(f"/api/builds/{build['id']}").json()["status"] == "succeeded"
    return before


def status(api, name="web"):
    return api.get(f"/api/deployments/{name}").json()


def test_deploy_serve_crash_rollout_scale_delete(cloud, tmp_path):
    api, gw = cloud
    src = tmp_path / "web"
    src.mkdir()
    (src / "main.py").write_text(APP)
    host = {"host": "web.vrtx.local"}

    # deploy: 2 replicas, reachable through the gateway, round-robin
    deploy(api, src)
    wait_for(lambda: status(api)["status"] == "running", "2 healthy replicas")
    seen = {gw.get("/", headers=host).json()["replica"] for _ in range(6)}
    assert len(seen) == 2
    assert gw.get("/", headers={"host": "web.localhost"}).json()["greeting"] == "hello v1"
    assert gw.get("/", headers={"host": "nope.vrtx.local"}).status_code == 404

    # logs come back from every replica
    logs = api.get("/api/deployments/web/logs", params={"tail": 50}).text
    assert "Uvicorn running" in logs and len({line.split("]")[0] for line in logs.splitlines() if line}) == 2

    # a replica crashes: it is replaced and the deployment heals
    victim = gw.get("/crash", headers=host)
    assert victim.status_code in (502, 503) or victim.status_code >= 500
    wait_for(lambda: status(api)["failures"] >= 1, "crash noticed")
    wait_for(lambda: status(api)["status"] == "running", "replacement healthy")
    assert "exited with code 3" in status(api)["last_error"]

    # same code again: nothing to roll out
    assert status(api)["version"] == 1
    deploy(api, src)
    assert status(api)["version"] == 1

    # new code: rolling update to v2, no failed requests while it happens
    (src / "main.py").write_text(APP.replace("hello v1", "hello v2"))
    deploy(api, src)
    codes = []
    end = time.time() + 30
    while time.time() < end:
        codes.append(gw.get("/", headers=host).status_code)
        d = status(api)
        if d["status"] == "running" and d["version"] == 2:
            break
    assert status(api)["version"] == 2 and status(api)["status"] == "running"
    assert set(codes) == {200}, f"requests failed during the rollout: {codes}"
    # the gateway refreshes its routes every second, so give it one refresh to drop the old replicas
    wait_for(lambda: {gw.get("/", headers=host).json()["greeting"] for _ in range(4)} == {"hello v2"},
             "gateway serves only v2", timeout=5)

    # scale to 3, then 0
    api.patch("/api/deployments/web", json={"replicas": 3})
    wait_for(lambda: status(api)["replicas"]["running"] == 3, "3 replicas")
    api.patch("/api/deployments/web", json={"replicas": 0})
    wait_for(lambda: status(api)["status"] == "stopped" and status(api)["replicas"]["total"] == 0, "scaled to 0")
    wait_for(lambda: gw.get("/", headers=host).status_code == 503, "gateway sees no replicas")

    # delete
    api.delete("/api/deployments/web")
    wait_for(lambda: api.get("/api/deployments/web").status_code == 404, "deployment removed")


def test_crashloop_and_bad_build(cloud, tmp_path):
    api, _ = cloud
    src = tmp_path / "bad"
    src.mkdir()
    (src / "main.py").write_text("import sys\nprint('no config, giving up')\nsys.exit(1)\n")
    deploy(api, src, name="bad", replicas=1)
    wait_for(lambda: status(api, "bad")["status"] == "crashloop", "crashloop", timeout=40)
    assert "no config, giving up" in status(api, "bad")["last_error"]

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_text("nothing to run")
    api.post("/api/deployments", json={"name": "empty"})
    build = api.post("/api/deployments/empty/builds", content=tarball(empty)).json()
    wait_for(lambda: api.get(f"/api/builds/{build['id']}").json()["status"] == "failed", "build fails")
    assert status(api, "empty")["status"] == "build failed"
    assert "could not tell how to run" in api.get(f"/api/builds/{build['id']}/logs").text
