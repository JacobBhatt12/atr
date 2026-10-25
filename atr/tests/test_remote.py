"""The Streamable HTTP transport. No network beyond 127.0.0.1, nothing billed.

    python -m pytest atr/tests -c /dev/null
"""
from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from atr.config import Config  # noqa: E402
from atr.remote import create_remote  # noqa: E402
from atr.server import ATRServer  # noqa: E402

FAKE = {"fake": {"command": sys.executable, "args": ["-m", "atr.tests.fake_server"]}}


def rpc(mid, method, params=None):
    return {"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}


INIT = rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})


def test_http_transport_rules():
    app = create_remote(Config(servers=FAKE), token="s3cret")

    async def go():
        await app.state.atr.start()
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://atr") as c:
                assert (await c.post("/mcp", json=INIT)).status_code == 401
                assert (await c.post("/mcp/wrong", json=INIT)).status_code == 401
                assert (await c.post("/mcp/s3cret", json=INIT, headers={"origin": "https://evil.example"})).status_code == 403

                r = await c.post("/mcp/s3cret", json=INIT, headers={"origin": "https://claude.ai"})
                assert r.status_code == 200 and r.json()["result"]["serverInfo"]["name"] == "atr"
                sid = r.headers["mcp-session-id"]
                bearer = {"authorization": "Bearer s3cret"}

                assert (await c.post("/mcp", json=rpc(2, "tools/list"), headers=bearer)).status_code == 400
                assert (await c.post("/mcp", json=rpc(2, "tools/list"), headers=bearer | {"mcp-session-id": "nope"})).status_code == 404
                h = bearer | {"mcp-session-id": sid}
                note = await c.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=h)
                assert note.status_code == 202

                names = [t["name"] for t in (await c.post("/mcp", json=rpc(2, "tools/list"), headers=h)).json()["result"]["tools"]]
                assert "atr_find_tools" in names and "atr_read" in names
                assert not {"atr_read_file", "atr_repo_map", "atr_search_code", "atr_run"} & set(names)  # off over HTTP

                batch = await c.post("/mcp", json=[rpc(3, "ping"), rpc(4, "tools/call", {"name": "atr_find_tools", "arguments": {"query": "send slack message"}})], headers=h)
                out = batch.json()
                assert [m["id"] for m in out] == [3, 4] and "fake__slack_send_message" in out[1]["result"]["content"][0]["text"]

                blocked = await c.post("/mcp", json=rpc(5, "tools/call", {"name": "atr_read_file", "arguments": {"path": "/etc/hosts"}}), headers=h)
                assert blocked.json()["result"]["isError"]
                assert (await c.get("/mcp", headers=h)).status_code == 405
                assert (await c.delete("/mcp", headers=h)).status_code == 204
                assert (await c.post("/mcp", json=rpc(6, "ping"), headers=h)).status_code == 404
        finally:
            await app.state.atr.close()
    asyncio.run(go())


def test_end_to_end_atr_behind_remote_atr():
    """Real sockets: outer ATR --HTTP--> remote ATR --stdio--> fake 50-tool server."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    remote = create_remote(Config(servers=FAKE, expose="all"), token="tok")
    server = uvicorn.Server(uvicorn.Config(remote, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started

    async def go():
        outer = ATRServer(Config(servers={"remote": {"url": f"http://127.0.0.1:{port}/mcp/tok"}}))
        await outer.start()
        try:
            assert outer.downstreams["remote"].error is None
            assert "remote__fake__github_list_issues" in outer.tools
            found = (await outer.call("atr_find_tools", {"query": "list github issues"}))["content"][0]["text"]
            assert "remote__fake__github_list_issues(" in found
            res = await outer.call("atr_call", {"tool": "remote__fake__github_list_issues", "arguments": {"owner": "a", "repo": "b"}})
            # read-only marks travel through a remote ATR too
            assert outer.is_read_only("remote__fake__github_list_issues")
            assert not outer.is_read_only("remote__fake__github_create_issue")
            assert not res["isError"] and "rows" in res["content"][0]["text"]
        finally:
            await outer.close()
    try:
        asyncio.run(go())
    finally:
        server.should_exit = True
        t.join(10)
