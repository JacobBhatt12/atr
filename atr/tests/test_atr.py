"""ATR tests. No GPU, no network, no dependencies beyond pytest.

    python -m pytest atr/tests -q
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest

from atr import tokens
from atr.compress import compress
from atr.config import Config
from atr.router import ToolIndex, signature, slim_schema
from atr.server import ATRServer
from atr.stash import query
from atr.tests import fake_server

FAKE = {"fake": {"command": sys.executable, "args": ["-m", "atr.tests.fake_server"]}}


# ------------------------------------------------------------------ compression

def test_prunes_nulls_and_noise_and_keeps_values():
    r = compress(json.dumps({"id": 7, "name": "x", "node_id": "abc", "body": None, "tags": [], "meta": {}}))
    assert "id: 7" in r.text and "name: x" in r.text
    for gone in ("node_id", "body", "tags", "meta", "null"):
        assert gone not in r.text
    assert not r.lossy


def test_list_of_objects_becomes_table_with_constant_columns_lifted():
    rows = [{"id": i, "kind": "user", "name": f"n{i}"} for i in range(5)]
    r = compress(json.dumps(rows, indent=2))
    lines = r.text.splitlines()
    assert lines[0] == "[5 rows; all rows: kind=user]"
    assert lines[1] == "id,name" and lines[2] == "0,n0"
    assert r.after < r.before / 2


def test_big_payload_meets_budget_and_marks_loss():
    data = fake_server.issues(1000)
    raw = json.dumps(data, indent=2)
    r = compress(raw, budget=1500)
    assert r.before > 100_000
    assert r.after <= 1500
    assert r.lossy and "more rows" in r.text
    assert r.saved_pct > 98


def test_never_makes_small_input_bigger():
    for s in ['{"a":1}', "ok", "[1,2,3]"]:
        assert compress(s).after <= tokens.count(s)


def test_logs_collapse_repeats_and_keep_the_error():
    r = compress(fake_server.logs(2000), budget=400)
    assert "similar lines" in r.text
    assert "ConnectionResetError" in r.text
    assert '\n  File "app/handler.py"' in r.text  # indentation survives
    assert r.after <= 400


def test_html_becomes_text():
    html = fake_server.call("web_fetch_url", {})["content"][0]["text"]
    r = compress(html)
    assert "<div" not in r.text and "var x" not in r.text and "color:red" not in r.text
    assert "Paragraph 0 of the article." in r.text


def test_json_lines_are_parsed():
    raw = "\n".join(json.dumps({"id": i, "ok": True}) for i in range(5))
    assert compress(raw).text.startswith("[5 rows; all rows: ok=true]")


# ------------------------------------------------------------------ stash queries

def test_query_filters_fields_and_groups():
    raw = json.dumps({"items": fake_server.issues(200)})
    res = query(raw, path="items", where=["state=open", "comments>8"], fields=["number", "user.login"], limit=5)
    assert res["returned"] <= 5 and res["total"] > 0
    assert set(res["rows"][0]) == {"number", "user.login"}
    g = query(raw, path="items", group_by="state")
    assert set(g["counts"]) == {"open", "closed"} and sum(g["counts"].values()) == 200


def test_query_text_search():
    res = query(fake_server.logs(500), search="error|traceback")
    assert res["matches"] == 3 and "ERROR" in res["lines"][0]


# ------------------------------------------------------------------ tool routing

def test_router_finds_the_right_tools():
    idx = ToolIndex({t["name"]: t for t in fake_server.tools()})
    assert idx.search("open a new github issue", 3)[0][0] == "github_create_issue"
    assert idx.search("post a message in slack", 3)[0][0] == "slack_send_message"
    assert "logs_rollback_deploy" in [n for n, _ in idx.search("roll back the last deploy", 3)]
    assert idx.search("run sql", 3)[0][0] == "db_query"


def test_slim_schema_and_signature():
    t = fake_server.tools()[0]
    slim = slim_schema(t["inputSchema"])
    assert "$schema" not in slim and "title" not in slim["properties"]["owner"]
    assert slim["required"] == ["owner", "repo"]
    assert signature("github_create_issue", t).startswith("github_create_issue(owner*:str, repo*:str, limit:int")


# ------------------------------------------------------------------ the server end to end

def run(coro):
    return asyncio.run(coro)


def test_server_hides_schemas_and_compresses_calls():
    async def go():
        srv = ATRServer(Config(servers=FAKE, budget=1500))
        await srv.start()
        try:
            assert len(srv.tools) == 50  # both pages of tools/list
            listed = (await srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}))["result"]["tools"]
            assert [t["name"] for t in listed] == ["atr_find_tools", "atr_read", "atr_call", "atr_query", "atr_read_file",
                                                   "atr_repo_map", "atr_search_code", "atr_diff", "atr_run", "atr_stats"]
            ann = {t["name"]: t["annotations"]["readOnlyHint"] for t in listed}
            assert [n for n, ro in ann.items() if not ro] == ["atr_call", "atr_run"]
            assert srv.stats["schema_exposed"] < srv.stats["schema_full"] / 5
            find = next(t for t in listed if t["name"] == "atr_find_tools")["description"]
            assert "Connected: fake (github" in find and "slack" in find  # the model knows what is behind ATR

            found = await srv.call("atr_find_tools", {"query": "list github issues"})
            assert "[r] fake__github_list_issues(" in found["content"][0]["text"]

            res = await srv.call("atr_read", {"tool": "fake__github_list_issues", "arguments": {"owner": "a", "repo": "b"}})
            text = res["content"][0]["text"]
            assert tokens.count(text) < 1700 and 'ref="r1"' in text
            assert "structuredContent" not in res

            q = await srv.call("atr_query", {"ref": "r1", "where": ["number=42"], "fields": ["title", "state"]})
            assert "Bug 42" in q["content"][0]["text"]

            bad = await srv.call("atr_call", {"tool": "fake__github_list_issue"})
            assert bad["isError"] and "github_list_issues" in bad["content"][0]["text"]
            assert "saved" in (await srv.call("atr_stats", {}))["content"][0]["text"]
        finally:
            await srv.close()
    run(go())


def test_pinned_tools_are_exposed_directly():
    async def go():
        srv = ATRServer(Config(servers=FAKE, pin=["fake__github_create_issue"]))
        await srv.start()
        try:
            names = [t["name"] for t in srv.list_tools()]
            assert "fake__github_create_issue" in names and len(names) == 11
            res = await srv.call("fake__github_create_issue", {"owner": "a", "repo": "b", "filter": "hi"})
            assert "number: 1001" in res["content"][0]["text"]
        finally:
            await srv.close()
    run(go())


def test_read_write_split():
    async def go():
        srv = ATRServer(Config(servers=FAKE, read_only=["fake__slack_send_*"], write=["fake__github_get_*"]))
        await srv.start()
        try:
            text = (await srv.call("atr_find_tools", {"query": "create github issue", "k": 3}))["content"][0]["text"]
            assert "[w] fake__github_create_issue(" in text and "[r]: run with atr_read" in text
            # a tool that changes things can't sneak through the no-approval tool
            blocked = await srv.call("atr_read", {"tool": "fake__github_create_issue", "arguments": {}})
            assert blocked["isError"] and "atr_call" in blocked["content"][0]["text"]
            ok = await srv.call("atr_call", {"tool": "fake__github_create_issue", "arguments": {"filter": "x"}})
            assert not ok["isError"]
            # atr_call can also run read-only tools (it just asks first)
            assert not (await srv.call("atr_call", {"tool": "fake__github_list_issues"}))["isError"]
            assert srv.is_read_only("fake__logs_tail_logs")             # marked by the server
            assert not srv.is_read_only("fake__logs_rollback_deploy")   # destructive
            assert srv.is_read_only("fake__slack_send_message")         # config read_only wins over the server
            assert not srv.is_read_only("fake__github_get_issue")       # config write wins over everything
        finally:
            await srv.close()
    run(go())


def test_unmarked_tools_need_approval():
    srv = ATRServer(Config())
    srv.tools = {"x__thing": (None, {"name": "thing"}), "x__risky": (None, {"name": "risky", "annotations":
                 {"readOnlyHint": True, "destructiveHint": True}})}
    assert not srv.is_read_only("x__thing") and not srv.is_read_only("x__risky")


def test_read_file_outlines_big_code(tmp_path):
    src = tmp_path / "big.py"
    src.write_text("\n".join(f"def f{i}(x):\n    return x + {i}  # " + "pad " * 20 for i in range(300)))
    srv = ATRServer(Config(read_roots=[str(tmp_path)], budget=800))
    text = srv.read_file(str(src), None, None)["content"][0]["text"]
    assert "Outline" in text and "1: def f0(x):" in text and tokens.count(text) < 1000
    assert srv.read_file("/etc/hosts", None, None)["isError"]  # outside read_roots
    hit = srv.read_file(str(src), "def f299", None)["content"][0]["text"]
    assert "1 matching lines" in hit


def test_stdio_handshake():
    """Talk to `python -m atr serve` the way an MCP client does."""
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
    p = subprocess.run([sys.executable, "-c", "import asyncio;from atr.server import serve;from atr.config import Config;"
                        "asyncio.run(serve(Config()))"],
                       input="\n".join(json.dumps(m) for m in msgs) + "\n", capture_output=True, text=True, timeout=30)
    out = [json.loads(ln) for ln in p.stdout.splitlines()]
    assert out[0]["result"]["serverInfo"]["name"] == "atr"
    assert out[0]["result"]["protocolVersion"] == "2025-06-18"
    assert len(out[1]["result"]["tools"]) == 10
    assert "Connected" not in out[1]["result"]["tools"][0]["description"]  # nothing connected, no line



def test_core_tools_always_load_and_search_names_the_rest():
    async def go():
        srv = ATRServer(Config(servers=FAKE))
        await srv.start()
        try:
            meta = {t["name"]: t.get("_meta") for t in srv.list_tools()}
            assert meta["atr_find_tools"] == {"anthropic/alwaysLoad": True} and meta["atr_read_file"] is None
            found = (await srv.call("atr_find_tools", {"query": "list github issues", "k": 2}))["content"][0]["text"]
            assert "also on fake:" in found and "slack_send_message" in found
        finally:
            await srv.close()
        bare = ATRServer(Config())
        assert all("_meta" not in t for t in bare.list_tools())  # nothing behind ATR: nothing forced into the prompt
    run(go())
