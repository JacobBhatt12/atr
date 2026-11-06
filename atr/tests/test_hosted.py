"""The hosted connector (atr/hosted.py, atr/web.py). Everything stays on 127.0.0.1.

    python -m pytest atr/tests -c /dev/null
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
import sys
import threading
import time

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")
pytest.importorskip("cryptography")
uvicorn = pytest.importorskip("uvicorn")

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, Response  # noqa: E402

from atr import web  # noqa: E402
from atr.config import Config  # noqa: E402
from atr.hosted import HostedServer, clean_servers, create_hosted, seal, unseal  # noqa: E402
from atr.daily import parse_site, site_info, site_search, weather  # noqa: E402
from atr.places import format_places, normalized_query, search_places  # noqa: E402
from atr.remote import create_remote  # noqa: E402

FAKE = {"fake": {"command": sys.executable, "args": ["-m", "atr.demo_server"]}}
ARTICLE = "<html><head><title>A post</title></head><body><nav>Home | About | Login</nav><article><h1>Big news</h1>" + \
    "".join(f"<p>Paragraph {i} says something long enough to count as real content for the extractor to keep, "
            f"with a <a href='/more/{i}'>link</a>.</p>" for i in range(60)) + "<p>The secret word is PINEAPPLE.</p></article></body></html>"
LINKS = "<html><head><title>Links</title></head><body><table>" + "".join(
    f"<tr><td>{i}.</td><td><a href='https://ex.com/{i}'>Story {i}</a></td><td><a href='/item?id={i}'>{i} comments</a></td></tr>"
    for i in range(40)) + "</table></body></html>"


def rpc(mid, method, params=None):
    return {"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}


INIT = rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})


@pytest.fixture
def secret(monkeypatch):
    monkeypatch.setenv("ATR_SECRET", "test-secret-0123456789")


@pytest.fixture
def private_ok(monkeypatch):
    monkeypatch.setattr(web, "ALLOW_PRIVATE", True)


def serve(app) -> tuple[uvicorn.Server, threading.Thread, int]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started
    return server, t, port


@pytest.fixture
def site():
    """A tiny website: an article, a links page, JSON, a redirect to localhost, and a 404."""
    app = FastAPI()
    app.get("/article")(lambda: HTMLResponse(ARTICLE))
    app.get("/links")(lambda: HTMLResponse(LINKS))
    app.get("/data")(lambda: JSONResponse([{"id": i, "state": "open" if i % 3 else "closed", "title": f"Issue {i}",
                                            "_links": {"self": "x"}} for i in range(300)]))
    app.get("/image")(lambda: Response(b"\x89PNG....", media_type="image/png"))
    server, t, port = serve(app)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    t.join(10)


# ---------------------------------------------------------------- links and validation

def test_personal_links_round_trip_and_tamper(secret):
    servers = clean_servers([{"name": "GitHub", "url": "https://api.example.com/mcp", "headers": {"Authorization": "Bearer x"}}])
    assert servers == {"github": {"url": "https://api.example.com/mcp", "headers": {"Authorization": "Bearer x"}}}
    blob = seal(servers)
    assert "Bearer" not in blob and unseal(blob) == servers
    with pytest.raises(ValueError):
        unseal(blob[:-4] + "AAAA")


def test_personal_link_needs_the_same_secret(secret, monkeypatch):
    blob = seal({"a": {"url": "https://a.example/mcp"}})
    monkeypatch.setenv("ATR_SECRET", "a-different-secret-9876")
    with pytest.raises(ValueError):
        unseal(blob)


@pytest.mark.parametrize("bad", [
    [],
    [{"name": "a", "url": "http://plain.example/mcp"}],
    [{"name": "a", "url": "https://x.example"}, {"name": "a", "url": "https://y.example"}],
    [{"name": "bad name!", "url": "https://x.example"}],
    [{"name": "a", "url": "https://x.example", "headers": {"X-Key": "v\r\nInjected: 1"}}],
    [{"name": f"s{i}", "url": "https://x.example"} for i in range(11)],
])
def test_clean_servers_rejects(bad):
    with pytest.raises(ValueError):
        clean_servers(bad)


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "100.64.0.1", "0.0.0.0",
                                "::1", "fe80::1", "::ffff:127.0.0.1", "fd00::1", "224.0.0.1"])
def test_private_addresses_are_blocked(ip):
    assert web.blocked_ip(ipaddress.ip_address(ip))


def test_public_addresses_are_allowed():
    assert not web.blocked_ip(ipaddress.ip_address("93.184.215.14"))
    assert not web.blocked_ip(ipaddress.ip_address("2606:4700::1111"))


@pytest.mark.parametrize("url", ["http://127.0.0.1/", "http://localhost:7860", "file:///etc/passwd", "ftp://x.example",
                                 "http://[::1]/", "http://169.254.169.254/latest/meta-data", "http://0x7f000001/"])
def test_check_url_refuses(url):
    with pytest.raises(web.FetchError):
        asyncio.run(web.check_url(url))


# ---------------------------------------------------------------- page conversion

def test_article_keeps_content_and_links():
    page = web.to_text("https://blog.example/post", "text/html;charset=utf-8", ARTICLE.encode())
    assert page.kind == "html" and page.title == "A post"
    assert "PINEAPPLE" in page.text and "Paragraph 10" in page.text
    assert "Home | About | Login" not in page.text  # menus dropped


def test_link_pages_keep_their_links_and_shorten_same_site_ones():
    page = web.to_text("https://news.example/", "text/html", LINKS.encode())
    assert "[Story 7](https://ex.com/7)" in page.text
    assert "[7 comments](/item?id=7)" in page.text


def test_office_documents():
    import io

    openpyxl = pytest.importorskip("openpyxl")
    pptx = pytest.importorskip("pptx")
    wb = openpyxl.Workbook()
    wb.active.title = "Sales"
    for row in [("Region", "Q1"), ("North", 120), ("South", 95)]:
        wb.active.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    page = web.to_text("https://x.example/report.xlsx", "application/octet-stream", buf.getvalue())
    assert page.kind == "xlsx" and "--- sheet Sales ---" in page.text and "North | 120" in page.text

    deck = pptx.Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[1])
    slide.shapes.title.text = "Roadmap"
    slide.placeholders[1].text = "Ship the hosted connector"
    buf = io.BytesIO()
    deck.save(buf)
    page = web.to_text("https://x.example/deck", DOCX_PPTX, buf.getvalue())
    assert page.kind == "pptx" and "Roadmap" in page.text and "Ship the hosted connector" in page.text


DOCX_PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation;charset=utf-8"


def test_binary_links_are_refused():
    with pytest.raises(web.FetchError):
        web.to_text("https://x.example/a.png", "image/png", b"\x89PNG")


# ---------------------------------------------------------------- free local-place search

def test_place_categories_and_compact_format():
    assert normalized_query("coffee shops open now") == "cafe"
    assert normalized_query("nearby book shops") == "bookstore"
    assert normalized_query("Acme Records") == "Acme Records"
    text = format_places("coffee", "Tampa", [{
        "name": "Example Cafe", "category": "cafe", "address": "1 Main St, Tampa, FL",
        "latitude": 27.95, "longitude": -82.46, "map": "https://www.openstreetmap.org/node/1",
        "hours": "Mo-Fr 08:00-17:00", "phone": "+1 555 0100"}])
    assert "Example Cafe" in text and "1 Main St" in text and "OpenStreetMap contributors" in text
    assert "ratings" in text and "Mo-Fr" in text


def test_place_search_parses_nominatim(monkeypatch):
    import atr.places as places

    places._cache.clear()
    monkeypatch.setattr(places, "_last_request", 0.0)

    async def go():
        async def handler(request):
            assert request.url.params["q"] == "cafe, Tampa, Florida"
            return httpx.Response(200, json=[{
                "osm_type": "node", "osm_id": 42, "lat": "27.95", "lon": "-82.46",
                "name": "Example Cafe", "type": "cafe", "display_name": "Example Cafe, Tampa",
                "address": {"house_number": "1", "road": "Main St", "city": "Tampa", "state": "Florida"},
                "extratags": {"opening_hours": "Mo-Fr 08:00-17:00", "contact:website": "https://example.test"},
                "namedetails": {"name": "Example Cafe"}}])

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await search_places(client, "coffee shops", "Tampa, Florida", 5)
            again = await search_places(client, "coffee shops", "Tampa, Florida", 5)
        assert result == again and result[0]["name"] == "Example Cafe"
        assert result[0]["address"] == "1 Main St, Tampa, Florida"
        assert result[0]["website"] == "https://example.test"
    asyncio.run(go())


def test_hosted_places_tool(monkeypatch):
    import atr.hosted as hosted

    async def fake_search(_client, query, location, limit):
        assert (query, location, limit) == ("coffee shops", "Tampa", 3)
        return [{"name": "Cafe", "category": "cafe", "address": "Tampa", "latitude": 1.0,
                 "longitude": 2.0, "map": "https://www.openstreetmap.org/node/1"}]

    monkeypatch.setattr(hosted, "search_places", fake_search)

    async def go():
        server = HostedServer()
        result = await server.call("atr_places", {"query": "coffee shops", "location": "Tampa", "limit": 3})
        await server.close()
        return result

    result = asyncio.run(go())
    assert not result["isError"] and "Cafe" in result["content"][0]["text"]


def test_site_info_and_site_search():
    homepage = """<html lang="en"><head><title>Widget Docs</title>
        <meta name="description" content="Documentation for Widget."><link rel="canonical" href="/docs/">
        </head><body><h1>Widget</h1><h2>Start here</h2>
        <a href="/docs/install-guide">Installation guide</a><a href="https://outside.example/news">News</a></body></html>"""
    sitemap = """<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://example.com/docs/install-widget</loc></url>
        <url><loc>https://example.com/docs/api-reference</loc></url></urlset>"""

    async def go():
        async def handler(request):
            if request.url.path == "/sitemap.xml":
                return httpx.Response(200, text=sitemap, headers={"content-type": "application/xml"})
            return httpx.Response(200, text=homepage, headers={"content-type": "text/html;charset=utf-8"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            info = await site_info(client, "https://example.com/")
            found = await site_search(client, "https://example.com", "install", 5)
        return info, found

    info, found = asyncio.run(go())
    assert "Title: Widget Docs" in info and "Documentation for Widget" in info
    assert "Canonical: https://example.com/docs/" in info and "Links on page: 2 (1 external domains)" in info
    assert "install widget" in found and "Installation guide" in found


def test_weather_uses_geocoding_and_free_forecast(monkeypatch):
    import atr.places as places

    places._geocode_cache.clear()
    monkeypatch.setattr(places, "_last_request", 0.0)

    async def go():
        async def handler(request):
            if request.url.host == "nominatim.openstreetmap.org":
                return httpx.Response(200, json=[{
                    "lat": "27.95", "lon": "-82.46", "display_name": "Tampa, Florida, United States",
                    "address": {"country_code": "us"}}])
            assert request.url.host == "api.open-meteo.com"
            assert request.url.params["temperature_unit"] == "fahrenheit"
            return httpx.Response(200, json={
                "timezone": "America/New_York",
                "current_units": {"temperature_2m": "°F", "apparent_temperature": "°F", "wind_speed_10m": "mp/h"},
                "current": {"time": "2026-11-06T10:00", "temperature_2m": 74, "apparent_temperature": 76,
                            "weather_code": 0, "wind_speed_10m": 4},
                "daily_units": {"temperature_2m_max": "°F", "temperature_2m_min": "°F"},
                "daily": {"time": ["2026-11-06"], "weather_code": [2], "temperature_2m_max": [82],
                          "temperature_2m_min": [68], "precipitation_probability_max": [10]}})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await weather(client, "Tampa, Florida", 1, "auto")

    text = asyncio.run(go())
    assert "Weather for Tampa" in text and "clear sky" in text and "high 82°F" in text
    assert "Open-Meteo" in text and "OpenStreetMap" in text


# ---------------------------------------------------------------- the connector over HTTP

def test_web_tools_over_http(site, private_ok):
    app = create_hosted()

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://atr", timeout=30) as c:
            r = await c.post("/mcp", json=INIT)
            assert r.status_code == 200 and "atr_fetch" in r.json()["result"]["instructions"]
            h = {"mcp-session-id": r.headers["mcp-session-id"]}
            names = [t["name"] for t in (await c.post("/mcp", json=rpc(2, "tools/list"), headers=h)).json()["result"]["tools"]]
            assert names == ["atr_fetch", "atr_search", "atr_places", "atr_site", "atr_weather", "atr_query", "atr_stats"]

            async def call(name, args):
                res = (await c.post("/mcp", json=rpc(3, "tools/call", {"name": name, "arguments": args}), headers=h)).json()
                return res["result"]["isError"], res["result"]["content"][0]["text"]

            err, text = await call("atr_fetch", {"url": f"{site}/article", "budget": 300})
            assert not err and text.startswith("A post · ") and "[lines 1-" in text and "PINEAPPLE" not in text
            offset = int(text.split("offset=", 1)[1].split(")", 1)[0])
            pages = 0
            while "more lines" in text and pages < 50:  # read on until the end, statelessly
                err, text = await call("atr_fetch", {"url": f"{site}/article", "budget": 300, "offset": offset})
                assert not err and f"[lines {offset + 1}-" in text
                if "offset=" in text:
                    offset = int(text.rsplit("offset=", 1)[1].split(")", 1)[0])
                pages += 1
            assert "PINEAPPLE" in text and "more lines" not in text

            err, text = await call("atr_fetch", {"url": f"{site}/article", "question": "What is the secret word?"})
            assert not err and "PINEAPPLE" in text and "parts of the live page" in text and len(text) < 1200
            err, text = await call("atr_fetch", {"url": f"{site}/article", "question": "whole page", "budget": 300})
            assert not err and "[lines 1-" in text and "parts of the live page" not in text

            err, text = await call("atr_fetch", {"url": f"{site}/article", "search": "Paragraph", "question": "secret"})
            assert not err and len(text) < 600 * 4 + 400  # many matches: capped, ranked by the question

            err, text = await call("atr_fetch", {"url": f"{site}/article", "search": "secret word", "context": 0})
            assert not err and "1 matching lines" in text and "PINEAPPLE" in text

            err, text = await call("atr_fetch", {"url": f"{site}/data", "budget": 300})
            assert not err and "_links" not in text and "atr_query" in text
            ref = text.split("Full result is ", 1)[1].split(":", 1)[0]
            err, text = await call("atr_query", {"ref": ref, "group_by": "state"})
            assert not err and "open: 200" in text and "closed: 100" in text

            err, text = await call("atr_fetch", {"url": f"{site}/image"})
            assert err and "image/png" in text
            err, text = await call("atr_fetch", {"url": f"{site}/missing"})
            assert err and "404" in text

            err, text = await call("atr_places", {"query": "coffee shops"})
            assert err and "query and location" in text
            err, text = await call("atr_site", {})
            assert err and "needs a url" in text
            err, text = await call("atr_weather", {})
            assert err and "needs a location" in text

            for tool, args in [("atr_run", {"command": "ls"}), ("atr_read_file", {"path": "/etc/hosts"}),
                               ("atr_find_tools", {"query": "x"}), ("atr_repo_map", {})]:
                err, text = await call(tool, args)
                assert err, tool

            err, text = await call("atr_stats", {})
            assert "tool results:" in text and "servers:" not in text

            # sessions are lenient: a session this server doesn't know (e.g. after a restart) just works
            r = await c.post("/mcp", json=rpc(9, "tools/list"), headers={"mcp-session-id": "from-before-a-restart"})
            assert r.status_code == 200 and r.json()["result"]["tools"]

            assert (await c.post("/mcp", json=INIT, headers={"origin": "https://evil.example"})).status_code == 403
            assert (await c.get("/mcp")).status_code == 405
            assert (await c.post("/mcp/u/not-a-real-link", json=INIT)).status_code == 404
            assert (await c.get("/healthz")).json()["ok"]
            home = await c.get("/")
            assert home.status_code == 200 and "http://atr/mcp" in home.text
    asyncio.run(go())


GUIDE = "# Widget\n\nWidget is a tool for widgets.\n\n" + "".join(
    f"## Topic {i}\n\n" + " ".join(f"Filler sentence {j} about topic {i} and nothing in particular." for j in range(12)) + "\n\n"
    for i in range(20)) + ("## Installation\n\nInstall it with `pip install widget`. It needs Python 3.11 or newer.\n\n"
                           "## License\n\nMIT. See [the license](/LICENSE) and [OSI](https://opensource.org/mit)[[1]](#cite).\n")


def test_focus_returns_the_parts_that_answer():
    from atr.focus import focus, sections
    secs = sections(GUIDE)
    assert len(secs) > 20 and all(s.lines and s.lines[0].strip() for s in secs)
    text, n, total = focus(GUIDE, "How do I install it and which Python version?", budget=300)
    assert "pip install widget" in text and "Python 3.11" in text and "[Installation · line " in text
    assert n < total and "Topic 3" not in text
    # same-site links and footnote markers are dropped, other links kept
    text, _, _ = focus(GUIDE, "license", budget=300, url="https://widget.dev/docs")
    assert "the license" in text and "(/LICENSE)" not in text and "[[1]]" not in text
    assert "(https://opensource.org/mit)" in text
    # "top"/"latest" questions get the start of the page, in order
    text, _, _ = focus(GUIDE, "what are the top topics?", budget=300)
    assert text.index("Widget is a tool") < text.index("Topic 0")


def test_fetch_refuses_private_addresses_without_the_test_switch(site):
    app = create_hosted()

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://atr") as c:
            r = await c.post("/mcp", json=INIT)
            h = {"mcp-session-id": r.headers["mcp-session-id"]}
            res = (await c.post("/mcp", json=rpc(2, "tools/call", {"name": "atr_fetch", "arguments": {"url": f"{site}/article"}}),
                                headers=h)).json()["result"]
            assert res["isError"] and "private or local" in res["content"][0]["text"]
    asyncio.run(go())


def test_sessions_do_not_share_refs(site, private_ok):
    app = create_hosted()

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://atr", timeout=30) as c:
            a = {"mcp-session-id": (await c.post("/mcp", json=INIT)).headers["mcp-session-id"]}
            b = {"mcp-session-id": (await c.post("/mcp", json=INIT)).headers["mcp-session-id"]}
            await c.post("/mcp", json=rpc(2, "tools/call", {"name": "atr_fetch", "arguments": {"url": f"{site}/article", "budget": 300}}), headers=a)
            res = (await c.post("/mcp", json=rpc(3, "tools/call", {"name": "atr_query", "arguments": {"ref": "r1"}}), headers=b)).json()
            assert res["result"]["isError"]
    asyncio.run(go())


def test_personal_link_puts_a_users_server_behind_atr(secret, private_ok):
    """Real sockets: chatbot -> hosted ATR (personal link) --HTTP--> the user's MCP server (here: an ATR with a fake 50-tool server)."""
    remote = create_remote(Config(servers=FAKE, expose="all"), token="tok")
    server, t, port = serve(remote)
    app = create_hosted()

    async def go():
        await remote.state.atr.start()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://atr", timeout=60) as c:
            made = await c.post("/api/link", json={"servers": [
                {"name": "mine", "url": f"http://127.0.0.1:{port}/mcp/tok"},
                {"name": "broken", "url": f"http://127.0.0.1:{port}/mcp/wrong-token"}]})
            assert made.status_code == 200, made.text
            body = made.json()
            assert body["servers"]["mine"]["tools"] > 40 and "error" in body["servers"]["broken"]
            path = body["url"].removeprefix("http://atr")
            assert path.startswith("/mcp/u/") and "tok" not in path

            r = await c.post(path, json=INIT)
            assert "atr_find_tools" in r.json()["result"]["instructions"]
            h = {"mcp-session-id": r.headers["mcp-session-id"]}
            tools = (await c.post(path, json=rpc(2, "tools/list"), headers=h)).json()["result"]["tools"]
            names = [x["name"] for x in tools]
            assert {"atr_fetch", "atr_find_tools", "atr_read", "atr_call", "atr_query"} <= set(names)
            assert len(names) < 12  # the 50 downstream tools stay hidden
            assert not any("_meta" in x for x in tools)

            async def call(name, args):
                res = (await c.post(path, json=rpc(3, "tools/call", {"name": name, "arguments": args}), headers=h)).json()
                return res["result"]["isError"], res["result"]["content"][0]["text"]

            err, text = await call("atr_find_tools", {"query": "list github issues"})
            assert not err and "mine__fake__github_list_issues" in text
            err, text = await call("atr_read", {"tool": "mine__fake__github_list_issues", "arguments": {"owner": "a", "repo": "b"}})
            assert not err and "rows" in text
            err, text = await call("atr_read", {"tool": "mine__fake__github_create_issue", "arguments": {}})
            assert err and "atr_call" in text  # changes go through atr_call

            # a second session on the same link is the same user: same server, refs carry over
            r2 = await c.post(path, json=INIT)
            assert len(app.state.tenants) == 1 and r2.headers["mcp-session-id"] != h["mcp-session-id"]
    try:
        asyncio.run(go())
    finally:
        server.should_exit = True
        t.join(10)


def test_links_are_off_without_a_secret(monkeypatch):
    monkeypatch.delenv("ATR_SECRET", raising=False)
    app = create_hosted()

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://atr") as c:
            r = await c.post("/api/link", json={"servers": [{"name": "a", "url": "https://a.example/mcp"}]})
            assert r.status_code == 503
            assert "turned off" in (await c.get("/")).text
    asyncio.run(go())


def test_rate_limit(monkeypatch):
    import atr.hosted as hosted

    monkeypatch.setattr(hosted, "CALLS_PER_MINUTE", 3)
    srv = hosted.HostedServer()

    async def go():
        await srv.start()
        outs = [await srv.call("atr_stats", {}) for _ in range(4)]
        await srv.close()
        return outs
    outs = asyncio.run(go())
    assert [o["isError"] for o in outs] == [False, False, False, True]


def test_cli_has_hosted_command():
    from atr.cli import main

    with pytest.raises(SystemExit) as e:
        main(["hosted", "--help"])
    assert e.value.code == 0


DDG_HTML = ('<div class="result results_links results_links_deep web-result "><h2 class="result__title">'
            '<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fex.com%2Fa%3Futm_source%3Dx&rut=1">'
            'Page <b>A</b></a></h2><a class="result__snippet" href="x">About &amp; A.</a></div>'
            '<div class="result results_links result--ad "><a class="result__a" href="https://ad.com">Ad</a></div>'
            '<div class="result results_links web-result "><a class="result__a" href="https://ex.com/b">Page B</a>'
            '<a class="result__snippet" href="x">About B.</a></div>')
BING_HTML = ('<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?!&amp;u=a1aHR0cHM6Ly9leC5jb20vYi8&amp;ntb=1">'
             'Page <strong>B</strong></a></h2><div class="b_caption"><p class="b_lineclamp2">'
             '<span class="news_dt">Mar 14, 2010</span>&nbsp;&#0183;&#32;A much longer description of page B, kept.</p></div></li>'
             '<li class="b_algo"><h2><a href="https://ex.com/c">Page C</a></h2><p class="b_lineclamp3">'
             '<span class="news_dt">2 days ago</span> · About C.</p></li>')


def test_search_parsers_and_merge():
    from atr.search import format_results, merge, parse_bing, parse_duckduckgo

    ddg, bing = parse_duckduckgo(DDG_HTML), parse_bing(BING_HTML)
    assert [r.url for r in ddg] == ["https://ex.com/a", "https://ex.com/b"]   # redirect undone, tracking and ad dropped
    assert ddg[0].title == "Page A" and ddg[0].snippet == "About & A."
    assert bing[0].url == "https://ex.com/b/" and bing[0].title == "Page B"   # Bing's base64 redirect decoded
    assert bing[0].snippet.startswith("A much longer") and "2010" not in bing[0].snippet
    assert bing[1].snippet.startswith("2 days ago")
    merged = merge({"duckduckgo": ddg, "bing": bing}, 5)
    assert merged[0].url.rstrip("/") == "https://ex.com/b" and merged[0].engines == ["duckduckgo", "bing"]
    assert merged[0].snippet.startswith("A much longer")   # the fuller snippet wins
    assert len(merged) == 3
    text = format_results("q", merged, {"duckduckgo": 2, "bing": 2, "brave": "HTTP 429"})
    assert text.startswith("q (DuckDuckGo, Bing):") and "ex.com/b" in text and "https://" not in text


def test_hosted_search_tool(monkeypatch):
    import atr.hosted as hosted
    from atr.search import Result

    async def fake_search(_client, query, limit):
        assert (query, limit) == ("mcp", 3)
        rs = [Result("https://ex.com/a", "Page A", "About A. " * 40)]
        return rs, {"duckduckgo": 1, "bing": "HTTP 202"}, {"duckduckgo": rs}

    monkeypatch.setattr(hosted, "web_search", fake_search)

    async def go():
        server = HostedServer()
        names = [t["name"] for t in server.list_tools()]
        result = await server.call("atr_search", {"query": "mcp", "limit": 3})
        empty = await server.call("atr_search", {"query": " "})
        await server.close()
        return names, result, empty

    names, result, empty = asyncio.run(go())
    assert "atr_search" in names
    assert not result["isError"] and "Page A | ex.com/a" in result["content"][0]["text"]
    assert len(result["content"][0]["text"]) < 400   # snippet shortened
    assert empty["isError"]


def test_stats_never_show_a_loss_and_vercel_keeps_no_total(monkeypatch):
    import atr.hosted as hosted

    async def go():
        srv = HostedServer()
        await srv.start()
        srv._count(76, 98)                        # ATR's note made a tiny result longer: counts as no change
        local = (await srv.call("atr_stats", {}))["content"][0]["text"]
        monkeypatch.setattr(hosted, "ON_VERCEL", True)
        online = (await srv.call("atr_stats", {}))["content"][0]["text"]
        await srv.close()
        return srv.stats, local, online

    stats, local, online = asyncio.run(go())
    assert stats["results_after"] <= stats["results_before"]
    assert "saved 0, 0%" in local and "-" not in local.split("(saved")[1].split(")")[0]
    assert "no running total" in online and "tool results:" not in online   # not a per-machine count that looks like the total


def test_big_savings_get_a_receipt_on_the_result(site, private_ok):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_hosted()), base_url="http://atr") as c:
            r = await c.post("/mcp", json=INIT)
            h = {"mcp-session-id": r.headers["mcp-session-id"]}
            res = await c.post("/mcp", json=rpc(2, "tools/call", {"name": "atr_fetch", "arguments": {
                "url": site + "/article", "question": "what is the secret word?"}}), headers=h)
            return res.json()["result"]["content"][0]["text"]
    text = asyncio.run(go())
    assert "[atr saved " in text and "→" in text.split("[atr saved ")[1]
