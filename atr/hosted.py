"""ATR as a shared online connector, for chatbots (claude.ai, ChatGPT, Le Chat, ...).

    pip install "atr-mcp[hosted]"
    ATR_SECRET=<long random string> atr hosted --port 7860

Anyone can add https://<host>/mcp as a connector and get:
  atr_fetch    any web page, PDF, Word, PowerPoint, Excel or JSON link, as compact text;
               with question=, only the parts that answer it (atr/focus.py)
  atr_search   web search (DuckDuckGo, Bing and Brave merged), ~60 tokens a result (atr/search.py)
  atr_places   local businesses and places from OpenStreetMap, with no API key
  atr_site     basic website information, or search a site's public sitemap/navigation
  atr_weather  current conditions and a seven-day forecast, with no API key
  atr_query    the rest of anything atr_fetch had to cut (search, paging, filters)
  atr_stats    tokens saved

A personal link (made on the home page) also puts the person's own online MCP servers behind
ATR, with their API keys: https://<host>/mcp/u/<blob>. The blob is the server list, encrypted
with ATR_SECRET, so this server keeps nothing on disk and links survive restarts. Those links
add atr_find_tools, atr_read and atr_call, the same as local ATR.

Nothing here touches the machine it runs on: no file, code or shell tools, and fetches and
downstream servers are limited to public internet addresses (see atr/web.py).

State: each personal link (or, without one, each MCP session) gets its own ATRServer, kept in
memory and dropped after IDLE_TTL or when there are more than MAX_TENANTS.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import time
import zlib
from collections import OrderedDict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from atr import __version__, tokens
from atr.compress import compress
from atr.config import Config
from atr.focus import FROM_TOP, focus
from atr.daily import site_info, site_search, weather
from atr.places import format_places, search_places
from atr.search import format_results, web_search
from atr.downstream import log
from atr.server import ATRServer, text_result
from atr.stash import query
from atr import web
from atr.web import FetchError, check_url, fetch_page, make_client

MAX_TENANTS = 300
IDLE_TTL = 2 * 3600
MAX_SERVERS = 10
FETCH_BUDGET = 4000            # tokens of a page shown by default
MAX_BUDGET = 20000
CALLS_PER_MINUTE = 40          # per tenant
MAX_CONCURRENT_FETCHES = 8     # whole server
MAX_BODY = 1024 * 1024         # biggest JSON-RPC request accepted
ALLOWED_ORIGINS = {"https://claude.ai", "https://chatgpt.com", "https://chat.openai.com", "https://chat.mistral.ai"}
NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

INSTRUCTIONS = (
    "ATR saves tokens. To read a web page, PDF, Word/PowerPoint/Excel file or JSON link, use atr_fetch. Pass "
    "question= with what you want to know: you get only the parts of the live page that answer it, word for word "
    "with line numbers (a few hundred tokens instead of the whole page). question=\"whole page\" reads it from "
    "the top; search= (a regex) returns only matching lines. The note at the end of a result says how to "
    "read more (offset=). Use atr_search to search the web (short results from several engines); atr_site for basic website info or to search within a public website; atr_places "
    "for local businesses; and atr_weather for current conditions and forecasts.")
PROXY_INSTRUCTIONS = (
    " Tools from the user's connected servers are hidden: atr_find_tools with a short task description, then "
    "atr_read for tools marked [r] (read-only) or atr_call for [w] (they change something and may need approval). "
    "Add question= (what you need from the result) to get only the parts or list items that answer it. "
    "Large results come back summarized with a ref; use atr_query on the ref instead of calling the tool again.")

ON_VERCEL = bool(os.environ.get("VERCEL"))
RECEIPT_MIN = 200             # tokens a result must save before it gets an "atr saved" note
QUESTION_BUDGET = 250         # tokens atr_fetch returns for question= by default
SEARCH_BUDGET = 600           # tokens search= returns by default
WH = re.compile(r"\b(who|what|when|where|which|why|how)\b", re.I)
WHOLE_PAGE = re.compile(r"^\s*(\*|all|everything|(the )?(whole|full|entire) (page|document|text))?\s*\.?\s*$", re.I)

# Kept short on purpose: chatbots re-read a tool's definition on every model call after loading it.
# context= and budget= still work but aren't advertised (budget= made models ask for more, not less).
FETCH_TOOL = {
    "name": "atr_fetch",
    "description": "Read a link. Use this, not web_fetch, when the answer must be current or exact (news, front "
                   "pages, prices, docs, READMEs, version numbers, figures): web_fetch can return an older cached "
                   "copy and paraphrases it, while this reads the live page (also PDF and Office files) and quotes "
                   "the sentences that answer the question.",
    "inputSchema": {"type": "object", "properties": {
        "url": {"type": "string"},
        "question": {"type": "string", "description": "What you want to know. \"whole page\" reads from the top."},
        "search": {"type": "string", "description": "Regex; returns matching lines instead."},
        "offset": {"type": "integer", "description": "Line to continue reading from."}},
        "required": ["url", "question"]},
    "annotations": {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": True, "title": "Read a link"},
}
SEARCH_TOOL = {
    "name": "atr_search",
    "description": "Search the web. Asks DuckDuckGo, Bing and Brave at once and returns one short merged list "
                   "(title, link, one-line snippet). Then atr_fetch a result with question= to read it.",
    "inputSchema": {"type": "object", "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5}},
        "required": ["query"]},
    "annotations": {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": True, "title": "Search the web"},
}
PLACES_TOOL = {
    "name": "atr_places",
    "description": "Find local businesses and places without an API key: cafes, restaurants, bookstores, pharmacies, "
                   "hotels, parks and more. Returns compact OpenStreetMap results with addresses, coordinates, map "
                   "links, and hours/phone/website when available. It has no ratings and coverage varies by area.",
    "inputSchema": {"type": "object", "properties": {
        "query": {"type": "string", "description": "Business or place category, such as coffee shops or bookstores."},
        "location": {"type": "string", "description": "City, neighborhood, postal code, or address."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 8}},
        "required": ["query", "location"]},
    "annotations": {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": True,
                    "title": "Find local places"},
}
SITE_TOOL = {
    "name": "atr_site",
    "description": "Inspect a public website, or search its sitemap and homepage links. Without query, returns title, "
                   "description, canonical URL, language, main headings, size and link counts. With query, finds "
                   "matching pages on that website. This is site search, not a search of the whole web.",
    "inputSchema": {"type": "object", "properties": {
        "url": {"type": "string", "description": "Website URL or domain."},
        "query": {"type": "string", "description": "Optional words to find within this website."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 8}},
        "required": ["url"]},
    "annotations": {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": True,
                    "title": "Inspect or search a website"},
}
WEATHER_TOOL = {
    "name": "atr_weather",
    "description": "Get current weather and a 1-7 day forecast for a city, postal code, or address. Free and keyless; "
                   "includes temperature, conditions, wind and precipitation chance.",
    "inputSchema": {"type": "object", "properties": {
        "location": {"type": "string"},
        "days": {"type": "integer", "minimum": 1, "maximum": 7, "default": 3},
        "units": {"type": "string", "enum": ["auto", "us", "metric"], "default": "auto"}},
        "required": ["location"]},
    "annotations": {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": True,
                    "title": "Check weather"},
}
PROXY_TOOLS = {"atr_find_tools", "atr_read", "atr_call"}


# ---------------------------------------------------------------- personal links

def _fernet():
    secret = os.environ.get("ATR_SECRET", "")
    if len(secret) < 16:
        return None
    from cryptography.fernet import Fernet

    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest()))


def clean_servers(raw) -> dict[str, dict]:
    """Validate a server list from the setup form: [{name, url, headers}] or {name: {url, headers}}."""
    items = list(raw.items()) if isinstance(raw, dict) else [(s.get("name"), s) for s in (raw or []) if isinstance(s, dict)]
    if not items:
        raise ValueError("add at least one server")
    if len(items) > MAX_SERVERS:
        raise ValueError(f"at most {MAX_SERVERS} servers per link")
    out: dict[str, dict] = {}
    for name, spec in items:
        name = str(name or "").strip().lower().replace(" ", "-")
        if not NAME.match(name):
            raise ValueError(f"server name {name!r}: use 1-32 letters, digits, - or _")
        if name in out:
            raise ValueError(f"two servers are named {name!r}")
        url = str(spec.get("url") or "").strip()
        if not (url.startswith("https://") or (web.ALLOW_PRIVATE and url.startswith("http://"))) or len(url) > 2000:
            raise ValueError(f"{name}: the URL must start with https://")
        headers = {}
        for k, v in (spec.get("headers") or {}).items():
            k, v = str(k).strip(), str(v).strip()
            if not k:
                continue
            if not re.fullmatch(r"[A-Za-z0-9-]{1,64}", k) or len(v) > 8192 or "\n" in v or "\r" in v:
                raise ValueError(f"{name}: header {k!r} isn't valid")
            headers[k] = v
        if len(headers) > 10:
            raise ValueError(f"{name}: at most 10 headers")
        out[name] = {"url": url, **({"headers": headers} if headers else {})}
    return out


def seal(servers: dict) -> str:
    f = _fernet()
    if f is None:
        raise RuntimeError("personal links are off: this server has no ATR_SECRET")
    return f.encrypt(zlib.compress(json.dumps({"v": 1, "servers": servers}, separators=(",", ":")).encode())).decode()


def unseal(blob: str) -> dict:
    from cryptography.fernet import InvalidToken

    f = _fernet()
    if f is None:
        raise ValueError("personal links are off on this server")
    try:
        data = json.loads(zlib.decompress(f.decrypt(blob.encode())))
    except (InvalidToken, ValueError, zlib.error):
        raise ValueError("this link isn't valid (it was changed, or the server's secret changed)") from None
    return clean_servers(data.get("servers") or {})


# ---------------------------------------------------------------- one user's server

class HostedServer(ATRServer):
    def __init__(self, servers: dict | None = None):
        super().__init__(Config(servers=dict(servers or {}), read_roots=[], coding=False, run=False))
        self.allow_read_files = False
        self.allow_run = False
        self.instructions = INSTRUCTIONS + (PROXY_INSTRUCTIONS if servers else "")
        self.client = make_client()
        self.calls: deque[float] = deque()
        self.last_used = time.time()

    async def start(self) -> None:
        bad = {}
        for name, spec in list(self.cfg.servers.items()):
            try:
                await check_url(spec["url"])
            except FetchError as e:
                bad[name] = str(e)
                del self.cfg.servers[name]
        await super().start()
        from atr.downstream import make_downstream

        for name, err in bad.items():
            d = make_downstream(name, {"url": "https://invalid.invalid"})
            d.error = err
            self.downstreams[name] = d

    async def close(self) -> None:
        await super().close()
        await self.client.aclose()

    def list_tools(self) -> list[dict]:
        tools = [FETCH_TOOL, SEARCH_TOOL, PLACES_TOOL, SITE_TOOL, WEATHER_TOOL]
        for t in super().list_tools():
            if t["name"] in PROXY_TOOLS and not self.cfg.servers:
                continue
            if t["name"] == "atr_query" or t["name"] == "atr_stats" or t["name"] in PROXY_TOOLS:
                tools.append({k: v for k, v in t.items() if k != "_meta"})
            elif not t["name"].startswith("atr_"):
                tools.append(t)  # pinned downstream tools
        return tools

    def rate_limited(self) -> bool:
        now = time.time()
        while self.calls and now - self.calls[0] > 60:
            self.calls.popleft()
        if len(self.calls) >= CALLS_PER_MINUTE:
            return True
        self.calls.append(now)
        return False

    async def call(self, name: str, a: dict) -> dict:
        if self.rate_limited():
            return text_result(f"Slow down: at most {CALLS_PER_MINUTE} ATR calls a minute. Try again shortly.", True)
        if name == "atr_fetch":
            return await self.fetch(a)
        if name == "atr_search":
            return await self.search(a)
        if name == "atr_places":
            return await self.places(a)
        if name == "atr_site":
            return await self.site(a)
        if name == "atr_weather":
            return await self.weather(a)
        allowed = {"atr_search", "atr_places", "atr_site", "atr_weather", "atr_query", "atr_stats"} | (
            PROXY_TOOLS if self.cfg.servers else set())
        if name.startswith("atr_") and name not in allowed:
            return text_result(f"{name} isn't available on the hosted connector.", True)
        return await super().call(name, a)

    async def search(self, a: dict) -> dict:
        query = str(a.get("query") or "").strip()
        if not query:
            return text_result("atr_search needs a query.", True)
        try:
            async with FETCH_SLOTS:
                results, status, by_engine = await web_search(self.client, query, int(a.get("limit") or 5))
        except Exception as e:  # noqa: BLE001 - search engine/network errors
            return text_result(f"Couldn't search: {type(e).__name__}: {e}"[:400], True)
        text = format_results(query, results, status)
        # "Before" is every engine's results with full snippets: what a plain search tool would return.
        before = sum(tokens.count(f"{r.title}\n{r.url}\n{r.snippet}") for rs in by_engine.values() for r in rs)
        self._count(max(before, tokens.count(text)), tokens.count(text))
        return text_result(text, not results)

    async def places(self, a: dict) -> dict:
        query, location = str(a.get("query") or "").strip(), str(a.get("location") or "").strip()
        if not query or not location:
            return text_result("atr_places needs both query and location.", True)
        try:
            results = await search_places(self.client, query, location, int(a.get("limit") or 8))
        except ValueError as e:
            return text_result(f"atr_places: {e}", True)
        except Exception as e:  # noqa: BLE001 - public search service/network errors
            return text_result(f"Couldn't search places: {type(e).__name__}: {e}"[:400], True)
        text = format_places(query, location, results)
        # The upstream JSON is larger than the compact text returned to the chatbot.
        before = tokens.count(json.dumps(results, ensure_ascii=False))
        self._count(before, tokens.count(text))
        return text_result(text)

    async def site(self, a: dict) -> dict:
        url, question = str(a.get("url") or "").strip(), str(a.get("query") or "").strip()
        if not url:
            return text_result("atr_site needs a url.", True)
        if "://" not in url:
            url = "https://" + url
        try:
            async with FETCH_SLOTS:
                text = (await site_search(self.client, url, question, int(a.get("limit") or 8))
                        if question else await site_info(self.client, url))
        except ValueError as e:
            return text_result(f"atr_site: {e}", True)
        except FetchError as e:
            return text_result(f"Couldn't inspect {url}: {e}", True)
        except Exception as e:  # noqa: BLE001 - public website/network errors
            return text_result(f"Couldn't inspect {url}: {type(e).__name__}: {e}"[:400], True)
        return text_result(text)

    async def weather(self, a: dict) -> dict:
        location = str(a.get("location") or "").strip()
        if not location:
            return text_result("atr_weather needs a location.", True)
        try:
            text = await weather(self.client, location, int(a.get("days") or 3), str(a.get("units") or "auto"))
        except ValueError as e:
            return text_result(f"atr_weather: {e}", True)
        except Exception as e:  # noqa: BLE001 - public weather/geocoding errors
            return text_result(f"Couldn't get weather: {type(e).__name__}: {e}"[:400], True)
        return text_result(text)

    async def fetch(self, a: dict) -> dict:
        url = str(a.get("url") or "").strip()
        if not url:
            return text_result("atr_fetch needs a url.", True)
        budget = max(200, min(int(a.get("budget") or FETCH_BUDGET), MAX_BUDGET))
        try:
            async with FETCH_SLOTS:
                page = await fetch_page(url, self.client)
        except FetchError as e:
            return text_result(f"Couldn't read {url}: {e}", True)
        except Exception as e:  # noqa: BLE001 - network errors, timeouts
            return text_result(f"Couldn't read {url}: {type(e).__name__}: {e}"[:400], True)
        if not page.text.strip():
            return text_result(f"{page.url}: no readable text (the page may need JavaScript or a sign-in).", True)

        full = page.text
        before = tokens.count(full)
        lines = full.splitlines()
        head = f"{page.title} · {page.url}" if page.title else page.url
        head += f" ({page.kind}, {len(lines)} lines)"

        search = a.get("search")
        if search:
            try:
                rx = re.compile(search, re.I)
            except re.error as e:
                return text_result(f"search isn't a valid regex: {e}", True)
            ctx = max(0, min(int(a.get("context") if a.get("context") is not None else 1), 10))
            hits = [i for i, ln in enumerate(lines) if rx.search(ln)]
            keep = sorted({j for i in hits for j in range(max(0, i - ctx), min(len(lines), i + ctx + 1))})
            out, prev = [], -2
            for j in keep:
                if j != prev + 1 and out:
                    out.append("…")
                out.append(f"{j + 1}: {lines[j]}")
                prev = j
            body = "\n".join(out) if out else "(no matching lines)"
            sbudget = int(a["budget"]) if a.get("budget") else SEARCH_BUDGET
            if a.get("question") and out and tokens.count(body) > sbudget:
                # too many matches: keep the ones that best answer the question
                body, _, _ = focus(body, str(a["question"]), budget=sbudget, url=page.url)
            text = f"{head}\n{len(hits)} matching lines for /{search}/:\n{body}"
            if tokens.count(text) > sbudget:
                text = text[: sbudget * 4] + "\n…"
            text += f"\n[atr: to read from a line on: atr_fetch(url=\"{page.url}\", offset=<line>)]"
        elif (a.get("question") and not WHOLE_PAGE.match(str(a["question"])) and not a.get("offset")
              and page.kind != "json"):
            q = str(a["question"])
            # the point of question= is a small answer: a big budget (models often pass the default
            # 4000) only widens it a little; read more with offset= or search=
            if a.get("budget"):
                qbudget = max(120, min(int(a["budget"]), QUESTION_BUDGET * 3))
            else:
                # a question with several parts ("who, when and which") gets room for each;
                # "top 5 stories" gets room for the whole list
                parts = len(WH.findall(q)) + len(re.findall(r",| and ", q)) // 2
                qbudget = min(QUESTION_BUDGET + 100 * max(0, parts - 1), QUESTION_BUDGET * 2)
                if FROM_TOP.search(q):
                    qbudget = max(qbudget, 450)
            if before <= qbudget:  # small page: all of it
                text = f"{head}\n\n{full}"
            else:
                # the model knows the URL it asked for: the title is enough
                body, n, total = focus(full, q, budget=qbudget, url=page.url)
                text = (f"{page.title or page.url}\n{body}\n"
                        f"[atr: {n} of {total} parts of the live page. Not here? search=\"word\" or offset=<line>]")
        elif page.kind == "json":
            r = compress(full, budget=budget)
            text = f"{head}\n{r.text}"
            if r.lossy:
                ref = self.stash.put(page.url, full)
                self.stats["stashed"] += 1
                text += (f"\n[atr: cut {r.before}→{r.after} tokens. Full result is {ref}: "
                         f"atr_query(ref=\"{ref}\", path=…, where=[…], fields=[…], search=…) for the rest]")
        else:
            start = max(0, min(int(a.get("offset") or 0), len(lines)))
            shown, used = [], tokens.count(head)
            for ln in lines[start:]:
                used += tokens.count(ln) + 1
                if used > budget and shown:
                    break
                shown.append(ln)
            end = start + len(shown)
            text = f"{head}\n\n" + "\n".join(shown)
            if start or end < len(lines):
                text = f"{head} [lines {start + 1}-{end}]\n\n" + "\n".join(shown)
            if end < len(lines):
                text += (f"\n\n[atr: {len(lines) - end} more lines. Read on: atr_fetch(url=\"{page.url}\", offset={end}); "
                         f"or find one thing: atr_fetch(url=\"{page.url}\", search=\"…\")]")
        after = tokens.count(text)
        if before - after >= RECEIPT_MIN:   # this server can't keep a running total (see stats_text), so say it here
            text += f"\n[atr saved {before - after:,} tokens ({before:,}→{after:,})]"
        self._count(before, after)
        return text_result(text)

    def stats_text(self) -> str:
        if ON_VERCEL:
            # Vercel runs each request on whichever machine is free, and every machine counts only its own
            # requests, so a total here would be wrong (it showed "1 call" after twenty). Each trimmed result
            # carries its own "atr saved" note instead, and the chat can add those up.
            return ("This online server runs on many machines at once, so it keeps no running total. "
                    "Each result ATR trimmed ends with a note like \"[atr saved 57,845 tokens (60,482→2,637)]\": "
                    "add those up for this chat. (Run `atr hosted` yourself, or use ATR on your computer, "
                    "to get a running total.)")
        text = super().stats_text()
        return text if self.cfg.servers else text.split("\n", 1)[0]

    def query_ref(self, a: dict) -> dict:
        try:
            label, raw = self.stash.get(a["ref"])
        except KeyError as e:
            return text_result(f"atr_query: {e}. Refs last for this conversation only; fetch the link again.", True)
        if not isinstance(raw, str) or raw.lstrip()[:1] in "{[":
            return super().query_ref(a)
        # plain text: answer with lines, not a compressed table of lines
        try:
            res = query(raw, search=a.get("search"), offset=int(a.get("offset") or 0),
                        limit=max(1, min(int(a.get("limit") or 100), 1000)))
        except re.error as e:
            return text_result(f"atr_query: {e}", True)
        lines = "\n".join(res["lines"]) or "(nothing there)"
        return text_result(f"{label} ({res['matches']} of {res['total_lines']} lines, from {res['offset']}):\n{lines}")


FETCH_SLOTS = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)


# ---------------------------------------------------------------- HTTP app

def _origin_ok(origin: str | None, own_host: str) -> bool:
    if not origin:
        return True  # connector backends call server-to-server, without an Origin
    host = origin.split("://", 1)[-1].split("/", 1)[0]
    return origin in ALLOWED_ORIGINS or host == own_host or host.split(":", 1)[0] in ("localhost", "127.0.0.1")


def _rpc_error(mid, code: int, msg: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": msg}}, status_code=status)


def create_hosted() -> FastAPI:
    tenants: OrderedDict[str, HostedServer] = OrderedDict()
    sessions: dict[str, str] = {}  # session id -> tenant key
    lock = asyncio.Lock()

    async def drop(key: str) -> None:
        srv = tenants.pop(key, None)
        for s in [s for s, k in sessions.items() if k == key]:
            del sessions[s]
        if srv is not None:
            asyncio.create_task(srv.close())

    async def tenant(key: str, servers: dict | None) -> HostedServer:
        async with lock:
            now = time.time()
            for k in [k for k, s in tenants.items() if now - s.last_used > IDLE_TTL]:
                await drop(k)
            srv = tenants.get(key)
            if srv is None:
                while len(tenants) >= MAX_TENANTS:
                    await drop(next(iter(tenants)))
                srv = tenants[key] = HostedServer(servers)
                asyncio.create_task(srv.start())
            tenants.move_to_end(key)
            srv.last_used = now
            return srv

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        for k in list(tenants):
            await tenants.pop(k).close()

    app = FastAPI(title="ATR", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.tenants = tenants

    def base_url(request: Request) -> str:
        public = os.environ.get("ATR_PUBLIC_URL", "").rstrip("/")
        if public:
            return public
        proto = request.headers.get("x-forwarded-proto", request.url.scheme).split(",")[0].strip()
        host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
        return f"{proto}://{host}"

    async def handle(request: Request, blob: str | None) -> Response:
        if not _origin_ok(request.headers.get("origin"), request.headers.get("host", "")):
            return JSONResponse({"error": "origin not allowed"}, status_code=403)
        servers = None
        if blob is not None:
            try:
                servers = unseal(blob)
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=404)
        if request.method == "GET":
            return Response(status_code=405, headers={"allow": "POST, DELETE"})
        sid = request.headers.get("mcp-session-id")
        if request.method == "DELETE":
            if sid in sessions and blob is None:
                await drop(sessions[sid])
            sessions.pop(sid or "", None)
            return Response(status_code=204)

        body = await request.body()
        if len(body) > MAX_BODY:
            return _rpc_error(None, -32600, "Request too large", status=413)
        try:
            payload = json.loads(body)
        except ValueError:
            return _rpc_error(None, -32700, "Parse error")
        msgs = payload if isinstance(payload, list) else [payload]
        if not msgs or len(msgs) > 20 or not all(isinstance(m, dict) for m in msgs):
            return _rpc_error(None, -32600, "Invalid request")

        headers = {}
        if any(m.get("method") == "initialize" for m in msgs) or not sid:
            sid = secrets.token_urlsafe(18)
            headers["mcp-session-id"] = sid
        # a personal link is one user, so all its sessions share one server (and its refs); without one,
        # each session is its own user. An unknown session (e.g. after a restart) just starts fresh.
        key = "u:" + hashlib.sha256(blob.encode()).hexdigest()[:24] if blob else sessions.get(sid, "s:" + sid)
        sessions[sid] = key
        if len(sessions) > MAX_TENANTS * 20:
            for s in list(sessions)[: len(sessions) // 2]:
                del sessions[s]
        srv = await tenant(key, servers)

        requests = [m for m in msgs if "method" in m and m.get("id") is not None]
        for m in msgs:
            if m not in requests and "method" in m:
                await srv.handle(m)
        if not requests:
            return Response(status_code=202, headers=headers)
        results = [await srv.handle(m) for m in requests]
        return JSONResponse(results if isinstance(payload, list) else results[0], headers=headers)

    @app.api_route("/mcp", methods=["GET", "POST", "DELETE"])
    async def mcp(request: Request):
        return await handle(request, None)

    @app.api_route("/mcp/u/{blob}", methods=["GET", "POST", "DELETE"])
    async def mcp_personal(blob: str, request: Request):
        return await handle(request, blob)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "version": __version__, "users": len(tenants)}

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        from atr.hosted_page import render

        return render(base_url(request), personal=_fernet() is not None)

    @app.post("/api/link")
    async def make_link(request: Request):
        if _fernet() is None:
            return JSONResponse({"error": "Personal links are off on this server."}, status_code=503)
        try:
            data = json.loads((await request.body())[:200_000])
            servers = clean_servers(data.get("servers"))
        except (ValueError, AttributeError, TypeError) as e:
            return JSONResponse({"error": str(e) or "bad request"}, status_code=400)
        # try them now, so a typo in a URL or key shows up here and not later in a chat
        probe = HostedServer(servers)
        try:
            await asyncio.wait_for(probe.start(), 45)
        except asyncio.TimeoutError:
            pass
        finally:
            await probe.close()
        def why(err: str | None) -> str:
            if not err:
                return "no tools (or it didn't answer in time)"
            err = err.split(": ", 1)[1] if err.startswith(tuple(f"{n}: " for n in servers)) else err
            if "<" in err and re.search(r"<(!doctype|html|head|body)", err, re.I):
                status = re.search(r"HTTP (\d+)", err)
                return f"that URL is a web page, not an MCP server{f' (HTTP {status.group(1)})' if status else ''}"
            return err[:200]

        report = {n: ({"tools": len(d.tools)} if d.tools else {"error": why(d.error)}) for n, d in probe.downstreams.items()}
        return {"url": f"{base_url(request)}/mcp/u/{seal(servers)}", "servers": report}

    return app


def run(host: str, port: int) -> None:
    import uvicorn

    if _fernet() is None:
        log("ATR_SECRET is not set (16+ characters): personal links are off; atr_fetch still works")
    log(f"hosted connector on http://{host}:{port}/mcp")
    uvicorn.run(create_hosted(), host=host, port=port, log_level="warning", proxy_headers=True,
                forwarded_allow_ips="*")
