"""A minimal MCP client for the servers ATR sits in front of (stdio transport).

Each downstream server is a child process speaking newline-delimited JSON-RPC 2.0, the same
way Claude Desktop or Cursor would run it. ATR is the client here: it does the initialize
handshake, lists tools (following pagination) and forwards tool calls.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import sys
from typing import Any, Awaitable, Callable

from atr import __version__

PROTOCOL = "2025-06-18"
LINE_LIMIT = 256 * 1024 * 1024  # tool results can be huge; the default 64 KiB line limit is not enough


class DownstreamError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[atr] {msg}", file=sys.stderr, flush=True)


class Downstream:
    def __init__(self, name: str, spec: dict, on_tools_changed: Callable[[str], Awaitable[None]] | None = None):
        self.name = name
        self.spec = spec
        self.proc: asyncio.subprocess.Process | None = None
        self.pending: dict[int, asyncio.Future] = {}
        self.ids = itertools.count(1)
        self.tools: list[dict] = []
        self.error: str | None = None
        self.on_tools_changed = on_tools_changed
        self._reader: asyncio.Task | None = None
        self._write_lock = asyncio.Lock()

    async def start(self, timeout: float = 60) -> None:
        cmd = self.spec.get("command")
        if not cmd:
            raise DownstreamError(f"{self.name}: needs a 'command' (stdio) or a 'url' (HTTP)")
        env = {**os.environ, **(self.spec.get("env") or {})}
        self.proc = await asyncio.create_subprocess_exec(
            cmd, *(self.spec.get("args") or []), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=None, env=env, cwd=self.spec.get("cwd"), limit=LINE_LIMIT)
        self._reader = asyncio.create_task(self._read_loop())
        init = await self.request("initialize", {
            "protocolVersion": PROTOCOL, "capabilities": {},
            "clientInfo": {"name": "atr", "version": __version__}}, timeout=timeout)
        await self.notify("notifications/initialized")
        self.server_info = init.get("serverInfo", {})
        await self.refresh_tools()

    async def refresh_tools(self) -> None:
        tools, cursor = [], None
        while True:
            res = await self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools.extend(res.get("tools") or [])
            cursor = res.get("nextCursor")
            if not cursor:
                break
        self.tools = tools

    async def call_tool(self, name: str, arguments: dict, timeout: float = 300) -> dict:
        return await self.request("tools/call", {"name": name, "arguments": arguments or {}}, timeout=timeout)

    async def _send(self, msg: dict) -> None:
        if not self.proc or self.proc.stdin is None or self.proc.returncode is not None:
            raise DownstreamError(f"{self.name}: server is not running")
        async with self._write_lock:
            self.proc.stdin.write((json.dumps(msg, separators=(",", ":")) + "\n").encode())
            await self.proc.stdin.drain()

    async def request(self, method: str, params: dict | None = None, timeout: float = 120) -> Any:
        rid = next(self.ids)
        fut = asyncio.get_running_loop().create_future()
        self.pending[rid] = fut
        try:
            await self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            raise DownstreamError(f"{self.name}: {method} timed out after {timeout:.0f}s") from None
        finally:
            self.pending.pop(rid, None)

    async def notify(self, method: str, params: dict | None = None) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    async def _read_loop(self) -> None:
        assert self.proc and self.proc.stdout
        try:
            while True:
                line = await self.proc.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except ValueError:
                    log(f"{self.name}: ignoring non-JSON output: {line[:120]!r}")
                    continue
                await self._dispatch(msg)
        except Exception as e:  # noqa: BLE001
            log(f"{self.name}: reader stopped: {e}")
        finally:
            self.error = self.error or "server exited"
            for fut in self.pending.values():
                if not fut.done():
                    fut.set_exception(DownstreamError(f"{self.name}: server exited"))

    async def _dispatch(self, msg: dict) -> None:
        if "id" in msg and ("result" in msg or "error" in msg) and "method" not in msg:
            fut = self.pending.get(msg["id"])
            if fut and not fut.done():
                if "error" in msg:
                    err = msg["error"] or {}
                    fut.set_exception(DownstreamError(f"{self.name}: {err.get('message', err)}"))
                else:
                    fut.set_result(msg["result"])
            return
        method = msg.get("method", "")
        if "id" in msg:  # the server is asking us something (ping, roots, sampling ...)
            if method == "ping":
                await self._send({"jsonrpc": "2.0", "id": msg["id"], "result": {}})
            elif method == "roots/list":
                await self._send({"jsonrpc": "2.0", "id": msg["id"], "result": {"roots": []}})
            else:
                await self._send({"jsonrpc": "2.0", "id": msg["id"],
                                  "error": {"code": -32601, "message": f"atr does not support {method}"}})
        elif method == "notifications/tools/list_changed":
            # a task, not an await: the reply to tools/list arrives through this same loop
            asyncio.create_task(self._tools_changed())

    async def _tools_changed(self) -> None:
        try:
            await self.refresh_tools()
            if self.on_tools_changed:
                await self.on_tools_changed(self.name)
        except Exception as e:  # noqa: BLE001
            log(f"{self.name}: could not refresh tools: {e}")

    async def close(self) -> None:
        if self.proc and self.proc.returncode is None:
            try:
                self.proc.stdin.close()
                await asyncio.wait_for(self.proc.wait(), 3)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                self.proc.kill()
        if self._reader:
            self._reader.cancel()


class HttpDownstream(Downstream):
    """A downstream server reached over Streamable HTTP: {"url": "https://...", "headers": {...}}.

    Same interface as the stdio one. Replies come back either as plain JSON or as an SSE
    stream that carries the reply; both are handled. Needs httpx (pip install -e '.[http]').
    """

    def __init__(self, name: str, spec: dict, on_tools_changed=None):
        super().__init__(name, spec, on_tools_changed)
        self.session: str | None = None
        self.protocol: str | None = None
        self.client = None

    async def start(self, timeout: float = 60) -> None:
        import httpx

        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=300.0),
                                        headers=self.spec.get("headers") or {})
        await self._initialize(timeout)
        await self.refresh_tools()

    async def _initialize(self, timeout: float) -> None:
        self.session = None
        init = await self.request("initialize", {
            "protocolVersion": PROTOCOL, "capabilities": {},
            "clientInfo": {"name": "atr", "version": __version__}}, timeout=timeout)
        self.protocol = init.get("protocolVersion", PROTOCOL)
        self.server_info = init.get("serverInfo", {})
        await self.notify("notifications/initialized")

    def _headers(self) -> dict:
        h = {"accept": "application/json, text/event-stream", "content-type": "application/json"}
        if self.session:
            h["mcp-session-id"] = self.session
        if self.protocol:
            h["mcp-protocol-version"] = self.protocol
        return h

    async def _post(self, msg: dict, timeout: float):
        if self.client is None:
            raise DownstreamError(f"{self.name}: not started")
        import httpx

        try:
            resp = await asyncio.wait_for(self.client.post(self.spec["url"], json=msg, headers=self._headers()), timeout)
        except asyncio.TimeoutError:
            raise DownstreamError(f"{self.name}: {msg.get('method')} timed out after {timeout:.0f}s") from None
        except httpx.HTTPError as e:
            raise DownstreamError(f"{self.name}: {e!r}") from None
        if resp.headers.get("mcp-session-id"):
            self.session = resp.headers["mcp-session-id"]
        return resp

    async def request(self, method: str, params: dict | None = None, timeout: float = 120) -> Any:
        rid = next(self.ids)
        msg = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
        resp = await self._post(msg, timeout)
        if resp.status_code == 404 and self.session and method != "initialize":
            await self._initialize(timeout)  # session expired on the server: start a new one, retry once
            resp = await self._post(msg, timeout)
        if resp.status_code >= 400:
            raise DownstreamError(f"{self.name}: HTTP {resp.status_code} for {method}: {resp.text[:200]}")
        ctype = resp.headers.get("content-type", "")
        replies = []
        if "text/event-stream" in ctype:
            data: list[str] = []
            for line in resp.text.splitlines() + [""]:
                if line.startswith("data:"):
                    data.append(line[5:].lstrip())
                elif not line.strip() and data:
                    try:
                        replies.append(json.loads("\n".join(data)))
                    except ValueError:
                        pass
                    data = []
        else:
            body = resp.json()
            replies = body if isinstance(body, list) else [body]
        for r in replies:
            if isinstance(r, dict) and r.get("id") == rid and ("result" in r or "error" in r):
                if "error" in r:
                    err = r["error"] or {}
                    raise DownstreamError(f"{self.name}: {err.get('message', err)}")
                return r["result"]
        raise DownstreamError(f"{self.name}: no reply to {method}")

    async def notify(self, method: str, params: dict | None = None) -> None:
        await self._post({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})}, 30)

    async def close(self) -> None:
        if self.client is None:
            return
        try:
            if self.session:
                await self.client.delete(self.spec["url"], headers=self._headers())
        except Exception:  # noqa: BLE001
            pass
        await self.client.aclose()
        self.client = None


def make_downstream(name: str, spec: dict, on_tools_changed=None) -> Downstream:
    if spec.get("url") and not spec.get("command"):
        return HttpDownstream(name, spec, on_tools_changed)
    return Downstream(name, spec, on_tools_changed)
