"""ATR over Streamable HTTP, the MCP transport that ChatGPT and claude.ai connectors use.

    atr serve --http                          # http://127.0.0.1:8789/mcp/<token>
    cloudflared tunnel --url http://127.0.0.1:8789     # gives you a public https URL (free)

Then add https://<your-tunnel>/mcp/<token> as a custom connector in claude.ai or ChatGPT.

Transport (MCP 2025-06-18 "Streamable HTTP"):
  POST /mcp   one JSON-RPC message or a batch; requests get a JSON reply, notifications 202
  GET  /mcp   405: ATR sends nothing unprompted, so there is no server-to-client stream
  DELETE /mcp ends the session
The initialize reply carries an Mcp-Session-Id header; later requests must send it back.

Security: a remote ATR can run your downstream tools with your credentials, so every request
needs the token (in the path, /mcp/<token>, for connectors that can't send headers, or as
`Authorization: Bearer <token>`). File and code reading is off unless you pass --read-files,
and atr_run (shell commands) unless you pass --allow-run.
Browser Origins other than localhost and the known connector sites are refused.
"""
from __future__ import annotations

import hmac
import json
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from atr.config import Config
from atr.server import ATRServer

ALLOWED_ORIGINS = {"https://claude.ai", "https://chatgpt.com", "https://chat.openai.com"}
SESSION_TTL = 24 * 3600


def _origin_ok(origin: str | None) -> bool:
    if not origin:
        return True  # server-to-server callers (the connector backends) send no Origin
    host = origin.split("://", 1)[-1].split(":", 1)[0].split("/", 1)[0]
    return origin in ALLOWED_ORIGINS or host in ("localhost", "127.0.0.1", "[::1]")


def _rpc_error(mid, code: int, msg: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": msg}}, status_code=status)


def create_remote(cfg: Config, token: str | None, read_files: bool = False, allow_run: bool = False) -> FastAPI:
    srv = ATRServer(cfg)
    srv.allow_read_files = read_files
    srv.allow_run = allow_run and cfg.run
    sessions: dict[str, float] = {}  # id -> last seen

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await srv.start()
        yield
        await srv.close()

    app = FastAPI(title="ATR MCP", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.atr = srv

    def authorized(request: Request, path_token: str | None) -> bool:
        if token is None:
            return True
        given = path_token or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        return bool(given) and hmac.compare_digest(given.encode(), token.encode())

    async def handle(request: Request, path_token: str | None = None) -> Response:
        if not _origin_ok(request.headers.get("origin")):
            return JSONResponse({"error": "origin not allowed"}, status_code=403)
        if not authorized(request, path_token):
            return JSONResponse({"error": "missing or wrong token"}, status_code=401)
        sid = request.headers.get("mcp-session-id")
        now = time.time()
        for s, seen in list(sessions.items()):
            if now - seen > SESSION_TTL:
                del sessions[s]

        if request.method == "GET":
            return Response(status_code=405, headers={"allow": "POST, DELETE"})
        if request.method == "DELETE":
            sessions.pop(sid or "", None)
            return Response(status_code=204)

        try:
            payload = json.loads(await request.body())
        except ValueError:
            return _rpc_error(None, -32700, "Parse error")
        msgs = payload if isinstance(payload, list) else [payload]
        if not msgs or not all(isinstance(m, dict) for m in msgs):
            return _rpc_error(None, -32600, "Invalid request")

        headers = {}
        if any(m.get("method") == "initialize" for m in msgs):
            sid = secrets.token_urlsafe(18)
            headers["mcp-session-id"] = sid
        elif not sid:
            return _rpc_error(None, -32600, "Missing Mcp-Session-Id header (send initialize first)")
        elif sid not in sessions:
            return _rpc_error(None, -32001, "Unknown or expired session; initialize again", status=404)
        sessions[sid] = now

        requests = [m for m in msgs if "method" in m and m.get("id") is not None]
        for m in msgs:
            if m not in requests and "method" in m:
                await srv.handle(m)  # notifications
        if not requests:
            return Response(status_code=202, headers=headers)
        results = [await srv.handle(m) for m in requests]
        body = results if isinstance(payload, list) else results[0]
        return JSONResponse(body, headers=headers)

    @app.api_route("/mcp", methods=["GET", "POST", "DELETE"])
    async def mcp(request: Request):
        return await handle(request)

    @app.api_route("/mcp/{path_token}", methods=["GET", "POST", "DELETE"])
    async def mcp_token(path_token: str, request: Request):
        return await handle(request, path_token)

    return app


def run(cfg: Config, host: str, port: int, token: str | None, read_files: bool, allow_run: bool = False,
        project: str | None = None) -> None:
    import uvicorn

    if project:  # over HTTP, the file and coding tools see only this folder
        cfg.project, cfg.read_roots = project, [project]
    if read_files or allow_run:
        print(f"[atr] file/code tools can see: {', '.join(cfg.read_roots)} (secrets like .env and keys are always refused)",
              flush=True)

    path = f"/mcp/{token}" if token else "/mcp"
    print(f"[atr] MCP over HTTP: http://{host}:{port}{path}", flush=True)
    if token:
        print("[atr] keep this URL secret: anyone with it can use your tools", flush=True)
    print(f"[atr] to reach it from claude.ai / ChatGPT: cloudflared tunnel --url http://{host}:{port}  "
          f"then use https://<tunnel-host>{path}", flush=True)
    if allow_run:
        print("[atr] --allow-run: anyone with the URL can run commands on this machine", flush=True)
    uvicorn.run(create_remote(cfg, token, read_files, allow_run), host=host, port=port, log_level="warning")
