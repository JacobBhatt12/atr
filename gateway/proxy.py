"""Gateway proxy: routes <name>.vrtx.local to the healthy replicas of deployment <name>.

    curl http://todo-api.vrtx.local:8080/todos           (with a hosts entry or local DNS)
    curl http://todo-api.localhost:8080/todos            (*.localhost resolves to 127.0.0.1 on its own)
    curl -H 'Host: todo-api.vrtx.local' localhost:8080/todos

The routing table comes from the control plane's /api/routes, refreshed every second.
Replicas are picked round-robin; a replica that refuses the connection is skipped and the
next one is tried (idempotent requests also when a replica drops mid-request).
Responses are streamed, so model deployments' SSE token streams pass straight through.
Month 7 swaps round-robin for consistent hashing on the prompt prefix.
"""
from __future__ import annotations

import asyncio
import itertools
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from cloud.common import DOMAIN

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
              "transfer-encoding", "upgrade", "host", "content-length"}
IDEMPOTENT = {"GET", "HEAD", "OPTIONS", "PUT", "DELETE"}
REFRESH_S = 1.0


def deployment_from_host(host: str) -> str | None:
    host = host.split(":")[0].lower().rstrip(".")
    for suffix in (f".{DOMAIN}", ".localhost"):
        if host.endswith(suffix):
            label = host[: -len(suffix)]
            return label.split(".")[-1] or None  # v1.qwen.vrtx.local -> qwen
    return None


class Router:
    def __init__(self, control_plane: str):
        self.cp = control_plane.rstrip("/")
        self.routes: dict[str, dict] = {}
        self.counters: dict[str, itertools.count] = {}
        self.ok = False

    async def refresh(self, client: httpx.AsyncClient) -> None:
        try:
            r = await client.get(f"{self.cp}/api/routes", timeout=2.0)
            r.raise_for_status()
            self.routes, self.ok = r.json()["routes"], True
        except (httpx.HTTPError, KeyError, ValueError):
            self.ok = False  # keep the last good table: the control plane may just be restarting

    def order(self, name: str) -> list[str]:
        """Backends for `name`, starting from the next one in round-robin order."""
        backends = self.routes.get(name, {}).get("backends", [])
        if not backends:
            return []
        start = next(self.counters.setdefault(name, itertools.count())) % len(backends)
        return backends[start:] + backends[:start]


def create_proxy(control_plane: str) -> FastAPI:
    router = Router(control_plane)
    client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None))  # read=None: long streams are fine

    async def refresher():
        while True:
            await router.refresh(client)
            await asyncio.sleep(REFRESH_S)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await router.refresh(client)
        task = asyncio.create_task(refresher())
        yield
        task.cancel()
        await client.aclose()

    app = FastAPI(title="VRTX Gateway Proxy", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.router = router

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
    async def proxy(path: str, request: Request):
        host = request.headers.get("host", "")
        name = deployment_from_host(host)
        if name is None:
            return JSONResponse({
                "error": f"no app for host {host!r}",
                "hint": f"use http://<name>.{DOMAIN}:<port>/ or http://<name>.localhost:<port>/",
                "apps": sorted(router.routes),
            }, status_code=404)
        if not router.routes.get(name, {}).get("backends"):
            await router.refresh(client)  # maybe it was deployed or became healthy in the last second
        if name not in router.routes:
            return JSONResponse({"error": f"no deployment named {name!r}", "apps": sorted(router.routes)}, 404)
        backends = router.order(name)
        if not backends:
            return JSONResponse({"error": f"{name} has no healthy replicas right now"}, status_code=503)

        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        client_host = request.client.host if request.client else ""
        headers |= {"x-forwarded-host": host, "x-forwarded-proto": request.url.scheme,
                    "x-forwarded-for": ", ".join(filter(None, [request.headers.get("x-forwarded-for"), client_host]))}
        body = await request.body()
        for backend in backends:
            upstream = client.build_request(request.method, f"{backend}/{path}", params=request.query_params,
                                            headers=headers, content=body)
            try:
                resp = await client.send(upstream, stream=True)
            except httpx.ConnectError:
                continue  # nothing was sent: always safe to try the next replica
            except httpx.TransportError as e:
                if request.method in IDEMPOTENT:
                    continue  # replica went away mid-request; repeating a GET is safe
                return JSONResponse({"error": f"upstream error: {e!r}"}, status_code=502)
            out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP_BY_HOP}
            out_headers["x-vrtx-replica"] = backend

            async def body_iter(resp=resp):
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    await resp.aclose()

            return StreamingResponse(body_iter(), status_code=resp.status_code, headers=out_headers)
        return JSONResponse({"error": f"every replica of {name} refused the connection"}, status_code=502)

    return app
