"""Run a whole single-machine VRTX Cloud in one process: control plane, one worker agent,
and the gateway proxy. This is what `vrtx up` starts, and what the tests start in a thread.

Each part can also run on its own (and on other machines):
    python -m cloud.api    --port 7070
    python -m cloud.agent  --control-plane http://cp-host:7070 --advertise http://this-host:7071
    python -m gateway.proxy --control-plane http://cp-host:7070 --port 8080
"""
from __future__ import annotations

import asyncio
import contextlib
import signal
import threading
from dataclasses import dataclass
from pathlib import Path

import uvicorn

from cloud.agent.agent import create_agent
from cloud.api.server import create_control_plane
from gateway.proxy import create_proxy


class _Server(uvicorn.Server):
    """uvicorn server that leaves signal handling to us (several servers share one loop)."""

    @contextlib.contextmanager
    def capture_signals(self):
        yield


@dataclass
class Stack:
    home: Path
    runtime: str = "auto"
    host: str = "127.0.0.1"
    api_port: int = 7070
    agent_port: int = 7071
    gateway_port: int = 8080
    node_id: str | None = None
    log_level: str = "warning"

    @property
    def api_url(self) -> str:
        return f"http://{self.host}:{self.api_port}"

    def servers(self) -> list[_Server]:
        cp = create_control_plane(self.home, self.runtime, self.gateway_port)
        agent = create_agent(self.api_url, self.node_id, self.runtime, f"http://{self.host}:{self.agent_port}", self.home)
        proxy = create_proxy(self.api_url)
        bind = "0.0.0.0" if self.host not in ("127.0.0.1", "localhost") else self.host
        return [
            _Server(uvicorn.Config(app, host=bind, port=port, log_level=self.log_level, lifespan="on"))
            for app, port in ((cp, self.api_port), (agent, self.agent_port), (proxy, self.gateway_port))
        ]

    async def serve(self, stop: asyncio.Event | None = None) -> None:
        cp, agent, proxy = self.servers()
        stop = stop or asyncio.Event()
        tasks = {name: asyncio.create_task(s.serve(), name=name)
                 for name, s in (("control plane", cp), ("agent", agent), ("gateway", proxy))}
        stopper = asyncio.create_task(stop.wait())
        done, _ = await asyncio.wait([stopper, *tasks.values()], return_when=asyncio.FIRST_COMPLETED)
        for t in done - {stopper}:  # a server quit on its own, e.g. its port was taken
            print(f"[vrtx] {t.get_name()} exited, shutting down the rest")
        stopper.cancel()
        # stop the agent first, so its replicas shut down while the control plane can still see them
        agent.should_exit = True
        await asyncio.wait([tasks["agent"]], timeout=60)
        cp.should_exit = proxy.should_exit = True
        await asyncio.gather(*tasks.values(), return_exceptions=True)

    def run_forever(self) -> None:
        async def main():
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop.set)
            await self.serve(stop)

        asyncio.run(main())

    def start_in_thread(self) -> "RunningStack":
        """For tests: run the stack on a background thread until .stop() is called."""
        running = RunningStack(self)
        running.start()
        return running


class RunningStack:
    def __init__(self, stack: Stack):
        self.stack = stack
        self.loop: asyncio.AbstractEventLoop | None = None
        self.stop_event: asyncio.Event | None = None
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        async def main():
            self.loop = asyncio.get_running_loop()
            self.stop_event = asyncio.Event()
            self.ready.set()
            await self.stack.serve(self.stop_event)

        asyncio.run(main())

    def start(self):
        self.thread.start()
        self.ready.wait(10)

    def stop(self, timeout: float = 30):
        if self.loop and self.stop_event:
            self.loop.call_soon_threadsafe(self.stop_event.set)
        self.thread.join(timeout)
