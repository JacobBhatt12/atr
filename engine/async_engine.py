"""Runs LLMEngine in a background thread so a FastAPI server can stream to many users at once.

Only the background thread ever touches the engine. Web requests talk to it through a
thread-safe inbox, and tokens come back through one asyncio queue per request.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import queue
import threading

from engine.llm_engine import LLMEngine
from engine.scheduler import SamplingParams

log = logging.getLogger("vrtx.engine")


class AsyncLLMEngine:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._inbox: queue.Queue = queue.Queue()
        self._streams: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._ids = itertools.count()
        self._thread = threading.Thread(target=self._loop, name="vrtx-engine", daemon=True)
        self._thread.start()

    # -------------------------------------------------------- engine thread
    def _emit(self, seq_id: int, item) -> None:
        stream = self._streams.get(seq_id)
        if stream is not None:
            loop, q = stream
            loop.call_soon_threadsafe(q.put_nowait, item)

    def _drain_inbox(self, block: bool) -> None:
        while True:
            try:
                kind, payload = self._inbox.get(block=block)
            except queue.Empty:
                return
            block = False
            if kind == "add":
                seq_id, prompt_ids, params = payload
                try:
                    self.engine.add_request(prompt_ids, params, seq_id=seq_id)
                except Exception as e:  # e.g. prompt too long
                    self._emit(seq_id, e)
            elif kind == "abort":
                self.engine.abort(payload)

    def _loop(self) -> None:
        while True:
            # Sleep until work arrives when idle; otherwise just pick up anything new.
            self._drain_inbox(block=not self.engine.has_unfinished())
            try:
                events = self.engine.step()
            except Exception as e:
                log.exception("engine step failed")
                for seq_id in list(self._streams):
                    self.engine.abort(seq_id)
                    self._emit(seq_id, e)
                continue
            for ev in events:
                self._emit(ev.seq.id, (ev.token_id, ev.finished))

    # ------------------------------------------------------------ public API
    async def generate(self, prompt_ids: list[int], params: SamplingParams):
        """Async iterator of generated token ids."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        seq_id = next(self._ids)
        self._streams[seq_id] = (loop, q)
        self._inbox.put(("add", (seq_id, prompt_ids, params)))
        finished = False
        try:
            while True:
                item = await q.get()
                if isinstance(item, Exception):
                    finished = True
                    raise item
                token_id, done = item
                if token_id is not None:
                    yield token_id
                if done:
                    finished = True
                    return
        finally:
            self._streams.pop(seq_id, None)
            if not finished:  # client disconnected: free its KV blocks right away
                self._inbox.put(("abort", seq_id))
