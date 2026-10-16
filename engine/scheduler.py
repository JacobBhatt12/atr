"""Continuous batching scheduler.

Every engine step (one forward pass) the scheduler decides which sequences run and how
many of their tokens to process. Sequences join and leave the batch at ANY step, so a
short reply never waits for a long one.

One idea handles prefill, decode, chunked prefill and recovery after preemption:
each sequence tracks `num_computed`, how many of its tokens already have K/V in the cache.
A step processes the next `n` uncomputed tokens. When that reaches the end of the
sequence, we sample a new token.

    prefill:  num_computed = 0,       uncomputed = whole prompt (maybe split into chunks)
    decode:   num_computed = len - 1, uncomputed = 1 (the token we just sampled)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

from engine.kv_cache import PagedKVCache


@dataclass
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    ignore_eos: bool = False  # benchmarks force a fixed output length


class SeqStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class Sequence:
    def __init__(self, seq_id: int, prompt_ids: list[int], params: SamplingParams):
        self.id = seq_id
        self.prompt_ids = list(prompt_ids)
        self.output_ids: list[int] = []
        self.params = params
        self.status = SeqStatus.WAITING
        self.block_table: list[int] = []
        self.num_computed = 0
        self.arrival_time = time.perf_counter()
        self.first_token_time: float | None = None
        self.finish_time: float | None = None
        self.finish_reason: str | None = None
        self.num_preemptions = 0

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def length(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def num_uncomputed(self) -> int:
        return self.length - self.num_computed


class Scheduler:
    def __init__(self, cache: PagedKVCache, max_num_seqs: int = 32, max_num_batched_tokens: int = 512):
        self.cache = cache
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    # ----------------------------------------------------------- bookkeeping
    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def _extra_blocks(self, seq: Sequence, n: int) -> int:
        return self.cache.blocks_needed(seq.num_computed + n) - len(seq.block_table)

    def _can_allocate(self, seq: Sequence, n: int) -> bool:
        return self._extra_blocks(seq, n) <= self.cache.allocator.num_free

    def _allocate(self, seq: Sequence, n: int) -> None:
        for _ in range(self._extra_blocks(seq, n)):
            seq.block_table.append(self.cache.allocator.allocate())

    def _free_blocks(self, seq: Sequence) -> None:
        for b in seq.block_table:
            self.cache.allocator.release(b)
        seq.block_table = []

    def _preempt(self, seq: Sequence) -> None:
        """Out of memory: drop this sequence's KV blocks and recompute them later.
        Its generated tokens are kept, so nothing the user saw is lost."""
        self._free_blocks(seq)
        seq.num_computed = 0
        seq.status = SeqStatus.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)

    def finish(self, seq: Sequence, reason: str) -> None:
        self._free_blocks(seq)
        seq.status = SeqStatus.FINISHED
        seq.finish_reason = reason
        seq.finish_time = time.perf_counter()
        if seq in self.running:
            self.running.remove(seq)
        elif seq in self.waiting:
            self.waiting.remove(seq)

    # -------------------------------------------------------------- schedule
    def schedule(self) -> list[tuple[Sequence, int]]:
        """Return [(sequence, num_tokens_to_process)] for the next forward pass."""
        budget = self.max_num_batched_tokens
        scheduled: list[tuple[Sequence, int]] = []
        preempted = False

        # 1) Sequences already running, oldest first. If memory runs out, evict the NEWEST.
        queue = deque(self.running)
        self.running = []
        while queue:
            seq = queue.popleft()
            n = min(seq.num_uncomputed, budget)
            if n == 0:  # token budget used up this step; it runs next step
                self.running.append(seq)
                continue
            while not self._can_allocate(seq, n):
                preempted = True
                if queue:
                    self._preempt(queue.pop())
                else:
                    self._preempt(seq)
                    seq = None
                    break
            if seq is None:
                break
            self._allocate(seq, n)
            scheduled.append((seq, n))
            self.running.append(seq)
            budget -= n

        # 2) Admit new sequences (first come, first served) while there is room.
        if not preempted:
            while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
                seq = self.waiting[0]
                n = min(seq.num_uncomputed, budget)
                if not self._can_allocate(seq, n):
                    break
                self.waiting.popleft()
                seq.status = SeqStatus.RUNNING
                self._allocate(seq, n)
                scheduled.append((seq, n))
                self.running.append(seq)
                budget -= n

        return scheduled
