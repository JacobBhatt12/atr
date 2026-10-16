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

Out of memory, a sequence is PREEMPTED in one of two ways:
    recompute: drop its blocks, set num_computed = 0, prefill again later (prefix caching
               makes this cheap, its old blocks are often still cached)
    swap:      copy its blocks to CPU RAM, copy them back later, num_computed is kept
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from engine.kv_cache import PagedKVCache, block_hash


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
    SWAPPED = "swapped"
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
        self.num_cached_tokens = 0     # prompt tokens served from the prefix cache
        self.swap_data = None          # CPU copy of the KV blocks while swapped out
        # prefix caching
        self.block_hashes: list[bytes] = []  # hash chain of every FULL block of all_ids
        self.num_registered = 0              # leading blocks already published to the cache
        # speculative decoding
        self.num_lookahead = 0         # draft tokens planned for this step
        self.draft_computed = 0        # tokens with K/V in the DRAFT model's cache
        self.spec_accepted = 0.0       # decayed counts for the acceptance-rate estimate
        self.spec_rejected = 0.0
        self.spec_idle = 0

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def length(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def num_uncomputed(self) -> int:
        return self.length - self.num_computed

    def hashes(self, block_size: int, num_blocks: int) -> list[bytes]:
        """Hashes of the first `num_blocks` full blocks, extending the chain as needed."""
        ids = None
        while len(self.block_hashes) < num_blocks:
            ids = ids or self.all_ids
            i = len(self.block_hashes)
            parent = self.block_hashes[-1] if self.block_hashes else None
            self.block_hashes.append(block_hash(parent, ids[i * block_size:(i + 1) * block_size]))
        return self.block_hashes[:num_blocks]


class Scheduler:
    def __init__(self, cache: PagedKVCache, max_num_seqs: int = 32, max_num_batched_tokens: int = 512,
                 preemption_mode: str = "recompute", swap_space_mb: int = 1024):
        if preemption_mode not in ("recompute", "swap"):
            raise ValueError("preemption_mode must be 'recompute' or 'swap'")
        self.cache = cache
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.preemption_mode = preemption_mode
        self.swap_space = swap_space_mb * 2**20
        self.swap_used = 0
        self.waiting: deque[Sequence] = deque()
        self.swapped: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0
        self.num_swaps = 0
        self.prefix_query_tokens = 0
        self.prefix_hit_tokens = 0
        # Speculative decoding asks for extra KV slots ("lookahead") for decode sequences.
        self.lookahead_fn: Callable[[Sequence], int] | None = None
        self.max_model_len = 1 << 30

    # ----------------------------------------------------------- bookkeeping
    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running or self.swapped)

    @property
    def caching(self) -> bool:
        return self.cache.allocator.caching

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
        seq.num_registered = 0
        seq.draft_computed = 0

    def _plan(self, seq: Sequence, budget: int) -> tuple[int, int]:
        """(tokens to process, speculative lookahead slots) for this step."""
        n = min(seq.num_uncomputed, budget)
        g = 0
        if self.lookahead_fn is not None and n == seq.num_uncomputed == 1:
            remaining = seq.params.max_tokens - len(seq.output_ids)
            g = max(0, min(self.lookahead_fn(seq), budget - 1, remaining - 1, self.max_model_len - seq.length - 1))
        return n, g

    def _preempt(self, seq: Sequence) -> None:
        """Out of memory: evict this sequence. Its generated tokens are kept either way,
        so nothing the user already saw is lost."""
        seq.num_preemptions += 1
        self.num_preemptions += 1
        nbytes = self.cache.swap_bytes(len(seq.block_table))
        if self.preemption_mode == "swap" and self.swap_used + nbytes <= self.swap_space:
            seq.swap_data = self.cache.swap_out(seq.block_table)
            self.swap_used += nbytes
            self.num_swaps += 1
            num_computed = seq.num_computed
            self._free_blocks(seq)
            seq.num_computed = num_computed
            seq.status = SeqStatus.SWAPPED
            self.swapped.appendleft(seq)
            return
        self._free_blocks(seq)
        seq.num_computed = 0
        seq.status = SeqStatus.WAITING
        self.waiting.appendleft(seq)

    def finish(self, seq: Sequence, reason: str) -> None:
        self._free_blocks(seq)
        if seq.swap_data is not None:
            self.swap_used -= self.cache.swap_bytes(seq.swap_data[0].shape[1])
            seq.swap_data = None
        seq.status = SeqStatus.FINISHED
        seq.finish_reason = reason
        seq.finish_time = time.perf_counter()
        for q in (self.running, self.waiting, self.swapped):
            if seq in q:
                q.remove(seq)
                break

    # --------------------------------------------------------- admission
    def _try_swap_in(self, seq: Sequence, budget: int) -> int | None:
        n, g = self._plan(seq, budget)
        need = self.cache.blocks_needed(seq.num_computed + n + g)
        if need > self.cache.allocator.num_free:
            return None
        seq.block_table = [self.cache.allocator.allocate() for _ in range(need)]
        k, _ = seq.swap_data
        self.cache.swap_in(seq.block_table[:k.shape[1]], seq.swap_data)
        self.swap_used -= self.cache.swap_bytes(k.shape[1])
        seq.swap_data = None
        seq.num_lookahead = g
        return n + g

    def _try_admit(self, seq: Sequence, budget: int) -> int | None:
        """Start a sequence that has no blocks yet, reusing cached prefix blocks if we can."""
        bs = self.cache.block_size
        alloc = self.cache.allocator
        hits: list[int] = []
        if self.caching:
            for h in seq.hashes(bs, seq.length // bs):
                b = alloc.lookup(h)
                if b is None:
                    break
                hits.append(b)
        # Always leave at least one token to compute: we need its logits.
        num_computed = min(len(hits) * bs, seq.length - 1)
        cow = len(hits) * bs > num_computed  # the last cached block will be written to
        seq.num_computed = num_computed
        n, g = self._plan(seq, budget)
        new_blocks = self.cache.blocks_needed(num_computed + n + g) - len(hits) + cow
        reuse = sum(alloc.is_evictable(b) for b in hits)  # these stop counting as free once shared
        if new_blocks > alloc.num_free - reuse:
            seq.num_computed = 0
            return None

        for b in hits:
            alloc.share(b)
        seq.block_table = list(hits)
        if cow:  # COPY-ON-WRITE: never write into a block someone else may be reading
            old = seq.block_table[-1]
            fresh = alloc.allocate()
            self.cache.copy_block(old, fresh)
            alloc.release(old)
            seq.block_table[-1] = fresh
        seq.num_registered = len(hits) - cow
        self._allocate(seq, n + g)
        if seq.num_preemptions == 0:
            self.prefix_query_tokens += len(seq.prompt_ids)
            self.prefix_hit_tokens += num_computed
            seq.num_cached_tokens = num_computed
        seq.num_lookahead = g
        return n + g

    # -------------------------------------------------------------- schedule
    def schedule(self) -> list[tuple[Sequence, int]]:
        """Return [(sequence, num_tokens_to_process)] for the next forward pass.
        Speculative lookahead slots are allocated too, but not counted in n."""
        budget = self.max_num_batched_tokens
        scheduled: list[tuple[Sequence, int]] = []
        preempted = False

        # 1) Sequences already running, oldest first. If memory runs out, evict the NEWEST.
        queue = deque(self.running)
        self.running = []
        while queue:
            seq = queue.popleft()
            n, g = self._plan(seq, budget)
            if n == 0:  # token budget used up this step; it runs next step
                seq.num_lookahead = 0
                self.running.append(seq)
                continue
            while not self._can_allocate(seq, n + g):
                preempted = True
                if queue:
                    self._preempt(queue.pop())
                else:
                    self._preempt(seq)
                    seq = None
                    break
            if seq is None:
                break
            self._allocate(seq, n + g)
            seq.num_lookahead = g
            scheduled.append((seq, n))
            self.running.append(seq)
            budget -= n + g

        # 2) Bring back swapped sequences, then admit new ones (first come, first served).
        if not preempted:
            for q, admit in ((self.swapped, self._try_swap_in), (self.waiting, self._try_admit)):
                while q and budget > 0 and len(self.running) < self.max_num_seqs:
                    seq = q[0]
                    used = admit(seq, budget)
                    if used is None:
                        break
                    q.popleft()
                    seq.status = SeqStatus.RUNNING
                    scheduled.append((seq, used - seq.num_lookahead))
                    self.running.append(seq)
                    budget -= used
                if q:  # keep arrival order: nothing new starts while older work is still waiting
                    break

        return scheduled

    def register_computed(self, seq: Sequence) -> None:
        """Publish newly completed full blocks to the prefix cache."""
        if not self.caching:
            return
        bs = self.cache.block_size
        full = min(seq.num_computed // bs, len(seq.block_table))
        if full > seq.num_registered:
            hashes = seq.hashes(bs, full)
            for i in range(seq.num_registered, full):
                self.cache.allocator.register(seq.block_table[i], hashes[i])
            seq.num_registered = full
