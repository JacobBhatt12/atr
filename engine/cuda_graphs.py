"""CUDA graphs for small steps: record the forward pass once, replay it every step.

Why: at batch 1 one Qwen2.5 decode step is ~300 kernel launches, and on a T4 host each
launch costs ~5-40 us of CPU time, more than the GPU work. Step time then follows the
layer count, not the bytes read, so a 0.5B draft cost 0.70x a 1.5B target step and
speculative decoding could not pay off. A replayed graph launches the whole step at once.

A graph freezes shapes and addresses, so every captured step runs on static buffers:

    B  sequences, padded up to a bucket (1, 2, 4, 8, 16, 24, 32, ...)
    Q  tokens per sequence (1 = decode; 2..gamma+1 = draft catch-up and spec verification)

Shorter items are padded at the FRONT (their real tokens stay last, so the causal rule
"key position <= query position" still holds), padding rows write their K/V into one
reserved null block that no sequence owns, and padding sequences attend only to it.
Q=1 graphs use the paged decode kernel, Q>1 graphs the prefill kernel with one tile per
sequence. Steps that do not fit (long prompt chunks, mixed prefill+decode) run eagerly.

Without CUDA (tests) the same static-buffer path runs eagerly, so the padding logic is
tested in CI through Triton's interpreter.
"""
from __future__ import annotations

import torch

from engine.batch import AttentionMetadata, BatchItem, run_model
PREFILL_BLOCK_M = 32  # engine.kernels.attention.PREFILL_BLOCK_M; importing it needs Triton

DECODE_BUCKETS = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256)


class GraphRunner:
    def __init__(self, model, cache, null_block: int, max_batch: int, max_model_len: int,
                 multi_q: tuple[int, ...] = (), multi_q_max_batch: int = 8, capture: bool | None = None):
        assert "attention" in model.kernels, "graphs need the Triton attention kernels"
        self.model, self.cache, self.null = model, cache, null_block
        self.bs = cache.block_size
        self.capture_graphs = torch.device(model.device).type == "cuda" if capture is None else capture
        buckets = [b for b in DECODE_BUCKETS if b < max_batch] + [max_batch]
        self.shapes = {1: buckets}
        for q in multi_q:
            if 1 < q <= min(PREFILL_BLOCK_M, self.bs):
                self.shapes[q] = [b for b in buckets if b <= multi_q_max_batch]
        self.max_b = max_batch
        self.max_q = max(self.shapes)
        self.width = max(min(-(-max_model_len // self.bs), cache.num_blocks), 1)
        dev = model.device
        n = self.max_b * self.max_q
        self.lbuf = torch.zeros(4, n, dtype=torch.long, device=dev)  # ids, positions, slots, logit rows
        self.tables = torch.zeros(self.max_b, self.width, dtype=torch.int32, device=dev)
        self.ctx = torch.ones(self.max_b, dtype=torch.int32, device=dev)
        self.arange = torch.arange(n, dtype=torch.long, device=dev)
        self.arange32 = self.arange.to(torch.int32)
        self.zeros32 = torch.zeros(self.max_b, dtype=torch.int32, device=dev)
        self.qlens = {q: torch.full((self.max_b,), q, dtype=torch.int32, device=dev) for q in self.shapes}
        self.qstarts: dict[tuple[int, int], torch.Tensor] = {}
        self.graphs: dict[tuple[int, int], tuple] = {}
        self.pool = None
        self.replays = 0
        self.eager_steps = 0

    # ------------------------------------------------------------ shapes
    def _bucket(self, n: int, q: int) -> int | None:
        for b in self.shapes.get(q, ()):
            if b >= n:
                return b
        return None

    def shape_for(self, items: list[BatchItem]) -> tuple[int, int] | None:
        if not items:
            return None
        q = max(len(it.tokens) for it in items)
        q = next((x for x in sorted(self.shapes) if x >= q), None)
        if q is None:
            return None
        b = self._bucket(len(items), q)
        if b is None:
            return None
        for it in items:
            if -(-(it.start + len(it.tokens)) // self.bs) > self.width:
                return None
        return b, q

    def _meta(self, b: int, q: int) -> AttentionMetadata:
        n = b * q
        slots = self.lbuf[2, :n]
        tables, ctx = self.tables[:b], self.ctx[:b]
        if q == 1:
            return AttentionMetadata(self.bs, slots, b, self.arange[:b], tables, ctx, [], None)
        meta = AttentionMetadata(self.bs, slots, 0, self.arange[:0], tables[:0], ctx[:0],
                                 [(i * q, q, 0) for i in range(b)], tables)
        # what attention.paged_attention would build on the host, as static device tensors
        qstarts = self.qstarts.get((b, q))
        if qstarts is None:
            qstarts = self.qstarts[(b, q)] = self.arange32[:b] * q
        meta.scratch["prefill_tiles"] = (qstarts, self.qlens[q][:b], ctx, self.arange32[:b], self.zeros32[:b])
        return meta

    def _forward(self, b: int, q: int, meta: AttentionMetadata) -> torch.Tensor:
        n = b * q
        return self.model.forward(self.lbuf[0, :n], self.lbuf[1, :n], meta, self.lbuf[3, :n], self.cache)

    # ------------------------------------------------------------ fill
    def _fill(self, items: list[BatchItem], b: int, q: int) -> int:
        bs, null = self.bs, self.null
        ids, pos, slots, rows = [], [], [], []
        width = 1
        tables, ctx = [], []
        for i, it in enumerate(items):
            k = len(it.tokens)
            pad = q - k
            table = it.block_table
            ids += [0] * pad + it.tokens
            pos += [0] * pad + list(range(it.start, it.start + k))
            slots += [null * bs + j % bs for j in range(pad)]
            slots += [table[p // bs] * bs + p % bs for p in range(it.start, it.start + k)]
            rows += range(i * q + q - it.num_logits, i * q + q)
            need = table[: -(-(it.start + k) // bs)]
            width = max(width, len(need))
            tables.append(need)
            ctx.append(it.start + k)
        for i in range(len(items), b):  # padding sequences: q tokens in the null block
            ids += [0] * q
            pos += range(q)
            slots += [null * bs + j % bs for j in range(q)]
            tables.append([null])
            ctx.append(q)
        num_rows = len(rows)
        rows += [0] * (b * q - num_rows)
        n = b * q
        self.lbuf[:, :n].copy_(torch.tensor([ids, pos, slots, rows], dtype=torch.long))
        self.tables[:b, :width].copy_(torch.tensor([t + [null] * (width - len(t)) for t in tables], dtype=torch.int32))
        self.ctx[:b].copy_(torch.tensor(ctx, dtype=torch.int32))
        return num_rows

    def _reset(self, b: int, q: int) -> None:
        self._fill([], b, q)

    # ------------------------------------------------------------ capture / run
    def capture(self) -> None:
        """Record every (batch bucket, Q) graph, biggest first so smaller ones reuse its memory."""
        if not self.capture_graphs:
            return
        for q in sorted(self.shapes, reverse=True):
            for b in sorted(self.shapes[q], reverse=True):
                self._capture(b, q)
        torch.cuda.synchronize()

    def _capture(self, b: int, q: int) -> None:
        meta = self._meta(b, q)
        self._reset(b, q)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):  # compile kernels, autotune, settle allocator before recording
            for _ in range(2):
                self._forward(b, q, meta)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = self._forward(b, q, meta)
        self.pool = g.pool()
        self.graphs[(b, q)] = (g, out)

    def run(self, items: list[BatchItem]) -> torch.Tensor:
        shape = self.shape_for(items)
        if shape is None:
            self.eager_steps += 1
            return run_model(self.model, self.cache, items)
        b, q = shape
        num_rows = self._fill(items, b, q)
        if self.capture_graphs:
            if (b, q) not in self.graphs:  # a shape added after capture()
                self._capture(b, q)
                self._fill(items, b, q)
            g, out = self.graphs[(b, q)]
            g.replay()
            self.replays += 1
            return out[:num_rows].clone()  # the next replay overwrites `out`
        self.replays += 1  # emulation: same buffers and padding, run eagerly
        return self._forward(b, q, self._meta(b, q))[:num_rows]
