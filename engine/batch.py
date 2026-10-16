"""Turns "these sequences, these tokens" into the flat tensors one forward pass needs.

Every scheduled sequence contributes a run of new tokens to ONE flat batch of T tokens.
Attention is the only operation that cares which sequence a token belongs to, and it
splits sequences into two groups:

    decode:  exactly one new token. All of them are handled by ONE batched call
             (one Triton kernel, or one padded gather + matmul in PyTorch).
    prefill: several new tokens (a prompt chunk, or speculative verification).
             Causal attention over the cached context plus the new tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class AttentionMetadata:
    block_size: int
    write_slots: torch.Tensor           # (T,)   cache slot for each new token's K and V
    # decode group
    num_decode: int
    decode_rows: torch.Tensor           # (Bd,)  row of each decode token in the flat batch
    decode_tables: torch.Tensor         # (Bd, max_blocks) int32, padded with 0
    decode_ctx_lens: torch.Tensor       # (Bd,)  int32
    # prefill group
    prefill: list[tuple[int, int, int]]  # (q_start, q_len, ctx_len) per sequence
    prefill_tables: torch.Tensor | None  # (Bp, max_blocks) int32
    max_decode_ctx: int = 0
    scratch: dict = field(default_factory=dict)  # per-step values shared by all layers


@dataclass
class BatchItem:
    block_table: list[int]
    start: int              # position of the first new token
    tokens: list[int]       # new tokens to run
    num_logits: int         # how many trailing positions need next-token logits


def _pad(tables: list[list[int]], device) -> torch.Tensor:
    width = max(len(t) for t in tables)
    return torch.tensor([t + [0] * (width - len(t)) for t in tables], dtype=torch.int32, device=device)


def build_batch(items: list[BatchItem], block_size: int, device):
    ids, positions, slots, logits_rows = [], [], [], []
    d_rows, d_tables, d_ctx = [], [], []
    prefill, p_tables = [], []
    offset = 0
    for it in items:
        n = len(it.tokens)
        end = it.start + n
        ids.extend(it.tokens)
        positions.extend(range(it.start, end))
        table = it.block_table
        slots.extend(table[p // block_size] * block_size + p % block_size for p in range(it.start, end))
        logits_rows.extend(range(offset + n - it.num_logits, offset + n))
        needed = table[: (end + block_size - 1) // block_size]
        if n == 1:
            d_rows.append(offset)
            d_tables.append(needed)
            d_ctx.append(end)
        else:
            prefill.append((offset, n, end))
            p_tables.append(needed)
        offset += n

    meta = AttentionMetadata(
        block_size=block_size,
        write_slots=torch.tensor(slots, dtype=torch.long, device=device),
        num_decode=len(d_rows),
        decode_rows=torch.tensor(d_rows, dtype=torch.long, device=device),
        decode_tables=_pad(d_tables, device) if d_tables else torch.zeros(0, 1, dtype=torch.int32, device=device),
        decode_ctx_lens=torch.tensor(d_ctx, dtype=torch.int32, device=device),
        prefill=prefill,
        prefill_tables=_pad(p_tables, device) if p_tables else None,
        max_decode_ctx=max(d_ctx, default=0),
    )
    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(positions, dtype=torch.long, device=device),
        meta,
        torch.tensor(logits_rows, dtype=torch.long, device=device),
    )


def run_model(model, cache, items: list[BatchItem]) -> torch.Tensor:
    """One forward pass. Returns logits for every requested row, in item order."""
    ids, positions, meta, rows = build_batch(items, cache.block_size, model.device)
    return model.forward(ids, positions, meta, rows, cache)
