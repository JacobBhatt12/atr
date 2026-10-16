"""PyTorch reference versions of every operation the model runs.

These run anywhere (CPU, Apple MPS, CUDA). engine/kernels/ has Triton versions of the
same functions with the same signatures; tests check the two agree.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.batch import AttentionMetadata


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    """x + residual, and the RMSNorm of that sum. Returns (sum, normed)."""
    x = x + residual
    return x, rms_norm(x, weight, eps)


def silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_and_cache(q, k, v, positions, inv_freq, k_cache, v_cache, slots):
    """Rotate q and k by their positions (RoPE), store k and v in the paged cache. Returns q."""
    freqs = positions.float()[:, None] * inv_freq[None, :]  # (T, d/2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos().to(q.dtype)[:, None, :], emb.sin().to(q.dtype)[:, None, :]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    k_cache[slots] = k
    v_cache[slots] = v
    return q


def _table_slots(table: torch.Tensor, block_size: int) -> torch.Tensor:
    """Block table (..., nb) -> cache slot of every position (..., nb * block_size)."""
    offs = torch.arange(block_size, device=table.device)
    return (table.long()[..., None] * block_size + offs).flatten(-2)


def paged_attention(q, k_cache, v_cache, meta: AttentionMetadata, scale: float) -> torch.Tensor:
    """q (T, nh, d) -> out (T, nh, d), attending over each sequence's cached keys/values."""
    T, nh, d = q.shape
    nkv = k_cache.shape[1]
    G = nh // nkv
    out = torch.empty_like(q)

    if meta.num_decode:
        # All decode sequences at once: gather their pages into a padded (B, S) block,
        # mask the padding, and do grouped-query attention with two batched matmuls.
        B = meta.num_decode
        slots = meta.scratch.get("decode_slots")
        if slots is None:
            slots = _table_slots(meta.decode_tables, meta.block_size)  # (B, S)
            valid = torch.arange(slots.shape[1], device=q.device) < meta.decode_ctx_lens[:, None].long()
            meta.scratch["decode_slots"] = slots
            meta.scratch["decode_valid"] = valid
        valid = meta.scratch["decode_valid"]
        K = k_cache[slots].permute(0, 2, 3, 1)  # (B, nkv, d, S)
        V = v_cache[slots].permute(0, 2, 1, 3)  # (B, nkv, S, d)
        Q = q[meta.decode_rows].view(B, nkv, G, d)
        scores = torch.matmul(Q, K).float() * scale  # (B, nkv, G, S)
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        out[meta.decode_rows] = torch.matmul(probs, V).reshape(B, nh, d)

    for i, (q_start, q_len, ctx_len) in enumerate(meta.prefill):
        key = ("prefill_slots", i)
        ctx_slots = meta.scratch.get(key)
        if ctx_slots is None:
            ctx_slots = meta.scratch[key] = _table_slots(meta.prefill_tables[i], meta.block_size)[:ctx_len]
        K, V = k_cache[ctx_slots], v_cache[ctx_slots]
        if G > 1:  # grouped-query attention: share each KV head across G query heads
            K = K.repeat_interleave(G, dim=1)
            V = V.repeat_interleave(G, dim=1)
        # causal: query at absolute pos (ctx_len - q_len + i) sees keys <= that pos
        mask = torch.ones(q_len, ctx_len, dtype=torch.bool, device=q.device).tril(ctx_len - q_len)
        o = F.scaled_dot_product_attention(
            q[q_start:q_start + q_len].transpose(0, 1), K.transpose(0, 1), V.transpose(0, 1), attn_mask=mask, scale=scale
        )
        out[q_start:q_start + q_len] = o.transpose(0, 1)
    return out


def linear(x: torch.Tensor, w, bias=None) -> torch.Tensor:
    """F.linear, or a dequantize-then-matmul for quantized weights."""
    if isinstance(w, torch.Tensor):
        return F.linear(x, w, bias)
    return F.linear(x, w.dequantize(), bias)
