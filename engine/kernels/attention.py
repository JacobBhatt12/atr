"""Paged attention kernels: read K and V straight out of scattered cache blocks.

The Month 2 PyTorch path ran one gather + one attention call per sequence per layer.
At batch 64 on Qwen2.5-0.5B (24 layers) that is 1,536 small launches per step, and it is
why step time grew almost linearly with batch size. Here each group is ONE launch:

DECODE (one new token per sequence)
    grid = (sequences, kv_heads). A program loads the G query heads that share one KV
    head (grouped-query attention), walks the sequence's context BLOCK_N tokens at a time
    by looking up block numbers in its block table, and keeps a running softmax
    (max m, denominator l, weighted sum acc) so the scores never touch global memory.
    K and V are read once per KV head, not once per query head.

PREFILL (many new tokens per sequence: prompt chunks, speculative verification)
    Flash-attention style. grid = (query tiles, query heads). A tile is up to BLOCK_M
    query rows of one sequence. Keys come from the paged cache (the new tokens' K/V were
    already written there by the RoPE kernel), with the causal rule
        key position <= query position
    and the loop stops at the tile's last query position, so it skips masked blocks.
"""
import torch
import triton
import triton.language as tl

from engine.batch import AttentionMetadata


@triton.jit
def _online_softmax_step(s, m, l, acc, v):
    m_new = tl.maximum(m, tl.max(s, 1))
    alpha = tl.exp(m - m_new)
    p = tl.exp(s - m_new[:, None])
    l = l * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    return m_new, l, acc


@triton.jit
def _paged_decode_kernel(Q, KC, VC, O, Tables, CtxLens, Rows,
                         stride_qt, stride_qh, stride_cs, stride_ch, stride_ot, stride_oh, stride_tb, scale,
                         PAGE: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    row = tl.load(Rows + b).to(tl.int64)
    ctx = tl.load(CtxLens + b)
    g = tl.arange(0, GP)
    d = tl.arange(0, D)
    gmask = (g < G)[:, None]
    heads = kvh * G + g
    q = tl.load(Q + row * stride_qt + heads[:, None] * stride_qh + d[None, :], mask=gmask, other=0.0)

    m = tl.full([GP], float("-inf"), tl.float32)
    l = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, D], tl.float32)
    for start in range(0, ctx, BLOCK_N):
        pos = start + tl.arange(0, BLOCK_N)
        pmask = pos < ctx
        blk = tl.load(Tables + b * stride_tb + pos // PAGE, mask=pmask, other=0).to(tl.int64)
        off = (blk * PAGE + pos % PAGE)[:, None] * stride_cs + kvh * stride_ch + d[None, :]
        k = tl.load(KC + off, mask=pmask[:, None], other=0.0)
        v = tl.load(VC + off, mask=pmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(pmask[None, :], s, float("-inf"))
        m, l, acc = _online_softmax_step(s, m, l, acc, v)
    out = acc / l[:, None]
    tl.store(O + row * stride_ot + heads[:, None] * stride_oh + d[None, :], out.to(O.dtype.element_ty), mask=gmask)


@triton.jit
def _paged_prefill_kernel(Q, KC, VC, O, Tables, QStarts, QLens, CtxLens, TileSeq, TileRow,
                          stride_qt, stride_qh, stride_cs, stride_ch, stride_ot, stride_oh, stride_tb, scale,
                          PAGE: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    tile = tl.program_id(0)
    h = tl.program_id(1)
    kvh = h // G
    sid = tl.load(TileSeq + tile)
    m0 = tl.load(TileRow + tile)
    q_start = tl.load(QStarts + sid)
    q_len = tl.load(QLens + sid)
    ctx = tl.load(CtxLens + sid)

    r = m0 + tl.arange(0, BLOCK_M)
    rmask = r < q_len
    qpos = ctx - q_len + r                     # absolute position of each query row
    d = tl.arange(0, D)
    trow = (q_start + r).to(tl.int64)
    q = tl.load(Q + trow[:, None] * stride_qt + h * stride_qh + d[None, :], mask=rmask[:, None], other=0.0)

    m = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    end = tl.minimum(ctx, ctx - q_len + m0 + BLOCK_M)  # causal: nothing past the tile's last row
    for start in range(0, end, BLOCK_N):
        pos = start + tl.arange(0, BLOCK_N)
        pmask = pos < ctx
        blk = tl.load(Tables + sid * stride_tb + pos // PAGE, mask=pmask, other=0).to(tl.int64)
        off = (blk * PAGE + pos % PAGE)[:, None] * stride_cs + kvh * stride_ch + d[None, :]
        k = tl.load(KC + off, mask=pmask[:, None], other=0.0)
        v = tl.load(VC + off, mask=pmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        allowed = (pos[None, :] <= qpos[:, None]) & pmask[None, :]
        s = tl.where(allowed, s, float("-inf"))
        m, l, acc = _online_softmax_step(s, m, l, acc, v)
    out = acc / l[:, None]
    tl.store(O + trow[:, None] * stride_ot + h * stride_oh + d[None, :], out.to(O.dtype.element_ty), mask=rmask[:, None])


DECODE_BLOCK_N = 64
PREFILL_BLOCK_M = 32
PREFILL_BLOCK_N = 32


def paged_attention(q, k_cache, v_cache, meta: AttentionMetadata, scale: float) -> torch.Tensor:
    T, nh, d = q.shape
    nkv = k_cache.shape[1]
    G = nh // nkv
    assert q.stride(-1) == 1 and k_cache.is_contiguous() and v_cache.is_contiguous()
    out = torch.empty(T, nh, d, dtype=q.dtype, device=q.device)

    if meta.num_decode:
        tables = meta.decode_tables
        _paged_decode_kernel[(meta.num_decode, nkv)](
            q, k_cache, v_cache, out, tables, meta.decode_ctx_lens, meta.decode_rows,
            q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), out.stride(0), out.stride(1),
            tables.stride(0), scale,
            PAGE=meta.block_size, G=G, GP=max(16, triton.next_power_of_2(G)), D=d, BLOCK_N=DECODE_BLOCK_N,
            num_warps=4,
        )

    if meta.prefill:
        tiles = meta.scratch.get("prefill_tiles")
        if tiles is None:  # built once per step, reused by every layer
            seq_ids, rows = [], []
            for i, (_, q_len, _) in enumerate(meta.prefill):
                for r in range(0, q_len, PREFILL_BLOCK_M):
                    seq_ids.append(i)
                    rows.append(r)
            info = torch.tensor([list(x) for x in meta.prefill], dtype=torch.int32).T.contiguous()
            packed = torch.cat([info.flatten(), torch.tensor(seq_ids + rows, dtype=torch.int32)]).to(q.device)
            B, n = len(meta.prefill), len(seq_ids)
            q_starts, q_lens, ctx_lens, tile_seq, tile_row = packed.split([B, B, B, n, n])
            tiles = meta.scratch["prefill_tiles"] = (q_starts, q_lens, ctx_lens, tile_seq, tile_row)
        q_starts, q_lens, ctx_lens, tile_seq, tile_row = tiles
        tables = meta.prefill_tables
        _paged_prefill_kernel[(tile_seq.shape[0], nh)](
            q, k_cache, v_cache, out, tables, q_starts, q_lens, ctx_lens, tile_seq, tile_row,
            q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), out.stride(0), out.stride(1),
            tables.stride(0), scale,
            PAGE=meta.block_size, G=G, D=d, BLOCK_M=PREFILL_BLOCK_M, BLOCK_N=PREFILL_BLOCK_N,
            num_warps=4,
        )
    return out
