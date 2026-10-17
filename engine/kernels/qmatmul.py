"""Weight-only quantized matmul: Y = X @ dequant(W)^T.

Two Triton paths, and a router that times them once per shape and keeps the faster one:

  fused    (small M, i.e. decode) each (BLOCK_K x BLOCK_N) tile of int8 / packed int4 codes
           is loaded, converted to fp16 (exact: the codes are small integers) and fed to
           the tensor cores; the group scale is applied to the tile's partial product, once
           per output column, not once per weight. Global memory sees 1 byte (int8) or half a
           byte (int4) per weight, which is what decode is bound by.
  dequant  (large M, i.e. prefill) one pass writes the fp16 weight, then cuBLAS. Big
           matmuls are compute-bound, and cuBLAS beats a hand-written kernel there.

First T4 run of the old kernel (scale applied per element, fixed 64x64 tiles, BM up to 64):
2-3x faster than the PyTorch dequant, but 4-7x slower than fp16 cuBLAS, and 5x slower than
the PyTorch dequant at M=64 (register spills). Hence the rewrite, the autotuned tile shapes,
and the router, which guarantees the quantized path is never slower than dequant + cuBLAS.
"""
import math
import os

import torch
import triton
import triton.language as tl

from engine.quant import QuantizedWeight

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


@triton.jit
def _qmatmul_kernel(X, Wq, S, Bias, Y, M, N, K,
                    stride_xm, stride_wn, stride_sn, stride_ym,
                    GROUP: tl.constexpr, BITS: tl.constexpr, HAS_BIAS: tl.constexpr,
                    BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
    # BK divides GROUP (and K/2 for int4), so a K tile never straddles two scale groups.
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mmask = (rm < M)[:, None]
    nmask = rn < N
    xp = X + rm[:, None] * stride_xm + rk[None, :]   # (BM, BK)
    wp = Wq + rn[None, :] * stride_wn + rk[:, None]  # (BK, BN): W^T tile, no transpose needed
    sp = S + rn * stride_sn
    acc = tl.zeros([BM, BN], tl.float32)

    if BITS == 8:
        for k0 in range(0, K, BK):
            x = tl.load(xp + k0, mask=mmask, other=0.0)
            w = tl.load(wp + k0, mask=nmask[None, :], other=0).to(x.dtype)
            s = tl.load(sp + k0 // GROUP, mask=nmask, other=0.0).to(tl.float32)
            acc += tl.dot(x, w) * s[None, :]
    else:
        HALF = K // 2  # byte j holds column j (low nibble) and column j + K/2 (high nibble)
        for k0 in range(0, HALF, BK):
            x_lo = tl.load(xp + k0, mask=mmask, other=0.0)
            x_hi = tl.load(xp + k0 + HALF, mask=mmask, other=0.0)
            p = tl.load(wp + k0, mask=nmask[None, :], other=0).to(tl.int32)
            lo = ((p & 15) - 8).to(x_lo.dtype)
            hi = ((p >> 4) - 8).to(x_lo.dtype)
            s_lo = tl.load(sp + k0 // GROUP, mask=nmask, other=0.0).to(tl.float32)
            s_hi = tl.load(sp + (k0 + HALF) // GROUP, mask=nmask, other=0.0).to(tl.float32)
            acc += tl.dot(x_lo, lo) * s_lo[None, :] + tl.dot(x_hi, hi) * s_hi[None, :]

    if HAS_BIAS:
        acc += tl.load(Bias + rn, mask=nmask, other=0.0).to(tl.float32)[None, :]
    tl.store(Y + rm[:, None] * stride_ym + rn[None, :], acc.to(Y.dtype.element_ty), mask=mmask & nmask[None, :])


@triton.jit
def _dequant_kernel(Wq, S, Out, N, K, stride_wn, stride_sn, stride_on,
                    GROUP: tl.constexpr, BITS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    rk = tl.program_id(1) * BK + tl.arange(0, BK)
    nm = (rn < N)[:, None]
    span = K if BITS == 8 else K // 2
    m = nm & (rk < span)[None, :]
    q = tl.load(Wq + rn[:, None] * stride_wn + rk[None, :], mask=m, other=0).to(tl.int32)
    out = Out + rn[:, None] * stride_on + rk[None, :]
    srow = S + rn[:, None] * stride_sn
    if BITS == 8:
        s = tl.load(srow + (rk // GROUP)[None, :], mask=m, other=0.0).to(tl.float32)
        tl.store(out, (q.to(tl.float32) * s).to(Out.dtype.element_ty), mask=m)
    else:
        s_lo = tl.load(srow + (rk // GROUP)[None, :], mask=m, other=0.0).to(tl.float32)
        s_hi = tl.load(srow + ((rk + span) // GROUP)[None, :], mask=m, other=0.0).to(tl.float32)
        tl.store(out, (((q & 15) - 8).to(tl.float32) * s_lo).to(Out.dtype.element_ty), mask=m)
        tl.store(out + span, (((q >> 4) - 8).to(tl.float32) * s_hi).to(Out.dtype.element_ty), mask=m)


_CONFIGS = [triton.Config({"BN": bn}, num_warps=w, num_stages=s)
            for bn, w in ((16, 2), (32, 4), (64, 4), (128, 4), (128, 8)) for s in (2, 3, 4)]
_tuned = None if INTERPRET else triton.autotune(_CONFIGS, key=["BM", "N", "K", "BITS", "GROUP"])(_qmatmul_kernel)

# (M bucket, N, K, bits, group, dtype) -> "fused" | "dequant"
_route: dict = {}


def _block_k(w: QuantizedWeight) -> int:
    span = w.group_size if w.bits == 8 else math.gcd(w.group_size, w.shape[1] // 2)
    return min(span & -span, 128)  # largest power of two dividing span, capped


def _bucket(M: int) -> int:
    return 16 if M <= 16 else 32 if M <= 32 else 64 if M <= 64 else 0


def dequantize(w: QuantizedWeight) -> torch.Tensor:
    N, K = w.shape
    out = torch.empty(N, K, dtype=w.dtype, device=w.qweight.device)
    span = K if w.bits == 8 else K // 2
    BN, BK = 32, 128
    _dequant_kernel[(triton.cdiv(N, BN), triton.cdiv(span, BK))](
        w.qweight, w.scales, out, N, K, w.qweight.stride(0), w.scales.stride(0), out.stride(0),
        GROUP=w.group_size, BITS=w.bits, BN=BN, BK=BK, num_warps=4)
    return out


def _dequant_linear(x, w, bias):
    return torch.nn.functional.linear(x, dequantize(w), bias)


def _fused(x, w, bias, BM, BK):
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
    args = (x, w.qweight, w.scales, bias if bias is not None else x, y, M, N, K,
            x.stride(0), w.qweight.stride(0), w.scales.stride(0), y.stride(0))
    meta = dict(GROUP=w.group_size, BITS=w.bits, HAS_BIAS=bias is not None, BM=BM, BK=BK)
    if _tuned is None:
        _qmatmul_kernel[(triton.cdiv(N, 32), triton.cdiv(M, BM))](*args, **meta, BN=32, num_warps=4)
    else:
        _tuned[lambda m: (triton.cdiv(N, m["BN"]), triton.cdiv(M, BM))](*args, **meta)
    return y


def _time_ms(fn, reps=20) -> float:
    for _ in range(3):
        fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / reps


def linear(x: torch.Tensor, w, bias=None) -> torch.Tensor:
    if isinstance(w, torch.Tensor):
        return torch.nn.functional.linear(x, w, bias)
    w: QuantizedWeight
    x = x.contiguous()
    M = x.shape[0]
    if M == 0:
        return torch.empty(0, w.shape[0], dtype=x.dtype, device=x.device)
    BM, BK = _bucket(M), _block_k(w)
    if not BM or BK < 16:
        return _dequant_linear(x, w, bias)
    if _tuned is not None:
        key = (BM, *w.shape, w.bits, w.group_size, x.dtype)
        route = _route.get(key)
        if route is None:  # first call for this shape: autotune the fused kernel, then race it
            fused = _time_ms(lambda: _fused(x, w, bias, BM, BK))
            deq = _time_ms(lambda: _dequant_linear(x, w, bias))
            route = _route[key] = "fused" if fused <= deq else "dequant"
        if route == "dequant":
            return _dequant_linear(x, w, bias)
    return _fused(x, w, bias, BM, BK)


@torch.inference_mode()
def warmup(weights, dtype) -> None:
    """Autotune and route every (shape, M bucket) up front, so it never lands inside a timed run."""
    if _tuned is None:
        return
    seen = set()
    for w in weights:
        if not isinstance(w, QuantizedWeight) or (w.shape, w.bits, w.group_size) in seen:
            continue
        seen.add((w.shape, w.bits, w.group_size))
        for M in (16, 32, 64):
            linear(torch.randn(M, w.shape[1], dtype=dtype, device=w.qweight.device), w)
