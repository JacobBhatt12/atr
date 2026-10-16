"""Weight-only quantized matmul: Y = X @ dequant(W)^T, dequantizing inside the kernel.

The PyTorch fallback rebuilds the full fp16 weight on every call, which costs MORE
memory traffic than plain fp16. Here each (BLOCK_N x BLOCK_K) tile of int8 / packed int4
codes is loaded, scaled in registers and fed to the tensor cores, so global memory only
ever sees 1 byte (int8) or half a byte (int4) per weight. Decode is memory-bound, so
fewer bytes per weight should mean proportionally faster matmuls at small batch.
"""
import torch
import triton
import triton.language as tl

from engine.quant import QuantizedWeight


@triton.jit
def _qmatmul_kernel(X, Wq, S, Bias, Y, M, N, K,
                    stride_xm, stride_wn, stride_sn, stride_ym,
                    GROUP: tl.constexpr, BITS: tl.constexpr, HAS_BIAS: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    rn = (pid_n * BN + tl.arange(0, BN)).to(tl.int64)
    rk = tl.arange(0, BK)
    mmask = (rm < M)[:, None]
    nmask = (rn < N)[:, None]
    acc = tl.zeros([BM, BN], tl.float32)

    if BITS == 8:
        for k0 in range(0, K, BK):
            cols = k0 + rk
            kmask = (cols < K)[None, :]
            x = tl.load(X + rm[:, None] * stride_xm + cols[None, :], mask=mmask & kmask, other=0.0)
            w = tl.load(Wq + rn[:, None] * stride_wn + cols[None, :], mask=nmask & kmask, other=0)
            s = tl.load(S + rn[:, None] * stride_sn + (cols // GROUP)[None, :], mask=nmask & kmask, other=0.0)
            wf = (w.to(tl.float32) * s.to(tl.float32)).to(x.dtype)
            acc += tl.dot(x, tl.trans(wf))
    else:
        HALF = K // 2  # byte j holds column j (low nibble) and column j + K/2 (high nibble)
        for k0 in range(0, HALF, BK):
            cols = k0 + rk
            kmask = (cols < HALF)[None, :]
            p = tl.load(Wq + rn[:, None] * stride_wn + cols[None, :], mask=nmask & kmask, other=0).to(tl.int32)
            lo = ((p & 15) - 8).to(tl.float32)
            hi = ((p >> 4) - 8).to(tl.float32)
            s_lo = tl.load(S + rn[:, None] * stride_sn + (cols // GROUP)[None, :], mask=nmask & kmask, other=0.0)
            s_hi = tl.load(S + rn[:, None] * stride_sn + ((cols + HALF) // GROUP)[None, :], mask=nmask & kmask, other=0.0)
            x_lo = tl.load(X + rm[:, None] * stride_xm + cols[None, :], mask=mmask & kmask, other=0.0)
            x_hi = tl.load(X + rm[:, None] * stride_xm + (cols + HALF)[None, :], mask=mmask & kmask, other=0.0)
            acc += tl.dot(x_lo, tl.trans((lo * s_lo.to(tl.float32)).to(x_lo.dtype)))
            acc += tl.dot(x_hi, tl.trans((hi * s_hi.to(tl.float32)).to(x_hi.dtype)))

    if HAS_BIAS:
        acc += tl.load(Bias + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
    tl.store(Y + rm[:, None] * stride_ym + rn[None, :], acc.to(Y.dtype.element_ty), mask=mmask & (rn < N)[None, :])


def linear(x: torch.Tensor, w, bias=None) -> torch.Tensor:
    if isinstance(w, torch.Tensor):
        return torch.nn.functional.linear(x, w, bias)
    w: QuantizedWeight
    x = x.contiguous()
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
    if M == 0:
        return y
    if M >= 256:  # big prefill chunks are compute-bound: one dequantize + cuBLAS wins there
        return torch.nn.functional.linear(x, w.dequantize(), bias)
    BM = 16 if M <= 16 else 32 if M <= 32 else 64
    BN, BK = 64, 64
    _qmatmul_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        x, w.qweight, w.scales, bias if bias is not None else x, y, M, N, K,
        x.stride(0), w.qweight.stride(0), w.scales.stride(0), y.stride(0),
        GROUP=w.group_size, BITS=w.bits, HAS_BIAS=bias is not None, BM=BM, BN=BN, BK=BK,
        num_warps=4,
    )
    return y
