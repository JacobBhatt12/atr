"""RMSNorm, optionally fused with the residual add that comes right before it.

Unfused, `x = x + r; h = rms_norm(x)` reads x and r, writes x, reads x again, and
launches about six PyTorch kernels (add, cast, pow, mean, rsqrt, mul, mul).
Fused: one launch, each row read once, written once.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(X, R, W, Y, stride_x, stride_r, stride_y, N, eps,
                     HAS_RESIDUAL: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    if HAS_RESIDUAL:
        r = tl.load(R + row * stride_r + cols, mask=mask, other=0.0)
        x = x + r  # same rounding as PyTorch: add in the model dtype
        tl.store(X + row * stride_x + cols, x, mask=mask)
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / N
    y = (xf * tl.rsqrt(var + eps)).to(x.dtype)
    w = tl.load(W + cols, mask=mask, other=0.0)
    tl.store(Y + row * stride_y + cols, w * y, mask=mask)


def _launch(x, r, weight, eps):
    T, N = x.shape
    assert x.stride(-1) == 1
    y = torch.empty_like(x)
    if T:
        _rms_norm_kernel[(T,)](
            x, r if r is not None else x, weight, y,
            x.stride(0), r.stride(0) if r is not None else 0, y.stride(0), N, eps,
            HAS_RESIDUAL=r is not None, BLOCK=triton.next_power_of_2(N), num_warps=4 if N <= 2048 else 8,
        )
    return y


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _launch(x, None, weight, eps)


def add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    """In place: x += residual. Returns (x, rms_norm(x))."""
    x = x.contiguous()
    y = _launch(x, residual.contiguous(), weight, eps)
    return x, y
