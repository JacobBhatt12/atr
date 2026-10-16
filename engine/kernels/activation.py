"""SwiGLU activation: silu(gate) * up in one pass over the fused gate|up matmul output."""
import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(X, Y, stride_x, stride_y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    gate = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    up = tl.load(X + row * stride_x + N + cols, mask=mask, other=0.0)
    g = gate.to(tl.float32)
    act = (g / (1.0 + tl.exp(-g))).to(gate.dtype)  # silu, rounded like PyTorch
    tl.store(Y + row * stride_y + cols, act * up, mask=mask)


def silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    T, N2 = gate_up.shape
    N = N2 // 2
    assert gate_up.stride(-1) == 1
    y = torch.empty(T, N, dtype=gate_up.dtype, device=gate_up.device)
    BLOCK = 1024
    if T:
        _silu_mul_kernel[(T, triton.cdiv(N, BLOCK))](gate_up, y, gate_up.stride(0), y.stride(0), N, BLOCK=BLOCK)
    return y
