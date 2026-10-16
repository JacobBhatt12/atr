"""Weight-only quantization: int8 and int4 with one scale per group of input columns.

A weight matrix W (N outputs x K inputs) is cut into groups of `group_size` consecutive
input columns. Each group in each row gets its own scale:

    scale = max(|w| in group) / qmax          qmax = 127 (int8) or 7 (int4)
    q     = round(w / scale)                  stored as int8, or packed 2 per byte for int4
    w    ~= q * scale                         at matmul time

Small groups follow the weights closely (better perplexity) but store more scales.
Activations stay fp16, so only the weight READS shrink, which is what decode is bound by.

int4 packing: byte j of a row holds column j in its low nibble and column j + K/2 in its
high nibble (stored as q + 8, in 0..15). That keeps both halves contiguous, so the Triton
kernel can unpack with a shift and a mask and never has to interleave.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QuantizedWeight:
    qweight: torch.Tensor   # int8 (N, K) or uint8 (N, K // 2)
    scales: torch.Tensor    # (N, K // group_size), same dtype as the model
    bits: int
    group_size: int
    shape: tuple[int, int]

    @property
    def dtype(self) -> torch.dtype:
        return self.scales.dtype

    def nbytes(self) -> int:
        return self.qweight.numel() * self.qweight.element_size() + self.scales.numel() * self.scales.element_size()

    def numel(self) -> int:
        return self.shape[0] * self.shape[1]

    def int_values(self) -> torch.Tensor:
        """The integer codes as int8 (N, K)."""
        if self.bits == 8:
            return self.qweight
        lo = (self.qweight & 0xF).to(torch.int8) - 8
        hi = (self.qweight >> 4).to(torch.int8) - 8
        return torch.cat([lo, hi], dim=1)

    def dequantize(self) -> torch.Tensor:
        q = self.int_values().float()
        s = self.scales.float().repeat_interleave(self.group_size, dim=1)
        return (q * s).to(self.dtype)


def quantize_weight(w: torch.Tensor, bits: int = 8, group_size: int = 128) -> QuantizedWeight:
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    N, K = w.shape
    if K % group_size:
        raise ValueError(f"input dim {K} is not a multiple of group_size {group_size}")
    if bits == 4 and K % 2:
        raise ValueError("int4 needs an even input dim")
    qmax = 127 if bits == 8 else 7
    g = w.float().view(N, K // group_size, group_size)
    scale = (g.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-8)
    q = torch.round(g / scale).clamp(-qmax - 1, qmax).to(torch.int8).view(N, K)
    if bits == 4:
        u = (q + 8).to(torch.uint8)
        q = u[:, : K // 2] | (u[:, K // 2:] << 4)
    return QuantizedWeight(q.contiguous(), scale.view(N, -1).to(w.dtype).contiguous(), bits, group_size, (N, K))


def pick_group_size(K: int, preferred: int = 128) -> int:
    g = preferred
    while K % g:
        g //= 2
    return g
