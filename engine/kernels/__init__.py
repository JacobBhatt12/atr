"""Triton kernels for VRTX Engine (Month 3).

    norm.py        fused residual add + RMSNorm
    rope.py        fused RoPE on q and k + write k, v into the paged cache
    activation.py  fused SiLU(gate) * up
    attention.py   paged attention: decode (one query per sequence) and prefill (flash-style, causal)
    qmatmul.py     int8 / int4 weight-only matmul that dequantizes inside the kernel

Every function here has the same signature as its PyTorch twin in engine/ops.py.
Triton needs an NVIDIA (or AMD) GPU. On a laptop the model uses engine/ops.py instead.
Tests can still run these kernels on CPU through Triton's interpreter: TRITON_INTERPRET=1.
"""
from __future__ import annotations

import os

try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:  # macOS, or a CPU-only install
    HAS_TRITON = False

KERNELS = ("norm", "rope", "act", "attention", "qmatmul")


def triton_usable(device) -> bool:
    import torch

    if not HAS_TRITON:
        return False
    if os.environ.get("TRITON_INTERPRET") == "1":
        return True
    return torch.device(device).type == "cuda"
