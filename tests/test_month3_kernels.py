"""Triton kernels vs the PyTorch reference ops.

Runs on a CUDA GPU, or anywhere with Triton installed and TRITON_INTERPRET=1 (CI does this
on a CPU runner). Skipped otherwise, e.g. on macOS where Triton is not available.
"""
import pytest
import torch

from engine import ops
from engine.kernels import triton_usable

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(not triton_usable(DEVICE), reason="needs Triton (CUDA, or TRITON_INTERPRET=1)")
DT = torch.float16 if DEVICE == "cuda" else torch.float32
TOL = dict(atol=2e-2, rtol=2e-2) if DT == torch.float16 else dict(atol=1e-4, rtol=1e-4)

from engine.batch import BatchItem, build_batch  # noqa: E402


def rand(*shape):
    return torch.randn(*shape, device=DEVICE).to(DT)


def test_rms_norm_and_fused_add():
    from engine.kernels import norm

    x, r, w = rand(7, 96), rand(7, 96), rand(96)
    torch.testing.assert_close(norm.rms_norm(x, w, 1e-6), ops.rms_norm(x, w, 1e-6), **TOL)
    want_x, want_h = ops.add_rms_norm(x, r, w, 1e-6)
    got_x, got_h = norm.add_rms_norm(x.clone(), r, w, 1e-6)
    torch.testing.assert_close(got_x, want_x, **TOL)
    torch.testing.assert_close(got_h, want_h, **TOL)


def test_silu_and_mul():
    from engine.kernels import activation

    x = rand(5, 2 * 1100)  # not a multiple of the block, exercises masking
    torch.testing.assert_close(activation.silu_and_mul(x), ops.silu_and_mul(x), **TOL)


def test_rope_and_cache():
    from engine.kernels import rope

    T, nh, nkv, d = 6, 6, 2, 32
    qkv = rand(T, (nh + 2 * nkv) * d)
    q, k, v = qkv.split([nh * d, nkv * d, nkv * d], dim=-1)  # strided views, like the model
    pos = torch.tensor([0, 1, 2, 7, 30, 31], device=DEVICE)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2, device=DEVICE).float() / d))
    slots = torch.tensor([5, 0, 9, 3, 11, 2], device=DEVICE)
    kc1, vc1 = torch.zeros(16, nkv, d, device=DEVICE, dtype=DT), torch.zeros(16, nkv, d, device=DEVICE, dtype=DT)
    kc2, vc2 = kc1.clone(), vc1.clone()
    want = ops.rope_and_cache(q.reshape(T, nh, d), k.reshape(T, nkv, d), v.reshape(T, nkv, d), pos, inv_freq, kc1, vc1, slots)
    got = rope.rope_and_cache(q.view(T, nh, d), k.view(T, nkv, d), v.view(T, nkv, d), pos, inv_freq, kc2, vc2, slots)
    torch.testing.assert_close(got, want, **TOL)
    torch.testing.assert_close(kc2, kc1, **TOL)
    torch.testing.assert_close(vc2, vc1, **TOL)


@pytest.mark.parametrize("nh,nkv", [(4, 2), (7, 1), (4, 4)])
def test_paged_attention_mixed_batch(nh, nkv):
    """Decode rows and prefill chunks (including a chunk on top of existing context) in one batch."""
    from engine.kernels import attention

    d, bs, num_blocks = 32, 4, 64
    g = torch.Generator().manual_seed(0)
    kc, vc = rand(num_blocks * bs, nkv, d), rand(num_blocks * bs, nkv, d)
    perm = torch.randperm(num_blocks, generator=g).tolist()  # scattered pages
    items, used = [], 0
    for start, n in [(9, 1), (0, 37), (40, 1), (20, 5), (1, 1), (64, 33)]:
        nb = (start + n + bs - 1) // bs
        items.append(BatchItem(perm[used:used + nb], start, [0] * n, 1))
        used += nb
    _, _, meta, _ = build_batch(items, bs, DEVICE)
    T = sum(len(it.tokens) for it in items)
    q = rand(T, nh, d)
    want = ops.paged_attention(q, kc, vc, meta, d ** -0.5)
    meta.scratch.clear()
    got = attention.paged_attention(q, kc, vc, meta, d ** -0.5)
    torch.testing.assert_close(got, want, **TOL)


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("M", [1, 5, 40, 300])
def test_quantized_matmul(bits, M):
    from engine.kernels import qmatmul
    from engine.quant import quantize_weight

    N, K = 96, 256
    w = rand(N, K)
    qw = quantize_weight(w, bits, group_size=64)
    x, b = rand(M, K), rand(N)
    want = torch.nn.functional.linear(x.float(), qw.dequantize().float(), b.float()).to(DT)
    got = qmatmul.linear(x, qw, b)
    torch.testing.assert_close(got, want, atol=5e-2 if DT == torch.float16 else 1e-3, rtol=2e-2)
    torch.testing.assert_close(qmatmul.dequantize(qw), qw.dequantize())


def test_full_model_with_all_kernels_matches_pytorch(tiny):
    from tests.conftest import prompts
    from engine.llm_engine import LLMEngine
    from engine.model import VRTXModel
    from engine.scheduler import SamplingParams

    hf, ours = tiny
    fast = VRTXModel(ours.cfg, hf.state_dict(), device=DEVICE, dtype=DT)
    fast.set_kernels(("norm", "rope", "act", "attention"))
    ref = VRTXModel(ours.cfg, hf.state_dict(), device=DEVICE, dtype=DT, kernels=())
    ps = prompts(4, seed=41)
    p = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True)
    a = LLMEngine(ref, num_kv_blocks=64, block_size=4, max_num_batched_tokens=16).generate(ps, p)
    b = LLMEngine(fast, num_kv_blocks=64, block_size=4, max_num_batched_tokens=16).generate(ps, p)
    for x, y in zip(a, b):
        assert y.output_ids[:4] == x.output_ids[:4]
