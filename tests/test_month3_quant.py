"""Weight-only int8 / int4 quantization with group scales."""
import pytest
import torch

from engine.batch import BatchItem, run_model
from engine.kv_cache import PagedKVCache
from engine.quant import pick_group_size, quantize_weight
from tests.conftest import make_hf, prompts, to_vrtx


@pytest.mark.parametrize("bits,tol", [(8, 0.01), (4, 0.2)])
def test_roundtrip_error_is_bounded_by_half_a_step(bits, tol):
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    qw = quantize_weight(w, bits, group_size=64)
    err = (qw.dequantize() - w).abs()
    step = qw.scales.repeat_interleave(64, dim=1)
    assert (err <= step / 2 + 1e-6).all()          # rounding error is at most half a quantization step
    assert err.mean() / w.abs().mean() < tol


def test_int4_packs_two_values_per_byte():
    w = torch.randn(8, 128)
    q4, q8 = quantize_weight(w, 4, 32), quantize_weight(w, 8, 32)
    assert q4.qweight.shape == (8, 64) and q4.qweight.dtype == torch.uint8
    assert q8.qweight.shape == (8, 128) and q8.qweight.dtype == torch.int8
    assert q4.int_values().min() >= -8 and q4.int_values().max() <= 7
    assert q4.nbytes() < q8.nbytes() < w.numel() * 2


def test_pick_group_size():
    assert pick_group_size(896) == 128 and pick_group_size(4864) == 128 and pick_group_size(96) == 32


@pytest.mark.parametrize("bits", [8, 4])
def test_quantized_model_logits_stay_close(bits):
    # Realistic weight scale: the default 0.2 init makes a chaotic network that amplifies any error.
    hf = make_hf(seed=0, hidden=128, heads=4, kv_heads=2, init=0.05)
    ref, q = to_vrtx(hf), to_vrtx(hf)
    q.quantize(bits, group_size=32)
    assert q.weight_bytes()["linear_bytes"] < ref.weight_bytes()["linear_bytes"] * (0.3 if bits == 8 else 0.2)
    prompt = prompts(1, seed=4, lo=20, hi=30)[0]
    out = []
    for m in (ref, q):
        cache = PagedKVCache(m.cfg, 16, 4)
        out.append(run_model(m, cache, [BatchItem(list(range(16)), 0, prompt, len(prompt))]))
    a, b = out
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    assert cos.min() > (0.999 if bits == 8 else 0.9), cos.min()
