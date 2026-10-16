"""Correctness tests for VRTX Engine, using a tiny randomly initialized Qwen2 model.

No download needed. The reference is Hugging Face's own implementation, so if our
forward pass, RoPE, GQA, paged cache or scheduler has a bug, these fail.
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from engine.batch import BatchItem, run_model  # noqa: E402
from engine.kv_cache import BlockAllocator, NoFreeBlocks, PagedKVCache  # noqa: E402
from engine.llm_engine import LLMEngine  # noqa: E402
from engine.model import ModelConfig, VRTXModel  # noqa: E402
from engine.scheduler import SamplingParams  # noqa: E402
from tests.conftest import GREEDY, hf_greedy, prompts  # noqa: E402


def test_prefill_logits_match_hf(tiny):
    hf, ours = tiny
    prompt = prompts(1)[0]
    engine = LLMEngine(ours, num_kv_blocks=16, block_size=4)
    seq = engine.add_request(prompt, SamplingParams(max_tokens=1, **GREEDY))
    batch = engine.scheduler.schedule()
    (s, n), = batch
    got = run_model(ours, engine.cache, [BatchItem(s.block_table, 0, prompt, n)])
    with torch.no_grad():
        want = hf(torch.tensor([prompt])).logits[0]
    assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()
    assert seq.id == s.id


def test_single_sequence_generation_matches_hf(tiny):
    hf, ours = tiny
    prompt = prompts(1, seed=7)[0]
    engine = LLMEngine(ours, num_kv_blocks=32, block_size=4)
    (seq,) = engine.generate([prompt], SamplingParams(max_tokens=20, **GREEDY))
    assert seq.output_ids == hf_greedy(hf, prompt, 20)


def test_batched_generation_matches_one_at_a_time(tiny):
    hf, ours = tiny
    ps = prompts(6, seed=3)
    engine = LLMEngine(ours, num_kv_blocks=128, block_size=4, max_num_seqs=6)
    seqs = engine.generate(ps, SamplingParams(max_tokens=15, **GREEDY))
    for p, s in zip(ps, seqs):
        assert s.output_ids == hf_greedy(hf, p, 15)


def test_chunked_prefill_matches(tiny):
    hf, ours = tiny
    ps = prompts(4, seed=5)
    # A 7-token budget forces prompts to be split across several steps.
    engine = LLMEngine(ours, num_kv_blocks=128, block_size=4, max_num_batched_tokens=7)
    seqs = engine.generate(ps, SamplingParams(max_tokens=10, **GREEDY))
    for p, s in zip(ps, seqs):
        assert s.output_ids == hf_greedy(hf, p, 10)


def test_preemption_under_memory_pressure_still_correct(tiny):
    hf, ours = tiny
    ps = prompts(6, seed=11)
    # Only 20 blocks x 4 slots = 80 tokens of cache for 6 sequences: preemption must happen.
    engine = LLMEngine(ours, num_kv_blocks=20, block_size=4, max_num_seqs=6)
    seqs = engine.generate(ps, SamplingParams(max_tokens=20, **GREEDY))
    assert engine.scheduler.num_preemptions > 0
    for p, s in zip(ps, seqs):
        assert s.output_ids == hf_greedy(hf, p, 20)
    assert engine.cache.num_used_blocks == 0  # everything freed


def test_requests_arriving_mid_flight(tiny):
    hf, ours = tiny
    a, b = prompts(2, seed=13)
    engine = LLMEngine(ours, num_kv_blocks=64, block_size=4)
    sa = engine.add_request(a, SamplingParams(max_tokens=12, **GREEDY))
    for _ in range(5):
        engine.step()
    sb = engine.add_request(b, SamplingParams(max_tokens=12, **GREEDY))  # joins a running batch
    while engine.has_unfinished():
        engine.step()
    assert sa.output_ids == hf_greedy(hf, a, 12)
    assert sb.output_ids == hf_greedy(hf, b, 12)


def test_allocator_refcounts():
    alloc = BlockAllocator(2)
    a = alloc.allocate()
    alloc.share(a)
    alloc.release(a)
    assert alloc.num_free == 1  # still held by one owner
    alloc.release(a)
    assert alloc.num_free == 2
    alloc.allocate()
    alloc.allocate()
    with pytest.raises(NoFreeBlocks):
        alloc.allocate()


def test_kv_bytes_per_token_formula():
    cfg = ModelConfig(151936, 896, 4864, 24, 14, 2, 64, 1e-6, 1e6, True, True, 32768)  # Qwen2.5-0.5B
    assert PagedKVCache.bytes_per_token(cfg, torch.float16) == 2 * 24 * 2 * 64 * 2  # 12 KB


GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None


@pytest.mark.skipif(GPU is None, reason="needs a cuda or mps GPU")
def test_fp16_on_gpu_matches_fp32_cpu(tiny):
    """The real deployment path (fp16 weights + KV cache on a GPU) must agree with the fp32 CPU reference."""
    hf, ours = tiny
    gpu = VRTXModel(ours.cfg, hf.state_dict(), device=GPU, dtype=torch.float16)
    ps = prompts(4, seed=17)
    ref = LLMEngine(ours, num_kv_blocks=64, block_size=4).generate(ps, SamplingParams(max_tokens=8, **GREEDY))
    got = LLMEngine(gpu, num_kv_blocks=64, block_size=4).generate(ps, SamplingParams(max_tokens=8, **GREEDY))
    assert gpu.embed.dtype == torch.float16 and gpu.embed.device.type == GPU
    # fp16 rounding can flip a near-tie late in a sequence, so require the first tokens to match.
    for r, g in zip(ref, got):
        assert g.output_ids[:4] == r.output_ids[:4]
