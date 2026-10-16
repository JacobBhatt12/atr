"""Correctness tests for VRTX Engine, using a tiny randomly initialized Qwen2 model.

No download needed. The reference is Hugging Face's own implementation, so if our
forward pass, RoPE, GQA, paged cache or scheduler has a bug, these fail.
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from engine.kv_cache import BlockAllocator, NoFreeBlocks, PagedKVCache  # noqa: E402
from engine.llm_engine import LLMEngine  # noqa: E402
from engine.model import ModelConfig, VRTXModel  # noqa: E402
from engine.scheduler import SamplingParams  # noqa: E402

GREEDY = dict(temperature=0.0, ignore_eos=True)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # grouped-query attention, like the real Qwen
        max_position_embeddings=512,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        initializer_range=0.2,
    )
    hf = transformers.Qwen2ForCausalLM(cfg).eval()
    ours = VRTXModel(ModelConfig.from_hf(hf.config), hf.state_dict())
    return hf, ours


def hf_greedy(hf, prompt, n):
    ids = list(prompt)
    with torch.no_grad():
        for _ in range(n):
            logits = hf(torch.tensor([ids])).logits[0, -1]
            ids.append(int(logits.argmax()))
    return ids[len(prompt):]


def prompts(k, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 97, (int(torch.randint(3, 40, (1,), generator=g)),), generator=g).tolist() for _ in range(k)]


def test_prefill_logits_match_hf(tiny):
    hf, ours = tiny
    prompt = prompts(1)[0]
    engine = LLMEngine(ours, num_kv_blocks=16, block_size=4)
    seq = engine.add_request(prompt, SamplingParams(max_tokens=1, **GREEDY))
    batch = engine.scheduler.schedule()
    (s, n), = batch
    slots = engine.cache.slots(s.block_table, 0, n)
    rows = torch.arange(n)
    got = ours.forward(torch.tensor(prompt), torch.arange(n), [(0, n, n, slots)], slots, rows, engine.cache)
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
