"""Swap preemption: out of memory, KV blocks go to CPU RAM and come back intact."""
from engine.llm_engine import LLMEngine
from engine.scheduler import SamplingParams
from tests.conftest import GREEDY, hf_greedy, prompts


def test_swap_preemption_matches_hf(tiny):
    hf, ours = tiny
    ps = prompts(6, seed=11)
    engine = LLMEngine(ours, num_kv_blocks=20, block_size=4, max_num_seqs=6, preemption_mode="swap")
    seqs = engine.generate(ps, SamplingParams(max_tokens=20, **GREEDY))
    assert engine.scheduler.num_swaps > 0
    for p, s in zip(ps, seqs):
        assert s.output_ids == hf_greedy(hf, p, 20)
    assert engine.cache.num_used_blocks == 0
    assert engine.scheduler.swap_used == 0


def test_swap_falls_back_to_recompute_when_swap_space_is_full(tiny):
    hf, ours = tiny
    ps = prompts(6, seed=11)
    engine = LLMEngine(ours, num_kv_blocks=20, block_size=4, max_num_seqs=6, preemption_mode="swap", swap_space_mb=0)
    seqs = engine.generate(ps, SamplingParams(max_tokens=20, **GREEDY))
    assert engine.scheduler.num_swaps == 0 and engine.scheduler.num_preemptions > 0
    for p, s in zip(ps, seqs):
        assert s.output_ids == hf_greedy(hf, p, 20)


def test_batched_decode_with_mixed_lengths(tiny):
    """Decode sequences with very different context lengths share one padded attention call."""
    hf, ours = tiny
    ps = prompts(5, seed=21, lo=2, hi=90)
    engine = LLMEngine(ours, num_kv_blocks=256, block_size=4, max_num_seqs=5)
    seqs = engine.generate(ps, SamplingParams(max_tokens=12, **GREEDY))
    for p, s in zip(ps, seqs):
        assert s.output_ids == hf_greedy(hf, p, 12)
