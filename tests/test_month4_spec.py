"""Speculative decoding: exact rejection rule, greedy equivalence, adaptive gamma."""
import torch

from engine.llm_engine import LLMEngine
from engine.scheduler import SamplingParams
from engine.speculative import expected_tokens, verify
from tests.conftest import GREEDY, hf_greedy, prompts


def test_verify_preserves_the_target_distribution():
    """Draft from q, accept/reject against p: the FIRST emitted token must follow p exactly."""
    g = torch.Generator().manual_seed(0)
    p = torch.tensor([[0.5, 0.3, 0.2, 0.0], [0.25, 0.25, 0.25, 0.25]])
    q = torch.tensor([[0.1, 0.1, 0.4, 0.4]])
    counts = torch.zeros(4)
    n = 20000
    for _ in range(n):
        d = int(torch.multinomial(q[0], 1, generator=g))
        out = verify([d], q, p, generator=g)
        counts[out[0]] += 1
    assert torch.allclose(counts / n, p[0], atol=0.015), counts / n


def test_verify_greedy_keeps_matching_prefix():
    V = 5
    onehot = lambda ids: torch.nn.functional.one_hot(torch.tensor(ids), V).float()  # noqa: E731
    p = onehot([1, 2, 3, 4])            # target argmaxes at 4 positions
    assert verify([1, 2, 3], onehot([1, 2, 3]), p) == [1, 2, 3, 4]   # all accepted + bonus
    assert verify([1, 0, 3], onehot([1, 0, 3]), p) == [1, 2]         # stop at first mismatch, take target's token


def test_expected_tokens_formula():
    assert expected_tokens(0.0, 4) == 1.0
    assert abs(expected_tokens(0.5, 2) - 1.75) < 1e-9
    assert expected_tokens(1.0, 3) == 4.0


def test_spec_greedy_matches_target_alone(tiny, tiny_draft):
    hf, ours = tiny
    ps = prompts(4, seed=31)
    engine = LLMEngine(ours, num_kv_blocks=256, block_size=4, draft_model=tiny_draft,
                       num_speculative_tokens=3, adaptive_speculation=False)
    seqs = engine.generate(ps, SamplingParams(max_tokens=16, **GREEDY))
    for p, s in zip(ps, seqs):
        assert s.output_ids == hf_greedy(hf, p, 16)
    st = engine.stats()
    assert st["spec_rounds"] > 0 and st["spec_proposed"] > 0
    assert engine.cache.num_used_blocks == 0


def test_self_draft_accepts_everything(tiny):
    """Drafting with the target itself: every greedy guess is right, so each round yields gamma + 1."""
    hf, ours = tiny
    ps = prompts(2, seed=32)
    engine = LLMEngine(ours, num_kv_blocks=256, block_size=4, draft_model=ours,
                       num_speculative_tokens=4, adaptive_speculation=False)
    seqs = engine.generate(ps, SamplingParams(max_tokens=21, **GREEDY))
    for p, s in zip(ps, seqs):
        assert s.output_ids == hf_greedy(hf, p, 21)
    st = engine.stats()
    assert st["spec_acceptance_rate"] == 1.0
    assert st["spec_tokens_per_round"] == 5.0


def test_spec_with_sampling_and_preemption_runs_clean(tiny, tiny_draft):
    _, ours = tiny
    ps = prompts(6, seed=33)
    engine = LLMEngine(ours, num_kv_blocks=30, block_size=4, max_num_seqs=6, draft_model=tiny_draft)
    seqs = engine.generate(ps, SamplingParams(max_tokens=24, temperature=0.8, top_k=20, top_p=0.95, ignore_eos=True))
    assert all(len(s.output_ids) == 24 for s in seqs)
    assert engine.scheduler.num_preemptions > 0
    assert engine.cache.num_used_blocks == 0


def test_adaptive_gamma_backs_off_when_drafts_fail(tiny, tiny_draft):
    _, ours = tiny
    engine = LLMEngine(ours, num_kv_blocks=64, block_size=4, draft_model=tiny_draft, num_speculative_tokens=6)
    spec = engine.spec
    seq = engine.add_request([1, 2, 3], SamplingParams(max_tokens=4))
    spec.cost_ratio = 0.3
    high = spec.gamma(seq)
    for _ in range(30):
        spec.observe(seq, 4, 0)  # every draft rejected
    assert high > 0 and spec.gamma(seq) == 0
