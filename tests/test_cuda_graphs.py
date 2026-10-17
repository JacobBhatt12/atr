"""CUDA-graph steps: padded static batches must give exactly the eager engine's tokens.

On CPU the runner emulates the graphs (same static buffers, padding and null block, run
eagerly), which needs the Triton kernels: CI runs it in Triton's interpreter.
"""
import pytest
import torch

from engine.kernels import triton_usable
from engine.llm_engine import LLMEngine
from engine.model import VRTXModel
from engine.scheduler import SamplingParams
from tests.conftest import GREEDY, hf_greedy, make_hf, prompts, to_vrtx

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(not triton_usable(DEVICE), reason="needs Triton (CUDA, or TRITON_INTERPRET=1)")
DT = torch.float16 if DEVICE == "cuda" else torch.float32


def model_of(tiny_pair):
    hf, ours = tiny_pair
    return VRTXModel(ours.cfg, hf.state_dict(), device=DEVICE, dtype=DT, kernels=("norm", "rope", "act", "attention"))


def test_graph_decode_matches_eager(tiny):
    m = model_of(tiny)
    ps = prompts(5, seed=51)
    p = SamplingParams(max_tokens=10, **GREEDY)
    kw = dict(num_kv_blocks=96, block_size=4, max_num_seqs=3, max_num_batched_tokens=24)
    eager = LLMEngine(m, cuda_graphs=False, **kw)
    a = eager.generate(ps, p)
    graphed = LLMEngine(m, cuda_graphs=True, **kw)
    b = graphed.generate(ps, p)
    n = 10 if DT == torch.float32 else 4  # fp16: allow late near-tie flips
    for x, y in zip(a, b):
        assert y.output_ids[:n] == x.output_ids[:n]
    st = graphed.stats()
    assert st["graph_replays"] > 0 and st["graph_eager_steps"] > 0  # decode steps graphed, prompts eager
    assert graphed.cache.num_used_blocks == 1  # only the reserved null block


@pytest.mark.skipif(DT != torch.float32, reason="exact token equality needs fp32")
def test_graph_spec_decoding_matches_target_alone(tiny, tiny_draft):
    hf, _ = tiny
    m = model_of(tiny)
    ps = prompts(3, seed=52)
    engine = LLMEngine(m, num_kv_blocks=256, block_size=4, draft_model=tiny_draft, num_speculative_tokens=3,
                       adaptive_speculation=False, max_num_seqs=4, cuda_graphs=True)
    seqs = engine.generate(ps, SamplingParams(max_tokens=14, **GREEDY))
    for p, s in zip(ps, seqs):
        assert s.output_ids == hf_greedy(hf, p, 14)
    assert engine.graphs.replays > 0 and engine.spec.runner.replays > 0
    assert engine.stats()["spec_rounds"] > 0


@pytest.mark.skipif(DEVICE != "cuda", reason="real graph capture needs CUDA")
def test_real_capture_with_spec_decoding(tiny):
    m = model_of(tiny)
    draft = to_vrtx(make_hf(seed=1, layers=1, hidden=32, heads=2, kv_heads=1), device=DEVICE, dtype=DT,
                    kernels=("norm", "rope", "act", "attention"))
    ps = prompts(3, seed=53)
    p = SamplingParams(max_tokens=12, **GREEDY)
    alone = LLMEngine(m, num_kv_blocks=256, block_size=4, cuda_graphs=False).generate(ps, p)
    engine = LLMEngine(m, num_kv_blocks=256, block_size=4, draft_model=draft, num_speculative_tokens=3,
                       adaptive_speculation=False, max_num_seqs=4, cuda_graphs=True)
    seqs = engine.generate(ps, p)
    assert engine.graphs.graphs and engine.spec.runner.graphs  # really captured
    assert engine.graphs.replays > 0 and engine.spec.runner.replays > 0
    for a, b in zip(alone, seqs):
        assert b.output_ids[:4] == a.output_ids[:4]
