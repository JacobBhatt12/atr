"""Shared fixtures: tiny randomly initialized Qwen2 models, so no download is needed."""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from engine.model import ModelConfig, VRTXModel  # noqa: E402

VOCAB = 97
GREEDY = dict(temperature=0.0, ignore_eos=True)


def make_hf(seed=0, layers=2, hidden=64, heads=4, kv_heads=2, init=0.2):
    torch.manual_seed(seed)
    cfg = transformers.Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,  # grouped-query attention, like the real Qwen
        max_position_embeddings=512,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        initializer_range=init,  # large on purpose: sharp logits make bugs visible
    )
    return transformers.Qwen2ForCausalLM(cfg).eval()


def to_vrtx(hf, **kw):
    return VRTXModel(ModelConfig.from_hf(hf.config), hf.state_dict(), **kw)


@pytest.fixture(scope="session")
def tiny():
    hf = make_hf()
    return hf, to_vrtx(hf)


@pytest.fixture(scope="session")
def tiny_draft():
    """A different (smaller, randomly initialized) model with the same vocabulary."""
    return to_vrtx(make_hf(seed=1, layers=1, hidden=32, heads=2, kv_heads=1))


def hf_greedy(hf, prompt, n):
    ids = list(prompt)
    with torch.no_grad():
        for _ in range(n):
            logits = hf(torch.tensor([ids])).logits[0, -1]
            ids.append(int(logits.argmax()))
    return ids[len(prompt):]


def prompts(k, seed=1, lo=3, hi=40):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, VOCAB, (int(torch.randint(lo, hi, (1,), generator=g)),), generator=g).tolist() for _ in range(k)]
