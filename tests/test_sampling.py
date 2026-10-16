import torch

from engine.sampling import sample


def test_greedy_picks_max():
    logits = torch.tensor([[0.1, 5.0, 0.3]])
    assert sample(logits, temperature=0).item() == 1


def test_top_k_one_is_greedy():
    logits = torch.tensor([[0.1, 0.2, 3.0, 0.4]])
    for _ in range(20):
        assert sample(logits, temperature=1.0, top_k=1).item() == 2


def test_top_p_keeps_only_dominant_token():
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    for _ in range(20):
        assert sample(logits, temperature=1.0, top_p=0.5).item() == 0
