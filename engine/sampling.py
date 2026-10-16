"""Token sampling: temperature, top-k and top-p (nucleus), written by hand."""
from __future__ import annotations

import torch


def _filter(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """Apply temperature, top-k and top-p. Returns logits with removed tokens set to -inf."""
    logits = logits.float() / temperature

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        kth_best = torch.topk(logits, k).values[..., -1, None]
        logits = logits.masked_fill(logits < kth_best, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        # drop a token if the tokens before it already reach top_p
        remove = (cumulative - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits


def probs_from_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """The distribution we actually sample from, shape (batch, vocab).

    Greedy (temperature 0) is a one-hot distribution on the argmax. Speculative decoding
    needs this explicit form: its accept/reject rule compares two distributions.
    """
    if temperature <= 0:
        return torch.zeros_like(logits, dtype=torch.float32).scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
    return torch.softmax(_filter(logits, temperature, top_k, top_p), dim=-1)


def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Pick the next token from logits of shape (batch, vocab). Returns shape (batch, 1)."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)  # greedy
    probs = torch.softmax(_filter(logits, temperature, top_k, top_p), dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _groups(params_list) -> dict[tuple, list[int]]:
    groups: dict[tuple, list[int]] = {}
    for i, p in enumerate(params_list):
        key = (0.0, 0, 1.0) if p.temperature <= 0 else (p.temperature, p.top_k, p.top_p)
        groups.setdefault(key, []).append(i)
    return groups


def sample_batch(logits: torch.Tensor, params_list) -> list[int]:
    """Sample one token per row. Rows with the same settings share one kernel call and
    one GPU->CPU copy, instead of one sync per sequence."""
    out = [0] * len(params_list)
    for (t, k, p), rows in _groups(params_list).items():
        idx = torch.tensor(rows, device=logits.device)
        tokens = sample(logits.index_select(0, idx), t, k, p)[:, 0].tolist()
        for r, tok in zip(rows, tokens):
            out[r] = tok
    return out


def probs_batch(logits: torch.Tensor, params_list) -> torch.Tensor:
    """probs_from_logits for rows that may each have their own settings."""
    groups = _groups(params_list)
    if len(groups) == 1:
        (t, k, p), = groups
        return probs_from_logits(logits, t, k, p)
    out = torch.empty(logits.shape, dtype=torch.float32, device=logits.device)
    for (t, k, p), rows in groups.items():
        idx = torch.tensor(rows, device=logits.device)
        out[idx] = probs_from_logits(logits.index_select(0, idx), t, k, p)
    return out
