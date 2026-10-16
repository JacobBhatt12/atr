"""Token sampling: temperature, top-k and top-p (nucleus), written by hand."""
import torch


def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Pick the next token from logits of shape (batch, vocab). Returns shape (batch, 1)."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)  # greedy

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

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
