"""Speculative decoding (Month 4).

A small DRAFT model guesses the next gamma tokens one at a time (cheap). The big TARGET
model then scores all of them in ONE forward pass, the same way it scores a prompt chunk.
Decode is memory-bound, so checking gamma + 1 positions costs about as much as one.

The accept/reject rule (Leviathan et al. 2023, Chen et al. 2023) keeps the output
distribution EXACTLY the target's. For draft token x with draft prob q(x), target prob p(x):

    accept x with probability min(1, p(x) / q(x))
    on the first rejection, sample from the residual  max(0, p - q) / sum(max(0, p - q))
    if all gamma are accepted, sample one bonus token from the target's last position

So every target pass yields between 1 and gamma + 1 tokens. With temperature 0 both
distributions are one-hot and the rule becomes "keep drafts while they equal the
target's argmax", so greedy output is identical to running the target alone.

ADAPTIVE GAMMA. If each draft token is accepted with probability a, a round of gamma
drafts yields E = (1 - a^(gamma+1)) / (1 - a) tokens and costs gamma * c + 1 target-step
equivalents, where c = draft step time / target step time (measured live). We track a
per sequence (decayed accept/reject counts) and pick the gamma with the most tokens per
unit of time, or gamma = 0 (plain decoding) when drafting does not pay.

KV CACHE. The draft model gets its own paged cache with the same number of blocks, so it
can reuse each sequence's block table: one allocation covers both models. Rejected
positions are rolled back by just lowering num_computed; those slots get overwritten.
"""
from __future__ import annotations

import time

import torch

from engine.batch import BatchItem, run_model
from engine.kv_cache import PagedKVCache
from engine.sampling import probs_batch
from engine.scheduler import Sequence


def verify(draft_tokens: list[int], q: torch.Tensor, p: torch.Tensor, generator=None) -> list[int]:
    """The exact speculative-sampling rule.

    draft_tokens: gamma proposed tokens.  q: (gamma, V) draft distributions they were sampled from.
    p: (gamma + 1, V) target distributions at the same positions plus one more.
    Returns the accepted drafts followed by one token from the target (residual or bonus).
    """
    g = len(draft_tokens)
    if g:
        idx = torch.tensor(draft_tokens, device=p.device)[:, None]
        pd = p[:g].gather(1, idx)[:, 0]
        qd = q.gather(1, idx)[:, 0]
        ratio = (pd / qd.clamp_min(1e-20)).clamp(max=1.0)
        u = torch.rand(g, device=p.device, generator=generator)
        k = int((u < ratio).int().cumprod(0).sum())
    else:
        k = 0
    if k < g:
        residual = (p[k] - q[k]).clamp_min(0)
        total = residual.sum()
        dist = residual / total if total > 0 else p[k]
    else:
        dist = p[g]
    last = int(torch.multinomial(dist, 1, generator=generator))
    return draft_tokens[:k] + [last]


def expected_tokens(alpha: float, gamma: int) -> float:
    if alpha >= 1.0:
        return gamma + 1.0
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


class SpecDecoder:
    def __init__(self, draft_model, num_blocks: int, block_size: int, num_speculative_tokens: int = 4,
                 adaptive: bool = True, decay: float = 0.9, probe_every: int = 8):
        self.model = draft_model
        self.cache = PagedKVCache(draft_model.cfg, num_blocks, block_size, draft_model.device, draft_model.dtype)
        self.max_gamma = num_speculative_tokens
        self.adaptive = adaptive
        self.decay = decay
        self.probe_every = probe_every
        self.runner = None  # a cuda_graphs.GraphRunner for the draft, set by the engine
        self.cost_ratio = 0.5  # draft step time / target step time, re-measured every step
        self._draft_time = None
        self._target_time = None
        self.proposed = 0
        self.accepted = 0
        self.rounds = 0
        self.tokens_from_rounds = 0

    # ---------------------------------------------------------- choose gamma
    def alpha(self, seq: Sequence) -> float:
        # Start optimistic (as if 4 of 5 drafts were accepted) so new sequences try speculation.
        a, r = seq.spec_accepted + 4 * 0.5, seq.spec_rejected + 1 * 0.5
        return a / (a + r)

    def gamma(self, seq: Sequence) -> int:
        if not self.adaptive:
            return self.max_gamma
        a, c = self.alpha(seq), self.cost_ratio
        best, best_rate = 0, 1.0
        for g in range(1, self.max_gamma + 1):
            rate = expected_tokens(a, g) / (g * c + 1)
            if rate > best_rate:
                best, best_rate = g, rate
        if best == 0:  # not worth it right now; probe now and then so a can recover
            seq.spec_idle += 1
            if seq.spec_idle >= self.probe_every:
                seq.spec_idle = 0
                return 1
        return best

    def observe(self, seq: Sequence, gamma: int, accepted: int) -> None:
        d = self.decay
        seq.spec_accepted = seq.spec_accepted * d + accepted
        seq.spec_rejected = seq.spec_rejected * d + (accepted < gamma)
        self.proposed += gamma
        self.accepted += accepted
        self.rounds += 1
        self.tokens_from_rounds += accepted + 1

    def record_target_time(self, seconds: float) -> None:
        self._target_time = seconds if self._target_time is None else 0.8 * self._target_time + 0.2 * seconds
        if self._draft_time is not None:
            self.cost_ratio = min(1.0, self._draft_time / self._target_time)

    # -------------------------------------------------------------- propose
    def propose(self, seqs: list[Sequence]) -> dict[int, tuple[list[int], torch.Tensor]]:
        """Run the draft model seq.num_lookahead times. Returns {seq.id: (tokens, q)}."""
        t0 = time.perf_counter()
        tokens = {s.id: [] for s in seqs}
        probs = {s.id: [] for s in seqs}
        rounds = max(s.num_lookahead for s in seqs)
        for j in range(rounds):
            active = [s for s in seqs if s.num_lookahead > j]
            items = []
            for s in active:
                if j == 0:  # catch up on every token the draft has not seen (the prompt, the last token(s))
                    items.append(BatchItem(s.block_table, s.draft_computed, s.all_ids[s.draft_computed:], 1))
                else:
                    items.append(BatchItem(s.block_table, s.length + j - 1, [tokens[s.id][-1]], 1))
            logits = self.runner.run(items) if self.runner else run_model(self.model, self.cache, items)
            q = probs_batch(logits, [s.params for s in active])
            greedy = [s.params.temperature <= 0 for s in active]
            if all(greedy):
                picked = q.argmax(-1)
            else:
                picked = torch.multinomial(q, 1)[:, 0]
            for s, tok, row in zip(active, picked.tolist(), q):
                tokens[s.id].append(tok)
                probs[s.id].append(row)
        for s in seqs:
            s.draft_computed = s.length + s.num_lookahead - 1
        elapsed = (time.perf_counter() - t0) / rounds
        self._draft_time = elapsed if self._draft_time is None else 0.8 * self._draft_time + 0.2 * elapsed
        return {s.id: (tokens[s.id], torch.stack(probs[s.id])) for s in seqs}

    def stats(self) -> dict:
        return {
            "spec_rounds": self.rounds,
            "spec_proposed": self.proposed,
            "spec_accepted": self.accepted,
            "spec_acceptance_rate": round(self.accepted / self.proposed, 4) if self.proposed else None,
            "spec_tokens_per_round": round(self.tokens_from_rounds / self.rounds, 3) if self.rounds else None,
            "spec_cost_ratio": round(self.cost_ratio, 3),
        }
