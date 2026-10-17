"""VRTX Engine: continuous batching over a paged KV cache, plus the Month 3 and 4 features.

    engine = LLMEngine.from_pretrained("models/qwen2.5-0.5b")
    outputs = engine.generate([ids_a, ids_b, ids_c], SamplingParams(max_tokens=64))

Options:
    kernels=None            Triton kernels on CUDA, PyTorch elsewhere (or pick: ("attention", "norm", ...))
    quant_bits=8 | 4        weight-only int8 / int4 for every transformer matmul
    enable_prefix_caching   reuse KV blocks of prompts that start the same way
    draft_model=...         speculative decoding with a smaller model of the same family
    preemption_mode="swap"  out of memory: copy KV to CPU instead of recomputing it
    cuda_graphs=None        record small steps (decode, draft, verify) as CUDA graphs: on by default with
                            the Triton kernels on CUDA; VRTX_CUDA_GRAPHS=0 turns it off
"""
from __future__ import annotations

import itertools
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from engine.batch import BatchItem, run_model
from engine.cuda_graphs import GraphRunner
from engine.kv_cache import PagedKVCache
from engine.model import VRTXModel, read_generation_eos
from engine.sampling import probs_batch, sample_batch
from engine.scheduler import SamplingParams, Scheduler, Sequence
from engine.speculative import SpecDecoder, verify


@dataclass
class TokenEvent:
    seq: Sequence
    token_id: int | None  # None when the sequence ended on a stop token
    finished: bool


def _download(path: str) -> str:
    if Path(path).is_dir():
        return path
    from huggingface_hub import snapshot_download  # a Hugging Face id: download it

    return snapshot_download(path, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])


class LLMEngine:
    def __init__(
        self,
        model: VRTXModel,
        tokenizer=None,
        *,
        stop_ids: set[int] | None = None,
        block_size: int = 16,
        num_kv_blocks: int | None = None,
        kv_cache_mb: int = 256,
        max_num_seqs: int = 32,
        max_num_batched_tokens: int = 512,
        enable_prefix_caching: bool = False,
        preemption_mode: str = "recompute",
        swap_space_mb: int = 1024,
        draft_model: VRTXModel | None = None,
        num_speculative_tokens: int = 4,
        adaptive_speculation: bool = True,
        cuda_graphs: bool | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.stop_ids = set(stop_ids or ())
        cfg = model.cfg
        if num_kv_blocks is None:
            per_token = PagedKVCache.bytes_per_token(cfg, model.dtype)
            if draft_model is not None:  # the draft's cache uses the same block tables, so split the budget
                per_token += PagedKVCache.bytes_per_token(draft_model.cfg, draft_model.dtype)
            num_kv_blocks = max(1, kv_cache_mb * 2**20 // (per_token * block_size))
        if draft_model is not None and enable_prefix_caching:
            raise ValueError("prefix caching and speculative decoding cannot be combined yet")
        self.cache = PagedKVCache(cfg, num_kv_blocks, block_size, model.device, model.dtype, enable_prefix_caching)
        self.scheduler = Scheduler(self.cache, max_num_seqs, max_num_batched_tokens, preemption_mode, swap_space_mb)
        self.max_model_len = cfg.max_position_embeddings
        self.scheduler.max_model_len = self.max_model_len
        self.spec = None
        if draft_model is not None:
            if draft_model.cfg.vocab_size != cfg.vocab_size:
                raise ValueError("draft and target models must share a vocabulary")
            self.spec = SpecDecoder(draft_model, num_kv_blocks, block_size, num_speculative_tokens, adaptive_speculation)
            self.scheduler.lookahead_fn = self.spec.gamma
        if cuda_graphs is None:
            cuda_graphs = (torch.device(model.device).type == "cuda" and os.environ.get("VRTX_CUDA_GRAPHS", "1") != "0"
                           and all("attention" in m.kernels for m in (model, draft_model) if m is not None))
        self.graphs = None
        if cuda_graphs:
            # One block no sequence ever owns: padding rows of a graph write their K/V there.
            null = self.cache.allocator.allocate()
            verify_q = tuple(range(2, num_speculative_tokens + 2)) if self.spec else ()
            self.graphs = GraphRunner(model, self.cache, null, max_num_seqs, self.max_model_len, multi_q=verify_q)
            self.graphs.capture()
            if self.spec:
                self.spec.runner = GraphRunner(draft_model, self.spec.cache, null, max_num_seqs, self.max_model_len,
                                               multi_q=(2,))
                self.spec.runner.capture()
        self._ids = itertools.count()
        self._by_id: dict[int, Sequence] = {}
        self.total_generated = 0
        self.total_steps = 0

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu", dtype: torch.dtype = torch.float32, *,
                        kernels=None, quant_bits: int | None = None, quant_group_size: int = 128,
                        draft: str | None = None,
                        **kwargs) -> "LLMEngine":
        from transformers import AutoTokenizer

        path = _download(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        stop = read_generation_eos(path)
        if tokenizer.eos_token_id is not None:
            stop.add(tokenizer.eos_token_id)
        model = VRTXModel.from_pretrained(path, device, dtype, kernels)
        if quant_bits:
            model.quantize(quant_bits, quant_group_size)
        draft_model = VRTXModel.from_pretrained(_download(draft), device, dtype, kernels) if draft else None
        return cls(model, tokenizer, stop_ids=stop, draft_model=draft_model, **kwargs)

    # ------------------------------------------------------------- requests
    def add_request(self, prompt_ids: list[int], params: SamplingParams | None = None, seq_id: int | None = None) -> Sequence:
        params = params or SamplingParams()
        if not prompt_ids:
            raise ValueError("empty prompt")
        total = len(prompt_ids) + params.max_tokens
        if self.cache.blocks_needed(total) > self.cache.num_blocks:
            raise ValueError(f"request needs {total} tokens of KV cache, more than the whole cache holds")
        seq = Sequence(next(self._ids) if seq_id is None else seq_id, prompt_ids, params)
        self._by_id[seq.id] = seq
        self.scheduler.add(seq)
        return seq

    def abort(self, seq_id: int) -> None:
        seq = self._by_id.pop(seq_id, None)
        if seq is not None and seq.finish_reason is None:
            self.scheduler.finish(seq, "abort")

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def _finish(self, seq: Sequence, reason: str) -> None:
        self.scheduler.finish(seq, reason)
        self._by_id.pop(seq.id, None)

    def _append(self, seq: Sequence, token: int, now: float, events: list[TokenEvent]) -> bool:
        """Add one sampled token. Returns True if the sequence just finished."""
        p = seq.params
        if seq.first_token_time is None:
            seq.first_token_time = now
        if not p.ignore_eos and token in self.stop_ids:
            self._finish(seq, "stop")
            events.append(TokenEvent(seq, None, True))
            return True
        seq.output_ids.append(token)
        self.total_generated += 1
        done = len(seq.output_ids) >= p.max_tokens or seq.length >= self.max_model_len
        if done:
            self._finish(seq, "length")
        events.append(TokenEvent(seq, token, done))
        return done

    # ----------------------------------------------------------------- step
    def step(self) -> list[TokenEvent]:
        batch = self.scheduler.schedule()
        if not batch:
            return []
        self.total_steps += 1

        spec_seqs = [seq for seq, _ in batch if seq.num_lookahead > 0]
        drafts = self.spec.propose(spec_seqs) if spec_seqs else {}

        t0 = time.perf_counter()
        items = []
        for seq, n in batch:
            start = seq.num_computed
            tokens = seq.all_ids[start:start + n]
            if seq.id in drafts:  # verify: last real token + every draft, scored in one pass
                tokens = tokens + drafts[seq.id][0]
                items.append(BatchItem(seq.block_table, start, tokens, len(tokens)))
            else:  # reached the newest token: we owe this sequence a new token
                items.append(BatchItem(seq.block_table, start, tokens, 1 if start + n == seq.length else 0))
        logits = self.graphs.run(items) if self.graphs else run_model(self.model, self.cache, items)

        events: list[TokenEvent] = []
        plain, plain_rows, row = [], [], 0
        spec_work = []
        for (seq, n), it in zip(batch, items):
            if seq.id in drafts:
                spec_work.append((seq, row, it.num_logits))
            else:
                seq.num_computed += n
                if it.num_logits:
                    plain.append(seq)
                    plain_rows.append(row)
            row += it.num_logits

        now = time.perf_counter()
        if plain:
            rows = torch.tensor(plain_rows, device=logits.device)
            for seq, token in zip(plain, sample_batch(logits.index_select(0, rows), [s.params for s in plain])):
                self._append(seq, token, now, events)

        for seq, row, k in spec_work:
            draft_tokens, q = drafts[seq.id]
            p = probs_batch(logits[row:row + k], [seq.params] * k)
            new = verify(draft_tokens, q, p)
            self.spec.observe(seq, len(draft_tokens), len(new) - 1)
            base = seq.length  # target now holds valid K/V for positions < base - 1 + len(new)
            seq.num_computed = base - 1 + len(new)
            seq.draft_computed = min(seq.draft_computed, seq.num_computed)
            for token in new:
                if self._append(seq, token, now, events):
                    break
        if self.spec is not None and spec_work:
            self.spec.record_target_time(time.perf_counter() - t0)

        for seq, _ in batch:
            if seq.finish_reason is None:
                self.scheduler.register_computed(seq)
        return events

    # ------------------------------------------------------ offline helper
    def generate(self, prompts: list[list[int]], params: SamplingParams | None = None) -> list[Sequence]:
        seqs = [self.add_request(p, params) for p in prompts]
        while self.has_unfinished():
            self.step()
        return seqs

    def stats(self) -> dict:
        s = self.scheduler
        out = {
            "running": len(s.running),
            "waiting": len(s.waiting),
            "swapped": len(s.swapped),
            "kv_blocks_used": self.cache.num_used_blocks,
            "kv_blocks_total": self.cache.num_blocks,
            "block_size": self.cache.block_size,
            "generated_tokens_total": self.total_generated,
            "steps_total": self.total_steps,
            "preemptions_total": s.num_preemptions,
            "swaps_total": s.num_swaps,
            "prefix_cache_hit_rate": round(s.prefix_hit_tokens / s.prefix_query_tokens, 4) if s.prefix_query_tokens else 0.0,
            "prefix_cached_blocks": self.cache.allocator.num_cached,
        }
        if self.graphs is not None:
            out["graph_replays"] = self.graphs.replays
            out["graph_eager_steps"] = self.graphs.eager_steps
        if self.spec is not None:
            out.update(self.spec.stats())
        return out
