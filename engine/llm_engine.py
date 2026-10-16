"""VRTX Engine, Month 2: continuous batching over a paged KV cache.

    engine = LLMEngine.from_pretrained("models/qwen2.5-0.5b")
    outputs = engine.generate([ids_a, ids_b, ids_c], SamplingParams(max_tokens=64))
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from engine.kv_cache import PagedKVCache
from engine.model import VRTXModel, read_generation_eos
from engine.sampling import sample
from engine.scheduler import SamplingParams, Scheduler, Sequence


@dataclass
class TokenEvent:
    seq: Sequence
    token_id: int | None  # None when the sequence ended on a stop token
    finished: bool


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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.stop_ids = set(stop_ids or ())
        cfg = model.cfg
        if num_kv_blocks is None:
            num_kv_blocks = PagedKVCache.blocks_for_budget(cfg, kv_cache_mb * 2**20, block_size, model.dtype)
        self.cache = PagedKVCache(cfg, num_kv_blocks, block_size, model.device, model.dtype)
        self.scheduler = Scheduler(self.cache, max_num_seqs, max_num_batched_tokens)
        self.max_model_len = cfg.max_position_embeddings
        self._ids = itertools.count()
        self._by_id: dict[int, Sequence] = {}
        self.total_generated = 0
        self.total_steps = 0

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu", dtype: torch.dtype = torch.float32, **kwargs) -> "LLMEngine":
        from transformers import AutoTokenizer

        if not Path(path).is_dir():  # a Hugging Face id: download it
            from huggingface_hub import snapshot_download

            path = snapshot_download(path, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
        tokenizer = AutoTokenizer.from_pretrained(path)
        stop = read_generation_eos(path)
        if tokenizer.eos_token_id is not None:
            stop.add(tokenizer.eos_token_id)
        model = VRTXModel.from_pretrained(path, device, dtype)
        return cls(model, tokenizer, stop_ids=stop, **kwargs)

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

    # ----------------------------------------------------------------- step
    def step(self) -> list[TokenEvent]:
        batch = self.scheduler.schedule()
        if not batch:
            return []
        self.total_steps += 1

        ids, positions, meta, ctx_list, write_slots, logits_rows, to_sample = [], [], [], [], [], [], []
        offset = 0
        for seq, n in batch:
            start, end = seq.num_computed, seq.num_computed + n
            ids.extend(seq.all_ids[start:end])
            positions.extend(range(start, end))
            ctx_slots = self.cache.slots(seq.block_table, 0, end)
            write_slots.append(ctx_slots[start:end])
            ctx_list.append(ctx_slots)
            meta.append((offset, n, end))
            if end == seq.length:  # reached the newest token: we owe this sequence a new token
                logits_rows.append(offset + n - 1)
                to_sample.append(seq)
            offset += n

        # Send every sequence's slot list to the GPU in ONE copy. Otherwise attention would
        # index the GPU cache with CPU tensors, a hidden copy per sequence per layer.
        device = self.model.device
        ctx_dev = torch.cat(ctx_list).to(device).split([len(c) for c in ctx_list])
        seq_meta = [(o, n, end, c) for (o, n, end), c in zip(meta, ctx_dev)]

        logits = self.model.forward(
            torch.tensor(ids, device=device),
            torch.tensor(positions, device=device),
            seq_meta,
            torch.cat(write_slots).to(device),
            torch.tensor(logits_rows, dtype=torch.long, device=device),
            self.cache,
        )
        for seq, n in batch:
            seq.num_computed += n

        events: list[TokenEvent] = []
        now = time.perf_counter()
        for row, seq in enumerate(to_sample):
            p = seq.params
            token = sample(logits[row:row + 1], p.temperature, p.top_k, p.top_p).item()
            if seq.first_token_time is None:
                seq.first_token_time = now
            if not p.ignore_eos and token in self.stop_ids:
                self._finish(seq, "stop")
                events.append(TokenEvent(seq, None, True))
                continue
            seq.output_ids.append(token)
            self.total_generated += 1
            done = len(seq.output_ids) >= p.max_tokens or seq.length >= self.max_model_len
            if done:
                self._finish(seq, "length")
            events.append(TokenEvent(seq, token, done))
        return events

    # ------------------------------------------------------ offline helper
    def generate(self, prompts: list[list[int]], params: SamplingParams | None = None) -> list[Sequence]:
        seqs = [self.add_request(p, params) for p in prompts]
        while self.has_unfinished():
            self.step()
        return seqs

    def stats(self) -> dict:
        return {
            "running": len(self.scheduler.running),
            "waiting": len(self.scheduler.waiting),
            "kv_blocks_used": self.cache.num_used_blocks,
            "kv_blocks_total": self.cache.num_blocks,
            "block_size": self.cache.block_size,
            "generated_tokens_total": self.total_generated,
            "steps_total": self.total_steps,
            "preemptions_total": self.scheduler.num_preemptions,
        }
