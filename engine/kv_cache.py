"""Paged KV cache.

GPU (or CPU) memory for keys/values is carved into fixed-size BLOCKS of `block_size`
token slots, like pages in an operating system. Each sequence owns a BLOCK TABLE:
a list of physical block numbers. Logical position p of a sequence lives at

    slot = block_table[p // block_size] * block_size + p % block_size

so a sequence's memory can be scattered anywhere, nothing is reserved up front,
and the most a sequence can waste is one partly filled block.
"""
from __future__ import annotations

import math
from collections import deque

import torch

from engine.model import ModelConfig


class NoFreeBlocks(RuntimeError):
    pass


class BlockAllocator:
    """Free list plus reference counts (ref counts let Month 4 share prefix blocks)."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._free = deque(range(num_blocks))
        self.ref = [0] * num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise NoFreeBlocks()
        b = self._free.popleft()
        self.ref[b] = 1
        return b

    def share(self, block: int) -> None:
        self.ref[block] += 1

    def release(self, block: int) -> None:
        self.ref[block] -= 1
        assert self.ref[block] >= 0, f"double free of block {block}"
        if self.ref[block] == 0:
            self._free.append(block)


class PagedKVCache:
    def __init__(self, cfg: ModelConfig, num_blocks: int, block_size: int, device="cpu", dtype=torch.float32):
        self.cfg, self.block_size, self.num_blocks = cfg, block_size, num_blocks
        shape = (num_blocks * block_size, cfg.num_kv_heads, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_layers)]
        self.allocator = BlockAllocator(num_blocks)

    # KV bytes per token = 2 (K and V) x layers x kv_heads x head_dim x bytes_per_value
    @staticmethod
    def bytes_per_token(cfg: ModelConfig, dtype=torch.float32) -> int:
        return 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * torch.tensor([], dtype=dtype).element_size()

    @classmethod
    def blocks_for_budget(cls, cfg: ModelConfig, budget_bytes: int, block_size: int, dtype=torch.float32) -> int:
        return max(1, budget_bytes // (cls.bytes_per_token(cfg, dtype) * block_size))

    def blocks_needed(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size)

    def slots(self, block_table: list[int], start: int, end: int) -> torch.Tensor:
        pos = torch.arange(start, end)
        blocks = torch.tensor(block_table, dtype=torch.long)[pos // self.block_size]
        return blocks * self.block_size + pos % self.block_size

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.allocator.num_free
