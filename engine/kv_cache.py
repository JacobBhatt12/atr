"""Paged KV cache.

GPU (or CPU) memory for keys/values is carved into fixed-size BLOCKS of `block_size`
token slots, like pages in an operating system. Each sequence owns a BLOCK TABLE:
a list of physical block numbers. Logical position p of a sequence lives at

    slot = block_table[p // block_size] * block_size + p % block_size

so a sequence's memory can be scattered anywhere, nothing is reserved up front,
and the most a sequence can waste is one partly filled block.

Month 4 adds PREFIX CACHING. A full block's contents depend only on the tokens in it
and every token before it, so we name each full block by a hash chain:

    hash(block i) = H(hash(block i-1), tokens in block i)

Two requests that start with the same system prompt get the same hashes, so the
second one reuses the first one's blocks (reference count + 1) and skips that prefill.
A block whose last owner finishes is not wiped: it parks in an LRU list and is only
recycled when memory runs out, so later requests can still hit it.
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict, deque

import torch

from engine.model import ModelConfig


class NoFreeBlocks(RuntimeError):
    pass


def block_hash(parent: bytes | None, tokens) -> bytes:
    h = hashlib.blake2b(parent or b"", digest_size=16)
    h.update(torch.tensor(list(tokens), dtype=torch.int64).numpy().tobytes())
    return h.digest()


class BlockAllocator:
    """Free list, reference counts, and (optionally) a hash -> block index for prefix caching."""

    def __init__(self, num_blocks: int, enable_prefix_caching: bool = False):
        self.num_blocks = num_blocks
        self.caching = enable_prefix_caching
        self._free = deque(range(num_blocks))
        self.ref = [0] * num_blocks
        self._hash_of: dict[int, bytes] = {}       # block -> hash of the tokens it holds
        self._block_of: dict[bytes, int] = {}      # hash -> block
        self._evictable: OrderedDict[int, None] = OrderedDict()  # ref 0 but still cached, oldest first

    @property
    def num_free(self) -> int:
        """Blocks available to allocate, counting cached blocks nobody is using."""
        return len(self._free) + len(self._evictable)

    def allocate(self) -> int:
        if self._free:
            b = self._free.popleft()
        elif self._evictable:  # recycle the least recently used cached block
            b, _ = self._evictable.popitem(last=False)
            del self._block_of[self._hash_of.pop(b)]
        else:
            raise NoFreeBlocks()
        self.ref[b] = 1
        return b

    def share(self, block: int) -> None:
        if self.ref[block] == 0:
            self._evictable.pop(block)
        self.ref[block] += 1

    def release(self, block: int) -> None:
        self.ref[block] -= 1
        assert self.ref[block] >= 0, f"double free of block {block}"
        if self.ref[block] == 0:
            if block in self._hash_of:
                self._evictable[block] = None  # newest at the end
            else:
                self._free.append(block)

    # --------------------------------------------------------- prefix cache
    def lookup(self, h: bytes) -> int | None:
        return self._block_of.get(h)

    def is_evictable(self, block: int) -> bool:
        return block in self._evictable

    def register(self, block: int, h: bytes) -> None:
        """Record that `block` holds the tokens named by `h`. First writer wins."""
        if self.caching and h not in self._block_of and block not in self._hash_of:
            self._block_of[h] = block
            self._hash_of[block] = h

    @property
    def num_cached(self) -> int:
        return len(self._block_of)


class PagedKVCache:
    def __init__(self, cfg: ModelConfig, num_blocks: int, block_size: int, device="cpu", dtype=torch.float32,
                 enable_prefix_caching: bool = False):
        self.cfg, self.block_size, self.num_blocks = cfg, block_size, num_blocks
        self.device, self.dtype = device, dtype
        shape = (num_blocks * block_size, cfg.num_kv_heads, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_layers)]
        self.allocator = BlockAllocator(num_blocks, enable_prefix_caching)

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

    def _paged(self, t: torch.Tensor) -> torch.Tensor:
        return t.view(self.num_blocks, self.block_size, *t.shape[1:])

    # ------------------------------------------------------ copy-on-write
    def copy_block(self, src: int, dst: int) -> None:
        for t in (*self.k, *self.v):
            p = self._paged(t)
            p[dst] = p[src]

    # --------------------------------------------------------------- swap
    def swap_out(self, block_table: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy a sequence's blocks to CPU memory: (layers, blocks, block_size, kv_heads, head_dim) x 2."""
        idx = torch.tensor(block_table, device=self.device)

        def gather(ts):
            return torch.stack([self._paged(t).index_select(0, idx) for t in ts]).cpu()

        return gather(self.k), gather(self.v)

    def swap_bytes(self, num_blocks: int) -> int:
        return num_blocks * self.block_size * self.bytes_per_token(self.cfg, self.dtype)

    def swap_in(self, block_table: list[int], data: tuple[torch.Tensor, torch.Tensor]) -> None:
        idx = torch.tensor(block_table, device=self.device)
        for ts, saved in zip((self.k, self.v), data):
            saved = saved.to(self.device, non_blocking=True)
            for li, t in enumerate(ts):
                self._paged(t).index_copy_(0, idx, saved[li])
