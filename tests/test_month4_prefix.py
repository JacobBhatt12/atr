"""Prefix caching: hashed blocks, reference counts, LRU eviction, copy-on-write."""
import torch

from engine.kv_cache import BlockAllocator, block_hash
from engine.llm_engine import LLMEngine
from engine.scheduler import SamplingParams
from tests.conftest import GREEDY, VOCAB, hf_greedy


def _shared(n, prefix_len, seed=0, suffix=(3, 9)):
    g = torch.Generator().manual_seed(seed)
    prefix = torch.randint(0, VOCAB, (prefix_len,), generator=g).tolist()
    return [prefix + torch.randint(0, VOCAB, (int(torch.randint(*suffix, (1,), generator=g)),), generator=g).tolist()
            for _ in range(n)]


def test_hash_chain_depends_on_everything_before():
    a = block_hash(None, [1, 2, 3, 4])
    assert a == block_hash(None, [1, 2, 3, 4])
    assert block_hash(a, [5, 6]) != block_hash(block_hash(None, [9, 9, 9, 9]), [5, 6])


def test_allocator_keeps_cached_blocks_until_memory_is_needed():
    alloc = BlockAllocator(3, enable_prefix_caching=True)
    b = alloc.allocate()
    alloc.register(b, b"h")
    alloc.release(b)
    assert alloc.num_free == 3 and alloc.lookup(b"h") == b  # free, but still findable
    alloc.share(b)                                          # a new request reuses it
    assert alloc.ref[b] == 1 and alloc.num_free == 2
    alloc.release(b)
    others = [alloc.allocate(), alloc.allocate()]           # plain free blocks go first
    assert b not in others and alloc.lookup(b"h") == b
    alloc.allocate()                                        # now it must be evicted
    assert alloc.lookup(b"h") is None


def test_shared_prefix_hits_cache_and_matches_hf(tiny):
    hf, ours = tiny
    ps = _shared(4, prefix_len=24)
    engine = LLMEngine(ours, num_kv_blocks=128, block_size=4, enable_prefix_caching=True)
    first = engine.generate(ps[:1], SamplingParams(max_tokens=8, **GREEDY))
    rest = engine.generate(ps[1:], SamplingParams(max_tokens=8, **GREEDY))
    for p, s in zip(ps, first + rest):
        assert s.output_ids == hf_greedy(hf, p, 8)
    assert all(s.num_cached_tokens == 24 for s in rest)
    assert engine.stats()["prefix_cache_hit_rate"] > 0.5
    assert engine.cache.num_used_blocks == 0


def test_identical_prompt_uses_copy_on_write(tiny):
    """Prompt length is a multiple of the block size, so a repeat hits EVERY block. The last
    token still has to be recomputed for its logits, and its block is copied, not overwritten."""
    hf, ours = tiny
    prompt = _shared(1, prefix_len=16, suffix=(1, 2))[0][:16]
    engine = LLMEngine(ours, num_kv_blocks=64, block_size=4, enable_prefix_caching=True)
    a = engine.add_request(prompt, SamplingParams(max_tokens=6, **GREEDY))
    while a.num_computed < 16:
        engine.step()
    b = engine.add_request(prompt, SamplingParams(max_tokens=6, **GREEDY))  # a is still running
    engine.step()
    assert b.block_table[:3] == a.block_table[:3]  # first three blocks shared
    assert b.block_table[3] != a.block_table[3]    # the written block was copied
    while engine.has_unfinished():
        engine.step()
    want = hf_greedy(hf, prompt, 6)
    assert a.output_ids == want and b.output_ids == want
    assert b.num_cached_tokens == 15


def test_prefix_cache_survives_preemption(tiny):
    hf, ours = tiny
    ps = _shared(6, prefix_len=12, seed=3)
    engine = LLMEngine(ours, num_kv_blocks=24, block_size=4, max_num_seqs=6, enable_prefix_caching=True)
    seqs = engine.generate(ps, SamplingParams(max_tokens=16, **GREEDY))
    assert engine.scheduler.num_preemptions > 0
    for p, s in zip(ps, seqs):
        assert s.output_ids == hf_greedy(hf, p, 16)
