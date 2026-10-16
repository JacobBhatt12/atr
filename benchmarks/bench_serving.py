"""Month 4 writeup benchmark: the same workloads on VRTX, vLLM and naive Hugging Face.

Workloads (section 14 of the README):
    short_chat     ~100 prompt tokens, 200 output tokens
    long_context   ~4,000 prompt tokens, 200 output tokens
    shared_prefix  1,024-token shared system prompt + short question, 128 output tokens

All requests arrive at t=0 and the engine may run up to --users of them at once, so
"users" is the number of concurrent streams. Every engine is forced to emit exactly
max_tokens (ignore EOS), greedy, so they all do the same amount of work.

Run:  python -m benchmarks.bench_serving --engine vrtx
      python -m benchmarks.bench_serving --engine vllm      (pip install vllm first)
      python -m benchmarks.bench_serving --engine hf --workloads short_chat --requests 16
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from benchmarks.common import (DEFAULT_MODEL, TOPICS, chat_ids, default_dtype, latency_summary, pick_device, save,
                               text_of_tokens)

WORKLOADS = {
    # name: (prompt builder args, output tokens, default requests)
    "short_chat": (100, 200, 128),
    "long_context": (4000, 200, 32),
    "shared_prefix": (1024, 128, 128),
}


def build_prompts(tok, workload: str, n: int) -> list[list[int]]:
    size, _, _ = WORKLOADS[workload]
    out = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        if workload == "short_chat":
            ids = chat_ids(tok, f"Here are some notes: {text_of_tokens(tok, 70, offset=97 * i)}\n"
                                f"Now explain {topic} in simple terms.")
        elif workload == "long_context":
            ids = chat_ids(tok, f"{text_of_tokens(tok, size, offset=211 * i)}\n\nSummarize the notes above, "
                                f"then relate them to {topic}.")
        else:
            system = "You are the support assistant for VRTX. Use these notes:\n" + text_of_tokens(tok, size)
            ids = chat_ids(tok, f"Briefly, what do the notes say about {topic}?", system)
        out.append(ids)
    return out


def kv_waste(prompts, out_tokens, reserve: int, block: int = 16) -> dict:
    """KV slots reserved vs used: a static cache that reserves `reserve` tokens per request
    (the classic approach) against our paged cache (at most one partly filled block)."""
    used = [len(p) + out_tokens for p in prompts]
    static = [max(reserve, u) for u in used]
    paged = [-(-u // block) * block for u in used]
    return {
        "static_reserved": sum(static), "paged_reserved": sum(paged), "used": sum(used),
        "static_waste_pct": round(100 * (1 - sum(used) / sum(static)), 1),
        "paged_waste_pct": round(100 * (1 - sum(used) / sum(paged)), 1),
    }


# ------------------------------------------------------------------ engines
def run_vrtx(args, device, dtype, workload, n):
    from engine.llm_engine import LLMEngine
    from engine.scheduler import SamplingParams

    engine = LLMEngine.from_pretrained(args.model, device=device, dtype=dtype, kv_cache_mb=args.kv_cache_mb,
                                       max_num_seqs=args.users, max_num_batched_tokens=args.max_batched_tokens,
                                       enable_prefix_caching=args.prefix_caching, quant_bits=args.quant)
    prompts = build_prompts(engine.tokenizer, workload, n)
    out_tokens = WORKLOADS[workload][1]
    # Warm up (compiles the Triton kernels) on an unrelated prompt, so it cannot seed the prefix cache.
    engine.generate([chat_ids(engine.tokenizer, "hi")], SamplingParams(max_tokens=4, temperature=0.0))
    engine.scheduler.prefix_hit_tokens = engine.scheduler.prefix_query_tokens = 0
    t = time.perf_counter()
    seqs = engine.generate(prompts, SamplingParams(max_tokens=out_tokens, temperature=0.0, ignore_eos=True))
    wall = time.perf_counter() - t
    st = engine.stats()
    res = {
        "throughput_tok_s": round(sum(len(s.output_ids) for s in seqs) / wall, 2), "wall_s": round(wall, 2),
        "preemptions": st["preemptions_total"], "prefix_cache_hit_rate": st["prefix_cache_hit_rate"],
        "kv_blocks_total": st["kv_blocks_total"], **latency_summary(seqs),
        "kv_memory": kv_waste(prompts, out_tokens, reserve=args.static_reserve),
    }
    return res, prompts


def run_vllm(args, device, dtype, workload, n):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams as VSP

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = build_prompts(tok, workload, n)
    out_tokens = WORKLOADS[workload][1]
    llm = run_vllm.cache.get(args.model)
    if llm is None:
        llm = run_vllm.cache[args.model] = LLM(
            model=args.model, dtype="float16", gpu_memory_utilization=args.vllm_gpu_util, max_num_seqs=args.users,
            enable_prefix_caching=args.prefix_caching, max_model_len=8192, seed=0)
    sp = VSP(max_tokens=out_tokens, temperature=0.0, ignore_eos=True)
    inputs = [{"prompt_token_ids": p} for p in prompts]
    llm.generate(inputs[:1], VSP(max_tokens=4), use_tqdm=False)
    t = time.perf_counter()
    outs = llm.generate(inputs, sp, use_tqdm=False)
    wall = time.perf_counter() - t
    res = {"throughput_tok_s": round(sum(len(o.outputs[0].token_ids) for o in outs) / wall, 2), "wall_s": round(wall, 2)}
    ttft = []
    for o in outs:  # request metrics exist in some vLLM versions only
        m = getattr(o, "metrics", None)
        if m is not None and getattr(m, "first_token_time", None) and getattr(m, "arrival_time", None):
            ttft.append(m.first_token_time - m.arrival_time)
    if ttft:
        res.update({"ttft_p50_s": round(statistics.median(ttft), 4), "ttft_p99_s": round(sorted(ttft)[int(0.99 * (len(ttft) - 1))], 4)})
    return res, prompts


run_vllm.cache = {}


@torch.inference_mode()
def run_hf(args, device, dtype, workload, n):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = build_prompts(tok, workload, n)
    out_tokens = WORKLOADS[workload][1]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    gen = dict(max_new_tokens=out_tokens, min_new_tokens=out_tokens, do_sample=False, temperature=None, top_p=None,
               top_k=None, repetition_penalty=1.0, pad_token_id=tok.eos_token_id)
    model.generate(torch.tensor([prompts[0]], device=device), **{**gen, "max_new_tokens": 4, "min_new_tokens": 4})
    t = time.perf_counter()
    finish = []
    for p in prompts:  # one at a time: everyone waits for the requests before them
        ids = torch.tensor([p], device=device)
        model.generate(ids, attention_mask=torch.ones_like(ids), **gen)
        finish.append(time.perf_counter() - t)
    wall = time.perf_counter() - t
    return {"throughput_tok_s": round(out_tokens * len(prompts) / wall, 2), "wall_s": round(wall, 2),
            "e2e_p50_s": round(statistics.median(finish), 4)}, prompts


ENGINES = {"vrtx": run_vrtx, "vllm": run_vllm, "hf": run_hf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=ENGINES, default="vrtx")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workloads", default="short_chat,long_context,shared_prefix")
    ap.add_argument("--requests", type=int, default=None, help="override every workload's request count")
    ap.add_argument("--users", type=int, default=64, help="max concurrent sequences")
    ap.add_argument("--kv-cache-mb", type=int, default=4096)
    ap.add_argument("--max-batched-tokens", type=int, default=2048)
    ap.add_argument("--prefix-caching", action="store_true")
    ap.add_argument("--quant", type=int, default=None, choices=[8, 4])
    ap.add_argument("--static-reserve", type=int, default=4096, help="tokens a static KV cache would reserve per request")
    ap.add_argument("--vllm-gpu-util", type=float, default=0.6)
    args = ap.parse_args()

    device = pick_device()
    dtype = default_dtype(device)
    rows = []
    for w in args.workloads.split(","):
        n = args.requests or WORKLOADS[w][2]
        res, prompts = ENGINES[args.engine](args, device, dtype, w, n)
        res = {"workload": w, "requests": n, "prompt_tokens_mean": round(sum(map(len, prompts)) / n, 1),
               "output_tokens": WORKLOADS[w][1], **res}
        rows.append(res)
        extra = f"   TTFT p50 {res['ttft_p50_s']:.3f}s" if "ttft_p50_s" in res else ""
        extra += f"   p99 {res['ttft_p99_s']:.3f}s" if "ttft_p99_s" in res else ""
        extra += f"   TPOT {res['tpot_mean_ms']:.1f} ms" if res.get("tpot_mean_ms") else ""
        print(f"  {args.engine:<5} {w:<14} {n:>4} req  {res['throughput_tok_s']:>9.1f} tok/s{extra}")
    tag = args.engine + ("_prefix" if args.prefix_caching else "") + (f"_int{args.quant}" if args.quant else "")
    save(f"serving_{tag}", device, {"engine": args.engine, "model": args.model.rstrip("/").split("/")[-1],
                                    "users": args.users, "prefix_caching": args.prefix_caching, "quant": args.quant,
                                    "rows": rows})


if __name__ == "__main__":
    main()
