"""Month 4: prefix caching on a shared-system-prompt workload.

Every request starts with the same long system prompt (think: one company's assistant)
and ends with its own short question. With prefix caching, only the first request pays
for the system prompt's prefill; the rest reuse its KV blocks.

Run:  python -m benchmarks.bench_prefix
      python -m benchmarks.bench_prefix --system-tokens 2048 --requests 128
"""
import argparse
import time

from benchmarks.common import (DEFAULT_MODEL, TOPICS, chat_ids, default_dtype, latency_summary, pick_device, save,
                               text_of_tokens)
from engine.llm_engine import LLMEngine
from engine.scheduler import SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--system-tokens", type=int, default=1024)
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--kv-cache-mb", type=int, default=1024)
    args = ap.parse_args()

    device = pick_device()
    dtype = default_dtype(device)
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, ignore_eos=True)
    rows = []
    for caching in (False, True):
        engine = LLMEngine.from_pretrained(args.model, device=device, dtype=dtype, kv_cache_mb=args.kv_cache_mb,
                                           max_num_seqs=args.batch, enable_prefix_caching=caching)
        tok = engine.tokenizer
        system = "You are the support assistant for VRTX. Use these notes:\n" + text_of_tokens(tok, args.system_tokens)
        prompts = [chat_ids(tok, f"Briefly, what does the documentation say about {TOPICS[i % len(TOPICS)]}?", system)
                   for i in range(args.requests)]
        engine.generate([chat_ids(tok, "hi")], SamplingParams(max_tokens=4, temperature=0.0))  # warm up
        t = time.perf_counter()
        seqs = engine.generate(prompts, params)
        wall = time.perf_counter() - t
        st = engine.stats()
        lat = latency_summary(seqs)
        r = {"prefix_caching": caching, "throughput_tok_s": round(sum(len(s.output_ids) for s in seqs) / wall, 2),
             "wall_s": round(wall, 2), "hit_rate": st["prefix_cache_hit_rate"], "steps": st["steps_total"],
             "prompt_tokens_mean": round(sum(len(p) for p in prompts) / len(prompts), 1), **lat}
        rows.append(r)
        print(f"  prefix caching {'ON ' if caching else 'OFF'}  {r['throughput_tok_s']:>8.1f} tok/s   wall {r['wall_s']:>6.1f}s   "
              f"TTFT p50 {r['ttft_p50_s']:.3f}s p99 {r['ttft_p99_s']:.3f}s   hit rate {r['hit_rate']:.1%}")
        del engine
    rows[1]["speedup"] = round(rows[1]["throughput_tok_s"] / rows[0]["throughput_tok_s"], 2)
    rows[1]["ttft_p50_speedup"] = round(rows[0]["ttft_p50_s"] / max(rows[1]["ttft_p50_s"], 1e-9), 2)
    save("prefix", device, {"model": args.model.rstrip("/").split("/")[-1], "dtype": str(dtype).split(".")[-1],
                            "system_tokens": args.system_tokens, "requests": args.requests,
                            "max_tokens": args.max_tokens, "batch": args.batch, "rows": rows})


if __name__ == "__main__":
    main()
