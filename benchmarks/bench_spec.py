"""Month 4: speculative decoding speedup vs gamma, with a fixed gamma and with adaptive gamma.

Target Qwen2.5-1.5B-Instruct, draft Qwen2.5-0.5B-Instruct (same tokenizer). Speculative
decoding helps most when the GPU is underused, so this runs at small batch sizes.

Run:  python -m benchmarks.bench_spec
      python -m benchmarks.bench_spec --gammas 0,2,4 --batches 1 --temperature 0.7
"""
import argparse
import time

import torch

from benchmarks.common import default_dtype, pick_device, save, topic_prompts
from engine.llm_engine import LLMEngine, _download
from engine.model import VRTXModel
from engine.scheduler import SamplingParams


def run(engine, prompts, params, batch):
    engine.scheduler.max_num_seqs = batch
    engine.generate(prompts[:1], SamplingParams(max_tokens=8, temperature=0.0))  # warm up
    if engine.spec is not None:
        s = engine.spec
        s.proposed = s.accepted = s.rounds = s.tokens_from_rounds = 0
    t = time.perf_counter()
    seqs = engine.generate(prompts, params)
    wall = time.perf_counter() - t
    out = {"throughput_tok_s": round(sum(len(x.output_ids) for x in seqs) / wall, 2), "wall_s": round(wall, 2)}
    if engine.spec is not None:
        out.update({k: v for k, v in engine.spec.stats().items() if k != "spec_rounds"})
    return out, [x.output_ids for x in seqs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--gammas", default="0,1,2,3,4,5,6")
    ap.add_argument("--batches", default="1,4")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--kv-cache-mb", type=int, default=1024)
    args = ap.parse_args()

    device = pick_device()
    dtype = default_dtype(device)
    common = dict(device=device, dtype=dtype, kv_cache_mb=args.kv_cache_mb)
    params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, top_p=1.0, top_k=0, ignore_eos=True)

    base = LLMEngine.from_pretrained(args.target, **common)
    prompts = topic_prompts(base.tokenizer, args.requests)
    rows, reference = [], {}
    for b in [int(x) for x in args.batches.split(",")]:
        r, outs = run(base, prompts, params, b)
        reference[b] = outs
        rows.append({"batch": b, "gamma": 0, "mode": "off", **r})
        print(f"  batch {b}  no speculation         {r['throughput_tok_s']:>7.1f} tok/s")
    draft_model = VRTXModel.from_pretrained(_download(args.draft), device, dtype)
    target_model = base.model
    tok, stop = base.tokenizer, base.stop_ids
    del base

    configs = [(g, False) for g in (int(x) for x in args.gammas.split(",")) if g > 0] + [(max(int(x) for x in args.gammas.split(",")), True)]
    for g, adaptive in configs:
        engine = LLMEngine(target_model, tok, stop_ids=stop, draft_model=draft_model, num_speculative_tokens=g,
                           adaptive_speculation=adaptive, kv_cache_mb=args.kv_cache_mb)
        for b in [int(x) for x in args.batches.split(",")]:
            r, outs = run(engine, prompts, params, b)
            if args.temperature == 0:  # greedy speculation must reproduce the target exactly
                same = sum(o == ref for o, ref in zip(outs, reference[b]))
                r["greedy_outputs_identical"] = f"{same}/{len(outs)}"
            rows.append({"batch": b, "gamma": g, "mode": "adaptive" if adaptive else "fixed", **r})
            print(f"  batch {b}  gamma {'<=' if adaptive else '= '}{g} {'adaptive' if adaptive else 'fixed   '}  "
                  f"{r['throughput_tok_s']:>7.1f} tok/s   accept {r.get('spec_acceptance_rate')}   "
                  f"tokens/round {r.get('spec_tokens_per_round')}   c {r.get('spec_cost_ratio')}")
        del engine
        if device == "cuda":
            torch.cuda.empty_cache()

    for r in rows:
        off = next(x for x in rows if x["batch"] == r["batch"] and x["mode"] == "off")
        r["speedup"] = round(r["throughput_tok_s"] / off["throughput_tok_s"], 2)
    save("spec", device, {"target": args.target, "draft": args.draft, "dtype": str(dtype).split(".")[-1],
                          "requests": args.requests, "max_tokens": args.max_tokens, "temperature": args.temperature,
                          "rows": rows})


if __name__ == "__main__":
    main()
