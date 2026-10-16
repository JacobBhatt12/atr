"""Month 2 headline benchmark: continuous batching ON vs OFF.

Same engine, same model, same prompts. The only change is how many sequences
may share a forward pass. Batch size 1 is how the Month 1 server behaved.

Run:  python -m benchmarks.bench_batching                     (uses models/qwen2.5-0.5b)
      python -m benchmarks.bench_batching --requests 32 --max-tokens 64
      python -m benchmarks.bench_batching --hf-baseline          (also time plain HF model.generate)
"""
import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch

from engine.llm_engine import LLMEngine
from engine.naive_engine import pick_device
from engine.scheduler import SamplingParams

ROOT = Path(__file__).resolve().parent.parent
TOPICS = [
    "how GPUs work", "the history of jazz", "why the sky is blue", "how to learn Python",
    "black holes", "how vaccines work", "the stock market", "how planes fly",
    "photosynthesis", "the Roman empire", "machine learning", "how the internet works",
]


def run(engine: LLMEngine, prompts, max_tokens: int, batch: int) -> dict:
    engine.scheduler.max_num_seqs = batch
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
    start = time.perf_counter()
    steps0, pre0 = engine.total_steps, engine.scheduler.num_preemptions
    seqs = engine.generate(prompts, params)
    wall = time.perf_counter() - start
    out = sum(len(s.output_ids) for s in seqs)
    lat = [s.finish_time - s.arrival_time for s in seqs]
    ttft = [s.first_token_time - s.arrival_time for s in seqs]
    return {
        "max_batch": batch,
        "wall_s": round(wall, 2),
        "output_tokens": out,
        "throughput_tok_s": round(out / wall, 2),
        "mean_latency_s": round(statistics.mean(lat), 2),
        "p50_ttft_s": round(statistics.median(ttft), 3),
        "engine_steps": engine.total_steps - steps0,
        "preemptions": engine.scheduler.num_preemptions - pre0,
    }


@torch.inference_mode()
def run_hf_baseline(model_path: str, prompts, max_tokens: int, device: str, dtype) -> dict:
    """Plain Hugging Face model.generate, one request at a time: the naive way to serve a model."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device).eval()
    gen = dict(max_new_tokens=max_tokens, min_new_tokens=max_tokens, do_sample=False,
               temperature=None, top_p=None, top_k=None, pad_token_id=model.config.eos_token_id)

    def one(p, **overrides):
        ids = torch.tensor([p], device=device)
        out = model.generate(ids, attention_mask=torch.ones_like(ids), **{**gen, **overrides})
        return out.shape[1] - ids.shape[1]

    one(prompts[0], max_new_tokens=4, min_new_tokens=4)  # warm up
    start = time.perf_counter()
    out, finish = 0, []
    for p in prompts:  # every request arrived at t=0 and waits for the ones before it
        out += one(p)
        finish.append(time.perf_counter() - start)
    wall = time.perf_counter() - start
    del model
    return {
        "engine": "hf_generate_sequential",
        "wall_s": round(wall, 2),
        "output_tokens": out,
        "throughput_tok_s": round(out / wall, 2),
        "mean_latency_s": round(statistics.mean(finish), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "qwen2.5-0.5b"))
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--batches", default="1,4,16")
    ap.add_argument("--kv-cache-mb", type=int, default=256, help="KV cache budget in MB")
    ap.add_argument("--hf-baseline", action="store_true", help="also time plain HF model.generate, one request at a time")
    args = ap.parse_args()

    device = pick_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    engine = LLMEngine.from_pretrained(args.model, device=device, dtype=dtype, kv_cache_mb=args.kv_cache_mb)
    tok = engine.tokenizer
    prompts = []
    for i in range(args.requests):
        msg = [{"role": "user", "content": f"Explain {TOPICS[i % len(TOPICS)]} in simple terms."}]
        prompts.append(tok(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)).input_ids)

    print(f"model={Path(args.model).name} device={device} params={engine.model.num_params()/1e6:.0f}M "
          f"dtype={str(dtype).split('.')[-1]} kv_blocks={engine.cache.num_blocks} requests={args.requests} max_tokens={args.max_tokens}")
    engine.generate(prompts[:1], SamplingParams(max_tokens=4, temperature=0.0))  # warm up

    results = []
    for b in [int(x) for x in args.batches.split(",")]:
        r = run(engine, prompts, args.max_tokens, b)
        results.append(r)
        print(f"  batch<= {b:>3}:  {r['throughput_tok_s']:>8.1f} tok/s   wall {r['wall_s']:>7.1f}s   "
              f"mean latency {r['mean_latency_s']:>6.1f}s   steps {r['engine_steps']}")
    base = results[0]["throughput_tok_s"]
    for r in results:
        r["speedup_vs_batch1"] = round(r["throughput_tok_s"] / base, 2)
    print(f"  speedup at batch<={results[-1]['max_batch']}: {results[-1]['speedup_vs_batch1']}x")

    hf = None
    if args.hf_baseline:
        hf = run_hf_baseline(args.model, prompts, args.max_tokens, device, dtype)
        print(f"  HF generate (1 at a time): {hf['throughput_tok_s']:>8.1f} tok/s   wall {hf['wall_s']:>7.1f}s   "
              f"mean latency {hf['mean_latency_s']:>6.1f}s")
        for r in results:
            r["speedup_vs_hf"] = round(r["throughput_tok_s"] / hf["throughput_tok_s"], 2)
        best = max(results, key=lambda r: r["throughput_tok_s"])
        print(f"  VRTX at batch<={best['max_batch']} is {best['speedup_vs_hf']}x faster than naive Hugging Face")

    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(exist_ok=True)
    report = {"date": datetime.now().isoformat(timespec="seconds"), "model": Path(args.model).name,
              "device": device, "dtype": str(dtype).split(".")[-1], "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
              "requests": args.requests, "max_tokens": args.max_tokens, "kv_cache_mb": args.kv_cache_mb,
              "runs": results, "hf_baseline": hf}
    path = out_dir / f"batching_{device}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"saved {path.relative_to(ROOT)}")
    print("RESULTS_JSON " + json.dumps(report))


if __name__ == "__main__":
    main()
