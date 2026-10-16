"""Month 3: quality vs memory vs speed for fp16, int8 and int4 weights.

Perplexity is measured on WikiText-2 (test split) when it can be downloaded
(needs pandas + pyarrow, both preinstalled on Kaggle), otherwise on this repo's README.

Run:  python -m benchmarks.bench_quant
      python -m benchmarks.bench_quant --ppl-tokens 8192 --configs base,int8,int4,int4g32

A config is "base" (the model dtype), or int8 / int4 with an optional group size: int4g32.
"""
import argparse
import math
import time

import torch

from benchmarks.common import DEFAULT_MODEL, default_dtype, filler_text, pick_device, save, topic_prompts
from engine.batch import BatchItem, run_model
from engine.kv_cache import PagedKVCache


def eval_text():
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download

        f = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
        return "wikitext-2-raw-v1 test", "".join(pd.read_parquet(f)["text"].tolist())
    except Exception as e:  # offline, or pandas missing
        print(f"  (WikiText-2 unavailable: {type(e).__name__}; using README.md)")
        return "README.md", filler_text()


@torch.inference_mode()
def perplexity(model, ids: list[int], window: int) -> float:
    """Non-overlapping windows; every token after the first in a window is scored."""
    bs = 16
    cache = PagedKVCache(model.cfg, window // bs + 1, bs, model.device, model.dtype)
    table = list(range(window // bs + 1))
    nll, count = 0.0, 0
    for start in range(0, len(ids) - 1, window):
        chunk = ids[start:start + window + 1]
        if len(chunk) < 2:
            break
        inp, tgt = chunk[:-1], torch.tensor(chunk[1:], device=model.device)
        logits = run_model(model, cache, [BatchItem(table, 0, inp, len(inp))])
        nll += torch.nn.functional.cross_entropy(logits, tgt, reduction="sum").item()
        count += len(tgt)
    return math.exp(nll / count)


def throughput(engine, prompts, max_tokens, batch):
    from engine.scheduler import SamplingParams

    engine.scheduler.max_num_seqs = batch
    engine.generate(prompts[:1], SamplingParams(max_tokens=4, temperature=0.0))
    t = time.perf_counter()
    seqs = engine.generate(prompts, SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True))
    return round(sum(len(s.output_ids) for s in seqs) / (time.perf_counter() - t), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--configs", default="base,int8,int4,int4g64,int4g32")
    ap.add_argument("--ppl-tokens", type=int, default=16384)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--speed-requests", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--batches", default="1,32")
    args = ap.parse_args()

    from engine.llm_engine import LLMEngine

    device = pick_device()
    dtype = default_dtype(device)
    source, text = eval_text()
    rows = []
    for name in args.configs.split(","):
        bits, group = None, 128
        if name != "base":
            bits, _, g = name.removeprefix("int").partition("g")
            bits, group = int(bits), int(g or 128)
        engine = LLMEngine.from_pretrained(args.model, device=device, dtype=dtype, quant_bits=bits, quant_group_size=group)
        ids = engine.tokenizer(text).input_ids[: args.ppl_tokens + 1]
        mem = engine.model.weight_bytes()
        ppl = perplexity(engine.model, ids, args.window)
        prompts = topic_prompts(engine.tokenizer, args.speed_requests)
        speed = {f"batch_{b}_tok_s": throughput(engine, prompts, args.max_tokens, int(b)) for b in args.batches.split(",")}
        r = {"config": name if bits else str(dtype).split(".")[-1], "bits": bits or torch.finfo(dtype).bits,
             "group_size": group if bits else None, "perplexity": round(ppl, 3), "linear_weight_mb": round(mem["linear_bytes"] / 2**20, 1),
             "total_weight_mb": round(mem["total_bytes"] / 2**20, 1), **speed}
        rows.append(r)
        print(f"  {r['config']:<8} ppl {ppl:>8.3f}   matmul weights {r['linear_weight_mb']:>7.1f} MB   "
              f"all weights {r['total_weight_mb']:>7.1f} MB   " + "   ".join(f"{k} {v}" for k, v in speed.items()))
        del engine
        if device == "cuda":
            torch.cuda.empty_cache()
    base = rows[0]
    for r in rows:
        r["linear_memory_saved_pct"] = round(100 * (1 - r["linear_weight_mb"] / base["linear_weight_mb"]), 1)
        r["ppl_increase_pct"] = round(100 * (r["perplexity"] / base["perplexity"] - 1), 2)
    save("quant", device, {"model": args.model.rstrip("/").split("/")[-1], "dtype": str(dtype).split(".")[-1],
                           "eval_text": source, "ppl_tokens": args.ppl_tokens, "window": args.window, "rows": rows})


if __name__ == "__main__":
    main()
