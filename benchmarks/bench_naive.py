"""Month 1 baseline: time to first token and tokens/sec for the naive engine.

Run:  python -m benchmarks.bench_naive
"""
import json
import statistics
from datetime import datetime
from pathlib import Path

from engine.naive_engine import NaiveEngine

PROMPTS = [
    "Explain how a GPU differs from a CPU.",
    "Write a short poem about the ocean.",
    "What is the capital of Japan and why is it important?",
    "Give three tips for learning to code.",
    "Summarize how the internet works in a paragraph.",
]


def main(max_new_tokens: int = 128):
    engine = NaiveEngine()
    print(f"model={engine.model_id} device={engine.device}")
    list(engine.stream([{"role": "user", "content": "hi"}], max_new_tokens=8))  # warm up

    runs = []
    for p in PROMPTS:
        list(engine.stream([{"role": "user", "content": p}], max_new_tokens=max_new_tokens, temperature=0))
        runs.append(engine.last_stats)
        s = engine.last_stats
        print(f"  ttft={s['ttft_s']*1000:7.1f} ms   {s['tokens_per_s']:6.1f} tok/s   {s['completion_tokens']} tokens")

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "model": engine.model_id,
        "device": engine.device,
        "median_ttft_ms": statistics.median(r["ttft_s"] for r in runs) * 1000,
        "median_tokens_per_s": statistics.median(r["tokens_per_s"] for r in runs),
    }
    print(json.dumps(result, indent=2))
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / f"naive_{engine.device}_{datetime.now():%Y%m%d_%H%M%S}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
