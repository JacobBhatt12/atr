"""Helpers shared by the benchmark scripts: device choice, prompts, timing, saving results."""
from __future__ import annotations

import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
DEFAULT_MODEL = str(ROOT / "models" / "qwen2.5-0.5b")

TOPICS = [
    "how GPUs work", "the history of jazz", "why the sky is blue", "how to learn Python",
    "black holes", "how vaccines work", "the stock market", "how planes fly",
    "photosynthesis", "the Roman empire", "machine learning", "how the internet works",
]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" else torch.float32


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def gpu_name(device: str) -> str | None:
    return torch.cuda.get_device_name() if device == "cuda" else None


def filler_text() -> str:
    """Deterministic English text for long prompts: this project's own README."""
    return (ROOT / "README.md").read_text()


def chat_ids(tok, user: str, system: str | None = None) -> list[int]:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)).input_ids


def text_of_tokens(tok, n: int, offset: int = 0) -> str:
    ids = tok(filler_text()).input_ids
    while len(ids) < offset + n:
        ids = ids + ids
    return tok.decode(ids[offset:offset + n])


def topic_prompts(tok, n: int) -> list[list[int]]:
    return [chat_ids(tok, f"Explain {TOPICS[i % len(TOPICS)]} in simple terms.") for i in range(n)]


def pct(xs, q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, max(0, round(q / 100 * (len(xs) - 1))))
    return xs[i]


def latency_summary(seqs) -> dict:
    ttft = [s.first_token_time - s.arrival_time for s in seqs]
    e2e = [s.finish_time - s.arrival_time for s in seqs]
    tpot = [(s.finish_time - s.first_token_time) / (len(s.output_ids) - 1) for s in seqs if len(s.output_ids) > 1]
    r = lambda x: round(x, 4)  # noqa: E731
    return {
        "ttft_p50_s": r(pct(ttft, 50)), "ttft_p95_s": r(pct(ttft, 95)), "ttft_p99_s": r(pct(ttft, 99)),
        "tpot_mean_ms": r(1000 * statistics.mean(tpot)) if tpot else None,
        "tpot_p99_ms": r(1000 * pct(tpot, 99)) if tpot else None,
        "e2e_p50_s": r(pct(e2e, 50)), "e2e_p99_s": r(pct(e2e, 99)),
        "ttft_all_s": [r(x) for x in ttft], "e2e_all_s": [r(x) for x in e2e],
    }


def save(kind: str, device: str, report: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {"date": datetime.now().isoformat(timespec="seconds"), "device": device, "gpu": gpu_name(device), **report}
    path = RESULTS / f"{kind}_{device}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"saved {path.relative_to(ROOT)}")
    print(f"RESULTS_JSON_{kind.upper()} " + json.dumps(report))
    return path


def timed(fn, device: str, warmup: int = 3, iters: int = 20) -> float:
    """Median seconds per call."""
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t)
    return statistics.median(times)
