"""Summarizes the claude.ai A/B test (see README.md here): python -m benchmarks.claude_ai.report

Reads runs.jsonl (one line per chat, from measure.js; label = <condition>-<task>-<run>) and
grades.json ({label: true/false}, whether the answer was right), and prints a table per task and
condition with medians.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDITIONS = {"A": "Built-in web tools only", "B": "ATR on, prompt doesn't mention it", "C": "ATR on, prompt says use atr_fetch",
              "B2": "Retest, ATR with optional question=, not mentioned",
              "C2": "Retest, ATR with optional question=, prompt says use atr_fetch",
              "C3": "Retest, ATR with required question=, prompt says use atr_fetch",
              "C4": "Snippets + short definition, prompt says use atr_fetch",
              "C5": "C4 + room for multi-part and top-N questions, capped search=, prompt says use atr_fetch"}
# atr_fetch's definition. claude.ai loads connector tools only when Claude looks them up (ToolSearch),
# so it is re-read on every model call after that lookup, and not at all if Claude never loads it.
ATR_SCHEMA_TOKENS = 226
SCHEMA_BY_CONDITION = {"B2": 293, "C2": 293, "C3": 306, "C4": 162, "C5": 162}  # atr_fetch grew when question= was added


def main() -> None:
    runs = [json.loads(ln) for ln in (HERE / "runs.jsonl").read_text().splitlines() if ln.strip()]
    grades = json.loads((HERE / "grades.json").read_text()) if (HERE / "grades.json").exists() else {}
    by = defaultdict(list)
    for r in runs:
        cond, task, _ = r["label"].split("-", 2)
        r["cond"], r["task"] = cond, task
        r["atr_used"] = any("atr_" in t for t in r["tools"])
        loaded = "ToolSearch" in r["tools"]
        r["schema"] = SCHEMA_BY_CONDITION.get(cond, ATR_SCHEMA_TOKENS) * (r["calls"] - 1) if loaded else 0
        by[(task, cond)].append(r)

    med = lambda xs: int(statistics.median(xs)) if xs else 0  # noqa: E731
    print("| Task | Setup | Runs | Correct | Used ATR | Tool results (tokens) | Context re-read (tokens) | Model calls |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    totals = defaultdict(lambda: [0, 0])
    for task in sorted({t for t, _ in by}):
        for cond in CONDITIONS:
            rs = by.get((task, cond), [])
            if not rs:
                continue
            ok = sum(1 for r in rs if grades.get(r["label"]) is True)
            ctx = [r["context_tokens"] + r["schema"] for r in rs]
            totals[cond][0] += med(ctx)
            totals[cond][1] += 1
            print(f"| {task} | {cond} | {len(rs)} | {ok}/{len(rs)} | {sum(r['atr_used'] for r in rs)}/{len(rs)} | "
                  f"{med([r['result_tokens'] for r in rs]):,} | {med(ctx):,} | {med([r['calls'] for r in rs])} |")
    print()
    for cond, label in CONDITIONS.items():
        if totals[cond][1]:
            print(f"{cond} ({label}): {totals[cond][0]:,} context tokens summed over task medians")
    print("\nContext includes atr_fetch's definition on each model call after Claude loads it. Tokens are characters / 4.")


if __name__ == "__main__":
    main()
