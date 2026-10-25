"""How many tokens ATR saves on typical MCP traffic.

    python -m benchmarks.bench_atr            (no GPU, no network, ~1 second)

Every scenario comes from atr/tests/fake_server.py, a 50-tool server whose definitions and
results are shaped like real ones (GitHub REST payloads, service logs, a fetched web page).
Token counts use tiktoken if installed, otherwise chars/4.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from atr import tokens
from atr.compress import compress
from atr.config import Config
from atr.server import ATRServer
from atr.tests import fake_server as fake


async def schemas() -> tuple[int, int, int]:
    srv = ATRServer(Config(servers={"fake": {"command": sys.executable, "args": ["-m", "atr.tests.fake_server"]}}))
    await srv.start()
    try:
        found = await srv.call("atr_find_tools", {"query": "list github issues"})
        return srv.stats["schema_full"], srv.stats["schema_exposed"], tokens.count(found["content"][0]["text"])
    finally:
        await srv.close()


def row(name: str, before: int, after: int, note: str = "") -> dict:
    return {"scenario": name, "before": before, "after": after,
            "saved_pct": round(100 * (1 - after / before), 1), "note": note}


def main() -> None:
    full, exposed, find = asyncio.run(schemas())
    rows = [row("50 tool schemas, sent every turn", full, exposed, f"+{find} tokens per atr_find_tools call")]
    cases = [
        ("1,000 GitHub issues (JSON), 1,500 budget", json.dumps(fake.issues(1000), indent=2), 1500),
        ("30 GitHub issues, no budget (lossless)", json.dumps(fake.issues(30), indent=2), None),
        ("2,000-line service log, 600 budget", fake.logs(2000), 600),
        ("Fetched web page (HTML)", fake.call("web_fetch_url", {})["content"][0]["text"], None),
    ]
    for name, raw, budget in cases:
        t = time.perf_counter()
        r = compress(raw, budget=budget)
        ms = (time.perf_counter() - t) * 1000
        rows.append(row(name, r.before, r.after, f"{ms:.0f} ms" + (", rest stashed for atr_query" if r.lossy else "")))

    w = max(len(r["scenario"]) for r in rows)
    print(f"{'scenario':<{w}}  {'before':>8}  {'after':>7}  {'saved':>6}  note")
    for r in rows:
        print(f"{r['scenario']:<{w}}  {r['before']:>8,}  {r['after']:>7,}  {r['saved_pct']:>5.1f}%  {r['note']}")
    out = Path(__file__).parent / "results" / f"atr_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"tokenizer": "tiktoken" if tokens._ENC else "chars/4", "rows": rows}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
