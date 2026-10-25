"""How many tokens ATR saves on typical MCP traffic.

    python -m benchmarks.bench_atr            (no GPU, no network, ~1 second)

MCP scenarios come from atr/tests/fake_server.py, a 50-tool server whose definitions and
results are shaped like real ones (GitHub REST payloads, service logs, a fetched web page).
Coding scenarios run in a throwaway project: a real `pytest -v` run (if pytest is installed), a real
git repo with a lockfile change, a re-read after an edit, and install/jest logs shaped like real ones.
"Before" is the raw output; Claude Code's own Bash tool would show at most 30,000 characters of it.
Token counts use tiktoken if installed, otherwise chars/4.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from atr import tokens
from atr.compress import compress, compress_output
from atr.config import Config
from atr.server import ATRServer
from atr import code
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


def coding(tmp: Path) -> list[dict]:
    rows = []
    # a project with 300 passing tests and 3 failing ones
    (tmp / "tests").mkdir()
    (tmp / "calc.py").write_text("def add(a, b): return a + b\ndef div(a, b): return a / b\n"
                                 "def price(s): return float(s.strip('$'))\n")
    for f in range(12):
        (tmp / "tests" / f"test_m{f}.py").write_text("from calc import add\n" + "".join(
            f"def test_add_{f}_{t}():\n    assert add({t}, {f}) == {t + f}\n" for t in range(25)))
    (tmp / "tests" / "test_bugs.py").write_text(
        "from calc import div, price\ndef test_div():\n    assert div(1, 0) == 0\n"
        "def test_price():\n    assert price('$1,200') == 1200\ndef test_list():\n    assert [1, 2, 3] == [1, 2, 4]\n")
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider"], cwd=tmp,
                             capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.TimeoutExpired):
        out = ""
    if "passed" in out:
        r = compress_output(out, budget=1500)
        rows.append(row("pytest -v: 303 tests, 3 failing", r.before, r.after, "all 3 failures and the summary kept"))
    # the same error once per test file (a broken jest config across 40 suites)
    blk = ("FAIL src/m{0}.test.js\n  ● Test suite failed to run\n\n    Cannot find module 'ts-jest' from 'jest.config.js'\n\n"
           "    > 1 | const {{ f{0} }} = require('./m{0}');\n        |                 ^\n\n"
           + "".join(f"      at Resolver.r{k} (node_modules/jest-resolve/build/resolver.js:{{0}}:{k})\n" for k in range(8)))
    raw = "\n".join(blk.format(i) for i in range(40)) + "\nTest Suites: 40 failed, 40 total\nTests:       0 total\n"
    r = compress_output(raw, budget=1500)
    rows.append(row("jest: one config error in 40 suites", r.before, r.after, "first error kept, the rest counted"))
    # a pip install log
    pkgs = [f"pkg{i}" for i in range(60)]
    raw = "".join(f"Collecting {p}\n  Downloading {p}-1.{i}.0-py3-none-any.whl.metadata (4.{i} kB)\n"
                  f"Downloading {p}-1.{i}.0-py3-none-any.whl (2{i}0 kB)\n   ━━━━━━━━━━━━━━━━━━━━ 2{i}0.0/2{i}0.0 kB 9.1 MB/s eta 0:00:00\n"
                  for i, p in enumerate(pkgs))
    raw += "Installing collected packages: " + ", ".join(pkgs) + "\nSuccessfully installed " + " ".join(
        f"{p}-1.{i}.0" for i, p in enumerate(pkgs)) + "\n"
    r = compress_output(raw, budget=600)
    rows.append(row("pip install, 60 packages", r.before, r.after, "the final 'Successfully installed' kept"))
    # git diff: two small code edits and a lockfile change
    repo = tmp / "repo"
    (repo / "src").mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", "-c", "user.email=b@b", "-c", "user.name=b", *a], cwd=repo,
                                    check=True, capture_output=True)
    for i in range(5):
        (repo / "src" / f"m{i}.js").write_text("".join(f"exports.f{k} = (a) => a + {k};\n" for k in range(40)))
    (repo / "package-lock.json").write_text("{}\n")
    try:
        git("init", "-q"); git("add", "-A"); git("commit", "-qm", "init")
        (repo / "package-lock.json").write_text("".join(
            f'    "node_modules/pkg{i}": {{\n      "version": "1.0.{i}",\n      "resolved": "https://registry.npmjs.org/pkg{i}/-/pkg{i}-1.0.{i}.tgz",\n'
            f'      "integrity": "sha512-{"x" * 80}"\n    }},\n' for i in range(300)))
        for i in (1, 3):
            t = (repo / "src" / f"m{i}.js").read_text().replace("a + 7;", "a * 7;")
            (repo / "src" / f"m{i}.js").write_text(t)
        text, plain = code.git_diff(repo)
        rows.append(row("git diff: 2 code edits + package-lock.json", plain, tokens.count(text),
                        "code changes shown, lockfile listed"))
    except (OSError, subprocess.CalledProcessError):
        pass
    # reading the same 200 lines again after a one-line edit
    srv = ATRServer(Config(read_roots=[str(tmp)], project=str(tmp)))
    big = tmp / "big.py"
    big.write_text("".join(f"def f{i}(x):\n    return x + {i}\n" for i in range(150)))
    first = asyncio.run(srv.call("atr_read_file", {"path": "big.py", "lines": "1-200"}))["content"][0]["text"]
    big.write_text(big.read_text().replace("return x + 42", "return x * 42"))
    again = asyncio.run(srv.call("atr_read_file", {"path": "big.py", "lines": "1-200"}))["content"][0]["text"]
    rows.append(row("re-reading 200 lines after a 1-line edit", tokens.count(first), tokens.count(again),
                    "only the changed line comes back"))
    return rows


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

    with tempfile.TemporaryDirectory() as tmp:
        rows += coding(Path(tmp))

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
