"""Draw every writeup chart from the JSON files in benchmarks/results/ into docs/charts/.

Uses the newest result of each kind (CUDA runs preferred). Charts whose data is
missing are skipped, so this can run after any subset of the benchmarks.

Run:  python -m benchmarks.plots
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.common import ROOT, RESULTS

OUT = ROOT / "docs" / "charts"

# Categorical slots, fixed order (validated palette): blue, orange, aqua, yellow.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# Peak numbers for the roofline (fp16 tensor-core FLOP/s, DRAM bytes/s).
PEAKS = {"Tesla T4": (65e12, 320e9), "NVIDIA L4": (121e12, 300e9), "NVIDIA A100": (312e12, 1.9e12),
         "Tesla P100": (18.7e12, 732e9)}


def latest(prefix: str):
    files = sorted(RESULTS.glob(f"{prefix}_*.json"), key=lambda p: p.name[-20:])
    cuda = [f for f in files if "_cuda_" in f.name]
    pick = (cuda or files)[-1:] if files else []
    return json.loads(pick[0].read_text()) if pick else None


def all_cuda(prefix: str):
    return [json.loads(f.read_text()) for f in sorted(RESULTS.glob(f"{prefix}_cuda_*.json"), key=lambda p: p.name[-20:])]


def _setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "axes.axisbelow": True, "lines.linewidth": 2, "lines.markersize": 7,
    })
    return plt


def _save(plt, fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print(f"  wrote docs/charts/{name}")


def _bar_labels(ax, bars, fmt="{:.0f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8, color=INK2, xytext=(0, 2), textcoords="offset points")


def throughput_vs_users(plt):
    runs = all_cuda("batching")
    if not runs:
        return
    series = {}
    hf = None
    for r in runs:  # newest run of each engine configuration wins
        label = "VRTX + Triton kernels" if r.get("kernels") else "VRTX, PyTorch ops"
        if r.get("quant"):
            label += f", int{r['quant']}"
        series[label] = r["runs"]
        hf = (r.get("hf_baseline") or {}).get("throughput_tok_s") or hf
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (label, rs) in enumerate(series.items()):
        ax.plot([x["max_batch"] for x in rs], [x["throughput_tok_s"] for x in rs], marker="o", color=C[i % 4], label=label)
        last = rs[-1]
        ax.annotate(f"{last['throughput_tok_s']:.0f}", (last["max_batch"], last["throughput_tok_s"]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    if hf:
        ax.axhline(hf, color=INK2, linestyle="--", linewidth=1.2, label=f"Hugging Face generate, 1 at a time ({hf:.0f})")
    ax.set_xscale("log", base=2)
    batches = sorted({x["max_batch"] for rs in series.values() for x in rs})
    ax.set_xticks(batches, [str(b) for b in batches])
    ax.minorticks_off()
    ax.set_xlabel("concurrent users (max batch)")
    ax.set_ylabel("output tokens / s")
    ax.set_title("Throughput vs concurrent users")
    ax.legend(loc="upper left")
    _save(plt, fig, "throughput_vs_users.png")


def latency_cdf(plt):
    r = latest("serving_vrtx")
    if not r:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, row in enumerate(r["rows"]):
        xs = sorted(row.get("ttft_all_s") or [])
        if not xs:
            continue
        ax.step(xs, [(k + 1) / len(xs) for k in range(len(xs))], where="post", color=C[i % 4],
                label=f"{row['workload']} ({row['requests']} req)")
    ax.set_xlabel("time to first token (s)")
    ax.set_ylabel("fraction of requests")
    ax.set_title(f"Time-to-first-token CDF, {r['users']} concurrent users")
    ax.legend(loc="lower right")
    _save(plt, fig, "latency_cdf.png")


def roofline(plt):
    r = latest("kernels")
    if not r or r.get("device") != "cuda":
        return
    peak_flops, peak_bw = PEAKS.get(r.get("gpu"), PEAKS["Tesla T4"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    import numpy as np

    xi = np.logspace(-1, 3, 200)
    ax.plot(xi, np.minimum(peak_flops, xi * peak_bw) / 1e9, color=INK2, linewidth=1.5, label="roofline")
    groups = {}
    for op in r["ops"]:
        groups.setdefault(op["kernel"].replace("int8_", "int_").replace("int4_", "int_"), []).append(op)
    for i, (name, ops) in enumerate(list(groups.items())[:4]):
        xs = [o["flops"] / o["bytes"] for o in ops]
        ax.scatter(xs, [o["triton_gflops"] for o in ops], color=C[i], s=40, label=name, edgecolor=SURFACE, linewidth=1.5)
    others = [o for k, v in list(groups.items())[4:] for o in v]
    if others:
        ax.scatter([o["flops"] / o["bytes"] for o in others], [o["triton_gflops"] for o in others], color=INK2, s=30,
                   marker="s", label="other kernels", edgecolor=SURFACE, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("achieved GFLOP/s")
    ax.set_title(f"Roofline, Triton kernels on {r.get('gpu')}")
    ax.legend(loc="lower right", fontsize=8)
    _save(plt, fig, "roofline.png")


def kernel_speedups(plt):
    r = latest("kernels")
    if not r:
        return
    ops = r["ops"]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(ops))))
    labels = [f"{o['kernel']}  {o['shape']}" for o in ops]
    bars = ax.barh(range(len(ops)), [o["speedup"] for o in ops], color=C[0], height=0.6)
    ax.axvline(1, color=INK2, linewidth=1)
    ax.set_yticks(range(len(ops)), labels, fontsize=7)
    ax.invert_yaxis()
    for b, o in zip(bars, ops):
        ax.annotate(f"{o['speedup']:.2f}x", (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=7, color=INK2)
    ax.set_xlabel("speedup over the PyTorch version (higher is better)")
    ax.set_title("Triton kernel speedups")
    ax.grid(axis="y", visible=False)
    _save(plt, fig, "kernel_speedups.png")


def perplexity_vs_memory(plt):
    r = latest("quant")
    if not r:
        return
    rows = r["rows"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([x["linear_weight_mb"] for x in rows], [x["perplexity"] for x in rows], color=C[0], marker="o")
    for x in rows:
        ax.annotate(f"{x['config']}\nppl {x['perplexity']:.2f}", (x["linear_weight_mb"], x["perplexity"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK2)
    ax.margins(x=0.1, y=0.12)
    ax.set_xlabel("transformer matmul weights (MB)")
    ax.set_ylabel(f"perplexity on {r['eval_text']} (lower is better)")
    ax.set_title(f"Perplexity vs memory, {r['model']}")
    _save(plt, fig, "perplexity_vs_memory.png")


def spec_vs_gamma(plt):
    runs = all_cuda("spec")
    greedy = [x for x in runs if x.get("temperature") == 0.0]
    r = (greedy or runs or [latest("spec")])[-1]  # the greedy sweep covers every gamma
    if not r:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    batches = sorted({x["batch"] for x in r["rows"]})
    for i, b in enumerate(batches):
        fixed = sorted([x for x in r["rows"] if x["batch"] == b and x["mode"] in ("off", "fixed")], key=lambda x: x["gamma"])
        ax.plot([x["gamma"] for x in fixed], [x["speedup"] for x in fixed], marker="o", color=C[i % 4], label=f"batch {b}, fixed gamma")
        ad = [x for x in r["rows"] if x["batch"] == b and x["mode"] == "adaptive"]
        if ad:
            ax.axhline(ad[0]["speedup"], color=C[i % 4], linestyle=":", linewidth=1.5,
                       label=f"batch {b}, adaptive ({ad[0]['speedup']:.2f}x)")
    ax.axhline(1, color=INK2, linewidth=1)
    ax.set_xlabel("draft tokens per round (gamma)")
    ax.set_ylabel("speedup vs no speculation")
    t = r.get("temperature", 0)
    ax.set_title(f"Spec decoding: {r['target'].split('/')[-1]} + {r['draft'].split('/')[-1]} draft"
                 + (", greedy" if t == 0 else f", temp {t:g}"))
    ax.legend(fontsize=8)
    _save(plt, fig, "spec_speedup_vs_gamma.png")


def prefix_cache(plt):
    r = latest("prefix")
    if not r:
        return
    off, on = r["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, key, title, fmt in ((axes[0], "throughput_tok_s", "Throughput (tok/s)", "{:.0f}"),
                                (axes[1], "ttft_p50_s", "Median time to first token (s)", "{:.2f}")):
        bars = ax.bar(["off", "on"], [off[key], on[key]], color=[C[1], C[0]], width=0.55)
        _bar_labels(ax, bars, fmt)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("prefix caching")
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Shared {r['system_tokens']}-token system prompt, {r['requests']} requests "
                 f"(hit rate {on['hit_rate']:.0%})", x=0.02, ha="left", fontweight="bold")
    _save(plt, fig, "prefix_cache.png")


def kv_waste(plt):
    r = latest("serving_vrtx")
    if not r or "kv_memory" not in r["rows"][0]:
        return
    rows = r["rows"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = range(len(rows))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], [row["kv_memory"]["static_waste_pct"] for row in rows], w, color=C[1],
                label="static cache, reserve max length per request")
    b2 = ax.bar([i + w / 2 for i in x], [row["kv_memory"]["paged_waste_pct"] for row in rows], w, color=C[0],
                label="VRTX paged cache (16-token blocks)")
    _bar_labels(ax, b1, "{:.0f}%")
    _bar_labels(ax, b2, "{:.1f}%")
    ax.set_xticks(list(x), [row["workload"] for row in rows])
    ax.set_ylabel("KV memory wasted (%)")
    ax.set_title("KV cache memory waste: before and after paging")
    ax.legend(fontsize=8)
    ax.grid(axis="x", visible=False)
    _save(plt, fig, "kv_waste.png")


def versus(plt):
    engines = {}
    for name, label in (("serving_hf", "Hugging Face naive"), ("serving_vrtx", "VRTX"), ("serving_vllm", "vLLM")):
        r = latest(name)
        if r:
            engines[label] = {row["workload"]: row["throughput_tok_s"] for row in r["rows"]}
    if len(engines) < 2:
        return
    workloads = list(next(iter(engines.values())).keys())
    for v in engines.values():
        workloads += [w for w in v if w not in workloads]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.8 / len(engines)
    for i, (label, vals) in enumerate(engines.items()):
        xs = [k + (i - (len(engines) - 1) / 2) * w for k in range(len(workloads))]
        have = [(x, vals[wl]) for x, wl in zip(xs, workloads) if wl in vals]  # no bar (not a 0) when not run
        bars = ax.bar([x for x, _ in have], [v for _, v in have], w * 0.92, color=C[i], label=label)
        _bar_labels(ax, bars)
    ax.set_xticks(range(len(workloads)), workloads)
    ax.set_ylabel("output tokens / s")
    ax.set_title(" vs ".join(reversed(list(engines))) + " (same GPU, same prompts)")
    ax.legend()
    ax.grid(axis="x", visible=False)
    _save(plt, fig, "vs_vllm.png")


def main():
    plt = _setup()
    for fn in (throughput_vs_users, latency_cdf, roofline, kernel_speedups, perplexity_vs_memory, spec_vs_gamma,
               prefix_cache, kv_waste, versus):
        try:
            fn(plt)
        except Exception as e:  # one broken result file should not stop the other charts
            print(f"  skipped {fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
