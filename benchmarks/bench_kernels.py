"""Month 3: every Triton kernel vs its PyTorch twin, at Qwen2.5-0.5B shapes, plus an
end-to-end ablation that turns the kernels on one at a time inside the real engine.

Needs a CUDA GPU.
Run:  python -m benchmarks.bench_kernels
      python -m benchmarks.bench_kernels --skip-ablation

Each kernel row records FLOPs and bytes moved, so benchmarks/plots.py can draw a roofline.
"""
import argparse
import time

import torch

from benchmarks.common import DEFAULT_MODEL, default_dtype, pick_device, save, timed, topic_prompts
from engine import ops
from engine.batch import BatchItem, build_batch
from engine.kernels import KERNELS, triton_usable

# Qwen2.5-0.5B
H, NH, NKV, D, I, LAYERS = 896, 14, 2, 64, 4864, 24


def row(name, shape, t_torch, t_triton, flops, nbytes):
    return {
        "kernel": name, "shape": shape,
        "torch_us": round(t_torch * 1e6, 2), "triton_us": round(t_triton * 1e6, 2),
        "speedup": round(t_torch / t_triton, 2),
        "flops": flops, "bytes": nbytes,
        "triton_gflops": round(flops / t_triton / 1e9, 2), "triton_gbs": round(nbytes / t_triton / 1e9, 2),
        "torch_gflops": round(flops / t_torch / 1e9, 2), "torch_gbs": round(nbytes / t_torch / 1e9, 2),
    }


def bench_ops(device, dtype):
    from engine.kernels import activation, attention, norm, qmatmul, rope
    from engine.quant import quantize_weight

    es = torch.tensor([], dtype=dtype).element_size()
    rnd = lambda *s: torch.randn(*s, device=device, dtype=dtype)  # noqa: E731
    out = []

    for T in (1, 64, 512):
        x, r, w = rnd(T, H), rnd(T, H), rnd(H)
        out.append(row("add_rms_norm", f"T={T}", timed(lambda: ops.add_rms_norm(x, r, w, 1e-6), device),
                       timed(lambda: norm.add_rms_norm(x, r, w, 1e-6), device), 5 * T * H, 4 * T * H * es))
        gu = rnd(T, 2 * I)
        out.append(row("silu_and_mul", f"T={T}", timed(lambda: ops.silu_and_mul(gu), device),
                       timed(lambda: activation.silu_and_mul(gu), device), 6 * T * I, 3 * T * I * es))
        qkv = rnd(T, (NH + 2 * NKV) * D)
        q, k, v = qkv.split([NH * D, NKV * D, NKV * D], dim=-1)
        q, k, v = q.view(T, NH, D), k.view(T, NKV, D), v.view(T, NKV, D)
        pos = torch.arange(T, device=device)
        inv = 1.0 / (1e6 ** (torch.arange(0, D, 2, device=device).float() / D))
        kc, vc = rnd(4096, NKV, D), rnd(4096, NKV, D)
        slots = torch.randperm(4096, device=device)[:T]
        out.append(row("rope_and_cache", f"T={T}",
                       timed(lambda: ops.rope_and_cache(q, k, v, pos, inv, kc, vc, slots), device),
                       timed(lambda: rope.rope_and_cache(q, k, v, pos, inv, kc, vc, slots), device),
                       6 * T * (NH + NKV) * D, (2 * NH + 4 * NKV) * T * D * es))

    # attention over a paged cache: decode batches, and a prefill chunk
    bs, ctx = 16, 512
    for B, q_len in ((1, 1), (8, 1), (32, 1), (64, 1), (1, 512), (4, 128)):
        blocks_each = (ctx + bs - 1) // bs
        num_blocks = B * blocks_each + 1
        perm = torch.randperm(num_blocks).tolist()
        items = [BatchItem(perm[i * blocks_each:(i + 1) * blocks_each], ctx - q_len, [0] * q_len, 1) for i in range(B)]
        _, _, meta, _ = build_batch(items, bs, device)
        kc, vc = rnd(num_blocks * bs, NKV, D), rnd(num_blocks * bs, NKV, D)
        q = rnd(B * q_len, NH, D)

        def t_ops():
            meta.scratch.clear()
            ops.paged_attention(q, kc, vc, meta, D ** -0.5)

        def t_tri():
            meta.scratch.clear()
            attention.paged_attention(q, kc, vc, meta, D ** -0.5)

        causal = 0.5 if q_len > 1 else 1.0
        flops = int(4 * B * q_len * ctx * NH * D * causal)
        nbytes = (2 * B * ctx * NKV * D + 2 * B * q_len * NH * D) * es
        name = "paged_decode_attention" if q_len == 1 else "paged_prefill_attention"
        out.append(row(name, f"B={B} q={q_len} ctx={ctx}", timed(t_ops, device), timed(t_tri, device), flops, nbytes))

    # quantized matmuls on the two biggest weights
    for (N, K, label) in ((2 * I, H, "gate_up"), (H, I, "down")):
        w = rnd(N, K)
        for bits in (8, 4):
            qw = quantize_weight(w, bits, 128)
            for M in (1, 16, 64):
                x = rnd(M, K)
                t_fp16 = timed(lambda: torch.nn.functional.linear(x, w), device)
                t_deq = timed(lambda: ops.linear(x, qw), device)
                t_tri = timed(lambda: qmatmul.linear(x, qw), device)
                r = row(f"int{bits}_matmul", f"{label} M={M}", t_deq, t_tri, 2 * M * N * K, qw.nbytes() + (M * K + M * N) * es)
                r["fp16_cublas_us"] = round(t_fp16 * 1e6, 2)
                r["speedup_vs_fp16"] = round(t_fp16 / t_tri, 2)
                out.append(r)
    return out


def ablation(model_path, device, dtype, requests, max_tokens, batch):
    """Same engine, same prompts, kernels switched on cumulatively."""
    from engine.llm_engine import LLMEngine
    from engine.scheduler import SamplingParams

    engine = LLMEngine.from_pretrained(model_path, device=device, dtype=dtype, kernels=(), max_num_seqs=batch)
    prompts = topic_prompts(engine.tokenizer, requests)
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
    steps = [(), ("attention",), ("attention", "rope"), ("attention", "rope", "norm"), ("attention", "rope", "norm", "act")]
    rows = []
    for ks in steps:
        engine.model.set_kernels(ks)
        engine.generate(prompts[:2], SamplingParams(max_tokens=4, temperature=0.0))  # warm up / compile
        t = time.perf_counter()
        seqs = engine.generate(prompts, params)
        wall = time.perf_counter() - t
        tok = sum(len(s.output_ids) for s in seqs)
        rows.append({"kernels": "+".join(ks) or "pytorch", "throughput_tok_s": round(tok / wall, 2), "wall_s": round(wall, 2)})
        print(f"  {rows[-1]['kernels']:<28} {rows[-1]['throughput_tok_s']:>8.1f} tok/s")
    base = rows[0]["throughput_tok_s"]
    for r in rows:
        r["speedup"] = round(r["throughput_tok_s"] / base, 2)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = pick_device()
    if not triton_usable(device):
        raise SystemExit("bench_kernels needs a CUDA GPU with Triton")
    dtype = default_dtype(device)
    print(f"device={device} {torch.cuda.get_device_name()} dtype={dtype} kernels={KERNELS}")
    rows = bench_ops(device, dtype)
    for r in rows:
        print(f"  {r['kernel']:<24} {r['shape']:<22} torch {r['torch_us']:>9.1f} us   triton {r['triton_us']:>9.1f} us   "
              f"{r['speedup']:>6.2f}x   {r['triton_gbs']:>7.1f} GB/s")
    abl = None if args.skip_ablation else ablation(args.model, device, dtype, args.requests, args.max_tokens, args.batch)
    save("kernels", device, {"dtype": str(dtype).split(".")[-1], "ops": rows, "ablation": abl,
                             "ablation_setup": {"requests": args.requests, "max_tokens": args.max_tokens, "batch": args.batch}})


if __name__ == "__main__":
    main()
