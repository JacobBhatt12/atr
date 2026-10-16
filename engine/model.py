"""VRTX's own transformer forward pass (Llama / Qwen2 family).

We only borrow the WEIGHTS from Hugging Face. Every operation (RMSNorm, RoPE,
attention over the paged KV cache, the MLP) is ours, which is what lets us batch
many sequences together and swap each op for a Triton kernel (engine/kernels/) on a GPU.

Shapes use these letters:
    T  = total tokens in this step, across ALL sequences (a "ragged" flat batch)
    nh = query heads, nkv = key/value heads, d = head dim
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from engine import ops
from engine.batch import AttentionMetadata


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    qkv_bias: bool
    max_position_embeddings: int

    @classmethod
    def from_hf(cls, c) -> "ModelConfig":
        head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        rope = getattr(c, "rope_parameters", None) or getattr(c, "rope_scaling", None) or {}
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if rope_type not in (None, "default"):
            raise NotImplementedError(f"RoPE scaling type '{rope_type}' is not supported yet")
        theta = getattr(c, "rope_theta", None) or rope.get("rope_theta") or 10000.0
        return cls(
            vocab_size=c.vocab_size,
            hidden_size=c.hidden_size,
            intermediate_size=c.intermediate_size,
            num_layers=c.num_hidden_layers,
            num_heads=c.num_attention_heads,
            num_kv_heads=getattr(c, "num_key_value_heads", None) or c.num_attention_heads,
            head_dim=head_dim,
            rms_norm_eps=c.rms_norm_eps,
            rope_theta=float(theta),
            tie_word_embeddings=bool(getattr(c, "tie_word_embeddings", False)),
            qkv_bias=getattr(c, "model_type", "") == "qwen2" or bool(getattr(c, "attention_bias", False)),
            max_position_embeddings=getattr(c, "max_position_embeddings", 4096),
        )


class SafetensorsWeights:
    """Reads tensors lazily from *.safetensors files, one at a time, to keep peak RAM low."""

    def __init__(self, path: str | Path):
        from safetensors import safe_open

        self._safe_open = safe_open
        self.files: dict[str, Path] = {}
        for f in sorted(Path(path).glob("*.safetensors")):
            with safe_open(str(f), framework="pt") as h:
                for k in h.keys():
                    self.files[k] = f
        if not self.files:
            raise FileNotFoundError(f"no .safetensors files in {path}")

    def __contains__(self, key: str) -> bool:
        return key in self.files

    def __getitem__(self, key: str) -> torch.Tensor:
        with self._safe_open(str(self.files[key]), framework="pt") as h:
            return h.get_tensor(key)


def _resolve_ops(device, kernels):
    """Pick the PyTorch or the Triton version of each op.

    kernels: None = every Triton kernel when usable (CUDA), or an iterable of names from
    engine.kernels.KERNELS to choose exactly which ones (benchmarks turn them on one by one).
    """
    from engine.kernels import KERNELS, triton_usable

    if kernels is None:
        kernels = KERNELS if triton_usable(device) and os.environ.get("VRTX_TRITON", "1") != "0" else ()
    kernels = set(kernels)
    if unknown := kernels - set(KERNELS):
        raise ValueError(f"unknown kernels {unknown}; choose from {KERNELS}")
    if kernels and not triton_usable(device):
        raise RuntimeError("Triton kernels need a CUDA GPU (or TRITON_INTERPRET=1 for tests)")
    o = SimpleNamespace(**{n: getattr(ops, n) for n in
                           ("rms_norm", "add_rms_norm", "silu_and_mul", "rope_and_cache", "paged_attention", "linear")})
    if "norm" in kernels:
        from engine.kernels import norm
        o.rms_norm, o.add_rms_norm = norm.rms_norm, norm.add_rms_norm
    if "rope" in kernels:
        from engine.kernels import rope
        o.rope_and_cache = rope.rope_and_cache
    if "act" in kernels:
        from engine.kernels import activation
        o.silu_and_mul = activation.silu_and_mul
    if "attention" in kernels:
        from engine.kernels import attention
        o.paged_attention = attention.paged_attention
    if "qmatmul" in kernels:
        from engine.kernels import qmatmul
        o.linear = qmatmul.linear
    return o, frozenset(kernels)


LINEAR_WEIGHTS = ("wqkv", "wo", "w_gate_up", "w_down")


class VRTXModel:
    def __init__(self, cfg: ModelConfig, weights, device: str = "cpu", dtype: torch.dtype = torch.float32,
                 kernels=None):
        self.cfg, self.device, self.dtype = cfg, device, dtype
        self.quant_bits: int | None = None

        def get(key: str) -> torch.Tensor:
            return weights[key].to(device=device, dtype=dtype).contiguous()

        self.embed = get("model.embed_tokens.weight")
        self.layers = []
        for i in range(cfg.num_layers):
            p = f"model.layers.{i}."
            # Fusion: q, k and v become ONE matmul, gate and up become ONE matmul.
            layer = {
                "ln1": get(p + "input_layernorm.weight"),
                "ln2": get(p + "post_attention_layernorm.weight"),
                "wqkv": torch.cat([get(p + f"self_attn.{n}_proj.weight") for n in "qkv"], 0),
                "bqkv": torch.cat([get(p + f"self_attn.{n}_proj.bias") for n in "qkv"], 0) if cfg.qkv_bias else None,
                "wo": get(p + "self_attn.o_proj.weight"),
                "w_gate_up": torch.cat([get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight")], 0),
                "w_down": get(p + "mlp.down_proj.weight"),
            }
            self.layers.append(layer)
        self.norm = get("model.norm.weight")
        if cfg.tie_word_embeddings or "lm_head.weight" not in weights:
            self.lm_head = self.embed
        else:
            self.lm_head = get("lm_head.weight")

        d = cfg.head_dim
        self.inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=device) / d))
        self.scale = d ** -0.5
        self.set_kernels(kernels)

    def set_kernels(self, kernels=None) -> None:
        self.ops, self.kernels = _resolve_ops(self.device, kernels)

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str = "cpu", dtype: torch.dtype = torch.float32,
                        kernels=None) -> "VRTXModel":
        from transformers import AutoConfig

        cfg = ModelConfig.from_hf(AutoConfig.from_pretrained(str(path)))
        return cls(cfg, SafetensorsWeights(path), device, dtype, kernels)

    # ----------------------------------------------------------- quantize
    def quantize(self, bits: int, group_size: int = 128) -> None:
        """Replace every transformer-layer matmul weight with int8 or int4 codes + group scales.
        Embeddings, the LM head and the norms stay in the model dtype."""
        from engine.quant import pick_group_size, quantize_weight

        for L in self.layers:
            for name in LINEAR_WEIGHTS:
                w = L[name]
                L[name] = quantize_weight(w, bits, pick_group_size(w.shape[1], group_size))
                del w
        self.quant_bits = bits
        if torch.device(self.device).type == "cuda":
            torch.cuda.empty_cache()

    def weight_bytes(self) -> dict:
        def nbytes(t):
            return t.nbytes() if hasattr(t, "nbytes") and callable(t.nbytes) else t.numel() * t.element_size()

        linear = sum(nbytes(L[n]) for L in self.layers for n in LINEAR_WEIGHTS)
        other = self.embed.numel() * self.embed.element_size() + self.norm.numel() * self.norm.element_size()
        if self.lm_head is not self.embed:
            other += self.lm_head.numel() * self.lm_head.element_size()
        for L in self.layers:
            other += sum(nbytes(L[n]) for n in ("ln1", "ln2", "bqkv") if L[n] is not None)
        return {"linear_bytes": linear, "other_bytes": other, "total_bytes": linear + other}

    # --------------------------------------------------------------- forward
    @torch.inference_mode()
    def forward(self, input_ids, positions, meta: AttentionMetadata, logits_rows, kv_cache) -> torch.Tensor:
        """Run one engine step over a flat batch of T new tokens (see engine/batch.py).

        input_ids, positions : (T,)
        meta                 : which tokens belong to which sequence, and where their K/V live
        logits_rows          : rows of the flat batch that need next-token logits
        Returns logits (len(logits_rows), vocab).
        """
        cfg, o = self.cfg, self.ops
        nh, nkv, d, eps = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.rms_norm_eps
        T = input_ids.shape[0]
        x = self.embed[input_ids]  # (T, H)
        h = o.rms_norm(x, self.layers[0]["ln1"], eps)

        for li, L in enumerate(self.layers):
            # ---- attention block
            qkv = o.linear(h, L["wqkv"], L["bqkv"])  # weights read ONCE for all T tokens
            q, k, v = qkv.split([nh * d, nkv * d, nkv * d], dim=-1)
            q = o.rope_and_cache(q.view(T, nh, d), k.view(T, nkv, d), v.view(T, nkv, d), positions, self.inv_freq,
                                 kv_cache.k[li], kv_cache.v[li], meta.write_slots)
            attn = o.paged_attention(q, kv_cache.k[li], kv_cache.v[li], meta, self.scale)
            x, h = o.add_rms_norm(x, o.linear(attn.reshape(T, nh * d), L["wo"]), L["ln2"], eps)

            # ---- MLP block (SwiGLU)
            mlp = o.linear(o.silu_and_mul(o.linear(h, L["w_gate_up"])), L["w_down"])
            if li + 1 < len(self.layers):
                x, h = o.add_rms_norm(x, mlp, self.layers[li + 1]["ln1"], eps)
            else:
                x = x + mlp

        # Only compute logits where we actually sample (saves a huge vocab matmul per prompt token).
        x = o.rms_norm(x[logits_rows].contiguous(), self.norm, eps)
        return F.linear(x, self.lm_head).float()

    def num_params(self) -> int:
        n = self.embed.numel() + self.norm.numel()
        if self.lm_head is not self.embed:
            n += self.lm_head.numel()
        for L in self.layers:
            n += sum(t.numel() for t in L.values() if t is not None)
        return n


def read_generation_eos(path: str | Path) -> set[int]:
    f = Path(path) / "generation_config.json"
    if not f.exists():
        return set()
    eos = json.loads(f.read_text()).get("eos_token_id")
    if eos is None:
        return set()
    return {eos} if isinstance(eos, int) else set(eos)
