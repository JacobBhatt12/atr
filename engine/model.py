"""VRTX's own transformer forward pass (Llama / Qwen2 family).

We only borrow the WEIGHTS from Hugging Face. Every operation below (RMSNorm, RoPE,
attention over the paged KV cache, the MLP) is ours, which is what lets us batch
many sequences together and, in Month 3, swap pieces for Triton kernels.

Shapes use these letters:
    T  = total tokens in this step, across ALL sequences (a "ragged" flat batch)
    nh = query heads, nkv = key/value heads, d = head dim
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


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


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class VRTXModel:
    def __init__(self, cfg: ModelConfig, weights, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.cfg, self.device, self.dtype = cfg, device, dtype

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

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str = "cpu", dtype: torch.dtype = torch.float32) -> "VRTXModel":
        from transformers import AutoConfig

        cfg = ModelConfig.from_hf(AutoConfig.from_pretrained(str(path)))
        return cls(cfg, SafetensorsWeights(path), device, dtype)

    # ------------------------------------------------------------------ RoPE
    def _rope(self, positions: torch.Tensor):
        freqs = positions.float()[:, None] * self.inv_freq[None, :]  # (T, d/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, d)
        return emb.cos().to(self.dtype)[:, None, :], emb.sin().to(self.dtype)[:, None, :]

    # --------------------------------------------------------------- forward
    @torch.inference_mode()
    def forward(self, input_ids, positions, seq_meta, write_slots, logits_rows, kv_cache) -> torch.Tensor:
        """Run one engine step.

        input_ids, positions : (T,)  every new token from every scheduled sequence, flattened
        seq_meta             : list of (q_start, q_len, ctx_len, ctx_slots) per sequence
                               ctx_slots = cache slot of every position 0..ctx_len-1
        write_slots          : (T,)  where each new token's K and V get stored
        logits_rows          : rows of the flat batch that need next-token logits
        Returns logits (len(logits_rows), vocab).
        """
        cfg = self.cfg
        nh, nkv, d = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        T = input_ids.shape[0]
        cos, sin = self._rope(positions)
        x = self.embed[input_ids]  # (T, H)

        for li, L in enumerate(self.layers):
            # ---- attention block
            h = rms_norm(x, L["ln1"], cfg.rms_norm_eps)
            qkv = F.linear(h, L["wqkv"], L["bqkv"])  # weights read ONCE for all T tokens
            q, k, v = qkv.split([nh * d, nkv * d, nkv * d], dim=-1)
            q = q.view(T, nh, d)
            k = k.view(T, nkv, d)
            v = v.view(T, nkv, d)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin

            k_cache, v_cache = kv_cache.k[li], kv_cache.v[li]  # (num_slots, nkv, d)
            k_cache[write_slots] = k
            v_cache[write_slots] = v

            attn_out = torch.empty(T, nh, d, dtype=x.dtype, device=x.device)
            for q_start, q_len, ctx_len, ctx_slots in seq_meta:
                # Gather this sequence's K/V from its scattered pages.
                # Month 3 replaces this Python loop with one Triton paged-attention kernel.
                K = k_cache[ctx_slots]
                V = v_cache[ctx_slots]
                if nkv != nh:  # grouped-query attention: share each KV head across nh/nkv query heads
                    K = K.repeat_interleave(nh // nkv, dim=1)
                    V = V.repeat_interleave(nh // nkv, dim=1)
                Q = q[q_start:q_start + q_len]
                mask = None
                if q_len > 1:  # causal: query at absolute pos (ctx_len - q_len + i) sees keys <= that pos
                    mask = torch.ones(q_len, ctx_len, dtype=torch.bool, device=x.device).tril(ctx_len - q_len)
                o = F.scaled_dot_product_attention(
                    Q.transpose(0, 1), K.transpose(0, 1), V.transpose(0, 1), attn_mask=mask
                )
                attn_out[q_start:q_start + q_len] = o.transpose(0, 1)

            x = x + F.linear(attn_out.reshape(T, nh * d), L["wo"])

            # ---- MLP block (SwiGLU)
            h = rms_norm(x, L["ln2"], cfg.rms_norm_eps)
            gate, up = F.linear(h, L["w_gate_up"]).chunk(2, dim=-1)
            x = x + F.linear(F.silu(gate) * up, L["w_down"])

        # Only compute logits where we actually sample (saves a huge vocab matmul per prompt token).
        x = rms_norm(x[logits_rows], self.norm, cfg.rms_norm_eps)
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
