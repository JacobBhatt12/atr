"""Deployment specs: what the user asks for, validated and filled with defaults.

An app spec (from vrtx.yaml or CLI flags):

    name: todo-api
    type: app
    replicas: 2
    port: 8000           # the port the app listens on inside its container ($PORT is set too)
    health: /health      # any HTTP answer below 500 counts as healthy
    run: uvicorn main:app --host 0.0.0.0 --port $PORT   # optional, detected otherwise
    env: {LOG_LEVEL: info}

A model spec:

    name: qwen-chat
    type: model
    model: Qwen/Qwen2.5-0.5B-Instruct    # Hugging Face id or a local folder
    quantization: int8
    speculative: {draft: Qwen/Qwen2.5-0.5B-Instruct}
    resources: {gpu_memory_gb: 4}
"""
from __future__ import annotations

from pathlib import Path

from cloud.common import valid_name

APP_DEFAULTS = {"port": 8000, "health": "/", "health_timeout_s": 120}
MODEL_DEFAULTS = {"port": 8000, "health": "/health", "health_timeout_s": 900}
QUANT_BITS = {None: 0, "none": 0, "fp16": 0, "int8": 8, "int4": 4}
KNOWN = {
    "name", "type", "replicas", "port", "health", "health_timeout_s", "run", "env", "resources",
    "model", "quantization", "speculative", "kv_cache_mb", "max_seqs", "scaling",
}


class SpecError(ValueError):
    pass


def normalize(raw: dict) -> dict:
    """Validate a spec and fill in defaults. Raises SpecError with a readable message."""
    spec = {k: v for k, v in raw.items() if v is not None}
    unknown = set(spec) - KNOWN
    if unknown:
        raise SpecError(f"unknown field(s): {', '.join(sorted(unknown))}")
    name = spec.get("name")
    if not name or not valid_name(name):
        raise SpecError(f"name {name!r} must be lowercase letters, digits and dashes (it becomes name.vrtx.local)")
    kind = spec.setdefault("type", "app")
    if kind not in ("app", "model"):
        raise SpecError("type must be 'app' or 'model'")
    for key, value in (MODEL_DEFAULTS if kind == "model" else APP_DEFAULTS).items():
        spec.setdefault(key, value)

    replicas = spec.setdefault("replicas", 1)
    if not isinstance(replicas, int) or not 0 <= replicas <= 64:
        raise SpecError("replicas must be an integer from 0 to 64")
    if not spec["health"].startswith("/"):
        raise SpecError("health must be a path like /health")
    spec["env"] = {str(k): str(v) for k, v in (spec.get("env") or {}).items()}

    res = dict(spec.get("resources") or {})
    spec["resources"] = {
        "gpu_memory_gb": float(res.get("gpu_memory_gb", 0)),
        "cpu": float(res.get("cpu", 0)),          # 0 = no limit
        "memory_mb": int(res.get("memory_mb", 0)),  # 0 = no limit
    }

    if kind == "model":
        if not spec.get("model"):
            raise SpecError("a model deployment needs model: <Hugging Face id or local folder>")
        quant = spec.get("quantization")
        if quant not in QUANT_BITS:
            raise SpecError("quantization must be int8, int4 or none")
        local = Path(spec["model"]).expanduser()
        if local.is_dir():
            spec["model"] = str(local.resolve())
    return spec


def engine_env(spec: dict) -> dict:
    """Environment variables that configure gateway/server.py for a model deployment."""
    env = {"VRTX_MODEL": spec["model"]}
    bits = QUANT_BITS[spec.get("quantization")]
    if bits:
        env["VRTX_QUANT"] = str(bits)
    draft = (spec.get("speculative") or {}).get("draft")
    if draft:
        env["VRTX_DRAFT_MODEL"] = draft
    if spec.get("kv_cache_mb"):
        env["VRTX_KV_CACHE_MB"] = str(spec["kv_cache_mb"])
    if spec.get("max_seqs"):
        env["VRTX_MAX_SEQS"] = str(spec["max_seqs"])
    return env | spec["env"]
