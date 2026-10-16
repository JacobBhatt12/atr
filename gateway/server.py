"""VRTX Gateway: OpenAI-compatible API on top of VRTX Engine.

Month 2: requests no longer wait in line. Every request joins the engine's
continuous batch, so many users stream at the same time.

Run:   uvicorn gateway.server:app --port 8000
Env:   VRTX_MODEL       local folder or Hugging Face id (default: models/qwen2.5-0.5b if present)
       VRTX_KV_CACHE_MB KV cache budget in MB (default 256)
       VRTX_MAX_SEQS    max sequences per batch (default 32)
"""
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from engine.async_engine import AsyncLLMEngine
from engine.llm_engine import LLMEngine
from engine.naive_engine import DEFAULT_MODEL, pick_device
from engine.scheduler import SamplingParams

LOCAL_MODEL = Path(__file__).resolve().parent.parent / "models" / "qwen2.5-0.5b"
MODEL = os.getenv("VRTX_MODEL") or (str(LOCAL_MODEL) if LOCAL_MODEL.is_dir() else DEFAULT_MODEL)
MODEL_NAME = Path(MODEL).name if Path(MODEL).is_dir() else MODEL

engine: LLMEngine | None = None
aengine: AsyncLLMEngine | None = None
started_at = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, aengine
    device = pick_device()
    engine = LLMEngine.from_pretrained(
        MODEL,
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        kv_cache_mb=int(os.getenv("VRTX_KV_CACHE_MB", "256")),
        max_num_seqs=int(os.getenv("VRTX_MAX_SEQS", "32")),
    )
    aengine = AsyncLLMEngine(engine)
    yield


app = FastAPI(title="VRTX Gateway", version="0.2.0", lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stream: bool = False


class IncrementalDetokenizer:
    """Turns token ids into text pieces without splitting multi-byte characters."""

    def __init__(self, tokenizer):
        self.tok, self.ids, self.sent = tokenizer, [], ""

    def add(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        if text.endswith("�"):
            return ""
        piece, self.sent = text[len(self.sent):], text
        return piece


def _chunk(rid: str, created: int, delta: dict, finish: str | None = None) -> str:
    body = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body)}\n\n"


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": engine.model.device, "engine": engine.stats()}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "vrtx"}]}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = [f"vrtx_{k} {v}" for k, v in engine.stats().items()]
    lines.append(f"vrtx_uptime_seconds {time.time() - started_at:.0f}")
    return "\n".join(lines) + "\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    tok = engine.tokenizer
    prompt = tok.apply_chat_template([m.model_dump() for m in req.messages], tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt).input_ids
    params = SamplingParams(max_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p, top_k=req.top_k)
    rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if req.stream:
        async def events():
            detok = IncrementalDetokenizer(tok)
            yield _chunk(rid, created, {"role": "assistant"})
            async for token_id in aengine.generate(prompt_ids, params):
                piece = detok.add(token_id)
                if piece:
                    yield _chunk(rid, created, {"content": piece})
            yield _chunk(rid, created, {}, finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    start = time.perf_counter()
    first = None
    out_ids = []
    try:
        async for token_id in aengine.generate(prompt_ids, params):
            first = first or time.perf_counter()
            out_ids.append(token_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    end = time.perf_counter()
    finish = "length" if len(out_ids) >= req.max_tokens else "stop"
    return {
        "id": rid,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": tok.decode(out_ids, skip_special_tokens=True)}, "finish_reason": finish}],
        "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(out_ids), "total_tokens": len(prompt_ids) + len(out_ids)},
        "vrtx": {
            "ttft_s": round((first or end) - start, 4),
            "tokens_per_s": round(len(out_ids) / (end - first), 2) if first and len(out_ids) > 1 else 0.0,
        },
    }
