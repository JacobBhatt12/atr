"""Month 1 baseline engine.

Runs a Hugging Face model with OUR OWN decode loop and sampler, one request at a time.
This is deliberately simple: it is the "before" number every later optimization is measured against.
"""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.sampling import sample

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class NaiveEngine:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None):
        self.model_id = model_id
        self.device = device or pick_device()
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(self.device).eval()
        self.stop_ids = {self.tokenizer.eos_token_id}
        gen_eos = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(gen_eos, int):
            self.stop_ids.add(gen_eos)
        elif isinstance(gen_eos, list):
            self.stop_ids.update(gen_eos)
        self.last_stats: dict = {}

    @torch.inference_mode()
    def stream(self, messages, max_new_tokens=256, temperature=0.7, top_p=0.9, top_k=50):
        """Yield text pieces as they are generated. Stats land in self.last_stats."""
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        start = time.perf_counter()
        first_token_at = None
        past = None
        next_input = input_ids
        generated: list[int] = []
        sent_text = ""

        for _ in range(max_new_tokens):
            out = self.model(input_ids=next_input, past_key_values=past, use_cache=True)
            past = out.past_key_values  # the KV cache: Month 2 replaces this with paged blocks
            token = sample(out.logits[:, -1, :], temperature, top_k, top_p)
            token_id = token.item()
            if first_token_at is None:
                first_token_at = time.perf_counter()
            if token_id in self.stop_ids:
                break
            generated.append(token_id)
            next_input = token.view(1, 1)

            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if text.endswith("�"):  # half of a multi-byte character, wait for the rest
                continue
            delta, sent_text = text[len(sent_text):], text
            if delta:
                yield delta

        end = time.perf_counter()
        n = len(generated)
        self.last_stats = {
            "prompt_tokens": input_ids.shape[1],
            "completion_tokens": n,
            "ttft_s": (first_token_at or end) - start,
            "total_s": end - start,
            "tokens_per_s": n / (end - (first_token_at or start)) if n > 1 else 0.0,
        }
