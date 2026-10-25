"""Token counting. Uses tiktoken when it is installed, otherwise a chars/4 estimate.

Claude and GPT tokenizers differ, but both land near 4 characters per token for English and
JSON, so the estimate is good enough to measure savings and enforce budgets.
"""
from __future__ import annotations

try:  # optional
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa: BLE001
    _ENC = None


def count(text: str) -> int:
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text, disallowed_special=()))
    return max(1, (len(text) + 3) // 4)
