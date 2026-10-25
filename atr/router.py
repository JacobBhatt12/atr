"""Just-in-time tools: rank tool definitions against a task and slim their schemas.

Sending 50 full tool schemas on every turn can cost thousands of tokens before the user has
typed anything. ATR exposes a handful of meta-tools instead; the model calls
atr_find_tools("open a github issue") and gets back only the few signatures that match.

Ranking is BM25 over each tool's name, description and parameter names: no model, no
download, microseconds per query. Good enough for tool names, which are short and literal.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

WORD = re.compile(r"[a-z0-9]+")
STOP = {"a", "an", "the", "to", "of", "for", "and", "or", "in", "on", "with", "by", "is", "it", "this",
        "that", "be", "from", "as", "at", "use", "using", "tool", "returns", "return", "given", "my", "me", "i"}
DROP_SCHEMA_KEYS = {"$schema", "title", "examples", "example", "additionalProperties", "$id", "$comment",
                    "markdownDescription", "deprecated", "readOnly", "writeOnly"}


def words(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # camelCase -> camel Case
    out = []
    for w in WORD.findall(text.lower()):
        if w in STOP:
            continue
        out.append(w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w)  # crude plural
    return out


def first_sentence(text: str | None, limit: int = 120) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    s = m.group(1) if m else text
    return s if len(s) <= limit else s[: limit - 1] + "…"


def slim_schema(schema: Any, desc_limit: int = 80) -> Any:
    """Same validation meaning, fewer tokens: drop titles/examples, shorten descriptions."""
    if isinstance(schema, list):
        return [slim_schema(x, desc_limit) for x in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k in DROP_SCHEMA_KEYS:
            continue
        if k == "description" and isinstance(v, str):
            v = first_sentence(v, desc_limit)
            if not v:
                continue
        elif k == "properties" and isinstance(v, dict):
            v = {name: slim_schema(p, desc_limit) for name, p in v.items()}
        else:
            v = slim_schema(v, desc_limit)
        out[k] = v
    return out


def _type(p: dict) -> str:
    t = p.get("type")
    if isinstance(t, list):
        t = "|".join(x for x in t if x != "null")
    if t == "array":
        inner = p.get("items", {})
        return f"{_type(inner) if isinstance(inner, dict) else 'any'}[]"
    if "enum" in p:
        vals = p["enum"][:6]
        return "|".join(map(str, vals)) + ("|…" if len(p["enum"]) > 6 else "")
    if not t and ("anyOf" in p or "oneOf" in p):
        opts = p.get("anyOf") or p.get("oneOf")
        return "|".join(sorted({_type(x) for x in opts if isinstance(x, dict) and x.get("type") != "null"})) or "any"
    return {"string": "str", "integer": "int", "number": "num", "boolean": "bool", "object": "obj"}.get(t, t or "any")


def signature(name: str, tool: dict) -> str:
    """One line per tool: name(req*:type, opt:type) — first sentence of the description."""
    schema = tool.get("inputSchema") or {}
    props = schema.get("properties") or {}
    req = set(schema.get("required") or [])
    params = [f"{p}{'*' if p in req else ''}:{_type(s) if isinstance(s, dict) else 'any'}" for p, s in props.items()]
    desc = first_sentence(tool.get("description"))
    return f"{name}({', '.join(params)})" + (f" — {desc}" if desc else "")


VERBS = {"get", "list", "create", "update", "delete", "remove", "search", "find", "add", "read", "write", "send",
         "set", "run", "fetch", "query", "edit", "move", "insert", "export", "import", "describe", "show", "make",
         "new", "open", "close", "start", "stop", "check", "ack", "tail", "merge", "rollback", "restart", "take",
         "do", "all", "one", "info", "by", "with", "page", "upload", "download", "post", "put", "patch"}


def topics(tool_names: list[str], n: int = 8) -> list[str]:
    """What a server's tools are about, from their names: github_create_issue -> github, issue."""
    counts: Counter = Counter()
    for name in tool_names:
        seen = set()
        for w in re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower()):
            if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
                w = w[:-1]  # files -> file, issues -> issue
            if w not in VERBS and len(w) > 1 and not w.isdigit() and w not in seen:
                seen.add(w)
                counts[w] += 1
    # most common first; ties alphabetical, so the text is identical on every start (prompt caching)
    return [w for w, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:n]]


class ToolIndex:
    def __init__(self, tools: dict[str, dict]):
        """tools: exposed name -> MCP tool definition."""
        self.tools = tools
        self.docs: dict[str, Counter] = {}
        for name, t in tools.items():
            props = (t.get("inputSchema") or {}).get("properties") or {}
            # the name counts three times: it is the most reliable signal
            text = " ".join([name] * 3 + [t.get("description") or ""] + list(props))
            self.docs[name] = Counter(words(text))
        self.avg = sum(sum(c.values()) for c in self.docs.values()) / max(1, len(self.docs))
        df: Counter = Counter()
        for c in self.docs.values():
            df.update(set(c))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        q = words(query)
        k1, b = 1.5, 0.75
        scores = []
        for name, c in self.docs.items():
            dl = sum(c.values())
            s = 0.0
            for w in q:
                f = c.get(w, 0)
                if not f:
                    # partial credit for prefixes ("issue" vs "issues", "repo" vs "repository")
                    f = 0.5 * sum(v for t, v in c.items() if len(w) >= 4 and (t.startswith(w) or w.startswith(t)) and len(t) >= 4)
                    if not f:
                        continue
                    idf = max((self.idf[t] for t in c if (t.startswith(w) or w.startswith(t)) and len(t) >= 4), default=0)
                else:
                    idf = self.idf.get(w, 0)
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / max(1e-9, self.avg)))
            if s > 0:
                scores.append((name, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]
