"""Keeps full tool results on the server so the model only ever receives a summary.

When a result is cut down, the original is stored under a short ref (r1, r2, ...). The model
can then pull exactly what it needs with atr_query instead of re-running the tool or having
the whole thing pasted into its context:

    atr_query(ref="r3", path="items", where=["state=open", "comments>5"], fields=["id","title"])
    atr_query(ref="r4", search="ERROR|Traceback")          # text: matching lines only
    atr_query(ref="r3", path="items", group_by="state")    # counts per value
"""
from __future__ import annotations

import re
from collections import Counter, OrderedDict
from typing import Any

from atr.compress import parse

OPS = ["!=", ">=", "<=", "~", "=", ">", "<"]  # longest first so ">=" is not read as ">"


class Stash:
    def __init__(self, max_items: int = 64):
        self.items: OrderedDict[str, tuple[str, Any]] = OrderedDict()
        self.max_items = max_items
        self._n = 0

    def put(self, label: str, raw: Any) -> str:
        self._n += 1
        ref = f"r{self._n}"
        self.items[ref] = (label, raw)
        while len(self.items) > self.max_items:
            self.items.popitem(last=False)
        return ref

    def get(self, ref: str) -> tuple[str, Any]:
        if ref not in self.items:
            known = ", ".join(self.items) or "none"
            raise KeyError(f"unknown ref {ref!r} (known: {known})")
        return self.items[ref]


def walk(v: Any, path: str | None) -> Any:
    """Dotted path into data: "data.items", "items.0.user", "items.*.user.login"."""
    if not path:
        return v
    head, _, rest = path.partition(".")
    if head == "*" and isinstance(v, list):
        return [walk(x, rest) for x in v]
    if isinstance(v, list):
        if head.lstrip("-").isdigit():
            return walk(v[int(head)], rest)
        return [walk(x, path) for x in v if isinstance(x, dict)]  # implicit map over rows
    if isinstance(v, dict):
        if head not in v:
            raise KeyError(f"no key {head!r} here (keys: {', '.join(list(v)[:30])})")
        return walk(v[head], rest)
    raise KeyError(f"cannot go into {type(v).__name__} with {head!r}")


def _get(row: Any, key: str) -> Any:
    try:
        return walk(row, key)
    except (KeyError, IndexError, TypeError):
        return None


def _num(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def match(row: Any, cond: str) -> bool:
    for op in OPS:
        if op in cond:
            key, want = (s.strip() for s in cond.split(op, 1))
            break
    else:
        raise ValueError(f"bad condition {cond!r}; use key=value, key!=value, key>n, key<n, key~regex")
    got = _get(row, key)
    if op == "~":
        return got is not None and re.search(want, str(got), re.I) is not None
    if op in ("=", "!="):
        eq = str(got).lower() == want.lower()
        return eq if op == "=" else not eq
    a, b = _num(got), _num(want)
    if a is None or b is None:
        return False
    return {">": a > b, "<": a < b, ">=": a >= b, "<=": a <= b}[op]


def query(raw: Any, path: str | None = None, where: list[str] | None = None, fields: list[str] | None = None,
          search: str | None = None, group_by: str | None = None, offset: int = 0, limit: int = 20) -> Any:
    value, is_data = parse(raw)
    if not is_data:
        lines = str(value).splitlines()
        hits = [(i + 1, ln) for i, ln in enumerate(lines) if not search or re.search(search, ln, re.I)]
        page = hits[offset: offset + limit]
        return {"total_lines": len(lines), "matches": len(hits), "offset": offset,
                "lines": [f"{n}: {ln}" for n, ln in page]}
    v = walk(value, path)
    if isinstance(v, dict) and not (where or group_by or search):
        return {k: v[k] for k in fields if k in v} if fields else v
    rows = v if isinstance(v, list) else [v]
    for cond in where or []:
        rows = [r for r in rows if match(r, cond)]
    if search:
        rx = re.compile(search, re.I)
        rows = [r for r in rows if rx.search(str(r))]
    if group_by:
        counts = Counter(str(_get(r, group_by)) for r in rows)
        return {"total": len(rows), "group_by": group_by, "counts": dict(counts.most_common(limit))}
    page = rows[offset: offset + limit]
    if fields:
        page = [{f: _get(r, f) for f in fields} if isinstance(r, dict) else r for r in page]
    return {"total": len(rows), "offset": offset, "returned": len(page), "rows": page}
