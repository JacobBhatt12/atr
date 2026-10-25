"""Turns bulky tool output into the smallest text that still answers the question.

What it does, cheapest and safest first:
  * parses JSON (and JSON lines) hiding inside text results
  * drops nulls, empty strings/lists/objects and known noise keys (_links, etag, node_id ...)
  * renders data as indented `key: value` lines instead of quoted, braced JSON
  * renders lists of similar objects as one CSV table; columns that are the same in every row
    are said once in the header instead of once per row
  * strips HTML to text, removes ANSI colors, collapses blank lines and repeated log lines
  * with a token budget: shortens long strings, samples big tables (first rows + count),
    keeps the head and tail of long text

Anything lossy is marked in the output, so the model knows there is more and can ask for it
(atr_query on the stashed full result).
"""
from __future__ import annotations

import csv
import html
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any

from atr import tokens

NOISE_KEYS = {"_links", "_meta", "etag", "node_id", "gravatar_id", "__typename", "_etag", "_rid", "_self"}
URL_KEY = re.compile(r".+_url$")
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
DIGITS = re.compile(r"\d+")
TAG = re.compile(r"<[^>]+>")
SCRIPT_STYLE = re.compile(r"<(script|style|noscript|svg)\b.*?</\1>", re.S | re.I)
BLOCK_TAG = re.compile(r"</?(p|div|br|li|tr|h[1-6]|section|article|header|footer|table|ul|ol)\b[^>]*>", re.I)


@dataclass
class Opts:
    level: int = 2
    max_str: int | None = 400     # longest string kept whole
    max_rows: int | None = None   # longest table / list kept whole
    drop_urls: bool = False


LEVELS = {
    1: Opts(level=1, max_str=None, max_rows=None),
    2: Opts(level=2, max_str=400, max_rows=None),
    3: Opts(level=3, max_str=160, max_rows=25, drop_urls=True),
}
# Tried in order when output is still over budget.
SQUEEZE = [Opts(3, 160, 25, True), Opts(3, 100, 10, True), Opts(3, 60, 5, True), Opts(3, 40, 3, True)]


@dataclass
class Result:
    text: str
    before: int           # tokens in
    after: int            # tokens out
    lossy: bool = False   # something was cut; the full data is worth stashing
    notes: list[str] = field(default_factory=list)

    @property
    def saved_pct(self) -> float:
        return 0.0 if not self.before else 100.0 * (1 - self.after / self.before)


# ---------------------------------------------------------------- parsing

def parse(raw: Any) -> tuple[Any, bool]:
    """(value, is_data). Text that is really JSON or JSON lines comes back as data."""
    if not isinstance(raw, str):
        return raw, True
    s = raw.strip()
    if s[:1] in "[{":
        try:
            return json.loads(s), True
        except ValueError:
            pass
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if len(lines) >= 2 and all(ln.lstrip()[:1] == "{" for ln in lines[:50]):
        try:
            return [json.loads(ln) for ln in lines], True
        except ValueError:
            pass
    return raw, False


# ---------------------------------------------------------------- data

def prune(v: Any, o: Opts) -> Any:
    """Drop empty values and noise keys, recursively. Returns None if nothing is left."""
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if k in NOISE_KEYS or (o.drop_urls and URL_KEY.match(str(k))):
                continue
            x = prune(x, o)
            if x is not None:
                out[k] = x
        return out or None
    if isinstance(v, list):
        out = [p for p in (prune(x, o) for x in v) if p is not None]
        return out or None
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    return v


def _cut(s: str, n: int | None) -> tuple[str, bool]:
    if n is None or len(s) <= n:
        return s, False
    return f"{s[:n]}…(+{len(s) - n} chars)", True


def _inline(v: Any) -> str:
    """Nested data on one line without JSON's quotes: {a:1 b:x} / [x; y]."""
    if isinstance(v, dict):
        return "{" + " ".join(f"{k}:{_inline(x)}" for k, x in v.items()) + "}"
    if isinstance(v, list):
        if len(v) == 1:
            return _inline(v[0])
        return "[" + "; ".join(_inline(x) for x in v) + "]"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _flatten(row: dict, prefix: str = "", depth: int = 0) -> dict:
    """{"user": {"login": "x"}} -> {"user.login": "x"} (two levels), so tables stay flat."""
    out = {}
    for k, v in row.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and depth < 2 and len(v) <= 12:
            out.update(_flatten(v, key + ".", depth + 1))
        else:
            out[key] = v
    return out


def _scalar(v: Any, o: Opts, st: dict) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    s = _inline(v) if isinstance(v, (dict, list)) else str(v)
    s, cut = _cut(s, o.max_str)
    st["lossy"] |= cut
    return s.replace("\r", "").replace("\n", "\\n")


def _is_scalar(v: Any) -> bool:
    return not isinstance(v, (dict, list))


def _table_cols(rows: list) -> list[str] | None:
    """Columns if `rows` reads well as a table: 2+ dicts with mostly the same keys."""
    if len(rows) < 2 or not all(isinstance(r, dict) for r in rows):
        return None
    cols: dict[str, None] = {}
    for r in rows:
        for k in r:
            cols.setdefault(k, None)
    avg = sum(len(r) for r in rows) / len(rows)
    if len(cols) > 40 or len(cols) > 1.6 * avg + 1:
        return None
    return list(cols)


def _csv_line(cells: list[str]) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(cells)
    return buf.getvalue()


def _table(rows: list[dict], cols: list[str], pad: str, o: Opts, st: dict) -> list[str]:
    head = []
    if o.level >= 2 and len(rows) >= 3:  # lift constant columns into the header
        const = [c for c in cols if all(c in r for r in rows) and _is_scalar(rows[0][c])
                 and all(r[c] == rows[0][c] for r in rows)]
        if const and len(const) < len(cols):
            head = [f"{c}={_scalar(rows[0][c], o, st)}" for c in const]
            cols = [c for c in cols if c not in const]
    shown = rows if o.max_rows is None else rows[: o.max_rows]
    title = f"{pad}[{len(rows)} rows" + (f"; all rows: {', '.join(head)}" if head else "") + "]"
    out = [title, pad + _csv_line(cols)]
    for r in shown:
        out.append(pad + _csv_line(["" if c not in r else _scalar(r[c], o, st) for c in cols]))
    if len(shown) < len(rows):
        st["lossy"] = True
        out.append(f"{pad}… +{len(rows) - len(shown)} more rows")
    return out


def _render(v: Any, depth: int, o: Opts, st: dict) -> list[str]:
    pad = "  " * depth
    if _is_scalar(v):
        return [pad + _scalar(v, o, st)]
    if isinstance(v, list):
        if o.level >= 2 and len(v) >= 2 and all(isinstance(r, dict) for r in v):
            v = [_flatten(r) for r in v]
        cols = _table_cols(v)
        if cols:
            return _table(v, cols, pad, o, st)
        items = v if o.max_rows is None else v[: o.max_rows]
        if all(_is_scalar(x) for x in items):
            line = pad + "[" + ", ".join(_scalar(x, o, st) for x in items)
            more = len(v) - len(items)
            if more:
                st["lossy"] = True
            return [line + (f", … +{more} more]" if more else "]")]
        out = []
        for x in items:
            sub = _render(x, depth + 1, o, st)
            out.append(pad + "- " + sub[0].lstrip())
            out.extend(sub[1:])
        if len(items) < len(v):
            st["lossy"] = True
            out.append(f"{pad}… +{len(v) - len(items)} more items")
        return out
    out = []
    for k, x in v.items():
        if _is_scalar(x):
            if isinstance(x, str) and "\n" in x and o.level >= 2:  # multi-line text as a block
                body, cut = _cut(x, None if o.max_str is None else o.max_str * 4)
                st["lossy"] |= cut
                out.append(f"{pad}{k}: |")
                out.extend(pad + "  " + ln for ln in clean_text(body, o).splitlines())
            else:
                out.append(f"{pad}{k}: {_scalar(x, o, st)}")
            continue
        sub = _render(x, depth + 1, o, st)
        if isinstance(x, list) and len(sub) == 1:
            out.append(f"{pad}{k}: {sub[0].lstrip()}")
        elif isinstance(x, list) and sub[0].lstrip().startswith("["):  # table title on the key line
            out.append(f"{pad}{k}: {sub[0].lstrip()}")
            out.extend(sub[1:])
        else:
            out.append(f"{pad}{k}:")
            out.extend(sub)
    return out


def render_data(v: Any, o: Opts) -> tuple[str, bool]:
    st = {"lossy": False}
    v = prune(v, o)
    if v is None:
        return "(empty)", False
    return "\n".join(_render(v, 0, o, st)), st["lossy"]


# ---------------------------------------------------------------- text

def html_to_text(s: str) -> str:
    s = SCRIPT_STYLE.sub(" ", s)
    s = BLOCK_TAG.sub("\n", s)
    s = TAG.sub("", s)
    return html.unescape(s)


def looks_like_html(s: str) -> bool:
    head = s[:2000].lower()
    return "<html" in head or "<!doctype html" in head or len(TAG.findall(head)) > 15


def _squash(ln: str) -> str:
    """Collapse runs of spaces inside a line; keep the indentation (code, YAML, tracebacks)."""
    body = ln.lstrip(" \t")
    indent = ln[: len(ln) - len(body)].replace("\t", "    ")
    return (indent + re.sub(r"[ \t]+", " ", body)).rstrip()


def clean_text(s: str, o: Opts) -> str:
    s = ANSI.sub("", s)
    if looks_like_html(s):
        s = html_to_text(s)
    lines = [_squash(ln.rsplit("\r", 1)[-1]) for ln in s.replace("\r\n", "\n").split("\n")]  # \r: progress bars
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            if out and out[-1] != "":
                out.append("")
            i += 1
            continue
        j = i + 1
        if o.level >= 2:  # same line apart from numbers (timestamps, ids, counters)
            shape = DIGITS.sub("#", ln)
            while j < len(lines) and lines[j].strip() and DIGITS.sub("#", lines[j]) == shape:
                j += 1
        else:
            while j < len(lines) and lines[j] == ln:
                j += 1
        n = j - i
        if n == 1:
            out.append(ln)
        elif all(lines[k] == ln for k in range(i, j)):
            out.append(f"{ln}  (×{n})")
        else:
            out.extend([ln, f"  … {n - 2} similar lines …", lines[j - 1]] if n > 2 else [ln, lines[j - 1]])
        i = j
    return "\n".join(out).strip()


def head_tail(s: str, budget: int) -> str:
    """Keep the start and the end (errors usually sit at the end) within `budget` tokens."""
    lines = s.splitlines()
    keep_chars = budget * 4
    head, tail, used = [], [], 0
    lo, hi = 0, len(lines) - 1
    turn = 0
    while lo <= hi:
        ln = lines[lo] if turn % 3 != 2 else lines[hi]  # 2 parts head, 1 part tail
        if used + len(ln) + 1 > keep_chars:
            break
        used += len(ln) + 1
        if turn % 3 != 2:
            head.append(ln)
            lo += 1
        else:
            tail.append(ln)
            hi -= 1
        turn += 1
    omitted = hi - lo + 1
    if omitted <= 0:
        return s
    if not head and not tail:  # one giant line
        return s[: keep_chars] + f"…(+{len(s) - keep_chars} chars)"
    return "\n".join(head + [f"… {omitted} lines omitted …"] + tail[::-1])


# ---------------------------------------------------------------- entry point

SNAPSHOT_BUDGET = 5000


def compress(raw: Any, budget: int | None = None, level: int = 2) -> Result:
    """Compress a tool result (str, or already-parsed JSON). `budget` is a soft token cap."""
    original = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, indent=2)
    before = tokens.count(original)
    value, is_data = parse(raw)
    notes: list[str] = []

    snap = None if is_data else compact_snapshot(value)

    def attempt(o: Opts) -> tuple[str, bool]:
        if is_data:
            return render_data(value, o)
        if snap is not None:
            return snap, False
        return clean_text(value, o), False

    o = LEVELS.get(level, LEVELS[2])
    text, lossy = attempt(o)
    if snap is not None and budget:
        # a compacted page is what the model works from; cutting it costs more turns than it saves
        budget = max(budget, SNAPSHOT_BUDGET)
    if budget and tokens.count(text) > budget:
        for o2 in SQUEEZE:
            if o2.level < o.level:
                continue
            text, lossy = attempt(o2)
            if tokens.count(text) <= budget:
                notes.append(f"squeezed (rows≤{o2.max_rows}, strings≤{o2.max_str})")
                break
        if tokens.count(text) > budget:
            text = head_tail(text, budget)
            lossy = True
            notes.append("cut to head and tail")

    if tokens.count(text) >= before and not lossy:  # never make things bigger
        text = original if not is_data else json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        if tokens.count(text) > before:
            text = original
    return Result(text=text, before=before, after=tokens.count(text), lossy=lossy, notes=notes)


# ---------------------------------------------------------------- command output (tests, builds, installs)

# Lines worth keeping wherever they are: failures, errors, tracebacks, file:line pointers, summaries.
IMPORTANT = re.compile(
    r"error|fail|exception|traceback|panic|fatal|assert|warning|\bwarn\b|denied|not found|cannot|can't|"
    r"unable to|undefined|unexpected|invalid|missing|refused|timed? ?out|segmentation|killed|abort|"
    r"\bnot ok\b|✗|✕|✘|×|^E\s|^>\s|^\s*File \"|\.\w{1,5}:\d+|^\s+at \S+ \(|^=+ .* =+$|^-+ .* -+$|"
    r"\b\d+ (passed|failed|errors?|skipped|warnings?|tests?|suites?)\b|^(Tests?|Test Suites|Ran \d+|OK\b|FAILED)",
    re.I)
# Lines that say nothing when things went well: passing tests, downloads, progress.
NOISE = re.compile(
    r"(PASSED|\bPASS\b|\bok\b|✓|✔)\s*(\[\s*\d+%\])?\s*$|^\s*(✓|✔|PASS\b)|^\s*(Downloading|Downloaded|Collecting|"
    r"Requirement already satisfied|Using cached|Installing collected|Obtaining|Preparing|Building wheel|"
    r"Created wheel|Stored in|Fetching|Resolving|Compiling|Checking|Fresh|Updating|Locking|Adding|Audited|"
    r"Progress|Reading|Unpacking|Selecting|Setting up|Processing|Get:\d|Hit:\d|Ign:\d|copying|creating|"
    r"running \w|\[\d+/\d+\]|remote:|Receiving objects|Resolving deltas|Counting objects)|^\s*[.sx]+\s*(\[\s*\d+%\])?$",
    re.I)


CARETS = re.compile(r"^[\s^~|]*\^[\s^~]*$")  # ^^^^ under a line: points at a column, says nothing new


def fold_blocks(text: str) -> str:
    """The same error block repeated (one per test file, one per package): keep the first, count the rest.
    A block starts at a line with no indent and runs until the next one (jest's FAIL x, cargo's error[E…])."""
    blocks: list[list[str]] = []
    for ln in text.splitlines():
        if CARETS.match(ln) and ln.strip():
            continue
        if not blocks or (ln[:1].strip() and blocks[-1] and any(x.strip() for x in blocks[-1])):
            blocks.append([])
        blocks[-1].append(ln)
    out: list[list[str]] = []
    first: dict[str, int] = {}
    repeats: dict[int, list[str]] = {}
    for blk in blocks:
        body = [ln for ln in blk if ln.strip()]
        shape = DIGITS.sub("#", "\n".join(body))
        if len(body) >= 3 and shape in first:
            repeats.setdefault(first[shape], []).append(body[0].strip()[:60])
            continue
        first.setdefault(shape, len(out))
        out.append(blk)
    for i, heads in repeats.items():
        names = ", ".join(heads[:3]) + (f", … {len(heads) - 3} more" if len(heads) > 3 else "")
        blk = out[i]
        end = len(blk)
        while end and not blk[end - 1].strip():
            end -= 1
        blk.insert(end, f"[atr: {len(heads)} more block(s) just like this one: {names}]")
    return "\n".join(ln for blk in out for ln in blk)


def compress_output(raw: str, budget: int | None = None, level: int = 2) -> Result:
    """Command output: when it is too long, keep what explains the result (failures, errors,
    tracebacks, warnings, the summary at the end) and drop passing tests and progress lines."""
    o = LEVELS.get(level, LEVELS[2])
    before = tokens.count(raw)
    text = fold_blocks(clean_text(raw, o))
    folded = "[atr: " in text and "[atr: " not in raw
    if not budget or tokens.count(text) <= budget:
        return Result(text=text, before=before, after=tokens.count(text), lossy=folded)
    lines = text.splitlines()
    n = len(lines)
    keep = [False] * n
    noise = [bool(NOISE.search(ln)) and not IMPORTANT.search(ln) for ln in lines]
    for i, ln in enumerate(lines):
        if not noise[i] and IMPORTANT.search(ln):
            for j in range(max(0, i - 2), min(n, i + 4)):  # a little context; tracebacks run downwards
                keep[j] = keep[j] or not noise[j]
    for i in list(range(min(3, n))) + list(range(max(0, n - 15), n)):  # how it started, how it ended
        keep[i] = keep[i] or not noise[i]
    out, gap, dropped = [], 0, 0
    for i, ln in enumerate(lines):
        if keep[i]:
            if gap:
                out.append(f"… {gap} lines …")
                gap = 0
            out.append(ln)
        else:
            gap += 1
            dropped += noise[i]
    if gap:
        out.append(f"… {gap} lines …")
    kept = "\n".join(out)
    notes = [f"kept {sum(keep)} of {n} lines (errors, failures, warnings, summary); {dropped} passing/progress lines dropped"]
    if tokens.count(kept) > budget:
        kept = head_tail(kept, budget)
        notes.append("cut to head and tail")
    kept = f"[atr: {'; '.join(notes)}]\n{kept}"
    return Result(text=kept, before=before, after=tokens.count(kept), lossy=True, notes=notes)


# ---------------------------------------------------------------- browser page snapshots (Playwright & co.)

ARIA_LINE = re.compile(r'^(?P<ind>\s*)- (?P<role>[\w/-]+)(?: "(?P<name>(?:[^"\\]|\\.)*)")?(?P<attrs>(?: \[[^\]]*\])*)'
                       r'(?P<colon>:)?(?: (?P<val>.*))?$')
# Containers that only say "a box is here": dropped when they carry no name or text; children move up.
LAYOUT_ROLES = {"generic", "group", "rowgroup", "row", "cell", "gridcell", "list", "listitem", "table", "none",
                "presentation", "region", "section", "paragraph", "separator", "LayoutTable", "LayoutTableRow",
                "LayoutTableCell", "Section", "contentinfo", "banner", "complementary", "main", "navigation",
                "article", "figure", "term", "definition", "strong", "emphasis", "Iframe", "iframe"}
JUNK_ATTRS = re.compile(r" \[(cursor=pointer|aria-hidden|box=[^\]]*)\]")


def _short_url(u: str) -> str:
    u = u.strip().strip('"')
    u = re.sub(r"^https?://(www\.)?", "", u)
    return u if len(u) <= 70 else u[:67] + "…"


def compact_snapshot(text: str) -> str | None:
    """An accessibility snapshot (Playwright MCP's YAML) with the layout boxes taken out: every visible
    text, every element you can act on (with its ref) and a short URL stay; empty wrappers, cursor
    hints and separators go. Returns None if `text` isn't a snapshot."""
    if "[ref=" not in text or not re.search(r"^\s*- \w+.*\[ref=\w+\]", text, re.M):
        return None
    out: list[str] = []
    stack: list[tuple[int, bool]] = []  # (original indent, kept?) of the open ancestors
    last_kept_at: dict[int, int] = {}   # depth -> index in out of the last kept line at that depth
    fenced = "```yaml" in text
    inside = not fenced
    for ln in text.splitlines():
        if fenced and ln.strip().startswith("```"):
            inside = ln.strip() != "```" or not inside
            out.append(ln)
            continue
        m = ARIA_LINE.match(ln) if inside else None
        if not m:
            out.append(ln)
            stack.clear()
            continue
        ind = len(m["ind"])
        while stack and stack[-1][0] >= ind:
            stack.pop()
        depth = sum(1 for _, k in stack if k)
        role, name, val = m["role"], m["name"], (m["val"] or "").strip()
        attrs = JUNK_ATTRS.sub("", m["attrs"] or "")
        if role == "/url":
            parent = last_kept_at.get(depth - 1)
            if parent is not None and "→" not in out[parent] and val and not val.startswith("javascript:"):
                out[parent] += f" → {_short_url(val)}"
            stack.append((ind, False))
            continue
        if role.startswith("/"):  # /placeholder and similar: fold into the parent line
            parent = last_kept_at.get(depth - 1)
            if parent is not None and val:
                out[parent] += f" ({role[1:]}: {val[:60]})"
            stack.append((ind, False))
            continue
        if "[aria-hidden" in (m["attrs"] or ""):
            stack.append((ind, False))
            continue
        if role == "text":
            v = val.strip('"').strip()
            if not v or not re.search(r"\w", v):  # "|", "(", ")" and other glue
                stack.append((ind, False))
                continue
            line = f"{' ' * depth}{v}"
        elif role in LAYOUT_ROLES and not name and not val:
            stack.append((ind, False))
            continue
        else:
            line = f"{' ' * depth}{role}" + (f' "{name}"' if name else "") + attrs + (f": {val}" if val else "")
        out.append(line)
        last_kept_at[depth] = len(out) - 1
        stack.append((ind, True))
    return "\n".join(out)
