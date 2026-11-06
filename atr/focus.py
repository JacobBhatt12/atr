"""Picks the parts of a long text that answer a question, word for word (atr_fetch question=).

    focus(page_text, "what BLEU score did the big model get?", budget=500)

Claude's built-in web fetch answers a question about a page with a summary from a second model.
This does the same job without a model: split the page into small sections, score each against
the question, and return the best ones as they are written, with their line numbers. Nothing is
paraphrased, so numbers, names and lists stay exact, and it costs nothing to run.

Sections follow the page's structure: a heading starts a new one, and lines are grouped up to
about SECTION_TOKENS so a list item or table row is never cut in half. Each section knows the
heading it sits under, which counts toward its score ("Installation" helps "install command").

Scoring is BM25 (the same as atr_find_tools), plus a bonus for words that appear next to each
other as in the question, and for numbers in the question. When the question asks for the top or
latest items, or many sections match about equally (a list of similar items), the start of the
page is returned instead, in page order.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from atr import tokens
from atr.router import words

SECTION_TOKENS = 110
HEADING = re.compile(r"^(#{1,6})\s+(.*\S)|^---\s*(page|slide|sheet)\s+(.+?)\s*---$", re.I)
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# "top 5 stories", "the latest posts", "list the first items": the answer is the start of the page
FROM_TOP = re.compile(r"\b(top|first|latest|newest|recent|front page|headlines?)\b", re.I)


@dataclass
class Section:
    start: int                  # first line (0-based)
    end: int                    # last line (inclusive)
    heading: str
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def sections(text: str) -> list[Section]:
    out: list[Section] = []
    heading = ""
    cur: Section | None = None
    used = 0
    for i, ln in enumerate(text.splitlines()):
        m = HEADING.match(ln.strip())
        if m:
            heading = (m.group(2) or f"{m.group(3)} {m.group(4)}").strip().strip("*")
            heading = re.sub(r"^Page:\s*", "", heading)
        cost = tokens.count(ln) + 1
        # a new section at each heading and at a blank line once the current one is big enough,
        # or anywhere once it is full
        # (a heading right after another heading, or after almost nothing, stays with it: a section
        # that is only a heading answers nothing)
        body = cur is not None and any(x.strip() and not HEADING.match(x.strip()) for x in cur.lines)
        new = (cur is None or (m is not None and body and used > 12) or used + cost > SECTION_TOKENS * 1.5
               or (not ln.strip() and used >= SECTION_TOKENS))
        if new:
            if cur and any(x.strip() for x in cur.lines):
                out.append(cur)
            cur, used = Section(i, i, heading), 0
        elif m is not None:
            cur.heading = heading  # the nearest heading names the section
        if not cur.lines and not ln.strip():
            cur.start = i + 1
            continue
        cur.lines.append(ln)
        cur.end = i
        used += cost
    if cur and any(x.strip() for x in cur.lines):
        out.append(cur)
    for s in out:  # trailing blank lines
        while s.lines and not s.lines[-1].strip():
            s.lines.pop()
            s.end -= 1
    return out


def _link_text(s: str) -> str:
    """Score on what people read, not on URLs: [text](url) -> text."""
    return re.sub(r"\]\([^)]*\)", "]", s)


def rank(question: str, secs: list[Section]) -> list[tuple[float, int]]:
    """(score, index) for every section, best first; ties go to the earlier section."""
    q = words(question)
    if not q or not secs:
        return [(0.0, i) for i in range(len(secs))]
    docs = [words(_link_text(s.text)) + words(s.heading) for s in secs]
    n = len(docs)
    avg = sum(map(len, docs)) / n or 1.0
    df = Counter(w for d in docs for w in set(d))
    qn = {x for x in NUMBER.findall(question) if len(x) > 1}  # "2014", "3.5"; not the 5 in "top 5"
    pairs = set(zip(q, q[1:]))
    scored = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        score = 0.0
        for w in set(q):
            if not tf[w]:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            # a very short section ("</details>", a lone heading) doesn't get the short-text boost
            score += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * max(len(d), avg / 2) / avg))
        if score:
            score += 0.6 * sum(1 for p in zip(d, d[1:]) if p in pairs)
            score += 0.8 * len(qn & set(NUMBER.findall(secs[i].text)))
        scored.append((score, i))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored


CITATION = re.compile(r"<sup>.*?</sup>|\[\\?\[[^\]]{1,12}\\?\]\]\([^)]*\)|\[\\?\[[^\]]{1,12}\\?\]\]")
EDIT_LINK = re.compile(r"\\?\[\s*\[edit\]\([^)]*\)\s*\\?\]|\[edit\]\([^)]*\)", re.I)
MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)\)")


def tidy(text: str, url: str = "") -> str:
    """Fewer tokens, same facts: links to the same site become plain text (the model can open the
    page again without a question to get them), footnote markers and [edit] links go."""
    from urllib.parse import urlsplit

    host = urlsplit(url).netloc

    def link(m: re.Match) -> str:
        label, href = m.group(1), m.group(2)
        if href.startswith(("/", "#")) or (host and urlsplit(href).netloc == host):
            return label
        return m.group(0)

    text = EDIT_LINK.sub("", text)
    text = CITATION.sub("", text)
    text = MD_LINK.sub(link, text)
    text = re.sub(r"^\s*\\?\[\s*\\?\]\s*$", "", text, flags=re.M)  # leftover "\[ \]" lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r"[ \t]+\n", "\n", text)


def sections_focus(text: str, question: str, budget: int = 500, url: str = "") -> tuple[str, int, int]:
    """Whole sections that best answer `question`, in page order, within `budget` tokens. Used for
    "top/latest" questions (the start of the page) and when no sentence matches.

    Returns (text, sections shown, sections in total). Each part starts with its heading and
    line numbers so the model can cite it or read around it with offset=.
    """
    secs = sections(text)
    if not secs:
        return "", 0, 0
    ranked = rank(question, secs)
    best = ranked[0][0]
    hits = sorted((sc for sc, _ in ranked if sc > 0), reverse=True)
    flat = len(hits) >= 6 and hits[0] < 1.3 * hits[len(hits) // 2]  # many sections match about equally
    if FROM_TOP.search(question) or flat or best <= 0:
        ranked = [(1.0, i) for i in range(len(secs))]  # page order
        best = 1.0
    keep: list[int] = []
    used = 0
    for score, i in ranked:
        if keep and (score <= 0 or score < best * 0.25):
            break
        cost = tokens.count(tidy(secs[i].text, url)) + 8
        if keep and used + cost > budget:
            continue
        keep.append(i)
        used += cost
        if used >= budget:
            break
    keep.sort()
    parts, last_heading, prev = [], None, None
    for i in keep:
        s = secs[i]
        label = f"lines {s.start + 1}-{s.end + 1}"
        if s.heading and s.heading != last_heading:
            label = f"{s.heading} · {label}"
        if prev is not None and i != prev + 1:
            parts.append("…")
        parts.append(f"[{label}]\n{tidy(s.text, url).strip()}")
        last_heading, prev = s.heading, i
    return "\n".join(parts), len(keep), len(secs)


# ---------------------------------------------------------------- snippets (the default)

SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(\"'0-9])")
# question words that say nothing about the answer ("how many", "what is", "does it have")
QSTOP = set("what which who whom whose when where why how many much does do did is are was were has have had be been "
            "the a an of to in on for at by with from it its this that these those there their they them use used "
            "using can could would should will tell me give list name please about".split())
WANTS_NUMBER = re.compile(r"\b(how (many|much|long|big|large|fast|old)|what (is|was|are) the (\w+ )?(number|value|rate|"
                          r"score|size|dimension|count|version|price|cost|amount|percentage)|which version|minimum|maximum)\b", re.I)
WANTS_COMMAND = re.compile(r"\b(install|command|run|start|setup|set up|how (do|to|can|should) (i|you|we))\b", re.I)
WANTS_DATE = re.compile(r"\b(when|what (year|date|month))\b", re.I)
DATE = re.compile(r"\b(1[89]\d\d|20\d\d|January|February|March|April|May|June|July|August|September|October|"
                  r"November|December)\b")
PER_SECTION = 3      # sentences kept from one section
TOP_SECTIONS = 15    # sections whose sentences are considered
SECTION_WEIGHT = 0.6
LONG_UNIT = 120      # a longer sentence is cut to the part around the question's words
# sections that cite or link out rather than say anything
BACK_MATTER = re.compile(r"^(references|notes|citations|bibliography|sources|external links|see also|further reading|"
                         r"footnotes)\b", re.I)
CODE_LINE = re.compile(r"^\s*(`|\$ |>>> |pip |uv |npm |npx |brew |apt |cargo |go |docker )")


def _units(s: Section) -> list[tuple[int, str]]:
    """(line offset, text) per sentence, table row or list item. Hard-wrapped prose (PDFs) is
    joined back into paragraphs first, so a sentence isn't cut at a line break."""
    paras: list = []
    for k, ln in enumerate(s.lines):
        st = ln.strip()
        if not st:
            paras.append(None)
            continue
        structural = st[0] in "|#-*>`" or bool(re.match(r"^\d+[.)] ", st))
        if paras and paras[-1] is not None and not structural and not paras[-1][2]:
            pk, pt, _ = paras[-1]
            paras[-1] = (pk, pt + " " + st, False)
        else:
            paras.append((k, st, structural))
    out = []
    for k, ln, _ in (p for p in paras if p):
        out += [(k, u) for u in (SENT.split(ln) if tokens.count(ln) > 45 else [ln])]
    return out


def _terms(text: str) -> set[str]:
    return set(words(_link_text(text).replace("_", " ")))


def _label(heading: str) -> str:
    """A heading as plain text: no links, anchors (¶) or markdown emphasis."""
    h = MD_LINK.sub(lambda m: m.group(1), heading).replace("¶", "")
    return re.sub(r"[*_`]+", "", h).strip()[:80]


def _window(text: str, q: set[str]) -> str:
    """The part of a very long sentence that covers the question's words (from the first to the
    last, as far as it fits), with a little before."""
    parts = text.split()
    hits = [i for i, w in enumerate(parts) if _terms(w) & q]
    if not hits:
        return text
    keep = int(LONG_UNIT * 0.7)
    a = max(0, hits[0] - 8)
    last = max([h for h in hits if h < a + keep] or [hits[0]])
    b = min(len(parts), max(a + keep, last + 12) if last + 12 - a <= keep * 1.3 else a + keep)
    return ("… " if a else "") + " ".join(parts[a:b]) + (" …" if b < len(parts) else "")


def focus(text: str, question: str, budget: int = 250, url: str = "") -> tuple[str, int, int]:
    """The sentences, rows and lines that best answer `question`, quoted word for word, in page
    order, within `budget` tokens. Returns (text, sections used, sections in total).

    Sections are ranked first (BM25); then the best one or two sentences of each top section
    are kept, like the snippets under search results, so several parts of the page fit in a few
    hundred tokens. Each snippet starts with its heading and line number.
    """
    secs = sections(text)
    if not secs:
        return "", 0, 0
    ranked = rank(question, secs)
    if FROM_TOP.search(question) or ranked[0][0] <= 0:
        return sections_focus(text, question, budget, url)
    q = set(words(question)) - QSTOP or set(words(question))
    qn = {x for x in NUMBER.findall(question) if len(x) > 1}
    n = len(secs)
    df = Counter(w for s in secs for w in _terms(s.text))
    idf = {w: math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) for w in q}
    wants_number, wants_date = WANTS_NUMBER.search(question), WANTS_DATE.search(question)
    wants_command = WANTS_COMMAND.search(question)
    best = ranked[0][0]
    cands = []  # (sentence score, section score, section, line, order, text)
    for sc, si in ranked[:TOP_SECTIONS]:
        if sc < best * 0.1:
            break
        s = secs[si]
        if BACK_MATTER.match(_label(s.heading)):
            continue
        head = set(words(s.heading))
        units = _units(s)
        for j, (k, u) in enumerate(units):
            terms = _terms(u)
            score = sum(idf[w] for w in q & (terms | head)) + 0.8 * len(qn & set(NUMBER.findall(u)))
            if score <= 0:
                continue
            if wants_number and re.search(r"\d", u):
                score *= 1.5
            if wants_date and DATE.search(u):
                score *= 1.5
            if wants_command and CODE_LINE.match(u):
                score *= 1.6
            cands.append((score, sc, si, k, j, u))
            # a command or code line right before a matching sentence usually is the answer
            # ("Install it with: / `uv add mcp`"): it rides along with half the score
            if j and CODE_LINE.match(units[j - 1][1]) and not (_terms(units[j - 1][1]) & q):
                cands.append((score * 0.9, sc, si, units[j - 1][0], j - 1, units[j - 1][1]))
    if not cands:
        return sections_focus(text, question, budget, url)
    top = max(c[0] for c in cands)
    cands.sort(key=lambda c: (-(c[0] / top + SECTION_WEIGHT * c[1] / best), c[2], c[3], c[4]))
    chosen, used, per = [], 0, Counter()
    for c in cands:
        if per[c[2]] >= PER_SECTION:
            continue
        t = tidy(c[5], url).strip()
        if tokens.count(t) > LONG_UNIT:
            t = _window(t, q)
        cost = tokens.count(t) + (tokens.count(_label(secs[c[2]].heading)) if not per[c[2]] else 0) + 5
        if chosen and used + cost > budget:
            continue
        chosen.append((c, t))
        used += cost
        per[c[2]] += 1
        if used >= budget * 0.95:
            break
    chosen.sort(key=lambda x: (secs[x[0][2]].start + x[0][3], x[0][4]))
    parts: list[list] = []
    last = None
    for c, t in chosen:
        s = secs[c[2]]
        if last and last[0] == c[2]:
            parts[-1][1] += (" " if c[3] == last[1] else " … ") + t
        else:
            head = _label(s.heading)
            label = f"{head} · line {s.start + c[3] + 1}" if head else f"line {s.start + c[3] + 1}"
            parts.append([label, t])
        last = (c[2], c[3])
    return "\n".join(f"[{a}] {b}" for a, b in parts), len(parts), n


# ---------------------------------------------------------------- JSON results (atr_read question=)

def main_list(value) -> tuple[list, list] | None:
    """The list that holds a result's items, and the keys that lead to it: a top-level list, or the
    biggest list one or two levels into an object ({"items": [...]}, {"data": {"rows": [...]}})."""
    if isinstance(value, list):
        return value, []
    if not isinstance(value, dict):
        return None
    best = None
    for k, v in value.items():
        found = main_list(v) if isinstance(v, dict) else ((v, []) if isinstance(v, list) else None)
        if found and (best is None or len(found[0]) > len(best[0])):
            best = (found[0], [k] + found[1])
    return best if best and len(best[0]) > 1 else None


def pick_items(items: list, question: str) -> list[int] | None:
    """Indices of the items that match `question`, best first, or None when the question doesn't
    single anything out (nothing matches, it asks for the first/latest ones, or nearly all match)."""
    import json

    if len(items) < 2 or FROM_TOP.search(question):
        return None
    texts = [json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x for x in items]
    docs = [set(words(_link_text(t))) for t in texts]
    # words that pick items out; one that is in nearly every item ("issues" in every issue's URL) doesn't
    telling = {w for w in set(words(question)) if 0 < sum(w in d for d in docs) <= 0.8 * len(items)}
    matched = [i for i, d in enumerate(docs) if d & telling]
    if not matched:
        return None
    secs = [Section(i, i, "", [texts[i]]) for i in range(len(items))]
    score = dict((i, sc) for sc, i in rank(question, secs))
    matched.sort(key=lambda i: (-score[i], i))
    best = score[matched[0]]
    return [i for i in matched if score[i] >= best * 0.25]
