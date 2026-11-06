"""Web search for the hosted connector (atr_search), free and keyless.

    results, engines_used = await web_search(client, "model context protocol", limit=5)
    text = format_results("model context protocol", results, engines_used)

Asks DuckDuckGo, Bing and Brave at the same time and merges their answers: a link that several
engines rank high comes first (reciprocal rank fusion), duplicates are dropped, and each result is
cut to a title, a short address and one or two sentences. Five results are about 200 tokens,
where a search tool that returns whole result pages or long snippets uses thousands.

Google doesn't answer automated requests without a paid API key, so it isn't included. If an
engine is slow or shows a bot check, the others still answer.
"""
from __future__ import annotations

import asyncio
import base64
import html
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, unquote, urlsplit, urlunsplit, urlencode

ENGINE_TIMEOUT = 8.0
SNIPPET_CHARS = 160
BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/130.0 Safari/537.36")
HEADERS = {"user-agent": BROWSER_UA, "accept": "text/html,application/xhtml+xml",
           "accept-language": "en-US,en;q=0.8"}
NAMES = {"duckduckgo": "DuckDuckGo", "bing": "Bing", "brave": "Brave"}
TRACKING = re.compile(r"^(utm_|fbclid|gclid|msclkid|ref$|ref_src$)")


@dataclass
class Result:
    url: str
    title: str
    snippet: str
    engines: list[str] = field(default_factory=list)
    score: float = 0.0


def _text(fragment: str) -> str:
    fragment = re.sub(r"<(script|style)\b.*?</\1>", " ", fragment, flags=re.S | re.I)
    # Bing's own date stamp is often wrong (years off); relative ones ("2 days ago") are kept.
    fragment = re.sub(r'<span class="news_dt">(?![^<]*ago)[^<]*</span>(&nbsp;|\s)*(&#0183;|·)?(&#32;|\s)*', "", fragment)
    fragment = re.sub(r"</?(strong|b|em|i|mark|span)\b[^>]*>", "", fragment)   # inline tags: no extra space
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def _clean_url(url: str) -> str:
    url = html.unescape(url).strip()
    if url.startswith("//"):
        url = "https:" + url
    parts = urlsplit(url)
    if parts.netloc.endswith("duckduckgo.com") and parts.path.startswith("/l/"):
        url = unquote(parse_qs(parts.query).get("uddg", [url])[0])
    elif parts.netloc.endswith("bing.com") and parts.path.startswith("/ck/"):
        u = parse_qs(parts.query).get("u", [""])[0]
        if u.startswith("a1"):
            try:
                url = base64.urlsafe_b64decode(u[2:] + "=" * (-len(u[2:]) % 4)).decode()
            except Exception:  # noqa: BLE001 - keep the redirect link if it can't be decoded
                pass
    parts = urlsplit(url)
    query = urlencode([(k, val) for k, vals in parse_qs(parts.query, keep_blank_values=True).items()
                       for val in vals if not TRACKING.match(k)])
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


def _key(url: str) -> str:
    p = urlsplit(url.lower())
    return p.netloc.removeprefix("www.") + p.path.rstrip("/") + ("?" + p.query if p.query else "")


def parse_duckduckgo(page: str) -> list[Result]:
    out = []
    for block in re.split(r'<div class="result results_links', page)[1:]:
        if "result--ad" in block[:200]:
            continue
        a = re.search(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S) or \
            re.search(r'href="([^"]+)"[^>]*class="result__a"[^>]*>(.*?)</a>', block, re.S)
        if not a:
            continue
        s = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.S)
        out.append(Result(_clean_url(a.group(1)), _text(a.group(2)), _text(s.group(1)) if s else ""))
    return out


def parse_bing(page: str) -> list[Result]:
    out = []
    for block in re.split(r'<li class="b_algo"', page)[1:]:
        a = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not a:
            continue
        s = re.search(r'<p class="b_lineclamp\d*"[^>]*>(.*?)</p>', block, re.S) or \
            re.search(r'class="b_caption"[^>]*>.*?<p[^>]*>(.*?)</p>', block, re.S)
        out.append(Result(_clean_url(a.group(1)), _text(a.group(2)), _text(s.group(1)) if s else ""))
    return out


def parse_brave(page: str) -> list[Result]:
    out = []
    for block in re.split(r'data-type="web"', page)[1:]:
        a = re.search(r'<a href="(https?://[^"]+)"', block)
        t = re.search(r'class="title search-snippet-title[^"]*"[^>]*title="([^"]*)"', block) or \
            re.search(r'class="title[^"]*"[^>]*>(.*?)</div>', block, re.S)
        if not a or not t:
            continue
        s = re.search(r'class="generic-snippet[^"]*">\s*<div class="content[^"]*">(.*?)</div>', block, re.S)
        out.append(Result(_clean_url(a.group(1)), _text(t.group(1)), _text(s.group(1)) if s else ""))
    return out


async def _engine(client, name: str, query: str) -> list[Result]:
    if name == "duckduckgo":
        r = await client.post("https://html.duckduckgo.com/html/", data={"q": query, "b": "", "kl": "us-en"},
                              headers=HEADERS, timeout=ENGINE_TIMEOUT)
        parse = parse_duckduckgo
    elif name == "bing":
        r = await client.get("https://www.bing.com/search", params={"q": query, "setlang": "en", "mkt": "en-US"},
                             headers=HEADERS, timeout=ENGINE_TIMEOUT, follow_redirects=True)
        parse = parse_bing
    else:
        r = await client.get("https://search.brave.com/search", params={"q": query, "source": "web"},
                             headers=HEADERS, timeout=ENGINE_TIMEOUT, follow_redirects=True)
        parse = parse_brave
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    results = [x for x in parse(r.text) if x.url.startswith("http") and x.title]
    if not results:
        raise RuntimeError("no results (possibly a bot check)")
    return results


def merge(by_engine: dict[str, list[Result]], limit: int) -> list[Result]:
    """Reciprocal rank fusion: each engine adds 1/(10 + rank) to a link's score."""
    merged: dict[str, Result] = {}
    for name, results in by_engine.items():
        for rank, r in enumerate(results[:15]):
            k = _key(r.url)
            if k not in merged:
                merged[k] = Result(r.url, r.title, r.snippet)
            m = merged[k]
            m.score += 1 / (10 + rank)
            m.engines.append(name)
            if len(r.snippet) > len(m.snippet):   # keep the most informative snippet
                m.snippet = r.snippet
    return sorted(merged.values(), key=lambda r: -r.score)[:limit]


async def web_search(client, query: str, limit: int = 5,
                     engines: tuple[str, ...] = ("duckduckgo", "bing", "brave")) -> tuple[list[Result], dict, dict]:
    """Returns (merged results, {engine: result count or error}, {engine: its own results})."""
    answers = await asyncio.gather(*(_engine(client, e, query) for e in engines), return_exceptions=True)
    by_engine, status = {}, {}
    for name, ans in zip(engines, answers):
        if isinstance(ans, BaseException):
            status[name] = f"{type(ans).__name__}: {ans}"[:120]
        else:
            by_engine[name], status[name] = ans, len(ans)
    return merge(by_engine, max(1, min(int(limit), 10))), status, by_engine


def _short(snippet: str) -> str:
    if len(snippet) <= SNIPPET_CHARS:
        return snippet
    cut = snippet[:SNIPPET_CHARS]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[:end + 1] if end > SNIPPET_CHARS * 0.5 else cut.rsplit(" ", 1)[0] + " …"


def format_results(query: str, results: list[Result], status: dict) -> str:
    used = [NAMES[e] for e, s in status.items() if isinstance(s, int)]
    if not results:
        failed = "; ".join(f"{NAMES[e]}: {s}" for e, s in status.items())
        return f"No web results for {query!r} ({failed}). Try other words, or atr_fetch a known page."
    lines = [f"{query} ({', '.join(used)}):"]
    for i, r in enumerate(results, 1):
        address = re.sub(r"^https?://(www\.)?", "", r.url).rstrip("/")
        lines.append(f"{i}. {r.title} | {address}")
        if r.snippet:
            lines.append(f"   {_short(r.snippet)}")
    lines.append("[atr_fetch(url, question) reads one]")
    return "\n".join(lines)
