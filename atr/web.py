"""Reads a web page or document as compact text (atr_fetch in the hosted connector).

    page = await fetch_page("https://example.com/post")   # -> Page(url, title, kind, text)

HTML becomes markdown with only the main content (no menus, ads or footers), keeping headings,
links and tables. PDF, Word, PowerPoint and Excel files become markdown too. JSON is left as
JSON, so the usual compressor can turn it into tables.

This runs on a shared server, so every request is checked before it is sent: only http(s),
and never to private, loopback, link-local or cloud-metadata addresses. Redirects are followed
by hand and each hop is checked again, and the address actually connected to is checked once
more after connecting (so a hostname can't switch to a private address between the two).
Downloads stop at MAX_BYTES.
"""
from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
import tempfile
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

from atr import __version__
from atr.compress import html_to_text, looks_like_html

MAX_BYTES = 15 * 1024 * 1024
MAX_REDIRECTS = 5
TIMEOUT = 25.0
USER_AGENT = f"Mozilla/5.0 (compatible; ATR/{__version__}; +https://github.com/JacobBhatt12/atr)"

DOC_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "application/epub+zip": ".epub",
}
DOC_EXTS = set(DOC_TYPES.values())


ALLOW_PRIVATE = False  # tests only: lets fetches reach 127.0.0.1


class FetchError(Exception):
    pass


@dataclass
class Page:
    url: str
    title: str
    kind: str   # html, json, text, pdf, docx, ...
    text: str


# ---------------------------------------------------------------- address checks

def blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return not ip.is_global or ip.is_multicast


async def check_url(url: str) -> None:
    """Raise FetchError unless `url` is http(s) and every address its host resolves to is public."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise FetchError("only http:// and https:// links can be fetched")
    host = parts.hostname
    if not host:
        raise FetchError("the link has no host name")
    try:
        ips = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(host, parts.port or 443, type=socket.SOCK_STREAM)
        except socket.gaierror:
            raise FetchError(f"can't find the site {host!r}") from None
        ips = [ipaddress.ip_address(i[4][0].split("%", 1)[0]) for i in infos]
    if not ips or (not ALLOW_PRIVATE and any(blocked_ip(ip) for ip in ips)):
        raise FetchError(f"{host} is a private or local address; only public sites can be fetched")


def _peer_ok(resp) -> bool:
    """The address the connection actually went to is public (None if the transport can't say)."""
    stream = resp.extensions.get("network_stream")
    addr = stream.get_extra_info("server_addr") if stream is not None else None
    if not addr or ALLOW_PRIVATE:
        return True
    try:
        return not blocked_ip(ipaddress.ip_address(str(addr[0]).split("%", 1)[0]))
    except ValueError:
        return True


# ---------------------------------------------------------------- download

async def download(url: str, client) -> tuple[str, str, bytes]:
    """GET `url` safely. Returns (final url, content type, body)."""
    for _ in range(MAX_REDIRECTS + 1):
        await check_url(url)
        async with client.stream("GET", url, follow_redirects=False) as resp:
            if not _peer_ok(resp):
                raise FetchError("that site points to a private address; only public sites can be fetched")
            if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("location"):
                url = urljoin(url, resp.headers["location"])
                continue
            if resp.status_code >= 400:
                raise FetchError(f"the site answered HTTP {resp.status_code}")
            size = int(resp.headers.get("content-length") or 0)
            if size > MAX_BYTES:
                raise FetchError(f"the file is {size // 1_000_000} MB; the limit is {MAX_BYTES // 1_000_000} MB")
            body = bytearray()
            async for chunk in resp.aiter_bytes():
                body.extend(chunk)
                if len(body) > MAX_BYTES:
                    raise FetchError(f"the file is over {MAX_BYTES // 1_000_000} MB")
            ctype = resp.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            charset = resp.charset_encoding or "utf-8"
            return str(resp.url), f"{ctype};charset={charset}", bytes(body)
    raise FetchError(f"more than {MAX_REDIRECTS} redirects")


# ---------------------------------------------------------------- conversion

def _decode(body: bytes, charset: str) -> str:
    try:
        return body.decode(charset or "utf-8", errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


class _Markdown(HTMLParser):
    """Whole-page HTML to markdown, keeping links (same-site ones as /paths). Used for pages that are mostly links and short
    items (news front pages, search results, tables), where main-content extraction falls apart."""

    SKIP = {"script", "style", "noscript", "svg", "template", "head", "iframe", "canvas", "select", "button"}
    BLOCK = {"p", "div", "section", "article", "main", "header", "footer", "nav", "aside", "form", "table", "tr",
             "ul", "ol", "dl", "dt", "dd", "blockquote", "pre", "figure", "figcaption", "br", "hr", "tbody", "thead"}

    def __init__(self, base: str):
        super().__init__(convert_charrefs=True)
        self.base, self.lines, self.cur, self.skip = base, [], [], 0
        self.href: str | None = None
        self.link_text: list[str] = []

    def _flush(self) -> None:
        line = re.sub(r"\s+", " ", "".join(self.cur)).strip().strip("|").strip()
        self.cur = []
        if line and re.search(r"\w", line) and (not self.lines or self.lines[-1] != line):
            self.lines.append(line)

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif self.skip:
            return
        elif tag in self.BLOCK:
            self._flush()
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._flush()
            self.cur.append("#" * int(tag[1]) + " ")
        elif tag == "li":
            self._flush()
            self.cur.append("- ")
        elif tag in ("td", "th"):
            self.cur.append(" | " if self.cur else "")
        elif tag == "a":
            href = dict(attrs).get("href") or ""
            if href and not href.startswith(("#", "javascript:", "mailto:")):
                full = urljoin(self.base, href)
                same = urlsplit(full)
                # same-site links as paths: the page's own URL is in the header, so nothing is lost
                if same.netloc == urlsplit(self.base).netloc and same.path:
                    full = same.path + (f"?{same.query}" if same.query else "")
                self.href, self.link_text = full, []

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif self.skip:
            return
        elif tag == "a" and self.href is not None:
            text = re.sub(r"\s+", " ", "".join(self.link_text)).strip()
            self.cur.append(f"[{text}]({self.href})" if text else "")
            self.href = None
        elif tag in self.BLOCK or tag in ("h1", "h2", "h3", "h4", "h5", "h6", "li"):
            self._flush()

    def handle_data(self, data):
        if self.skip:
            return
        (self.link_text if self.href is not None else self.cur).append(data)

    def markdown(self) -> str:
        self._flush()
        return "\n".join(self.lines)


def page_markdown(html: str, url: str) -> str:
    p = _Markdown(url)
    try:
        p.feed(html)
        p.close()
    except Exception:  # noqa: BLE001 - broken HTML: keep what was parsed
        pass
    return p.markdown()


def fragmented(md: str) -> bool:
    """Main-content extraction gave one word per line (a page of links and short items)."""
    lines = [ln for ln in md.splitlines() if ln.strip()]
    return len(lines) > 20 and sum(len(ln) < 25 for ln in lines) / len(lines) > 0.6


def html_to_markdown(html: str, url: str) -> tuple[str, str]:
    """Main content as markdown, and the page title."""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    title = re.sub(r"\s+", " ", unescape(m.group(1))).strip() if m else ""
    md = None
    try:
        import trafilatura

        md = trafilatura.extract(html, url=url, output_format="markdown", include_links=True, include_tables=True,
                                 include_comments=False, include_images=False, favor_recall=True,
                                 deduplicate=True)
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - a page trafilatura can't parse still has text
        pass
    if md and len(md.strip()) > 200 and not fragmented(md):
        return md.strip(), title
    whole = page_markdown(html, url)
    return (whole or html_to_text(html)).strip(), title


def pdf_to_text(body: bytes) -> str | None:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return None
    try:
        doc = pdfium.PdfDocument(body)
        pages = []
        for i in range(len(doc)):
            text = doc[i].get_textpage().get_text_range().replace("\r\n", "\n").strip()
            if text:
                pages.append(f"--- page {i + 1} ---\n{text}")
        return "\n\n".join(pages)
    except Exception:  # noqa: BLE001 - let markitdown try
        return None


def office_to_text(body: bytes, ext: str) -> str | None:
    """Word, PowerPoint and Excel with small, focused readers. None if the reader isn't installed."""
    import io

    try:
        if ext == ".docx":
            import mammoth

            return page_markdown(mammoth.convert_to_html(io.BytesIO(body)).value, "")
        if ext == ".pptx":
            from pptx import Presentation

            out = []
            for i, slide in enumerate(Presentation(io.BytesIO(body)).slides, 1):
                out.append(f"--- slide {i} ---")
                for shape in slide.shapes:
                    if getattr(shape, "has_table", False) and shape.has_table:
                        out += [" | ".join(c.text.strip() for c in row.cells) for row in shape.table.rows]
                    elif getattr(shape, "has_text_frame", False) and shape.has_text_frame and shape.text_frame.text.strip():
                        out.append(shape.text_frame.text.strip())
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
                    out.append("Notes: " + slide.notes_slide.notes_text_frame.text.strip())
            return "\n".join(out)
        if ext == ".xlsx":
            from openpyxl import load_workbook

            wb = load_workbook(io.BytesIO(body), read_only=True, data_only=True)
            out = []
            for ws in wb.worksheets:
                out.append(f"--- sheet {ws.title} ---")
                for n, row in enumerate(ws.iter_rows(values_only=True)):
                    if n >= 5000:
                        out.append("… more rows not shown")
                        break
                    cells = ["" if v is None else str(v) for v in row]
                    while cells and not cells[-1]:
                        cells.pop()
                    if cells:
                        out.append(" | ".join(cells))
            return "\n".join(out)
    except ImportError:
        return None
    except Exception as e:  # noqa: BLE001
        raise FetchError(f"couldn't read the {ext} file: {e}") from None
    return None


def doc_to_markdown(body: bytes, ext: str) -> str:
    text = pdf_to_text(body) if ext == ".pdf" else office_to_text(body, ext)
    if text and text.strip():
        return text
    try:
        from markitdown import MarkItDown
    except ImportError:
        raise FetchError(f"reading {ext} files isn't installed on this server") from None
    fd, path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(body)
        return MarkItDown().convert(path).text_content or ""
    except Exception as e:  # noqa: BLE001
        raise FetchError(f"couldn't read the {ext} file: {e}") from None
    finally:
        os.unlink(path)


def to_text(url: str, ctype: str, body: bytes) -> Page:
    mime, _, charset = ctype.partition(";charset=")
    ext = DOC_TYPES.get(mime) or next((e for e in DOC_EXTS if urlsplit(url).path.lower().endswith(e)), None)
    if ext and (mime in DOC_TYPES or mime in ("", "application/octet-stream", "binary/octet-stream")):
        return Page(url, "", ext[1:], doc_to_markdown(body, ext))
    if mime.startswith(("image/", "audio/", "video/", "font/")) or mime in ("application/zip", "application/octet-stream"):
        raise FetchError(f"that link is a {mime or 'binary'} file, not a page or document")
    text = _decode(body, charset)
    if "json" in mime:
        return Page(url, "", "json", text)
    if "html" in mime or (("xml" in mime or not mime) and looks_like_html(text)):
        md, title = html_to_markdown(text, url)
        return Page(url, title, "html", md)
    return Page(url, "", "text", text)


async def fetch_page(url: str, client) -> Page:
    url = url.strip()
    if "://" not in url:
        url = "https://" + url
    final, ctype, body = await download(url, client)
    return await asyncio.to_thread(to_text, final, ctype, body)


def make_client():
    import httpx

    return httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT), headers={
        "user-agent": USER_AGENT, "accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        "accept-language": "en-US,en;q=0.8"})
