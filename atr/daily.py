"""Small, keyless everyday tools for the hosted ATR connector."""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import unquote, urljoin, urlsplit

from atr.places import geocode_location
from atr.web import download

MAX_SITEMAPS = 3


@dataclass
class SiteDetails:
    title: str = ""
    description: str = ""
    canonical: str = ""
    language: str = ""
    headings: list[str] = field(default_factory=list)
    links: list[tuple[str, str]] = field(default_factory=list)


class _SiteParser(HTMLParser):
    def __init__(self, base: str):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.details = SiteDetails()
        self.in_title = False
        self.heading: str | None = None
        self.link: str | None = None
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "html":
            self.details.language = attrs.get("lang", "")
        elif tag == "title":
            self.in_title = True
            self.text = []
        elif tag in ("h1", "h2") and len(self.details.headings) < 8:
            self.heading = tag
            self.text = []
        elif tag == "a" and attrs.get("href"):
            self.link = urljoin(self.base, attrs["href"])
            self.text = []
        elif tag == "meta":
            name = (attrs.get("name") or attrs.get("property") or "").lower()
            if name in ("description", "og:description") and not self.details.description:
                self.details.description = attrs.get("content", "").strip()
        elif tag == "link" and "canonical" in (attrs.get("rel") or "").lower():
            self.details.canonical = urljoin(self.base, attrs.get("href", ""))

    def handle_data(self, data):
        if self.in_title or self.heading or self.link:
            self.text.append(data)

    def handle_endtag(self, tag):
        text = re.sub(r"\s+", " ", "".join(self.text)).strip()
        if tag == "title" and self.in_title:
            self.details.title = text
            self.in_title = False
            self.text = []
        elif tag == self.heading:
            if text:
                self.details.headings.append(text)
            self.heading = None
            self.text = []
        elif tag == "a" and self.link:
            if text and self.link.startswith(("http://", "https://")):
                self.details.links.append((text, self.link))
            self.link = None
            self.text = []


def parse_site(html_text: str, url: str) -> SiteDetails:
    parser = _SiteParser(url)
    try:
        parser.feed(html_text)
        parser.close()
    except Exception:  # broken pages can still yield useful metadata
        pass
    return parser.details


def _decode(body: bytes, ctype: str) -> str:
    match = re.search(r"charset=([^;\s]+)", ctype, re.I)
    return body.decode(match.group(1) if match else "utf-8", errors="replace")


async def site_info(client, url: str) -> str:
    final, ctype, body = await download(url, client)
    mime = ctype.split(";", 1)[0]
    lines = [f"URL: {final}", f"Content type: {mime or 'unknown'}", f"Size: {len(body):,} bytes"]
    if "html" not in mime:
        return "\n".join(lines)
    details = parse_site(_decode(body, ctype), final)
    if details.title:
        lines.insert(0, f"Title: {details.title}")
    if details.description:
        lines.append(f"Description: {details.description}")
    if details.canonical:
        lines.append(f"Canonical: {details.canonical}")
    if details.language:
        lines.append(f"Language: {details.language}")
    if details.headings:
        lines.append("Main headings: " + " | ".join(details.headings))
    external = {urlsplit(link).netloc for _, link in details.links if urlsplit(link).netloc != urlsplit(final).netloc}
    lines.append(f"Links on page: {len(details.links)} ({len(external)} external domains)")
    return "\n".join(lines)


def _terms(query: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]{2,}", query.casefold()) if t not in {"the", "and", "for", "with"}]


def _rank_links(links: list[tuple[str, str]], query: str, limit: int) -> list[tuple[str, str]]:
    terms = _terms(query)
    scored = []
    seen = set()
    for label, link in links:
        if link in seen:
            continue
        seen.add(link)
        haystack = unquote(f"{label} {urlsplit(link).path}").casefold().replace("-", " ").replace("_", " ")
        score = sum(3 if re.search(rf"\b{re.escape(term)}\b", haystack) else 1 if term in haystack else 0
                    for term in terms)
        if score:
            scored.append((score, label or link, link))
    scored.sort(key=lambda item: (-item[0], len(item[2])))
    return [(label, link) for _, label, link in scored[:limit]]


async def site_search(client, url: str, query: str, limit: int = 8) -> str:
    """Search a site's sitemap and homepage links without a search-engine API."""
    if not query.strip():
        raise ValueError("query is required for site search")
    limit = max(1, min(int(limit), 10))
    if "://" not in url:
        url = "https://" + url
    parts = urlsplit(url)
    root = f"{parts.scheme}://{parts.netloc}/"
    links: list[tuple[str, str]] = []
    local_sitemap = urljoin(url if url.endswith("/") else url.rsplit("/", 1)[0] + "/", "sitemap.xml")
    sitemap_urls = list(dict.fromkeys((local_sitemap, urljoin(root, "sitemap.xml"))))
    checked = 0
    while sitemap_urls and checked < MAX_SITEMAPS:
        sitemap = sitemap_urls.pop(0)
        checked += 1
        try:
            _, _, body = await download(sitemap, client)
            xml = ET.fromstring(body)
        except Exception:  # missing or malformed sitemap: homepage links remain useful
            continue
        locations = [html.unescape(node.text.strip()) for node in xml.iter() if node.tag.endswith("loc") and node.text]
        if xml.tag.endswith("sitemapindex"):
            sitemap_urls.extend(locations[: MAX_SITEMAPS - checked])
        else:
            links.extend((unquote(urlsplit(link).path).strip("/").replace("-", " ") or link, link)
                         for link in locations[:5000])
    try:
        final, ctype, body = await download(root, client)
        if "html" in ctype:
            links.extend(parse_site(_decode(body, ctype), final).links)
    except Exception:
        pass
    links = [(label, link) for label, link in links if urlsplit(link).netloc == parts.netloc]
    matches = _rank_links(links, query, limit)
    if not matches:
        return (f"No sitemap or homepage links on {root} matched {query!r}. This searches public site navigation, "
                "not pages hidden behind sign-in or JavaScript. Try atr_fetch on a likely page instead.")
    lines = [f"{len(matches)} pages on {root} matching {query!r}:"]
    lines.extend(f"{i}. {label}\n   {link}" for i, (label, link) in enumerate(matches, 1))
    return "\n".join(lines)


WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "rime fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 80: "light rain showers", 81: "rain showers",
    82: "heavy rain showers", 95: "thunderstorm", 96: "thunderstorm with hail", 99: "heavy hail storm",
}


async def weather(client, location: str, days: int = 3, units: str = "auto") -> str:
    place = await geocode_location(client, location)
    if not place:
        return f"Couldn't find the location {location!r}. Try a city plus state or country."
    days = max(1, min(int(days), 7))
    if units not in ("auto", "us", "metric"):
        raise ValueError("units must be auto, us, or metric")
    if units == "auto":
        units = "us" if place["country_code"] in {"us", "bs", "bz", "ky", "pw", "fm", "mh", "lr"} else "metric"
    params = {
        "latitude": place["latitude"], "longitude": place["longitude"],
        "current": "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto", "forecast_days": days,
    }
    if units == "us":
        params.update({"temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch"})
    response = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
    response.raise_for_status()
    data = response.json()
    current, daily = data.get("current") or {}, data.get("daily") or {}
    current_units, daily_units = data.get("current_units") or {}, data.get("daily_units") or {}
    code = int(current.get("weather_code", -1))
    lines = [f"Weather for {place['name']} ({data.get('timezone', 'local time')}):",
             f"Now ({current.get('time', 'unknown')}): {WEATHER_CODES.get(code, f'code {code}')}, "
             f"{current.get('temperature_2m')} {current_units.get('temperature_2m', '')}; feels like "
             f"{current.get('apparent_temperature')} {current_units.get('apparent_temperature', '')}; wind "
             f"{current.get('wind_speed_10m')} {current_units.get('wind_speed_10m', '')}."]
    for i, date in enumerate(daily.get("time") or []):
        code = int((daily.get("weather_code") or [-1] * days)[i])
        lines.append(f"{date}: {WEATHER_CODES.get(code, f'code {code}')}; high "
                     f"{daily['temperature_2m_max'][i]}{daily_units.get('temperature_2m_max', '')}, low "
                     f"{daily['temperature_2m_min'][i]}{daily_units.get('temperature_2m_min', '')}; "
                     f"precipitation chance {daily['precipitation_probability_max'][i]}%.")
    lines.append("Forecast: Open-Meteo. Location: OpenStreetMap contributors.")
    return "\n".join(lines)
