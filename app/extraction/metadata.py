"""Metadata extraction: description, canonical URL, publication date, author, language.

All functions work on the *pristine* soup (before boilerplate removal) because metadata lives in
<head>, JSON-LD scripts and header/byline areas that cleaning would delete.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from app.crawler.url_utils import URLValidationError, normalize_url

_WS = re.compile(r"\s+")
_DATE_FORMATS = ("%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y", "%Y/%m/%d", "%Y.%m.%d")


def clean(text: str | None) -> str:
    return _WS.sub(" ", (text or "").replace("\xa0", " ")).strip()


def collect_meta(soup: BeautifulSoup) -> dict[str, str]:
    """Map lowercase meta key (name / property / itemprop / http-equiv) -> first non-empty content."""
    metas: dict[str, str] = {}
    for tag in soup.find_all("meta"):
        content = clean(tag.get("content"))
        if not content:
            continue
        for attr in ("name", "property", "itemprop", "http-equiv"):
            key = tag.get(attr)
            if isinstance(key, str) and key.strip():
                metas.setdefault(key.strip().lower(), content)
    return metas


def extract_jsonld(soup: BeautifulSoup) -> list[dict]:
    """Return every JSON-LD object (flattening @graph) found in the page."""
    nodes: list[dict] = []

    def flatten(data, depth: int = 0) -> None:
        if depth > 4:
            return
        if isinstance(data, list):
            for item in data:
                flatten(item, depth + 1)
        elif isinstance(data, dict):
            nodes.append(data)
            graph = data.get("@graph")
            if isinstance(graph, (list, dict)):
                flatten(graph, depth + 1)

    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        raw = script.string or script.get_text()
        try:
            flatten(json.loads(raw))
        except (ValueError, TypeError):
            continue
    return nodes


# ---------------------------------------------------------------------------------- description
def find_description(metas: dict[str, str]) -> str | None:
    for key in ("description", "og:description", "twitter:description"):
        if metas.get(key):
            return metas[key][:1000]
    return None


# ---------------------------------------------------------------------------------- canonical
def find_canonical(soup: BeautifulSoup, base_url: str) -> str | None:
    for link in soup.find_all("link", href=True):
        rel = link.get("rel") or []
        rel = rel.split() if isinstance(rel, str) else rel
        if "canonical" in [r.lower() for r in rel]:
            try:
                url = normalize_url(link["href"].strip(), base_url)
            except URLValidationError:
                return None
            return url if url.startswith(("http://", "https://")) else None
    return None


# ---------------------------------------------------------------------------------- dates
def parse_date(value: str | None) -> str | None:
    """Parse many date formats; return ISO-8601 ('2024-01-05' or full timestamp) or None."""
    if not value:
        return None
    v = " ".join(value.split())[:64]
    dt: datetime | None = None
    has_time = True
    try:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        has_time = len(v) > 10
    except ValueError:
        for fmt in _DATE_FORMATS:
            try:
                dt, has_time = datetime.strptime(v, fmt), False
                break
            except ValueError:
                continue
        if dt is None:
            try:
                dt = parsedate_to_datetime(v)
            except (TypeError, ValueError, IndexError):
                return None
    if not (1990 <= dt.year <= datetime.now(timezone.utc).year + 1):
        return None
    return dt.isoformat() if has_time else dt.date().isoformat()


_DATE_META_KEYS = (
    "article:published_time", "og:article:published_time", "datepublished", "pubdate", "publishdate",
    "publish_date", "og:published_time", "dc.date.issued", "dc.date", "date", "sailthru.date",
    "parsely-pub-date",
)


def find_published(soup: BeautifulSoup, metas: dict[str, str], jsonld: list[dict]) -> str | None:
    for key in _DATE_META_KEYS:
        if (d := parse_date(metas.get(key))):
            return d
    for node in jsonld:
        for key in ("datePublished", "dateCreated"):
            if isinstance(node.get(key), str) and (d := parse_date(node[key])):
                return d
    # <time> elements: explicit itemprop anywhere, otherwise only inside article/main (conservative:
    # a wrong date, e.g. from a comment, is worse than no date).
    tag = soup.find("time", attrs={"itemprop": "datePublished"})
    if tag and (d := parse_date(tag.get("datetime") or tag.get_text())):
        return d
    for tag in soup.find_all("time", attrs={"datetime": True}):
        if tag.find_parent(["article", "main"]) and (d := parse_date(tag["datetime"])):
            return d
    return None


# ---------------------------------------------------------------------------------- author
def _clean_author(value) -> str | None:
    if not isinstance(value, str):
        return None
    v = clean(value)
    v = re.sub(r"^by\s+", "", v, flags=re.I).strip()
    if not v or v.lower().startswith(("http://", "https://")) or len(v) > 100:
        return None
    return v


def _jsonld_authors(value) -> list[str]:
    names: list[str] = []
    if isinstance(value, str):
        names.append(value)
    elif isinstance(value, dict):
        names.extend(_jsonld_authors(value.get("name")))
    elif isinstance(value, list):
        for item in value:
            names.extend(_jsonld_authors(item))
    return names


def find_author(soup: BeautifulSoup, metas: dict[str, str], jsonld: list[dict]) -> str | None:
    for key in ("author", "article:author", "dc.creator", "dc.contributor", "sailthru.author",
                "parsely-author"):
        if (a := _clean_author(metas.get(key))):
            return a
    for node in jsonld:
        names = [n for n in (_clean_author(x) for x in _jsonld_authors(node.get("author"))) if n]
        if names:
            return ", ".join(dict.fromkeys(names))[:200]
    tag = soup.find(attrs={"itemprop": "author"})
    if tag is not None:
        inner = tag.find(attrs={"itemprop": "name"})
        if (a := _clean_author((inner or tag).get_text())):
            return a
    link = soup.find("a", rel="author")
    if link is not None and (a := _clean_author(link.get_text())):
        return a
    return None


# ---------------------------------------------------------------------------------- language
def normalize_lang(value: str | None) -> str | None:
    if not value:
        return None
    primary = value.strip().lower().replace("_", "-").split("-")[0]
    return primary if re.fullmatch(r"[a-z]{2,3}", primary) else None


def detect_language(soup: BeautifulSoup, metas: dict[str, str], text: str) -> str | None:
    """Declared language (html lang / content-language / og:locale) wins; otherwise statistical
    detection on the extracted text (needs >= 50 characters)."""
    html = soup.find("html")
    declared = (
        normalize_lang(html.get("lang") if html is not None else None)
        or normalize_lang(metas.get("content-language"))
        or normalize_lang(metas.get("og:locale"))
    )
    if declared:
        return declared
    if len(text) < 50:
        return None
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0  # deterministic results
        return normalize_lang(detect(text[:5000]))
    except Exception:  # noqa: BLE001 - detection is best-effort
        return None
