"""Link discovery and robots-meta handling (lightweight; full extraction lives in app/extraction)."""
from __future__ import annotations

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from app.crawler.url_utils import URLValidationError, normalize_url

_SKIP_PREFIXES = ("#", "mailto:", "tel:", "javascript:", "data:", "sms:", "ftp:")
_META_ROBOTS = re.compile(r"^robots$", re.I)


def robots_meta(soup: BeautifulSoup) -> tuple[bool, bool]:
    """Return (noindex, nofollow) from <meta name="robots">."""
    noindex = nofollow = False
    for tag in soup.find_all("meta", attrs={"name": _META_ROBOTS}):
        directives = {d.strip().lower() for d in (tag.get("content") or "").split(",")}
        noindex |= bool(directives & {"noindex", "none"})
        nofollow |= bool(directives & {"nofollow", "none"})
    return noindex, nofollow


def parse_page_links(html: str, base_url: str) -> tuple[list[str], bool, bool]:
    """Return (links, noindex, nofollow). `links` is empty when the page says nofollow."""
    soup = BeautifulSoup(html, "lxml")
    noindex, nofollow = robots_meta(soup)
    if nofollow:
        return [], noindex, nofollow

    base = base_url
    base_tag = soup.find("base", href=True)
    if base_tag:
        base = urljoin(base_url, base_tag["href"].strip())

    seen: dict[str, None] = {}
    for a in soup.find_all("a", href=True):
        rel = a.get("rel") or []
        if isinstance(rel, str):
            rel = rel.split()
        if any(r.lower() == "nofollow" for r in rel):
            continue
        href = a["href"].strip()
        if not href or href.lower().startswith(_SKIP_PREFIXES):
            continue
        try:
            seen[normalize_url(href, base)] = None
        except URLValidationError:
            continue
    return list(seen), noindex, nofollow


def extract_links(html: str, base_url: str) -> list[str]:
    return parse_page_links(html, base_url)[0]
