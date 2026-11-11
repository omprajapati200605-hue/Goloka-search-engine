"""HTML -> ExtractedPage.

Pipeline
  1. Parse with BeautifulSoup + lxml.
  2. From the pristine DOM: metadata (title, description, canonical, date, author, language
     declaration) and ALL links (navigation links are useful for discovery).
  3. Remove clutter: scripts, styles, nav, aside, footer, hidden elements, ARIA landmarks that are
     not content, page-level headers.
  4. Pick the main-content container (<main>, single <article>, role=main, common content ids,
     else <body>), then drop remaining boilerplate blocks (cookie banners, sidebars, share bars...)
     inside it, with guards so a wrapper that merely has e.g. "with-sidebar" in its class is kept.
  5. Extract headings, paragraphs and block-structured text; normalize whitespace.
  6. Content hash = SHA-256 of (title + normalized text), insensitive to whitespace changes.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, Tag
from bs4.element import CData, Comment, NavigableString, PageElement

from app.crawler.url_utils import URLValidationError, normalize_url
from app.extraction import metadata as md
from app.storage.db import utcnow_iso

MAX_LINKS = 1000

_DROP_TAGS = [
    "script", "style", "noscript", "template", "iframe", "object", "embed", "svg", "canvas", "nav",
    "aside", "footer", "input", "button", "select", "textarea", "option", "datalist", "dialog",
    "audio", "video", "map",
]
_DROP_ROLES = {"navigation", "contentinfo", "complementary", "search", "dialog", "alertdialog"}
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)
_BOILERPLATE_TOKENS = {
    "cookie", "cookies", "gdpr", "consent", "advert", "advertisement", "ads", "adsbygoogle", "popup",
    "modal", "newsletter", "subscribe", "breadcrumb", "breadcrumbs", "sidebar", "comments", "related",
    "social", "share", "sharing", "navbar", "nav", "toolbar", "pagination", "pager", "sponsored",
    "promo",
}
_BLOCK_TAGS = [
    "p", "div", "section", "article", "main", "header", "footer", "aside", "nav", "li", "ul", "ol", "dl",
    "dt", "dd", "tr", "td", "th", "table", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre",
    "figure", "figcaption", "hr", "address", "details", "summary", "form", "fieldset",
]
_CONTENT_IDS = ("main-content", "content", "main", "primary", "article-body", "post-content")
_ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")
_WS = re.compile(r"\s+")


@dataclass
class ExtractedPage:
    url: str                       # original (requested) URL
    final_url: str                 # URL after redirects
    domain: str
    title: str
    text: str                      # main readable content
    meta_description: str | None
    canonical_url: str | None
    headings: list[dict]           # [{"level": 1, "text": "..."}]
    paragraphs: list[str]
    published_date: str | None
    author: str | None
    language: str | None
    internal_links: list[str]
    external_links: list[str]
    content_hash: str
    crawled_at: str
    extra: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def has_content(self) -> bool:
        return bool(self.text.strip())

    def metadata(self) -> dict:
        """JSON-serialisable metadata stored alongside the document."""
        return {
            "final_url": self.final_url,
            "meta_description": self.meta_description,
            "canonical_url": self.canonical_url,
            "headings": self.headings,
            "published_date": self.published_date,
            "author": self.author,
            "language": self.language,
            "internal_links": self.internal_links,
            "external_links": self.external_links,
            "word_count": self.word_count,
        }

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------------------------ helpers
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\xa0", " ")
    return _WS.sub(" ", _ZERO_WIDTH.sub("", text)).strip()


def compute_content_hash(title: str, text: str) -> str:
    payload = f"{normalize_text(title)}\n{normalize_text(text)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _site_key(host: str) -> str:
    return host[4:] if host.startswith("www.") else host


def _collect_links(soup: BeautifulSoup, page_url: str, page_host: str) -> tuple[list[str], list[str]]:
    base = page_url
    base_tag = soup.find("base", href=True)
    if base_tag:
        try:
            base = normalize_url(base_tag["href"].strip(), page_url)
        except URLValidationError:
            pass
    internal: dict[str, None] = {}
    external: dict[str, None] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.lower().startswith(("#", "mailto:", "tel:", "javascript:", "data:", "sms:")):
            continue
        try:
            link = normalize_url(href, base)
        except URLValidationError:
            continue
        if not link.startswith(("http://", "https://")) or link == page_url:
            continue
        host = urlsplit(link).hostname or ""
        (internal if _site_key(host) == _site_key(page_host) else external)[link] = None
    return list(internal)[:MAX_LINKS], list(external)[:MAX_LINKS]


def _tokens(tag: Tag) -> set[str]:
    classes = tag.get("class") or []
    classes = classes.split() if isinstance(classes, str) else classes
    raw = " ".join(classes) + " " + str(tag.get("id") or "")
    return {t for t in re.split(r"[^a-z0-9]+", raw.lower()) if t}


def _text_len(tag: Tag) -> int:
    return len(tag.get_text(" ", strip=True))


def _strip_definite_clutter(root: Tag) -> None:
    for node in root.find_all(string=lambda s: isinstance(s, Comment)):
        node.extract()
    for tag in root.find_all(_DROP_TAGS):
        tag.extract()
    for tag in list(root.find_all(True)):
        if tag.name in ("html", "body"):
            continue
        attrs = tag.attrs
        style = attrs.get("style")
        aria_hidden = str(attrs.get("aria-hidden", "")).strip().lower() == "true"
        if "hidden" in attrs or aria_hidden or (isinstance(style, str) and _HIDDEN_STYLE.search(style)):
            tag.extract()
            continue
        role = str(attrs.get("role", "")).strip().lower()
        if role in _DROP_ROLES:
            tag.extract()
        elif (tag.name == "header" or role == "banner") and not tag.find_parent(["article", "main"]):
            tag.extract()


def _pick_container(soup: BeautifulSoup) -> Tag:
    body = soup.body or soup
    candidate: Tag | None = None
    mains = body.find_all("main")
    if len(mains) == 1:
        candidate = mains[0]
    if candidate is None:
        roles = body.find_all(attrs={"role": "main"})
        if len(roles) == 1:
            candidate = roles[0]
    if candidate is None:
        articles = body.find_all("article")
        if len(articles) == 1:
            candidate = articles[0]
    if candidate is None:
        bodies = body.find_all(attrs={"itemprop": "articleBody"})
        if len(bodies) == 1:
            candidate = bodies[0]
    if candidate is None:
        for element_id in _CONTENT_IDS:
            found = body.find(id=element_id)
            if found is not None:
                candidate = found
                break
    if candidate is not None and _text_len(candidate) >= 0.25 * _text_len(body):
        return candidate
    return body


def _strip_boilerplate_blocks(container: Tag) -> None:
    total = max(_text_len(container), 1)
    for tag in list(container.find_all(True)):
        if tag is container or tag.name in ("article", "main"):
            continue
        if not (_tokens(tag) & _BOILERPLATE_TOKENS):
            continue
        if tag.find(["main", "article", "h1"]) or _text_len(tag) > 0.6 * total:
            continue  # wrapper that holds real content (e.g. class="content-with-sidebar")
        tag.extract()


_BREAK = "\x1e"  # explicit block boundary; raw newlines in the source HTML are plain whitespace
_BLOCK_SET = frozenset(_BLOCK_TAGS)


def _block_lines(container: Tag) -> list[str]:
    """Read-only, iterative walk that emits text with a break marker at block boundaries
    (mutating the tree per block tag is quadratic on pages with thousands of siblings)."""
    parts: list[str] = []
    stack: list[tuple[PageElement, bool]] = [(container, False)]
    while stack:
        node, closing = stack.pop()
        if closing:
            parts.append(_BREAK)
        elif isinstance(node, Tag):
            if node.name == "br":
                parts.append(_BREAK)
                continue
            if node.name in _BLOCK_SET:
                parts.append(_BREAK)
                stack.append((node, True))
            stack.extend((child, False) for child in reversed(node.contents))
        elif type(node) in (NavigableString, CData):
            parts.append(str(node))
    lines: list[str] = []
    for raw in "".join(parts).split(_BREAK):
        line = normalize_text(raw)
        if line and (not lines or lines[-1] != line):
            lines.append(line)
    return lines


# ------------------------------------------------------------------------------------ public API
def extract_page(
    html: str,
    url: str,
    *,
    final_url: str | None = None,
    crawled_at: str | None = None,
) -> ExtractedPage:
    """Extract a structured page from raw HTML. Never raises on malformed or empty HTML."""
    final = normalize_url(final_url or url)
    page_host = urlsplit(final).hostname or ""
    soup = BeautifulSoup(html or "", "lxml")

    # ---- pristine-DOM extraction
    metas = md.collect_meta(soup)
    jsonld = md.extract_jsonld(soup)
    title_tag = soup.find("title")
    first_h1 = soup.find("h1")
    title = (
        normalize_text(title_tag.get_text()) if title_tag else ""
    ) or metas.get("og:title", "") or (normalize_text(first_h1.get_text()) if first_h1 else "")
    description = md.find_description(metas)
    canonical = md.find_canonical(soup, final)
    published = md.find_published(soup, metas, jsonld)
    author = md.find_author(soup, metas, jsonld)
    internal_links, external_links = _collect_links(soup, final, page_host)

    # ---- content extraction
    _strip_definite_clutter(soup)
    container = _pick_container(soup)
    _strip_boilerplate_blocks(container)

    headings = [
        {"level": int(h.name[1]), "text": t}
        for h in container.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        if (t := normalize_text(h.get_text()))
    ]
    paragraphs = [
        t for p in container.find_all("p") if not p.find_parent("p") and (t := normalize_text(p.get_text()))
    ]
    text = "\n".join(_block_lines(container))
    language = md.detect_language(soup, metas, text)

    return ExtractedPage(
        url=url,
        final_url=final,
        domain=page_host,
        title=title,
        text=text,
        meta_description=description,
        canonical_url=canonical,
        headings=headings,
        paragraphs=paragraphs,
        published_date=published,
        author=author,
        language=language,
        internal_links=internal_links,
        external_links=external_links,
        content_hash=compute_content_hash(title, text),
        crawled_at=crawled_at or utcnow_iso(),
    )
