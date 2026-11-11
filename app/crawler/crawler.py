"""Asynchronous breadth-first crawler.

Flow per URL:  policy check (scope, robots.txt, restricted paths) -> safe fetch -> restricted/login
page detection -> duplicate-content check -> on_page callback -> enqueue discovered links.

`max_pages` bounds the number of distinct URLs the crawl will visit (successes and failures).
Persistence of pages is delegated to the `on_page` callback (wired up in later stages).
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from app.config import Settings, get_settings
from app.crawler.fetcher import FetchResult, SafeFetcher
from app.crawler.links import parse_page_links
from app.crawler.robots import RobotsPolicy
from app.crawler.url_utils import (
    Resolver,
    assert_public_host,
    collapse_whitespace,
    domain_allowed,
    is_restricted_path,
    looks_like_html_url,
    normalize_domain,
    system_resolver,
    validate_url,
)
from app.storage.db import utcnow_iso

log = logging.getLogger("goloka.crawler")

_PASSWORD_INPUT = re.compile(r"<input\b[^>]*\btype\s*=\s*[\"']?password", re.I)


@dataclass
class PageResult:
    url: str            # normalized URL that was requested
    final_url: str      # URL after redirects
    depth: int
    status_code: int
    html: str
    fetched_at: str
    content_hash: str   # hash of whitespace-normalized HTML (full content hash comes with extraction)


@dataclass
class UrlOutcome:
    url: str
    depth: int
    status: str         # fetched | duplicate | blocked | restricted | skipped | error
    http_status: int | None = None
    detail: str | None = None
    at: str = field(default_factory=utcnow_iso)


@dataclass
class CrawlSummary:
    seed_url: str
    max_pages: int
    max_depth: int
    allowed_domains: list[str]
    started_at: str
    finished_at: str | None = None
    outcomes: list[UrlOutcome] = field(default_factory=list)

    def count(self, *statuses: str) -> int:
        return sum(1 for o in self.outcomes if o.status in statuses)

    @property
    def pages_crawled(self) -> int:
        return self.count("fetched")

    @property
    def pages_failed(self) -> int:
        return self.count("error")

    @property
    def pages_skipped(self) -> int:
        return self.count("duplicate", "blocked", "restricted", "skipped")


PageCallback = Callable[[PageResult], Any]
OutcomeCallback = Callable[[UrlOutcome], Any]


async def _maybe_await(value: Any) -> None:
    if inspect.isawaitable(value):
        await value


class _Policy:
    """Fetch policy: domain scope + restricted paths + robots.txt."""

    def __init__(self, allowed: set[str], robots: RobotsPolicy) -> None:
        self._allowed = allowed
        self._robots = robots

    async def check(self, url: str) -> str | None:
        host = urlsplit(url).hostname or ""
        if not domain_allowed(host, self._allowed):
            return f"domain not allowed: {host}"
        if is_restricted_path(url):
            return "restricted path (login/account area)"
        if not await self._robots.allowed(url):
            return "disallowed by robots.txt"
        return None

    async def delay(self, url: str) -> float:
        return await self._robots.crawl_delay(url)


class Crawler:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        fetcher: SafeFetcher | None = None,
        resolver: Resolver = system_resolver,
        on_page: PageCallback | None = None,
        on_outcome: OutcomeCallback | None = None,
    ) -> None:
        self._s = settings or get_settings()
        self._resolver = resolver
        self._owns_fetcher = fetcher is None
        self._fetcher = fetcher or SafeFetcher(self._s, resolver=resolver)
        self._on_page = on_page
        self._on_outcome = on_outcome

    async def aclose(self) -> None:
        if self._owns_fetcher:
            await self._fetcher.aclose()

    async def crawl(
        self,
        seed_url: str,
        *,
        max_pages: int | None = None,
        max_depth: int | None = None,
        allowed_domains: list[str] | tuple[str, ...] = (),
    ) -> CrawlSummary:
        s = self._s
        seed = validate_url(seed_url)  # URLValidationError / SSRFError for bad seeds
        seed_host = urlsplit(seed).hostname or ""
        await assert_public_host(seed_host, self._resolver)

        pages_limit = s.crawl_default_max_pages if max_pages is None else max_pages
        depth_limit = s.crawl_default_max_depth if max_depth is None else max_depth
        if pages_limit < 1:
            raise ValueError("max_pages must be at least 1")
        if depth_limit < 0:
            raise ValueError("max_depth must be 0 or greater")
        pages_limit = min(pages_limit, s.crawl_hard_max_pages)
        depth_limit = min(depth_limit, s.crawl_hard_max_depth)
        allowed = {seed_host} | {normalize_domain(d) for d in allowed_domains}

        summary = CrawlSummary(seed, pages_limit, depth_limit, sorted(allowed), utcnow_iso())
        robots = RobotsPolicy(lambda u: self._fetcher.fetch(u, html_only=False), s.user_agent)
        policy = _Policy(allowed, robots)

        queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
        seen: set[str] = set()
        content_seen: set[str] = set()

        async def record(outcome: UrlOutcome) -> None:
            summary.outcomes.append(outcome)
            if self._on_outcome:
                try:
                    await _maybe_await(self._on_outcome(outcome))
                except Exception:  # noqa: BLE001 - callbacks must not kill the crawl
                    log.exception("on_outcome callback failed")

        def enqueue(url: str, depth: int) -> None:
            if url in seen or len(seen) >= pages_limit:
                return
            seen.add(url)
            queue.put_nowait((url, depth))

        def in_scope(url: str) -> bool:
            return (
                domain_allowed(urlsplit(url).hostname or "", allowed)
                and looks_like_html_url(url)
                and not is_restricted_path(url)
            )

        async def process(url: str, depth: int) -> None:
            res: FetchResult = await self._fetcher.fetch(url, policy=policy)
            if not res.ok:
                kind = res.error_kind
                if kind == "blocked" and (res.error or "").startswith("restricted"):
                    status = "restricted"
                elif kind == "restricted":
                    status = "restricted"
                elif kind in ("blocked", "ssrf_blocked"):
                    status = "blocked"
                elif kind in ("non_html", "invalid_url"):
                    status = "skipped"
                else:
                    status = "error"
                await record(UrlOutcome(url, depth, status, res.status_code, f"{kind}: {res.error}"))
                return

            final = res.final_url
            if final != url:
                if final in seen:
                    await record(UrlOutcome(url, depth, "duplicate", res.status_code,
                                            f"redirect target already seen: {final}"))
                    return
                seen.add(final)

            if _PASSWORD_INPUT.search(res.text):
                await record(UrlOutcome(url, depth, "restricted", res.status_code,
                                        "login form detected (not crawled)"))
                return

            content_hash = hashlib.sha256(collapse_whitespace(res.text).encode("utf-8")).hexdigest()
            if content_hash in content_seen:
                await record(UrlOutcome(url, depth, "duplicate", res.status_code, "duplicate content"))
                return
            content_seen.add(content_hash)

            links, noindex, _nofollow = parse_page_links(res.text, final)
            if noindex:
                await record(UrlOutcome(url, depth, "skipped", res.status_code, "meta robots noindex"))
            else:
                page = PageResult(url, final, depth, res.status_code or 200, res.text,
                                  res.fetched_at, content_hash)
                callback_error: Exception | None = None
                if self._on_page:
                    try:
                        await _maybe_await(self._on_page(page))
                    except Exception as exc:  # noqa: BLE001
                        log.exception("on_page callback failed for %s", url)
                        callback_error = exc
                if callback_error is None:
                    await record(UrlOutcome(url, depth, "fetched", res.status_code))
                else:  # keep discovering links even if storing this page failed
                    await record(UrlOutcome(url, depth, "error", res.status_code,
                                            f"on_page failed: {callback_error}"))

            if depth < depth_limit:
                for link in links:
                    if in_scope(link):
                        enqueue(link, depth + 1)

        async def worker() -> None:
            while True:
                url, depth = await queue.get()
                try:
                    await process(url, depth)
                except Exception as exc:  # noqa: BLE001
                    log.exception("unexpected crawl error for %s", url)
                    await record(UrlOutcome(url, depth, "error", None, f"internal: {type(exc).__name__}: {exc}"))
                finally:
                    queue.task_done()

        enqueue(seed, 0)
        workers = [asyncio.create_task(worker()) for _ in range(s.crawl_concurrency)]
        try:
            await queue.join()
        finally:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        summary.finished_at = utcnow_iso()
        return summary
