"""Safe HTTP fetcher: manual redirects, SSRF checks on every hop, size/time limits, rate limiting.

The fetcher never bypasses access controls: 401/403/429/503 responses are reported and never
retried or worked around.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urljoin, urlsplit

import httpx

from app.config import Settings
from app.crawler.url_utils import (
    DNSResolutionError,
    Resolver,
    SSRFError,
    URLValidationError,
    assert_public_host,
    system_resolver,
    validate_url,
)
from app.crawler.pinned import PinnedConnectBlocked, make_pinned_transport
from app.storage.db import utcnow_iso

HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_CHARSET = re.compile(r"charset=([\w\-]+)", re.I)


@dataclass
class FetchResult:
    url: str
    final_url: str
    status_code: int | None = None
    content_type: str | None = None
    body: bytes = b""
    text: str = ""
    redirects: list[str] = field(default_factory=list)
    error_kind: str | None = None  # None = success
    error: str | None = None
    fetched_at: str = field(default_factory=utcnow_iso)

    @property
    def ok(self) -> bool:
        return self.error_kind is None


class FetchPolicy(Protocol):
    async def check(self, url: str) -> str | None:
        """Return None if the URL may be fetched, otherwise a human-readable reason."""

    async def delay(self, url: str) -> float:
        """Extra per-domain delay (e.g. robots Crawl-delay) in seconds."""


class DomainRateLimiter:
    """Enforces a minimum interval between requests to the same host."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._next: dict[str, float] = {}

    async def wait(self, host: str, delay: float) -> None:
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            loop = asyncio.get_running_loop()
            pause = self._next.get(host, 0.0) - loop.time()
            if pause > 0:
                await asyncio.sleep(pause)
            self._next[host] = loop.time() + max(delay, 0.0)


def _decode(body: bytes, content_type: str | None) -> str:
    charset = "utf-8"
    if content_type and (m := _CHARSET.search(content_type)):
        charset = m.group(1)
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


class SafeFetcher:
    def __init__(
        self,
        settings: Settings,
        *,
        resolver: Resolver = system_resolver,
        limiter: DomainRateLimiter | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._s = settings
        self._resolve = resolver
        self._limiter = limiter or DomainRateLimiter()
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.crawl_timeout_seconds),
            follow_redirects=False,
            transport=make_pinned_transport(resolver),   # connects only to IPs validated at connection time
            trust_env=False,                              # never route through env proxies: they would resolve DNS
            headers={
                "User-Agent": settings.user_agent,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _fail(url: str, current: str, kind: str, msg: str, redirects: list[str],
              status: int | None = None) -> FetchResult:
        return FetchResult(url=url, final_url=current, status_code=status, redirects=redirects,
                           error_kind=kind, error=msg)

    async def fetch(self, url: str, *, policy: FetchPolicy | None = None,
                    html_only: bool = True) -> FetchResult:
        redirects: list[str] = []
        visited: set[str] = set()
        current = url
        s = self._s

        for _hop in range(s.crawl_max_redirects + 1):
            # 1. syntactic validation + SSRF checks (every hop, including redirects)
            try:
                current = validate_url(current)
                host = urlsplit(current).hostname or ""
                await assert_public_host(host, self._resolve)
            except SSRFError as exc:
                return self._fail(url, current, "ssrf_blocked", str(exc), redirects)
            except URLValidationError as exc:
                return self._fail(url, current, "invalid_url", str(exc), redirects)
            except DNSResolutionError as exc:
                return self._fail(url, current, "dns_error", str(exc), redirects)

            if current in visited:
                return self._fail(url, current, "redirect_loop", "Redirect loop detected", redirects)
            visited.add(current)

            # 2. crawl policy (domain scope, robots.txt, restricted paths)
            extra_delay = 0.0
            if policy is not None:
                reason = await policy.check(current)
                if reason:
                    return self._fail(url, current, "blocked", reason, redirects)
                extra_delay = await policy.delay(current)

            # 3. per-domain politeness
            await self._limiter.wait(host, max(s.crawl_per_domain_delay_seconds, extra_delay))

            # 4. request (redirects handled manually so each hop is re-validated)
            try:
                async with asyncio.timeout(s.crawl_timeout_seconds * 3):
                    async with self._client.stream("GET", current) as resp:
                        status = resp.status_code
                        ctype_raw = resp.headers.get("content-type")
                        ctype = (ctype_raw or "").split(";", 1)[0].strip().lower()

                        if status in REDIRECT_STATUSES:
                            location = resp.headers.get("location")
                            if not location:
                                return self._fail(url, current, "http_error",
                                                  f"HTTP {status} without Location", redirects, status)
                            redirects.append(current)
                            current = urljoin(current, location)
                            continue

                        if status in (401, 403):
                            return self._fail(url, current, "restricted",
                                              f"HTTP {status}: access restricted", redirects, status)
                        if status in (404, 410):
                            return self._fail(url, current, "not_found", f"HTTP {status}", redirects, status)
                        if status == 429:
                            return self._fail(url, current, "rate_limited", "HTTP 429 Too Many Requests",
                                              redirects, status)
                        if status >= 400 or status < 200:
                            return self._fail(url, current, "http_error", f"HTTP {status}", redirects, status)

                        if html_only and ctype not in HTML_TYPES:
                            return self._fail(url, current, "non_html",
                                              f"Unsupported content type: {ctype or 'none'}", redirects, status)

                        declared = resp.headers.get("content-length", "")
                        if declared.isdigit() and int(declared) > s.crawl_max_page_bytes:
                            return self._fail(url, current, "too_large",
                                              f"Content-Length {declared} exceeds limit", redirects, status)

                        chunks: list[bytes] = []
                        size = 0
                        async for chunk in resp.aiter_bytes():
                            size += len(chunk)
                            if size > s.crawl_max_page_bytes:
                                return self._fail(url, current, "too_large",
                                                  "Response body exceeds size limit", redirects, status)
                            chunks.append(chunk)
                        body = b"".join(chunks)
                        return FetchResult(
                            url=url, final_url=current, status_code=status, content_type=ctype or None,
                            body=body, text=_decode(body, ctype_raw), redirects=redirects,
                        )
            except (httpx.TimeoutException, TimeoutError):
                return self._fail(url, current, "timeout", "Request timed out", redirects)
            except httpx.HTTPError as exc:
                if isinstance(exc.__cause__, PinnedConnectBlocked):
                    return self._fail(url, current, "ssrf_blocked", str(exc.__cause__), redirects)
                return self._fail(url, current, "network_error", f"{type(exc).__name__}: {exc}", redirects)

        return self._fail(url, current, "too_many_redirects",
                          f"More than {s.crawl_max_redirects} redirects", redirects)
