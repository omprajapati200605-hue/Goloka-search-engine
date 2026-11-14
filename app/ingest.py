"""Glue: Crawler -> HTML extraction -> Document storage.

Use `make_ingest_callback(repo)` as the crawler's `on_page` callback. Extraction (CPU-bound) and the
SQLite write run in a worker thread so the crawler's event loop stays responsive.
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from app.crawler.crawler import PageResult
from app.extraction.extractor import extract_page
from app.storage.repository import DocumentRepository, UpsertResult


@dataclass
class IngestStats:
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    duplicate: int = 0
    empty: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, result: UpsertResult) -> None:
        with self._lock:
            setattr(self, result.action, getattr(self, result.action) + 1)

    @property
    def stored(self) -> int:
        return self.created + self.updated + self.unchanged

    def as_dict(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in ("created", "updated", "unchanged", "duplicate", "empty")}


# Parsing a large page (Wikipedia pages are 0.5-2 MB of HTML) briefly needs ~75 MB. Several crawl workers parsing
# at once would multiply that and run a 512 MB server out of memory, so parsing is done one page at a time. It
# costs nothing in speed: parsing is CPU-bound and already shares one core.
_PARSE_LOCK = threading.Lock()


def ingest_page(repo: DocumentRepository, page: PageResult) -> UpsertResult:
    """Extract and store one crawled page (synchronous)."""
    with _PARSE_LOCK:
        extracted = extract_page(page.html, page.url, final_url=page.final_url, crawled_at=page.fetched_at)
    return repo.upsert_page(extracted)


def make_ingest_callback(
    repo: DocumentRepository, stats: IngestStats | None = None
) -> Callable[[PageResult], Awaitable[UpsertResult]]:
    async def on_page(page: PageResult) -> UpsertResult:
        result = await asyncio.to_thread(ingest_page, repo, page)
        if stats is not None:
            stats.record(result)
        return result

    return on_page
