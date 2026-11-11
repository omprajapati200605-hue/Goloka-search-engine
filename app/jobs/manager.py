"""Runs crawl jobs in the background: pending -> running -> completed | failed.

One worker takes jobs from a FIFO queue, so two jobs never crawl the same site at the same time (the per-domain
politeness delay lives inside one crawl). Each job: crawl -> extract -> store -> (optionally) index. Progress,
per-URL outcomes and errors are written to SQLite as the job runs, so they survive a restart; a job that was
running when the server stopped is marked failed ("interrupted"), and jobs still pending are run again.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from app.config import Settings
from app.crawler.crawler import Crawler, UrlOutcome
from app.crawler.url_utils import (DNSResolutionError, Resolver, URLValidationError, assert_public_host,
                                   normalize_domain, system_resolver, validate_url)
from urllib.parse import urlsplit

from app.engine import IndexingInProgress
from app.ingest import IngestStats, make_ingest_callback
from app.jobs.store import COMPLETED, FAILED, Job, JobStore
from app.storage.repository import DocumentRepository

log = logging.getLogger("goloka.jobs")

Indexer = Callable[[], object]          # SearchEngine.index_pending (blocking); returns an IndexReport


class JobRejected(Exception):
    """The job could not be accepted. `code` is a stable machine-readable reason."""

    def __init__(self, code: str, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.code, self.message, self.status_code = code, message, status_code


class JobManager:
    def __init__(self, settings: Settings, repo: DocumentRepository, store: JobStore, indexer: Indexer, *,
                 resolver: Resolver = system_resolver, index_retry_seconds: float = 0.5,
                 index_max_wait_seconds: float = 120.0) -> None:
        self._s, self._repo, self._store, self._indexer = settings, repo, store, indexer
        self._resolver = resolver
        self._retry, self._max_wait = index_retry_seconds, index_max_wait_seconds
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._current: str | None = None

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        interrupted = self._store.fail_interrupted("interrupted: the server stopped while this job was running")
        if interrupted:
            log.warning("marked %d interrupted job(s) as failed", interrupted)
        for job_id in self._store.pending_ids():
            self._queue.put_nowait(job_id)
        self._worker = asyncio.create_task(self._work(), name="goloka-job-worker")

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None

    # ------------------------------------------------------------------ submitting
    async def submit(self, url: str, *, max_pages: int | None = None, max_depth: int | None = None,
                     allowed_domains: list[str] | None = None, auto_index: bool = True) -> Job:
        s = self._s
        try:
            seed = validate_url(url)
            await assert_public_host(urlsplit(seed).hostname or "", self._resolver)
        except URLValidationError as exc:
            raise JobRejected("unsafe_url", str(exc)) from exc
        except DNSResolutionError as exc:
            raise JobRejected("dns_error", f"Could not resolve the host: {exc}") from exc
        pages = s.crawl_default_max_pages if max_pages is None else max_pages
        depth = s.crawl_default_max_depth if max_depth is None else max_depth
        if not 1 <= pages <= s.crawl_hard_max_pages:
            raise JobRejected("invalid_limit", f"max_pages must be between 1 and {s.crawl_hard_max_pages}")
        if not 0 <= depth <= s.crawl_hard_max_depth:
            raise JobRejected("invalid_limit", f"max_depth must be between 0 and {s.crawl_hard_max_depth}")
        try:
            domains = sorted({normalize_domain(d) for d in allowed_domains or []})
        except ValueError as exc:
            raise JobRejected("invalid_domain", str(exc)) from exc
        if self._store.count_active() >= s.jobs_max_pending:
            raise JobRejected("queue_full", f"Too many unfinished jobs (limit {s.jobs_max_pending}); try later", 429)
        job = self._store.create(seed, pages, depth, domains, auto_index)
        self._queue.put_nowait(job.id)
        return job

    async def wait_idle(self, timeout: float = 30.0) -> None:
        """Wait until the queue is drained and nothing is running (used by tests and graceful shutdown)."""
        await asyncio.wait_for(self._queue.join(), timeout)

    # ------------------------------------------------------------------ worker
    async def _work(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self._run(job_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one broken job must not stop the worker
                log.exception("job %s crashed", job_id)
                self._store.finish(job_id, FAILED, error=f"internal error ({type(exc).__name__}); see server log")
            finally:
                self._queue.task_done()

    async def _run(self, job_id: str) -> None:
        job = self._store.get(job_id)
        if job is None or not self._store.mark_running(job_id):
            return
        self._current = job_id
        stats = IngestStats()
        counts = {"processed": 0, "crawled": 0, "failed": 0, "skipped": 0}
        persist_lock = asyncio.Lock()

        async def persist(phase: str | None = None) -> None:
            async with persist_lock:  # snapshot taken inside the lock so writes stay in order
                snap = dict(counts)
                await asyncio.to_thread(self._store.update_progress, job_id, processed=snap["processed"],
                                        crawled=snap["crawled"], failed=snap["failed"], skipped=snap["skipped"],
                                        stored=stats.as_dict(), phase=phase)

        async def on_outcome(o: UrlOutcome) -> None:
            counts["processed"] += 1
            if o.status == "fetched":
                counts["crawled"] += 1
            elif o.status == "error":
                counts["failed"] += 1
            else:
                counts["skipped"] += 1
            await asyncio.to_thread(self._store.record_url, job_id, o.url, o.depth, o.status, o.http_status,
                                    o.detail, o.at)
            await persist()

        crawler = Crawler(self._s, resolver=self._resolver, on_page=make_ingest_callback(self._repo, stats),
                          on_outcome=on_outcome)
        first_problem: str | None = None
        try:
            summary = await crawler.crawl(job.seed_url, max_pages=job.max_pages, max_depth=job.max_depth,
                                          allowed_domains=job.allowed_domains)
            first_problem = next((f"{o.url}: {o.detail}" for o in summary.outcomes if o.status != "fetched"), None)
        except (URLValidationError, DNSResolutionError, ValueError) as exc:
            await persist()
            self._store.finish(job_id, FAILED, error=f"crawl could not start: {exc}")
            return
        except asyncio.CancelledError:
            self._store.finish(job_id, FAILED, error="interrupted: the server stopped while this job was running")
            raise
        finally:
            await crawler.aclose()
            self._current = None

        await persist()
        if counts["crawled"] == 0:
            reason = first_problem or "the crawler found nothing to fetch"
            self._store.finish(job_id, FAILED, error=f"no pages could be crawled; first problem: {reason}")
            return

        index_result = None
        if job.auto_index:
            await persist("indexing")
            index_result = await self._index()
        self._store.finish(job_id, COMPLETED, index_result=index_result)

    async def _index(self) -> dict:
        """Run the indexer, waiting if a manual /index run is active. Indexing trouble never fails the crawl."""
        waited = 0.0
        while True:
            try:
                report = await asyncio.to_thread(self._indexer)
                return {"status": report.status, "pending": report.pending, "warnings": report.warnings,
                        "keyword": report.keyword, "vectors": report.vectors}
            except IndexingInProgress:
                if waited >= self._max_wait:
                    return {"status": "skipped", "error": "another indexing run was active; call POST /api/v1/index"}
                await asyncio.sleep(self._retry)
                waited += self._retry
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.exception("automatic indexing failed")
                return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
