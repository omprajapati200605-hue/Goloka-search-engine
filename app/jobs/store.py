"""Persistence for crawl jobs (tables crawl_jobs / crawl_urls). Thread-safe: one connection + RLock."""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.storage.db import init_db, utcnow_iso

PENDING, RUNNING, COMPLETED, FAILED = "pending", "running", "completed", "failed"
ACTIVE = (PENDING, RUNNING)


@dataclass
class Job:
    id: str
    seed_url: str
    status: str
    phase: str                       # queued | crawling | indexing | done
    max_pages: int
    max_depth: int
    allowed_domains: list[str]
    auto_index: bool
    urls_processed: int
    pages_crawled: int
    pages_failed: int
    pages_skipped: int
    stored: dict[str, int] = field(default_factory=dict)
    index_result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def finished(self) -> bool:
        return self.status in (COMPLETED, FAILED)


def _row_to_job(r) -> Job:
    return Job(
        id=r["id"], seed_url=r["seed_url"], status=r["status"], phase=r["phase"], max_pages=r["max_pages"],
        max_depth=r["max_depth"], allowed_domains=json.loads(r["allowed_domains"] or "[]"),
        auto_index=bool(r["auto_index"]), urls_processed=r["urls_processed"], pages_crawled=r["pages_crawled"],
        pages_failed=r["pages_failed"], pages_skipped=r["pages_skipped"],
        stored=json.loads(r["stored"]) if r["stored"] else {},
        index_result=json.loads(r["index_result"]) if r["index_result"] else None,
        error=r["error"], created_at=r["created_at"], started_at=r["started_at"], finished_at=r["finished_at"])


class JobStore:
    def __init__(self, db_path: str | Path) -> None:
        self._conn = init_db(db_path)
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create(self, seed_url: str, max_pages: int, max_depth: int, allowed_domains: list[str],
               auto_index: bool) -> Job:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._conn.execute(
                "INSERT INTO crawl_jobs (id, seed_url, status, max_pages, max_depth, allowed_domains, auto_index, "
                "phase, created_at) VALUES (?, ?, 'pending', ?, ?, ?, ?, 'queued', ?)",
                (job_id, seed_url, max_pages, max_depth, json.dumps(allowed_domains), int(auto_index), utcnow_iso()))
            self._conn.commit()
        return self.get(job_id)  # type: ignore[return-value]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM crawl_jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def list(self, *, status: str | None = None, limit: int = 50, offset: int = 0) -> tuple[list[Job], int]:
        where, params = ("WHERE status = ?", [status]) if status else ("", [])
        with self._lock:
            total = self._conn.execute(f"SELECT COUNT(*) AS c FROM crawl_jobs {where}", params).fetchone()["c"]
            rows = self._conn.execute(
                f"SELECT * FROM crawl_jobs {where} ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
                [*params, limit, offset]).fetchall()
        return [_row_to_job(r) for r in rows], total

    def count_active(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) AS c FROM crawl_jobs WHERE status IN ('pending','running')").fetchone()["c"]

    def counts_by_status(self) -> dict[str, int]:
        with self._lock:
            return {r["status"]: r["c"] for r in self._conn.execute(
                "SELECT status, COUNT(*) AS c FROM crawl_jobs GROUP BY status")}

    def pending_ids(self) -> list[str]:
        with self._lock:
            return [r["id"] for r in self._conn.execute(
                "SELECT id FROM crawl_jobs WHERE status = 'pending' ORDER BY created_at, id")]

    def mark_running(self, job_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE crawl_jobs SET status='running', phase='crawling', started_at=? "
                "WHERE id=? AND status='pending'", (utcnow_iso(), job_id))
            self._conn.commit()
            return cur.rowcount == 1

    def update_progress(self, job_id: str, *, processed: int, crawled: int, failed: int, skipped: int,
                        stored: dict[str, int] | None = None, phase: str | None = None) -> None:
        """Counters only ever grow, so an out-of-order write can never move progress backwards."""
        with self._lock:
            self._conn.execute(
                "UPDATE crawl_jobs SET urls_processed=MAX(urls_processed, ?), pages_crawled=MAX(pages_crawled, ?), "
                "pages_failed=MAX(pages_failed, ?), pages_skipped=MAX(pages_skipped, ?), "
                "stored=COALESCE(?, stored), phase=COALESCE(?, phase) WHERE id=? AND status='running'",
                (processed, crawled, failed, skipped, json.dumps(stored) if stored is not None else None,
                 phase, job_id))
            self._conn.commit()

    def finish(self, job_id: str, status: str, *, error: str | None = None,
               index_result: dict[str, Any] | None = None) -> None:
        if status not in (COMPLETED, FAILED):
            raise ValueError(f"not a final status: {status}")
        with self._lock:
            self._conn.execute(
                "UPDATE crawl_jobs SET status=?, phase='done', error=?, index_result=COALESCE(?, index_result), "
                "finished_at=? WHERE id=?",
                (status, error, json.dumps(index_result) if index_result is not None else None,
                 utcnow_iso(), job_id))
            self._conn.commit()

    def fail_interrupted(self, reason: str) -> int:
        """Jobs left 'running' by a stopped server can never finish: mark them failed with the reason."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE crawl_jobs SET status='failed', phase='done', error=?, finished_at=? WHERE status='running'",
                (reason, utcnow_iso()))
            self._conn.commit()
            return cur.rowcount

    # ---- per-URL outcomes
    def record_url(self, job_id: str, url: str, depth: int, status: str, http_status: int | None,
                   detail: str | None, at: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO crawl_urls (job_id, url, depth, status, http_status, detail, fetched_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)", (job_id, url, depth, status, http_status, detail, at))
            self._conn.commit()

    def list_urls(self, job_id: str, *, status: str | None = None, limit: int = 100,
                  offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        where, params = "WHERE job_id = ?", [job_id]
        if status:
            where += " AND status = ?"
            params.append(status)
        with self._lock:
            total = self._conn.execute(f"SELECT COUNT(*) AS c FROM crawl_urls {where}", params).fetchone()["c"]
            rows = self._conn.execute(
                f"SELECT url, depth, status, http_status, detail, fetched_at FROM crawl_urls {where} "
                "ORDER BY fetched_at, url LIMIT ? OFFSET ?", [*params, limit, offset]).fetchall()
        return [dict(r) for r in rows], total
