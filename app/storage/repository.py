"""Persistent document storage (SQLite now; plain SQL so PostgreSQL migration stays easy).

Key behaviours
  * Stable IDs: uuid5(URL), so re-crawling the same URL always maps to the same document.
  * Change detection: identical content_hash -> only `crawled_at` is refreshed ("unchanged").
    Changed content -> text/metadata replaced, `last_updated_at` bumped, `indexed_hash` kept so the
    document shows up in `needs_indexing()` until the indexers call `mark_indexed()`.
  * Cross-URL duplicate content for a NEW URL is not stored ("duplicate").
  * Pages without readable text are not stored ("empty").
  * Thread-safe: one connection guarded by an RLock (the API and crawl workers use threads).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from app.extraction.extractor import ExtractedPage
from app.storage.db import init_db, utcnow_iso

STATUS_OK = "ok"
_COLUMNS = ("id, url, title, domain, text, metadata, content_hash, crawled_at, last_updated_at, "
            "crawl_status, indexed_hash")
_COLUMNS_NO_TEXT = _COLUMNS.replace(" text,", " '' AS text,")


def document_id_for(url: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, url))


@dataclass
class Document:
    id: str
    url: str
    title: str
    domain: str
    text: str
    metadata: dict[str, Any]
    content_hash: str
    crawled_at: str
    last_updated_at: str
    crawl_status: str
    indexed_hash: str | None = None

    @property
    def needs_indexing(self) -> bool:
        return self.crawl_status == STATUS_OK and self.indexed_hash != self.content_hash

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        d = {
            "id": self.id, "url": self.url, "title": self.title, "domain": self.domain,
            "metadata": self.metadata, "content_hash": self.content_hash,
            "crawled_at": self.crawled_at, "last_updated_at": self.last_updated_at,
            "crawl_status": self.crawl_status, "indexed": not self.needs_indexing,
        }
        if include_text:
            d["text"] = self.text
        return d


@dataclass
class UpsertResult:
    action: str  # created | updated | unchanged | duplicate | empty
    document: Document | None = None
    duplicate_of: str | None = None

    @property
    def stored(self) -> bool:
        return self.action in ("created", "updated", "unchanged")


@dataclass
class StorageStats:
    total: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    indexed: int = 0
    pending_indexing: int = 0
    last_crawled_at: str | None = None


def _row_to_doc(row: sqlite3.Row) -> Document:
    return Document(
        id=row["id"], url=row["url"], title=row["title"], domain=row["domain"], text=row["text"],
        metadata=json.loads(row["metadata"] or "{}"), content_hash=row["content_hash"],
        crawled_at=row["crawled_at"], last_updated_at=row["last_updated_at"],
        crawl_status=row["crawl_status"], indexed_hash=row["indexed_hash"],
    )


class DocumentRepository:
    def __init__(self, db_path: str | Path, *, clock: Callable[[], str] = utcnow_iso) -> None:
        self._conn = init_db(db_path)
        self._lock = threading.RLock()
        self._clock = clock

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "DocumentRepository":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ writes
    def upsert_page(self, page: ExtractedPage, *, crawl_status: str = STATUS_OK) -> UpsertResult:
        """Insert or refresh the document for `page.final_url`."""
        if not page.has_content:
            return UpsertResult("empty")
        url = page.final_url
        doc_id = document_id_for(url)
        metadata = page.metadata()
        metadata["requested_url"] = page.url
        meta_json = json.dumps(metadata, ensure_ascii=False)
        now = self._clock()

        with self._lock, self._conn:
            existing = self._conn.execute(
                f"SELECT {_COLUMNS_NO_TEXT} FROM documents WHERE id = ?", (doc_id,)).fetchone()

            if existing is None:
                dup = self._conn.execute(
                    "SELECT id FROM documents WHERE content_hash = ? AND crawl_status = ? LIMIT 1",
                    (page.content_hash, STATUS_OK)).fetchone()
                if dup is not None:
                    return UpsertResult("duplicate", duplicate_of=dup["id"])
                self._conn.execute(
                    "INSERT INTO documents (id, url, title, domain, text, metadata, content_hash, "
                    "crawled_at, last_updated_at, crawl_status, indexed_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                    (doc_id, url, page.title, page.domain, page.text, meta_json, page.content_hash,
                     page.crawled_at, now, crawl_status))
                return UpsertResult("created", self.get(doc_id))

            revived = existing["crawl_status"] != STATUS_OK and crawl_status == STATUS_OK
            if existing["content_hash"] == page.content_hash and not revived:
                self._conn.execute(
                    "UPDATE documents SET crawled_at = ?, crawl_status = ? WHERE id = ?",
                    (page.crawled_at, crawl_status, doc_id))
                return UpsertResult("unchanged", self.get(doc_id))

            # content changed (or a previously gone page came back): replace and force re-indexing
            self._conn.execute(
                "UPDATE documents SET title = ?, domain = ?, text = ?, metadata = ?, content_hash = ?, "
                "crawled_at = ?, last_updated_at = ?, crawl_status = ?, "
                "indexed_hash = CASE WHEN ? THEN NULL ELSE indexed_hash END WHERE id = ?",
                (page.title, page.domain, page.text, meta_json, page.content_hash, page.crawled_at,
                 now, crawl_status, 1 if revived else 0, doc_id))
            return UpsertResult("updated", self.get(doc_id))

    def mark_status(self, url: str, status: str, *, crawled_at: str | None = None) -> bool:
        """Set crawl_status for a URL (e.g. 'gone' after a 404). Returns True if a row changed."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE documents SET crawl_status = ?, crawled_at = COALESCE(?, crawled_at) WHERE id = ?",
                (status, crawled_at, document_id_for(url)))
            return cur.rowcount > 0

    def mark_indexed(self, document_id: str, content_hash: str) -> bool:
        """Record that `content_hash` has been indexed. Ignored if the content changed meanwhile."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE documents SET indexed_hash = ? WHERE id = ? AND content_hash = ?",
                (content_hash, document_id, content_hash))
            return cur.rowcount > 0

    def delete(self, document_id: str) -> bool:
        with self._lock, self._conn:
            return self._conn.execute("DELETE FROM documents WHERE id = ?", (document_id,)).rowcount > 0

    # ------------------------------------------------------------------ reads
    def get(self, document_id: str) -> Document | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_COLUMNS} FROM documents WHERE id = ?", (document_id,)).fetchone()
        return _row_to_doc(row) if row else None

    def get_by_url(self, url: str) -> Document | None:
        return self.get(document_id_for(url))

    def get_many(self, ids: list[str]) -> dict[str, Document]:
        if not ids:
            return {}
        out: dict[str, Document] = {}
        with self._lock:
            for i in range(0, len(ids), 500):  # stay below SQLite's variable limit
                chunk = ids[i:i + 500]
                marks = ",".join("?" * len(chunk))
                for row in self._conn.execute(
                        f"SELECT {_COLUMNS} FROM documents WHERE id IN ({marks})", chunk):
                    out[row["id"]] = _row_to_doc(row)
        return out

    def list(self, *, limit: int = 50, offset: int = 0, domain: str | None = None,
             status: str | None = None, include_text: bool = False) -> tuple[list[Document], int]:
        """Return (page of documents newest-crawled first, total matching)."""
        where, params = [], []
        if domain:
            where.append("domain = ?")
            params.append(domain.lower())
        if status:
            where.append("crawl_status = ?")
            params.append(status)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        cols = _COLUMNS if include_text else _COLUMNS_NO_TEXT
        with self._lock:
            total = self._conn.execute(f"SELECT COUNT(*) AS c FROM documents{clause}", params).fetchone()["c"]
            rows = self._conn.execute(
                f"SELECT {cols} FROM documents{clause} ORDER BY crawled_at DESC, id LIMIT ? OFFSET ?",
                [*params, max(limit, 0), max(offset, 0)]).fetchall()
        return [_row_to_doc(r) for r in rows], total

    def count(self, status: str | None = STATUS_OK) -> int:
        with self._lock:
            if status is None:
                return self._conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"]
            return self._conn.execute(
                "SELECT COUNT(*) AS c FROM documents WHERE crawl_status = ?", (status,)).fetchone()["c"]

    def needs_indexing(self, limit: int | None = None) -> list[Document]:
        """Searchable documents whose current content has not been indexed yet."""
        sql = (f"SELECT {_COLUMNS} FROM documents WHERE crawl_status = ? "
               "AND (indexed_hash IS NULL OR indexed_hash != content_hash) ORDER BY crawled_at, id")
        params: list[Any] = [STATUS_OK]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            return [_row_to_doc(r) for r in self._conn.execute(sql, params)]

    def iter_documents(self, *, status: str | None = STATUS_OK, batch_size: int = 200) -> Iterator[Document]:
        """Stream documents in stable id order (used to rebuild indexes after a restart)."""
        last_id = ""
        while True:
            sql = f"SELECT {_COLUMNS} FROM documents WHERE id > ?"
            params: list[Any] = [last_id]
            if status is not None:
                sql += " AND crawl_status = ?"
                params.append(status)
            sql += " ORDER BY id LIMIT ?"
            params.append(batch_size)
            with self._lock:
                rows = self._conn.execute(sql, params).fetchall()
            if not rows:
                return
            for row in rows:
                yield _row_to_doc(row)
            last_id = rows[-1]["id"]

    def stats(self) -> StorageStats:
        with self._lock:
            by_status = {r["crawl_status"]: r["c"] for r in self._conn.execute(
                "SELECT crawl_status, COUNT(*) AS c FROM documents GROUP BY crawl_status")}
            indexed = self._conn.execute(
                "SELECT COUNT(*) AS c FROM documents WHERE crawl_status = ? AND indexed_hash = content_hash",
                (STATUS_OK,)).fetchone()["c"]
            last = self._conn.execute("SELECT MAX(crawled_at) AS m FROM documents").fetchone()["m"]
        ok = by_status.get(STATUS_OK, 0)
        return StorageStats(total=sum(by_status.values()), by_status=by_status, indexed=indexed,
                            pending_indexing=ok - indexed, last_crawled_at=last)
