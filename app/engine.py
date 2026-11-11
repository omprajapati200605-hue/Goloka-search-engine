"""SearchEngine: the one object the API talks to.

It owns the SQLite repository, the BM25 index, the vector index and the hybrid searcher, and knows how to
bring the indexes up to date (`index_pending`). Nothing here starts a crawl; crawling arrives with the
background-job system and simply stores documents that `index_pending` then indexes.

Concurrency: FastAPI runs the (synchronous) route handlers in a thread pool. One re-entrant lock guards the
in-memory indexes, so a search never sees a half-updated index; a second `index_pending` call while one is
running is refused (`IndexingInProgress`) instead of queueing behind it.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.search.embedder import Embedder, EmbeddingError, SentenceTransformerEmbedder
from app.search.hybrid import HybridConfig, HybridSearcher, SearchResponse
from app.search.keyword_index import KeywordIndex, default_index_path
from app.search.vector_index import VectorIndex, default_vector_path
from app.storage.chunk_store import ChunkStore
from app.storage.repository import DocumentRepository

log = logging.getLogger("goloka.engine")


class IndexingInProgress(RuntimeError):
    """Another indexing run is already active."""


@dataclass
class IndexReport:
    status: str                              # completed | partial
    documents: int                           # searchable documents in storage
    keyword: dict[str, int]
    vectors: dict[str, int] | None           # None when semantic indexing could not run
    pending: int                             # documents still not fully indexed afterwards
    fully_indexed: int
    warnings: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    finished_at: float = field(default_factory=time.time)


class SearchEngine:
    def __init__(self, settings: Settings, *, embedder: Embedder | None = None) -> None:
        settings.ensure_dirs()
        self.settings = settings
        self.repo = DocumentRepository(settings.db_path)
        self.keyword_path = default_index_path(settings)
        self.vector_path = default_vector_path(settings)
        self.keyword = KeywordIndex.load(self.keyword_path)
        self.embedder: Embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
        self.chunk_store = ChunkStore(settings.db_path)
        self.vectors = VectorIndex.load(
            self.vector_path, self.embedder, self.chunk_store,
            chunk_size_words=settings.chunk_size_words, chunk_overlap_words=settings.chunk_overlap_words)
        self.searcher = HybridSearcher(self.repo, self.keyword, self.vectors, HybridConfig.from_settings(settings))
        self._lock = threading.RLock()
        self._index_gate = threading.Lock()
        self.semantic_status = "not_loaded"          # not_loaded | ready | unavailable
        self.last_semantic_error: str | None = None
        self.last_index: IndexReport | None = None

    # ------------------------------------------------------------------ indexing
    def index_pending(self) -> IndexReport:
        """Bring the keyword and vector indexes up to date with the stored documents. Idempotent."""
        if not self._index_gate.acquire(blocking=False):
            raise IndexingInProgress("an indexing run is already in progress")
        started = time.perf_counter()
        warnings: list[str] = []
        try:
            with self._lock:
                kw = self.keyword.sync(self.repo)
                self.keyword.save(self.keyword_path)
                keyword_stats = {"added": kw.added, "updated": kw.updated, "unchanged": kw.unchanged,
                                 "removed": kw.removed}

                vector_stats: dict[str, int] | None = None
                try:
                    vs = self.vectors.sync(self.repo)
                    self.vectors.save(self.vector_path)
                    vector_stats = {"added": vs.added, "updated": vs.updated, "unchanged": vs.unchanged,
                                    "removed": vs.removed, "empty": vs.empty, "failed": vs.failed}
                    self.semantic_status, self.last_semantic_error = "ready", None
                    if vs.failed:
                        warnings.append(f"{vs.failed} document(s) could not be embedded; see server log")
                except EmbeddingError as exc:
                    self.semantic_status, self.last_semantic_error = "unavailable", str(exc)
                    warnings.append(f"semantic indexing skipped ({exc}); keyword search is up to date")

                self._mark_fully_indexed()
                stats = self.repo.stats()
            report = IndexReport(
                status="completed" if stats.pending_indexing == 0 else "partial",
                documents=stats.by_status.get("ok", 0), keyword=keyword_stats, vectors=vector_stats,
                pending=stats.pending_indexing, fully_indexed=stats.indexed, warnings=warnings,
                duration_ms=(time.perf_counter() - started) * 1000)
            self.last_index = report
            return report
        finally:
            self._index_gate.release()

    def _mark_fully_indexed(self) -> None:
        """A document counts as indexed only if BOTH indexes hold its current content."""
        signature = self.vectors.signature
        for doc in self.repo.needs_indexing():
            in_keyword = self.keyword.content_hash_of(doc.id) == doc.content_hash
            has_nothing_to_embed = not (doc.text or "").strip() and not (doc.title or "").strip()
            in_vectors = has_nothing_to_embed or self.chunk_store.document_state(doc.id) == (doc.content_hash, signature)
            if in_keyword and in_vectors:
                self.repo.mark_indexed(doc.id, doc.content_hash)

    # ------------------------------------------------------------------ searching
    def _finish(self, response: SearchResponse) -> SearchResponse:
        if any("semantic search unavailable" in w for w in response.warnings):
            self.semantic_status = "unavailable"
            self.last_semantic_error = next(w for w in response.warnings if "semantic search unavailable" in w)
        elif (response.mode in ("hybrid", "semantic") and response.query.strip() and len(self.vectors) > 0
              and self.semantic_status != "ready"):
            # the query was embedded successfully against a non-empty vector index, so the model works
            self.semantic_status, self.last_semantic_error = "ready", None
        pending = self.repo.stats().pending_indexing
        if pending:
            response.warnings.append(
                f"{pending} stored document(s) are not indexed yet and cannot be found; run POST /api/v1/index")
        return response

    def search(self, query: str, k: int = 10, *, method: str | None = None,
               min_score: float | None = None) -> SearchResponse:
        with self._lock:
            return self._finish(self.searcher.search(query, k, method=method, min_score=min_score))

    def search_keyword(self, query: str, k: int = 10) -> SearchResponse:
        with self._lock:
            return self._finish(self.searcher.search_keyword(query, k))

    def search_semantic(self, query: str, k: int = 10, *, min_score: float | None = None) -> SearchResponse:
        with self._lock:
            return self._finish(self.searcher.search_semantic(query, k, min_score=min_score))

    # ------------------------------------------------------------------ status
    def snapshot(self) -> dict[str, Any]:
        stats = self.repo.stats()
        return {
            "stats": stats,
            "keyword_documents": len(self.keyword),
            "vector_chunks": len(self.vectors),
            "semantic_status": self.semantic_status,
            "semantic_error": self.last_semantic_error,
            "embedding_model": self.settings.embedding_model,
            "last_index": self.last_index,
        }

    def close(self) -> None:
        for closable in (self.chunk_store, self.repo):
            try:
                closable.close()
            except Exception:  # noqa: BLE001 - closing must never raise during shutdown
                log.exception("error while closing %s", type(closable).__name__)
