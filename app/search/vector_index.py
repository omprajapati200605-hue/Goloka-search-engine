"""Semantic search: chunk -> embed -> FAISS (cosine similarity via normalised inner product).

Storage layout
  * vectors   : FAISS `IndexIDMap2(IndexFlatIP)` saved to <vector_dir>/chunks.faiss (+ .meta.json)
  * metadata  : SQLite tables chunks / vector_documents (see app/storage/chunk_store.py)
SQLite is the source of truth for *what* is indexed; the FAISS file only holds the vectors. If the two
disagree after a crash, `reconcile()` (run automatically by `load()`) repairs it: orphan vectors are
dropped and documents with missing vectors are reset so the next `sync()` re-embeds them.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app.config import Settings
from app.search.chunker import CHUNKER_VERSION, split_into_chunks
from app.search.embedder import Embedder, EmbeddingError, SentenceTransformerEmbedder
from app.storage.chunk_store import ChunkStore
from app.storage.repository import DocumentRepository

log = logging.getLogger("goloka.vector_index")
META_VERSION = 1


def default_vector_path(settings: Settings) -> Path:
    return settings.vector_dir / "chunks.faiss"


def _meta_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


@dataclass(frozen=True)
class SemanticHit:
    doc_id: str
    score: float        # cosine similarity of the best-matching chunk (-1..1, higher = more similar)
    chunk_index: int
    chunk_text: str     # the matching passage, usable as a snippet
    faiss_id: int


@dataclass
class VectorSyncStats:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    empty: int = 0
    failed: int = 0


@dataclass
class ReconcileReport:
    orphan_vectors_removed: int = 0
    documents_reset: int = 0


class VectorIndex:
    def __init__(self, embedder: Embedder, store: ChunkStore, *, chunk_size_words: int = 200,
                 chunk_overlap_words: int = 40) -> None:
        split_into_chunks("", size_words=chunk_size_words, overlap_words=chunk_overlap_words)  # validates
        self._embedder = embedder
        self._store = store
        self._size, self._overlap = chunk_size_words, chunk_overlap_words
        self._index: faiss.IndexIDMap2 | None = None
        self._lock = threading.RLock()
        self.last_reconcile = ReconcileReport()

    # ------------------------------------------------------------------ basics
    @property
    def signature(self) -> str:
        return f"{self._embedder.name}|{self._size}|{self._overlap}|c{CHUNKER_VERSION}"

    def __len__(self) -> int:
        """Number of chunk vectors currently in the FAISS index."""
        return 0 if self._index is None else int(self._index.ntotal)

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.ascontiguousarray(self._embedder.embed(texts), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise EmbeddingError("embedder returned an unexpected shape")
        faiss.normalize_L2(vectors)  # cosine similarity even if the embedder does not normalise
        return vectors

    def _ensure_index(self, dim: int) -> faiss.IndexIDMap2:
        if self._index is None:
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        elif self._index.d != dim:
            raise EmbeddingError(f"embedding dimension {dim} does not match the index ({self._index.d})")
        return self._index

    def _faiss_ids(self) -> set[int]:
        if self._index is None or self._index.ntotal == 0:
            return set()
        return {int(i) for i in faiss.vector_to_array(self._index.id_map)}

    def _faiss_remove(self, ids: list[int]) -> None:
        if ids and self._index is not None:
            self._index.remove_ids(np.array(ids, dtype=np.int64))

    # ------------------------------------------------------------------ indexing
    def add_document(self, doc_id: str, title: str, text: str, content_hash: str) -> str:
        """Embed and index a document. Returns 'added', 'updated', 'unchanged' or 'empty'.

        Existing vectors of other documents are never touched. Re-indexing the same unchanged
        version (same content hash and same model/chunking signature) is a no-op."""
        state = self._store.document_state(doc_id)
        if state == (content_hash, self.signature):
            return "unchanged"
        chunks = split_into_chunks(text, size_words=self._size, overlap_words=self._overlap)
        if not chunks and (title or "").strip():
            chunks = [title.strip()]
        if not chunks:
            self.remove_document(doc_id)
            return "empty"
        title = (title or "").strip()
        vectors = self._embed([f"{title}\n{c}" if title else c for c in chunks])  # slow part: no lock held
        with self._lock:
            old_ids, new_ids = self._store.replace_document(doc_id, content_hash, self.signature, chunks)
            index = self._ensure_index(vectors.shape[1])
            self._faiss_remove(old_ids)
            index.add_with_ids(vectors, np.array(new_ids, dtype=np.int64))
        return "updated" if state is not None else "added"

    def remove_document(self, doc_id: str) -> bool:
        with self._lock:
            old_ids = self._store.remove_document(doc_id)
            self._faiss_remove(old_ids)
        return bool(old_ids)

    def sync(self, repo: DocumentRepository) -> VectorSyncStats:
        """Embed new/changed searchable documents, drop ones that are gone. Unchanged documents are skipped."""
        stats = VectorSyncStats()
        before = self._store.indexed_documents()
        seen: set[str] = set()
        for doc in repo.iter_documents():
            seen.add(doc.id)
            try:
                action = self.add_document(doc.id, doc.title, doc.text, doc.content_hash)
            except EmbeddingError:
                raise  # model unavailable: every document would fail, so stop and report
            except Exception:  # noqa: BLE001 - one bad document must not block the rest
                log.exception("could not embed document %s", doc.id)
                stats.failed += 1
                continue
            setattr(stats, action, getattr(stats, action) + 1)
        for doc_id in set(self._store.indexed_documents()) | set(before):
            if doc_id not in seen:
                self.remove_document(doc_id)
                stats.removed += 1
        return stats

    # ------------------------------------------------------------------ searching
    def search(self, query: str, k: int = 10, *, min_score: float | None = None) -> list[SemanticHit]:
        """Top-k documents by their best-matching chunk, most similar first."""
        query = (query or "").strip()
        if not query or k <= 0 or len(self) == 0:
            return []
        qvec = self._embed([query])
        with self._lock:
            index = self._index
            if index is None or index.ntotal == 0:
                return []
            if qvec.shape[1] != index.d:
                raise EmbeddingError(f"query embedding dimension {qvec.shape[1]} does not match the index ({index.d})")
            total = int(index.ntotal)
            fetch = min(total, max(k * 5, 20))
            while True:
                scores, ids = index.search(qvec, fetch)
                chunks = self._store.get_chunks(int(i) for i in ids[0] if i >= 0)
                best: dict[str, SemanticHit] = {}
                for score, fid in zip(scores[0], ids[0]):
                    chunk = chunks.get(int(fid)) if fid >= 0 else None
                    if chunk is None or chunk.document_id in best:   # stale vector or weaker chunk of same doc
                        continue
                    best[chunk.document_id] = SemanticHit(
                        chunk.document_id, float(score), chunk.chunk_index, chunk.text, int(fid))
                if len(best) >= k or fetch >= total:
                    break
                fetch = min(total, fetch * 4)  # many chunks came from few documents: look deeper
        hits = list(best.values())[:k]
        return [h for h in hits if min_score is None or h.score >= min_score]

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._index is None:
                for p in (path, _meta_path(path)):
                    p.unlink(missing_ok=True)
                return
            tmp, meta_tmp = path.with_name(path.name + ".tmp"), _meta_path(path).with_name(path.name + ".meta.tmp")
            faiss.write_index(self._index, str(tmp))
            meta_tmp.write_text(json.dumps({
                "version": META_VERSION, "embedder": self._embedder.name,
                "dimension": int(self._index.d), "vectors": int(self._index.ntotal)}), encoding="utf-8")
            os.replace(tmp, path)
            os.replace(meta_tmp, _meta_path(path))

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder, store: ChunkStore, **kwargs) -> "VectorIndex":
        """Load the saved vectors (if compatible) and reconcile them with the database. Never raises on a
        missing/corrupt file: the index simply starts empty and sync() re-embeds what is missing."""
        index = cls(embedder, store, **kwargs)
        path = Path(path)
        if path.exists() and _meta_path(path).exists():
            try:
                meta = json.loads(_meta_path(path).read_text(encoding="utf-8"))
                if meta["version"] != META_VERSION or meta["embedder"] != embedder.name:
                    log.warning("Vector index %s was built with another model/version; it will be rebuilt", path)
                else:
                    loaded = faiss.read_index(str(path))
                    if isinstance(loaded, faiss.IndexIDMap2) and loaded.d == meta["dimension"]:
                        index._index = loaded
                    else:
                        log.warning("Vector index %s has an unexpected format; it will be rebuilt", path)
            except Exception as exc:  # noqa: BLE001 - corrupt file: rebuild instead of crashing
                log.warning("Vector index %s is unreadable (%s); it will be rebuilt", path, exc)
        index.last_reconcile = index.reconcile()
        return index

    @classmethod
    def from_settings(cls, settings: Settings, embedder: Embedder | None = None) -> "VectorIndex":
        settings.ensure_dirs()
        embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
        store = ChunkStore(settings.db_path)
        return cls.load(default_vector_path(settings), embedder, store,
                        chunk_size_words=settings.chunk_size_words, chunk_overlap_words=settings.chunk_overlap_words)

    def reconcile(self) -> ReconcileReport:
        """Make FAISS and SQLite agree: drop vectors without metadata, reset documents without vectors."""
        report = ReconcileReport()
        with self._lock:
            in_faiss, in_db = self._faiss_ids(), self._store.all_ids()
            orphans = sorted(in_faiss - in_db)
            if orphans:
                self._faiss_remove(orphans)
                report.orphan_vectors_removed = len(orphans)
            missing = in_db - in_faiss
            if missing:
                affected = set(self._store.documents_for_ids(missing).values())
                for doc_id in affected:
                    self._faiss_remove(self._store.remove_document(doc_id))  # next sync() re-embeds it
                report.documents_reset = len(affected)
        if report.orphan_vectors_removed or report.documents_reset:
            log.warning("Vector index repaired: %s", report)
        return report
