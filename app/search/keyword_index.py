"""Keyword (BM25) search index over document titles and content.

* Titles and bodies are indexed as two separate fields, so hybrid ranking can weigh them
  independently. `score` = content_score + title_boost * title_score.
* Incremental: add/update/remove documents at any time; the BM25 statistics are rebuilt lazily
  on the next search after a change.
* Persistent: `save()` / `load()` write a gzip-JSON snapshot (atomic replace). `sync(repo)` then
  brings the index in line with the document database, re-tokenizing only new/changed documents.
"""
from __future__ import annotations

import gzip
import heapq
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.config import Settings
from app.search.bm25 import DEFAULT_B, DEFAULT_K1, BM25Backend, make_backend
from app.search.tokenizer import TOKENIZER_VERSION, analyze_document, analyze_query
from app.storage.repository import DocumentRepository

log = logging.getLogger("goloka.keyword_index")

SNAPSHOT_VERSION = 1
DEFAULT_TITLE_BOOST = 2.0


def default_index_path(settings: Settings) -> Path:
    return settings.data_dir / "bm25_index.json.gz"


@dataclass(frozen=True)
class KeywordHit:
    doc_id: str
    score: float          # combined: content_score + title_boost * title_score
    title_score: float
    content_score: float


@dataclass
class SyncStats:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0


@dataclass
class _Entry:
    content_hash: str
    title_tokens: list[str]
    content_tokens: list[str]


class _FieldIndex:
    """BM25 over one field for the documents that have at least one token in it."""

    def __init__(self, backend: BM25Backend, doc_ids: list[str]) -> None:
        self.backend = backend
        self.doc_ids = doc_ids

    def score(self, terms: Sequence[str]) -> dict[str, float]:
        return {self.doc_ids[pos]: s for pos, s in self.backend.score(terms).items()}


class KeywordIndex:
    def __init__(self, *, backend: str = "rank_bm25", k1: float = DEFAULT_K1, b: float = DEFAULT_B,
                 title_boost: float = DEFAULT_TITLE_BOOST) -> None:
        self._backend_name, self._k1, self._b = backend, k1, b
        make_backend(backend, k1=k1, b=b)  # fail fast on a bad backend name
        self.title_boost = title_boost
        self._entries: dict[str, _Entry] = {}
        self._title: _FieldIndex | None = None
        self._content: _FieldIndex | None = None
        self._dirty = True
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ maintenance
    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._entries

    def content_hash_of(self, doc_id: str) -> str | None:
        entry = self._entries.get(doc_id)
        return entry.content_hash if entry else None

    def add_document(self, doc_id: str, title: str, text: str, content_hash: str) -> bool:
        """Index (or re-index) a document. Returns False, doing nothing, if it is unchanged."""
        with self._lock:
            current = self._entries.get(doc_id)
            if current is not None and current.content_hash == content_hash:
                return False
            self._entries[doc_id] = _Entry(content_hash, analyze_document(title), analyze_document(text))
            self._dirty = True
            return True

    def remove_document(self, doc_id: str) -> bool:
        with self._lock:
            if self._entries.pop(doc_id, None) is None:
                return False
            self._dirty = True
            return True

    def _ensure_built(self) -> None:
        if not self._dirty:
            return

        def build(field: str) -> _FieldIndex:
            ids = sorted(i for i, e in self._entries.items() if getattr(e, field))
            backend = make_backend(self._backend_name, k1=self._k1, b=self._b)
            backend.build([getattr(self._entries[i], field) for i in ids])
            return _FieldIndex(backend, ids)

        self._title, self._content = build("title_tokens"), build("content_tokens")
        self._dirty = False

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 10) -> list[KeywordHit]:
        """Top-k documents for the query, best first (ties broken by document id)."""
        terms = analyze_query(query)
        if not terms or k <= 0:
            return []
        with self._lock:
            if not self._entries:
                return []
            self._ensure_built()
            title_scores = self._title.score(terms)
            content_scores = self._content.score(terms)
        hits = [
            KeywordHit(
                doc_id=d,
                score=content_scores.get(d, 0.0) + self.title_boost * title_scores.get(d, 0.0),
                title_score=title_scores.get(d, 0.0),
                content_score=content_scores.get(d, 0.0),
            )
            for d in title_scores.keys() | content_scores.keys()
        ]
        return heapq.nsmallest(k, hits, key=lambda h: (-h.score, h.doc_id))

    # ------------------------------------------------------------------ sync with the database
    def sync(self, repo: DocumentRepository) -> SyncStats:
        """Make the index match the searchable ('ok') documents in the repository."""
        stats = SyncStats()
        seen: set[str] = set()
        for doc in repo.iter_documents():
            seen.add(doc.id)
            if self.content_hash_of(doc.id) == doc.content_hash:
                stats.unchanged += 1
                continue
            is_new = doc.id not in self
            self.add_document(doc.id, doc.title, doc.text, doc.content_hash)
            if is_new:
                stats.added += 1
            else:
                stats.updated += 1
        with self._lock:
            stale = [i for i in self._entries if i not in seen]
        for doc_id in stale:
            self.remove_document(doc_id)
            stats.removed += 1
        return stats

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "version": SNAPSHOT_VERSION,
                "tokenizer_version": TOKENIZER_VERSION,
                "docs": {i: {"h": e.content_hash, "t": e.title_tokens, "c": e.content_tokens}
                         for i, e in self._entries.items()},
            }
        tmp = path.with_name(path.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)  # atomic: readers never see a half-written file

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "KeywordIndex":
        """Load a snapshot. A missing, corrupt or outdated file yields an empty index (never raises);
        call sync() afterwards to rebuild from the database."""
        index = cls(**kwargs)
        path = Path(path)
        if not path.exists():
            return index
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                payload = json.load(fh)
            if payload["version"] != SNAPSHOT_VERSION or payload["tokenizer_version"] != TOKENIZER_VERSION:
                log.warning("Keyword index snapshot %s is outdated; it will be rebuilt", path)
                return index
            entries = {
                doc_id: _Entry(str(d["h"]), list(map(str, d["t"])), list(map(str, d["c"])))
                for doc_id, d in payload["docs"].items()
            }
        except (OSError, EOFError, ValueError, KeyError, TypeError, AttributeError) as exc:
            log.warning("Keyword index snapshot %s is unreadable (%s); it will be rebuilt", path, exc)
            return index
        index._entries = entries
        index._dirty = True
        return index
