"""Persistent vector metadata: which chunks exist, which FAISS id each one has, and which document
versions have been embedded (with which model/chunking signature).

Lives in the same SQLite file as the documents (tables `chunks`, `vector_documents`, `vector_state`).
FAISS ids come from a monotonic counter and are never reused.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from app.storage.db import init_db, utcnow_iso


@dataclass(frozen=True)
class Chunk:
    faiss_id: int
    document_id: str
    chunk_index: int
    text: str


class ChunkStore:
    def __init__(self, db_path: str | Path) -> None:
        self._conn: sqlite3.Connection = init_db(db_path)
        self._lock = threading.RLock()
        with self._lock, self._conn:
            self._conn.execute("INSERT OR IGNORE INTO vector_state (key, value) VALUES ('next_faiss_id', 1)")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ writes
    def replace_document(self, document_id: str, content_hash: str, signature: str,
                         chunks: Sequence[str]) -> tuple[list[int], list[int]]:
        """Atomically swap a document's chunks. Returns (old_faiss_ids, new_faiss_ids)."""
        with self._lock, self._conn:
            old = [r[0] for r in self._conn.execute(
                "SELECT faiss_id FROM chunks WHERE document_id = ?", (document_id,))]
            self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            counter = self._conn.execute("SELECT value FROM vector_state WHERE key = 'next_faiss_id'").fetchone()[0]
            highest = self._conn.execute("SELECT COALESCE(MAX(faiss_id), 0) FROM chunks").fetchone()[0]
            start = max(counter, highest + 1)
            new = list(range(start, start + len(chunks)))
            self._conn.executemany(
                "INSERT INTO chunks (faiss_id, document_id, chunk_index, text) VALUES (?, ?, ?, ?)",
                [(fid, document_id, i, text) for i, (fid, text) in enumerate(zip(new, chunks))])
            self._conn.execute("UPDATE vector_state SET value = ? WHERE key = 'next_faiss_id'", (start + len(chunks),))
            self._conn.execute(
                "INSERT OR REPLACE INTO vector_documents (document_id, content_hash, signature, chunk_count, indexed_at) "
                "VALUES (?, ?, ?, ?, ?)", (document_id, content_hash, signature, len(chunks), utcnow_iso()))
        return old, new

    def remove_document(self, document_id: str) -> list[int]:
        """Forget a document's chunks and embedding state. Returns the FAISS ids that were removed."""
        with self._lock, self._conn:
            old = [r[0] for r in self._conn.execute(
                "SELECT faiss_id FROM chunks WHERE document_id = ?", (document_id,))]
            self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            self._conn.execute("DELETE FROM vector_documents WHERE document_id = ?", (document_id,))
        return old

    def clear(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM chunks")
            self._conn.execute("DELETE FROM vector_documents")

    # ------------------------------------------------------------------ reads
    def document_state(self, document_id: str) -> tuple[str, str] | None:
        """(content_hash, signature) of the embedded version of a document, if any."""
        with self._lock:
            row = self._conn.execute(
                "SELECT content_hash, signature FROM vector_documents WHERE document_id = ?", (document_id,)).fetchone()
        return (row[0], row[1]) if row else None

    def indexed_documents(self) -> dict[str, tuple[str, str]]:
        with self._lock:
            return {r[0]: (r[1], r[2]) for r in self._conn.execute(
                "SELECT document_id, content_hash, signature FROM vector_documents")}

    def all_ids(self) -> set[int]:
        with self._lock:
            return {r[0] for r in self._conn.execute("SELECT faiss_id FROM chunks")}

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def get_chunks(self, faiss_ids: Iterable[int]) -> dict[int, Chunk]:
        ids = [int(i) for i in faiss_ids]
        out: dict[int, Chunk] = {}
        with self._lock:
            for i in range(0, len(ids), 500):
                part = ids[i:i + 500]
                marks = ",".join("?" * len(part))
                for r in self._conn.execute(
                        f"SELECT faiss_id, document_id, chunk_index, text FROM chunks WHERE faiss_id IN ({marks})", part):
                    out[r[0]] = Chunk(r[0], r[1], r[2], r[3])
        return out

    def documents_for_ids(self, faiss_ids: Iterable[int]) -> dict[int, str]:
        return {i: c.document_id for i, c in self.get_chunks(faiss_ids).items()}
