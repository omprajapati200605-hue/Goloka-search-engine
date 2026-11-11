"""SQLite connection handling and schema initialization."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = 1


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with sane defaults (WAL, FKs, Row factory)."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


# Columns added after the first release. Existing databases get them via ALTER TABLE (idempotent).
_ADDED_COLUMNS = {
    "crawl_jobs": {
        "auto_index": "INTEGER NOT NULL DEFAULT 1",
        "phase": "TEXT NOT NULL DEFAULT 'queued'",
        "urls_processed": "INTEGER NOT NULL DEFAULT 0",
        "stored": "TEXT",          # JSON: documents created/updated/unchanged/duplicate/empty
        "index_result": "TEXT",    # JSON: outcome of the automatic indexing step
    },
}


def _apply_added_columns(conn: sqlite3.Connection) -> None:
    for table, columns in _ADDED_COLUMNS.items():
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, ddl in columns.items():
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Create tables if missing and record the schema version. Idempotent."""
    conn = connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    _apply_added_columns(conn)
    row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    if row["v"] is None:
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, utcnow_iso()),
        )
    conn.commit()
    return conn
