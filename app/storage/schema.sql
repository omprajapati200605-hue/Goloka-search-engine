-- GOLOKA schema. Portable SQL (SQLite now, PostgreSQL-compatible shape later):
-- plain types, no SQLite-only features beyond AUTOINCREMENT-free text primary keys.
-- Timestamps are ISO-8601 UTC strings; metadata is JSON text (JSONB on PostgreSQL).

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL,
    applied_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id               TEXT PRIMARY KEY,
    url              TEXT NOT NULL UNIQUE,
    title            TEXT NOT NULL DEFAULT '',
    domain           TEXT NOT NULL,
    text             TEXT NOT NULL DEFAULT '',
    metadata         TEXT NOT NULL DEFAULT '{}',
    content_hash     TEXT NOT NULL,
    crawled_at       TEXT NOT NULL,
    last_updated_at  TEXT NOT NULL,
    crawl_status     TEXT NOT NULL DEFAULT 'ok',
    indexed_hash     TEXT            -- content_hash at last successful indexing; NULL = never indexed
);
CREATE INDEX IF NOT EXISTS idx_documents_domain       ON documents(domain);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_crawled_at   ON documents(crawled_at);

CREATE TABLE IF NOT EXISTS crawl_jobs (
    id            TEXT PRIMARY KEY,
    seed_url      TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',   -- pending | running | completed | failed
    max_pages     INTEGER NOT NULL,
    max_depth     INTEGER NOT NULL,
    allowed_domains TEXT NOT NULL DEFAULT '[]',
    pages_crawled INTEGER NOT NULL DEFAULT 0,
    pages_failed  INTEGER NOT NULL DEFAULT 0,
    pages_skipped INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_crawl_jobs_status ON crawl_jobs(status);

-- Per-URL crawl outcomes (also drives URL de-duplication across a job).
CREATE TABLE IF NOT EXISTS crawl_urls (
    job_id      TEXT NOT NULL,
    url         TEXT NOT NULL,
    depth       INTEGER NOT NULL,
    status      TEXT NOT NULL,        -- queued | fetched | duplicate | blocked | error | skipped
    http_status INTEGER,
    detail      TEXT,
    fetched_at  TEXT,
    PRIMARY KEY (job_id, url)
);

-- Vector chunk metadata (FAISS row position -> document/chunk). Populated in Stage 6.
CREATE TABLE IF NOT EXISTS chunks (
    faiss_id     INTEGER PRIMARY KEY,
    document_id  TEXT    NOT NULL,
    chunk_index  INTEGER NOT NULL,
    text         TEXT    NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

-- Stage 6: which document versions have been embedded, and how (signature = model|chunking settings).
CREATE TABLE IF NOT EXISTS vector_documents (
    document_id  TEXT PRIMARY KEY,
    content_hash TEXT    NOT NULL,
    signature    TEXT    NOT NULL,
    chunk_count  INTEGER NOT NULL,
    indexed_at   TEXT    NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Monotonic counter so FAISS ids are never reused (a stale vector can never alias a new chunk).
CREATE TABLE IF NOT EXISTS vector_state (
    key   TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
