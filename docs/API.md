# GOLOKA API reference

Base URL: `http://127.0.0.1:8000`. Interactive docs: `/docs` (Swagger) and `/redoc`. Machine-readable schema: `/openapi.json`
(a copy is committed as `docs/openapi.json`; regenerate with `python -m scripts.export_openapi`).

All example responses below are real output from the running application. Scores in the examples come from a deterministic
test embedder, so your numbers will differ. The shapes are exact.

## Conventions

**Authentication.** Off by default. If the server sets `GOLOKA_API_KEY`, every endpoint except `GET /api/v1/health`, `/docs` and
`/openapi.json` needs the header `X-API-Key: <key>`. Missing or wrong key gives `401`.

**Errors.** Every error has the same shape:
```json
{"error": {"code": "not_found", "message": "Document 'nope' does not exist"}}
```
Validation errors add `details: [{"location": "query.q", "message": "..."}]`. Codes:

| HTTP | code | Meaning |
|---|---|---|
| 401 | `unauthorized` | API key required or wrong |
| 404 | `not_found` | Unknown document or job (or unknown route) |
| 405 | `method_not_allowed` | Wrong HTTP method |
| 409 | `indexing_in_progress` | An indexing run is already active |
| 422 | `validation_error` | Bad or missing parameter |
| 422 | `unsafe_url` | Crawl seed is not a public http/https URL (loopback, private IPs, credentials, bad port, ...) |
| 422 | `dns_error` | Crawl seed host does not resolve |
| 422 | `invalid_limit` / `invalid_domain` | `max_pages`, `max_depth` or `allowed_domains` out of range |
| 429 | `queue_full` | Too many unfinished crawl jobs (`GOLOKA_JOBS_MAX_PENDING`) |
| 500 | `internal_error` | Unexpected server error (details are in the server log, never in the response) |

**Never invented.** Search results are read from stored documents. If nothing matches, `results` is an empty list.
`bm25_score` is `null` when the keyword search did not match the document, and `semantic_score` is `null` when the semantic
search did not, so a missing signal is never filled with a made-up number.

---

## Crawling

### POST /api/v1/crawl
Queue a background crawl. Returns **202** immediately with a job.

Body (JSON):

| field | type | default | notes |
|---|---|---|---|
| `url` | string, required | | public `http`/`https` URL |
| `max_pages` | int ≥ 1 | `GOLOKA_CRAWL_DEFAULT_MAX_PAGES` (50) | at most `GOLOKA_CRAWL_HARD_MAX_PAGES` (1000) |
| `max_depth` | int ≥ 0 | `GOLOKA_CRAWL_DEFAULT_MAX_DEPTH` (2) | at most `GOLOKA_CRAWL_HARD_MAX_DEPTH` (6) |
| `allowed_domains` | string[] (max 20) | `[]` | extra domains the crawl may follow links to. The seed's own domain is always allowed |
| `auto_index` | bool | `true` | index the pages when the crawl ends |

```
curl -X POST http://127.0.0.1:8000/api/v1/crawl -H "Content-Type: application/json" -d "{\"url\": \"https://example.org/\", \"max_pages\": 10}"
```
Response (`Location: /api/v1/jobs/<id>` header is also set):
```json
{
  "id": "80f47f33c73e46ad81c39638d3bcaa11",
  "seed_url": "https://example.org/",
  "status": "pending",
  "phase": "queued",
  "max_pages": 10,
  "max_depth": 2,
  "allowed_domains": [],
  "auto_index": true,
  "progress": {"urls_processed": 0, "pages_crawled": 0, "pages_failed": 0, "pages_skipped": 0, "max_pages": 10, "percent": 0.0},
  "documents": {},
  "index": null,
  "error": null,
  "created_at": "2026-11-10T19:16:34+00:00",
  "started_at": null,
  "finished_at": null
}
```
The crawler obeys robots.txt (including `Crawl-delay`), never follows login/account paths, never retries 401/403/429 responses,
and refuses any connection to a non-public address, including after redirects and DNS changes.

### GET /api/v1/jobs/{job_id}
Status, progress and errors of one job.

* `status`: `pending` → `running` → `completed` | `failed`
* `phase`: `queued` → `crawling` → `indexing` → `done`
* `progress.percent` is `urls_processed / max_pages`. It is an upper-bound estimate, because a crawl ends early when it runs out of
  links. Completed jobs show 100.
* `documents`: how pages were stored (`created`, `updated`, `unchanged`, `duplicate`, `empty`).
* `index`: result of the automatic indexing step (`status` `completed` | `partial` | `failed` | `skipped`). A problem here does
  **not** make the job fail, because the pages are already stored; call `POST /api/v1/index` to retry.
* `error`: set when `status` is `failed`. A job fails when no page could be crawled (the message names the first problem, e.g.
  `disallowed by robots.txt` or `HTTP 500`), when the crawl cannot start, or when the server stopped mid-job (`interrupted`).

```json
{
  "id": "80f47f33c73e46ad81c39638d3bcaa11",
  "seed_url": "https://example.org/",
  "status": "completed",
  "phase": "done",
  "max_pages": 10,
  "max_depth": 2,
  "allowed_domains": [],
  "auto_index": true,
  "progress": {"urls_processed": 3, "pages_crawled": 2, "pages_failed": 1, "pages_skipped": 0, "max_pages": 10, "percent": 100.0},
  "documents": {"created": 2, "updated": 0, "unchanged": 0, "duplicate": 0, "empty": 0},
  "index": {
    "status": "completed", "pending": 0, "warnings": [],
    "keyword": {"added": 2, "updated": 0, "unchanged": 0, "removed": 0},
    "vectors": {"added": 2, "updated": 0, "unchanged": 0, "removed": 0, "empty": 0, "failed": 0}
  },
  "error": null,
  "created_at": "2026-11-10T19:16:34+00:00",
  "started_at": "2026-11-10T19:16:34+00:00",
  "finished_at": "2026-11-10T19:16:34+00:00"
}
```

### GET /api/v1/jobs
List jobs, newest first. Query: `limit` (1–200, default 50), `offset`, `status` (`pending|running|completed|failed`).
Returns `{"items": [<job>...], "total", "limit", "offset"}`.

### GET /api/v1/jobs/{job_id}/urls
What happened to each URL. Query: `limit` (1–500, default 100), `offset`, `status`
(`fetched|duplicate|blocked|restricted|skipped|error`). Use `?status=error` to see what failed and why.
```json
{
  "items": [
    {"url": "https://example.org/", "depth": 0, "status": "fetched", "http_status": 200, "detail": null, "fetched_at": "2026-11-10T19:16:34+00:00"},
    {"url": "https://example.org/gone", "depth": 1, "status": "error", "http_status": 404, "detail": "not_found: HTTP 404", "fetched_at": "2026-11-10T19:16:34+00:00"}
  ],
  "total": 3, "limit": 100, "offset": 0
}
```

---

## Indexing

### POST /api/v1/index
Build or update the keyword (BM25) and vector indexes for every stored document that is new or changed. Normally you do not need
this, because crawl jobs index automatically. Use it after `auto_index: false`, or to retry after the embedding model was
unavailable. Safe to repeat: unchanged documents are skipped and never re-embedded. Returns **409** if a run is already active.
```json
{
  "status": "completed",
  "documents": 2,
  "keyword": {"added": 0, "updated": 0, "unchanged": 2, "removed": 0},
  "vectors": {"added": 0, "updated": 0, "unchanged": 2, "removed": 0, "empty": 0, "failed": 0},
  "fully_indexed": 2,
  "pending": 0,
  "warnings": [],
  "duration_ms": 1.84
}
```
`status` is `partial` when something is still pending (for example the embedding model could not load). Then `vectors` is `null`,
`warnings` says why, and keyword search is still fully up to date.

---

## Search

All three search endpoints take `q` (1–500 characters, not blank) and `k` (1–50, default 10), and return the same envelope:

```json
{
  "query": "tomato sauce",
  "mode": "hybrid",
  "method": "weighted",
  "total_returned": 1,
  "took_ms": 1.31,
  "results": [
    {
      "document_id": "2169ce09-b1ee-51ab-83da-4ce8dcb09920",
      "title": "Cooking pasta",
      "url": "https://example.org/pasta",
      "domain": "example.org",
      "snippet": "Cooking pasta Boil pasta in salted water, then add tomato sauce and basil.",
      "bm25_score": 1.1621629374657765,
      "semantic_score": 0.31622782349586487,
      "combined_score": 0.526491129398346,
      "title_score": null,
      "content_score": 1.1621629374657765,
      "components": {"content": 1.0, "semantic": 0.31622782349586487, "title": 0.0},
      "ranks": {"content": 1, "semantic": 1}
    }
  ],
  "warnings": []
}
```

| field | meaning |
|---|---|
| `combined_score` | The ranking score. Hybrid: 0..1. Keyword: the raw BM25 score (can exceed 1). Semantic: cosine similarity |
| `bm25_score` / `semantic_score` | Raw per-signal scores, or `null` if that signal did not match this document |
| `title_score`, `content_score` | The two BM25 fields (title is boosted ×2 inside `bm25_score`) |
| `components` | Normalised per-signal scores that were fused (hybrid only) |
| `ranks` | 1-based rank of the document within each signal (hybrid only) |
| `warnings` | Degradations, e.g. `semantic search unavailable (...)`, or `N stored document(s) are not indexed yet ...` |

### GET /api/v1/search
Hybrid: BM25 on content + BM25 on title + semantic similarity, fused. Extra parameters:
* `method`: `weighted` (default; weights `GOLOKA_HYBRID_*_WEIGHT`, defaults 0.4 / 0.4 / 0.2) or `rrf` (reciprocal rank fusion,
  `GOLOKA_RRF_K`).
* `min_score` (0–1): drop results whose combined score is lower.

If the embedding model is unavailable the response falls back to keyword results and says so in `warnings`.
```
curl "http://127.0.0.1:8000/api/v1/search?q=tomato+sauce&k=5&method=rrf"
```

### GET /api/v1/search/keyword
BM25 only. Finds literal words (after lowercasing, no stemming, English stop-words removed).

### GET /api/v1/search/semantic
Embedding similarity only. Finds by meaning, e.g. `noodles` finds a page about pasta. Extra parameter `min_score` (−1..1,
default `GOLOKA_HYBRID_MIN_SEMANTIC_SCORE` = 0.2). Each result's `snippet` is the best-matching passage.

---

## Documents

### GET /api/v1/documents
Stored documents, newest crawl first, without the full text. Query: `limit` (1–200, default 50), `offset`, `domain`, `status`
(`ok` is searchable; `gone` is a page that disappeared).
```json
{
  "items": [{
    "id": "2169ce09-b1ee-51ab-83da-4ce8dcb09920",
    "url": "https://example.org/pasta",
    "title": "Cooking pasta",
    "domain": "example.org",
    "content_hash": "01cffc16…",
    "crawled_at": "2026-11-10T19:16:34+00:00",
    "last_updated_at": "2026-11-10T19:16:34+00:00",
    "crawl_status": "ok",
    "indexed": true,
    "word_count": 13
  }],
  "total": 2, "limit": 1, "offset": 0
}
```
`indexed` is true only when the current content is in both the keyword and the vector index.

### GET /api/v1/documents/{document_id}
The same fields plus `text` (the extracted main text) and `metadata` (`final_url`, `meta_description`, `canonical_url`, `headings`,
`published_date`, `author`, `language`, `internal_links`, `external_links`, `word_count`, ...). `404` if unknown.
Document ids are stable: the same URL always gets the same id.

---

## System

### GET /api/v1/health
Always open (no API key). `status` is `degraded` when the semantic model failed to load or run, otherwise `ok`.
```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime_seconds": 12.3,
  "documents": {"total": 2, "by_status": {"ok": 2}, "indexed": 2, "pending_indexing": 0, "last_crawled_at": "2026-11-10T19:16:34+00:00"},
  "indexes": {
    "keyword_documents": 2,
    "vector_chunks": 2,
    "semantic": {"model": "sentence-transformers/all-MiniLM-L6-v2", "status": "ready", "last_error": null}
  },
  "last_index": null,
  "jobs": {"completed": 1}
}
```
`semantic.status`: `not_loaded` (the model is loaded on first use), `ready`, or `unavailable` (see `last_error`).
