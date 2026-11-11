# GOLOKA

A backend search engine core in Python. It crawls web pages, extracts their text, stores them in SQLite, and searches them
three ways: keyword (BM25), semantic (sentence-transformers + FAISS), and a hybrid of both. A FastAPI service exposes it, and
crawls run as background jobs. A small search website is served by the same server.

```
crawl (robots.txt, SSRF-safe) -> extract text -> SQLite -> BM25 index + FAISS vectors -> hybrid ranking -> FastAPI
```

## Install (Windows, VS Code terminal in the project folder)

Requires Python 3.12 or newer.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```
On macOS/Linux use `source .venv/bin/activate` and `cp .env.example .env`.

`.env` is optional. Every setting has a default (see the table below).

## Run

```
python -m scripts.run_api
```
Open http://127.0.0.1:8000 for the search website (search, and add pages to search), and http://127.0.0.1:8000/docs for interactive API docs. The first semantic search or indexing run downloads the embedding
model (about 90 MB) and needs internet once; after that it is cached.

## Try it

Windows `cmd` (use `^` for line breaks, and escape quotes inside JSON as shown):
```
curl -X POST http://127.0.0.1:8000/api/v1/crawl -H "Content-Type: application/json" -d "{\"url\": \"https://example.org/\", \"max_pages\": 10}"
curl http://127.0.0.1:8000/api/v1/jobs/<job id from the reply>
curl "http://127.0.0.1:8000/api/v1/search?q=your+query"
```
Crawl only sites you are allowed to crawl. GOLOKA obeys robots.txt, but that does not replace a site's terms of use. Set
`GOLOKA_USER_AGENT` to something that identifies you before crawling anything real.

Typical flow: `POST /crawl` -> poll `GET /jobs/{id}` until `completed` -> `GET /search?q=...`. Crawl jobs index automatically.

## The search website

Open http://127.0.0.1:8000. **Search** finds pages by keywords, by meaning, or both, and shows for each result how strongly it
matched each way. **Add pages** starts a crawl and shows its progress. Goloka only searches pages you have added, so a question
such as "who is the first great mathematician" returns the added pages that best match it, not a direct answer. To try it, add
`https://en.wikipedia.org/wiki/History_of_mathematics` (the button on the Add pages screen fills this in), wait for the crawl to
finish, then search. Wikipedia may refuse crawlers with a generic name: set `GOLOKA_USER_AGENT` in `.env` to a name with your
contact details and restart the server.

The page draws crawled text only as plain text and is served with a strict Content-Security-Policy, so a hostile page cannot run
script in your browser.

## Endpoints

| Method and path | Purpose |
|---|---|
| `POST /api/v1/crawl` | Start a background crawl job (returns a job id) |
| `GET /api/v1/jobs`, `GET /api/v1/jobs/{id}`, `GET /api/v1/jobs/{id}/urls` | Job status, progress, errors, per-URL outcomes |
| `POST /api/v1/index` | Index stored documents (new or changed ones only) |
| `GET /api/v1/search?q=` | Hybrid search (`method=weighted` or `rrf`) |
| `GET /api/v1/search/keyword?q=` | BM25 only |
| `GET /api/v1/search/semantic?q=` | Embeddings only |
| `GET /api/v1/documents`, `GET /api/v1/documents/{id}` | Browse stored documents |
| `GET /api/v1/health` | Status, counts, index and model state |

Full reference with real example responses: [docs/API.md](docs/API.md). Schema: [docs/openapi.json](docs/openapi.json).

## Test

```
python -m pytest
```
Tests never touch the internet: the network is mocked (`respx`), DNS is faked, and a deterministic stand-in replaces the
embedding model. One test that loads the real model is skipped unless you opt in:
```
set GOLOKA_RUN_MODEL_TESTS=1
python -m pytest tests/test_embedder.py
```

## Configuration

Environment variables with the prefix `GOLOKA_`, or a `.env` file. The most useful ones:

| Variable | Default | Meaning |
|---|---|---|
| `GOLOKA_DATA_DIR`, `GOLOKA_DB_PATH`, `GOLOKA_VECTOR_DIR` | `./data/...` | Where the database and indexes live |
| `GOLOKA_USER_AGENT` | `GOLOKABot/0.1 ...` | Sent with every request. Make it identify you |
| `GOLOKA_CRAWL_PER_DOMAIN_DELAY_SECONDS` | `1.0` | Minimum pause between requests to one host (robots `Crawl-delay` can raise it) |
| `GOLOKA_CRAWL_DEFAULT_MAX_PAGES` / `_HARD_MAX_PAGES` | `50` / `1000` | Default and absolute page limits |
| `GOLOKA_CRAWL_DEFAULT_MAX_DEPTH` / `_HARD_MAX_DEPTH` | `2` / `6` | Default and absolute depth limits |
| `GOLOKA_CRAWL_MAX_PAGE_BYTES` | `2000000` | Larger responses are skipped |
| `GOLOKA_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Changing it re-embeds everything on the next index run |
| `GOLOKA_HYBRID_BM25_WEIGHT` / `_SEMANTIC_WEIGHT` / `_TITLE_WEIGHT` | `0.4` / `0.4` / `0.2` | Weighted fusion |
| `GOLOKA_HYBRID_MIN_SEMANTIC_SCORE` | `0.2` | Similarity below this is not a semantic match |
| `GOLOKA_API_KEY` | unset | If set, every endpoint except `/health` needs the header `X-API-Key` |
| `GOLOKA_CORS_ORIGINS` | none | JSON list of browser origins allowed to call the API |
| `GOLOKA_API_HOST` / `_PORT` | `127.0.0.1` / `8000` | Bind address |

## Project layout

```
app/
  config.py            settings
  engine.py            owns the database and indexes; indexing and search entry points
  ingest.py            crawler page -> extracted document -> storage
  api/                 FastAPI app, routes, schemas, error format, API-key check
  crawler/             URL safety (SSRF), robots.txt parser, safe fetcher, pinned connections, crawler
  extraction/          HTML -> title, main text, metadata
  storage/             SQLite schema, document repository, chunk store
  search/              tokenizer, BM25, chunker, embedder, vector index, hybrid fusion, snippets
  jobs/                job store and background job manager
  web/                 the search website: index.html and static/ (app.js, styles.css), no build step
scripts/               run_api.py, export_openapi.py, init_db.py, dev_cli.py (try the engine without the API)
tests/                 pytest suite
docs/                  API.md, openapi.json
```

## How the pieces behave

* **Crawling.** Only public http/https hosts on ports 80/443. Every request, including each redirect, is checked, and the
  connection itself is pinned to an address that was validated at connection time, so DNS tricks cannot reach internal
  networks. robots.txt is obeyed. Login and account paths are skipped. 401/403/429 are reported, never retried or bypassed.
  No CAPTCHA handling of any kind.
* **Change detection.** Each page's content is hashed. Re-crawling an unchanged page does nothing; a changed page is
  re-indexed; duplicates are skipped.
* **Indexing.** A document counts as indexed only when both the BM25 and vector indexes hold its current content. Index files
  are saved to `data/` and loaded on start, so restarts do not re-embed anything.
* **Hybrid search.** Content BM25, title BM25 and semantic similarity are normalised and fused (weighted, or reciprocal rank
  fusion). If the model is unavailable, search falls back to keyword results and says so in `warnings`.
* **Jobs.** One job runs at a time. Progress and per-URL outcomes are stored. A job running when the server stops is marked
  `failed` (interrupted); queued jobs run again after a restart.

## Known limits (please read)

* **Not yet run against the real web or the real model in the development environment.** All automated tests use mocked
  network and a stand-in embedder. The crawler, extraction and the model behave as the tests describe, but your first real crawl
  and first real model download are the real-world check. Tell me what they show.
* **The semantic threshold (`0.2`) is a starting guess, not tuned.** Real MiniLM similarities depend on your data. If semantic
  results look too loose or too strict, adjust `GOLOKA_HYBRID_MIN_SEMANTIC_SCORE`.
* **Environment proxies are ignored by the crawler.** A proxy would resolve DNS itself, which would defeat the connection
  pinning. On a network that only allows web access through a proxy, crawling will fail.
* **Text search is basic:** no stemming (`run` does not match `running`) and English stop-words only. Language is detected and
  stored, but not used for ranking.
* **Scale:** vector search is exact (FAISS flat) and the BM25 index is held in memory, which suits thousands to low hundreds of
  thousands of pages, not web scale. SQLite is a single-writer database, so run one server process.
* **Pages that need JavaScript** to show their content are not rendered; only the HTML the server sends is read.
* **While an index run is active, searches wait** for it (they share a lock).
* **No built-in rate limiting or user accounts.** The optional API key is a single shared secret. Do not expose the port to the
  internet without putting it behind a proper gateway. By default it listens on 127.0.0.1 only.
* `/docs` and `/openapi.json` stay open even when an API key is set.
