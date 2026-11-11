"""Background crawl jobs: real app, real crawler/extractor/SQLite/BM25/FAISS, mocked network, fake DNS."""
import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings
from app.engine import SearchEngine
from app.jobs.store import JobStore
from app.storage.db import init_db
from tests.conftest import fake_resolver
from tests.fake_embedder import HashingEmbedder


def page(title, body, links=()):
    anchors = "".join(f'<a href="{h}">l</a>' for h in links)
    return f"<html><head><title>{title}</title></head><body><main><h1>{title}</h1><p>{body}</p>{anchors}</main></body></html>"


def make_settings(tmp_path, **kw):
    return Settings(_env_file=None, data_dir=tmp_path / "data", db_path=tmp_path / "data" / "goloka.db",
                    vector_dir=tmp_path / "data" / "vectors", crawl_per_domain_delay_seconds=0.0,
                    crawl_max_page_bytes=50_000, crawl_timeout_seconds=2.0, **kw)


@pytest.fixture
def env(tmp_path):
    settings = make_settings(tmp_path)
    engine = SearchEngine(settings, embedder=HashingEmbedder(concepts={"noodles": "food", "pasta": "food"}))
    with TestClient(create_app(engine=engine, resolver=fake_resolver)) as client:
        yield type("Env", (), dict(client=client, engine=engine, settings=settings))
    engine.close()


def wait_done(client, job_id, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {job}")


def site(mock):
    mock.get("https://site.test/robots.txt").respond(404)
    mock.get("https://site.test/").respond(200, html=page("Home", "Welcome to the garden", ["/pasta", "/space", "/missing"]))
    mock.get("https://site.test/pasta").respond(200, html=page("Cooking", "Boil pasta in salted water with sauce"))
    mock.get("https://site.test/space").respond(200, html=page("Space", "Telescopes reveal distant galaxies"))
    mock.get("https://site.test/missing").respond(404)


# ---- happy path ----------------------------------------------------------------------------
def test_crawl_job_runs_indexes_and_is_searchable(mock, env):
    site(mock)
    r = env.client.post("/api/v1/crawl", json={"url": "https://site.test/", "max_pages": 10})
    assert r.status_code == 202 and r.headers["location"] == f"/api/v1/jobs/{r.json()['id']}"
    body = r.json()
    assert body["status"] in ("pending", "running") and body["max_pages"] == 10 and body["max_depth"] == 2
    job = wait_done(env.client, body["id"])
    assert job["status"] == "completed" and job["phase"] == "done" and job["error"] is None
    assert job["progress"]["pages_crawled"] == 3 and job["progress"]["pages_failed"] == 1
    assert job["progress"]["urls_processed"] == 4 and job["progress"]["percent"] == 100.0
    assert job["documents"]["created"] == 3 and job["started_at"] and job["finished_at"]
    assert job["index"]["status"] == "completed" and job["index"]["pending"] == 0

    hit = env.client.get("/api/v1/search", params={"q": "noodles"}).json()        # found by meaning, no manual /index
    assert hit["results"][0]["url"] == "https://site.test/pasta"
    assert env.client.get("/api/v1/health").json()["jobs"] == {"completed": 1}


def test_job_urls_endpoint_shows_errors(mock, env):
    site(mock)
    job = wait_done(env.client, env.client.post("/api/v1/crawl", json={"url": "https://site.test/"}).json()["id"])
    urls = env.client.get(f"/api/v1/jobs/{job['id']}/urls").json()
    assert urls["total"] == 4 and {u["status"] for u in urls["items"]} == {"fetched", "error"}
    err = env.client.get(f"/api/v1/jobs/{job['id']}/urls", params={"status": "error"}).json()
    assert err["total"] == 1 and err["items"][0]["url"] == "https://site.test/missing" and err["items"][0]["http_status"] == 404
    assert env.client.get(f"/api/v1/jobs/{job['id']}/urls", params={"limit": 2}).json()["items"].__len__() == 2


def test_auto_index_false_leaves_pages_pending(mock, env):
    site(mock)
    job = wait_done(env.client, env.client.post("/api/v1/crawl", json={"url": "https://site.test/", "auto_index": False}).json()["id"])
    assert job["status"] == "completed" and job["index"] is None
    health = env.client.get("/api/v1/health").json()["documents"]
    assert health["pending_indexing"] == 3
    assert env.client.post("/api/v1/index").json()["status"] == "completed"


def test_recrawl_reports_unchanged(mock, env):
    site(mock)
    wait_done(env.client, env.client.post("/api/v1/crawl", json={"url": "https://site.test/"}).json()["id"])
    second = wait_done(env.client, env.client.post("/api/v1/crawl", json={"url": "https://site.test/"}).json()["id"])
    assert second["documents"]["unchanged"] == 3 and second["documents"]["created"] == 0


def test_limits_are_respected(mock, env):
    site(mock)
    job = wait_done(env.client, env.client.post("/api/v1/crawl", json={"url": "https://site.test/", "max_pages": 1}).json()["id"])
    assert job["progress"]["pages_crawled"] == 1 and env.engine.repo.count() == 1
    job = wait_done(env.client, env.client.post("/api/v1/crawl", json={"url": "https://site.test/", "max_depth": 0}).json()["id"])
    assert job["progress"]["urls_processed"] == 1


def test_robots_disallow_fails_job_with_reason(mock, env):
    mock.get("https://site.test/robots.txt").respond(200, text="User-agent: *\nDisallow: /")
    job = wait_done(env.client, env.client.post("/api/v1/crawl", json={"url": "https://site.test/"}).json()["id"])
    assert job["status"] == "failed" and "no pages could be crawled" in job["error"] and "robots.txt" in job["error"]
    assert env.engine.repo.count() == 0 and job["index"] is None


def test_unreachable_seed_fails_job(mock, env):
    mock.get("https://site.test/robots.txt").respond(404)
    mock.get("https://site.test/").respond(500)
    job = wait_done(env.client, env.client.post("/api/v1/crawl", json={"url": "https://site.test/"}).json()["id"])
    assert job["status"] == "failed" and job["progress"]["pages_failed"] == 1 and "500" in job["error"]


def test_embedding_failure_does_not_fail_the_crawl(mock, tmp_path):
    from app.search.embedder import EmbeddingError

    class Down(HashingEmbedder):
        def embed(self, texts):
            raise EmbeddingError("model offline")

    engine = SearchEngine(make_settings(tmp_path), embedder=Down())
    try:
        with TestClient(create_app(engine=engine, resolver=fake_resolver)) as c:
            site(mock)
            job = wait_done(c, c.post("/api/v1/crawl", json={"url": "https://site.test/"}).json()["id"])
            assert job["status"] == "completed" and job["index"]["status"] == "partial"
            assert any("model offline" in w for w in job["index"]["warnings"])
            assert c.get("/api/v1/search/keyword", params={"q": "galaxies"}).json()["total_returned"] == 1
    finally:
        engine.close()


def test_indexer_crash_is_recorded_not_fatal(mock, env):
    def boom():
        raise RuntimeError("disk full")

    env.client.app.state.jobs._indexer = boom
    site(mock)
    job = wait_done(env.client, env.client.post("/api/v1/crawl", json={"url": "https://site.test/"}).json()["id"])
    assert job["status"] == "completed" and job["index"]["status"] == "failed" and "disk full" in job["index"]["error"]


def test_job_waits_for_a_running_manual_index(mock, env):
    env.client.app.state.jobs._retry = 0.02
    site(mock)
    assert env.engine._index_gate.acquire(blocking=False)
    r = env.client.post("/api/v1/crawl", json={"url": "https://site.test/"}).json()
    time.sleep(0.3)
    env.engine._index_gate.release()
    job = wait_done(env.client, r["id"])
    assert job["status"] == "completed" and job["index"]["status"] == "completed"


# ---- submission validation -----------------------------------------------------------------
@pytest.mark.parametrize("url,code", [
    ("http://127.0.0.1/", "unsafe_url"), ("http://localhost/", "unsafe_url"), ("ftp://site.test/", "unsafe_url"),
    ("http://169.254.169.254/latest/meta-data", "unsafe_url"), ("http://private.test/", "unsafe_url"),
    ("http://user:pw@site.test/", "unsafe_url"), ("not a url", "unsafe_url"),
])
def test_unsafe_seeds_are_rejected_and_no_job_is_created(env, url, code):
    r = env.client.post("/api/v1/crawl", json={"url": url})
    assert r.status_code == 422 and r.json()["error"]["code"] == code
    assert env.client.get("/api/v1/jobs").json()["total"] == 0


@pytest.mark.parametrize("body,code", [
    ({"url": "https://site.test/", "max_pages": 100000}, "invalid_limit"),
    ({"url": "https://site.test/", "max_depth": 99}, "invalid_limit"),
    ({"url": "https://site.test/", "allowed_domains": ["bad domain"]}, "invalid_domain"),
])
def test_invalid_limits_and_domains_rejected(env, body, code):
    r = env.client.post("/api/v1/crawl", json=body)
    assert r.status_code == 422 and r.json()["error"]["code"] == code


@pytest.mark.parametrize("body", [
    {}, {"url": ""}, {"url": "https://site.test/", "max_pages": 0}, {"url": "https://site.test/", "max_depth": -1},
    {"url": "https://site.test/", "surprise": 1}, {"url": "https://site.test/", "allowed_domains": "x.test"},
])
def test_malformed_body_is_422_validation_error(env, body):
    r = env.client.post("/api/v1/crawl", json=body)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_queue_limit_returns_429(tmp_path, mock):
    engine = SearchEngine(make_settings(tmp_path, jobs_max_pending=1), embedder=HashingEmbedder())
    try:
        with TestClient(create_app(engine=engine, resolver=fake_resolver)) as c:
            mock.get("https://site.test/robots.txt").respond(404)
            mock.get("https://site.test/").respond(200, html=page("Home", "hello"))
            # fill the queue without the worker consuming: pause it by cancelling it
            c.portal.call(c.app.state.jobs.stop)
            assert c.post("/api/v1/crawl", json={"url": "https://site.test/"}).status_code == 202
            r = c.post("/api/v1/crawl", json={"url": "https://site.test/"})
            assert r.status_code == 429 and r.json()["error"]["code"] == "queue_full"
    finally:
        engine.close()


def test_jobs_listing_filters_and_404(mock, env):
    site(mock)
    ok = wait_done(env.client, env.client.post("/api/v1/crawl", json={"url": "https://site.test/"}).json()["id"])
    mock.get("https://bad.test/robots.txt").respond(404)
    mock.get("https://bad.test/").respond(503)
    bad = wait_done(env.client, env.client.post("/api/v1/crawl", json={"url": "https://bad.test/"}).json()["id"])
    assert env.client.get("/api/v1/jobs").json()["total"] == 2
    assert [j["id"] for j in env.client.get("/api/v1/jobs", params={"status": "failed"}).json()["items"]] == [bad["id"]]
    assert [j["id"] for j in env.client.get("/api/v1/jobs", params={"status": "completed"}).json()["items"]] == [ok["id"]]
    assert env.client.get("/api/v1/jobs", params={"status": "bogus"}).status_code == 422
    assert env.client.get("/api/v1/jobs/nope").status_code == 404
    assert env.client.get("/api/v1/jobs/nope/urls").json()["error"]["code"] == "not_found"


# ---- persistence / recovery ----------------------------------------------------------------
def test_restart_fails_interrupted_jobs_and_runs_pending_ones(mock, tmp_path):
    settings = make_settings(tmp_path)
    store = JobStore(settings.db_path)
    running = store.create("https://site.test/", 5, 1, [], True)
    store.mark_running(running.id)
    waiting = store.create("https://site.test/", 5, 1, [], True)
    store.close()
    site(mock)
    engine = SearchEngine(settings, embedder=HashingEmbedder())
    try:
        with TestClient(create_app(engine=engine, resolver=fake_resolver)) as c:
            interrupted = c.get(f"/api/v1/jobs/{running.id}").json()
            assert interrupted["status"] == "failed" and "interrupted" in interrupted["error"]
            assert wait_done(c, waiting.id)["status"] == "completed"
    finally:
        engine.close()


def test_job_history_survives_restart(mock, tmp_path):
    settings = make_settings(tmp_path)
    site(mock)
    e1 = SearchEngine(settings, embedder=HashingEmbedder())
    with TestClient(create_app(engine=e1, resolver=fake_resolver)) as c:
        job_id = wait_done(c, c.post("/api/v1/crawl", json={"url": "https://site.test/"}).json()["id"])["id"]
    e1.close()
    e2 = SearchEngine(settings, embedder=HashingEmbedder())
    try:
        with TestClient(create_app(engine=e2, resolver=fake_resolver)) as c:
            job = c.get(f"/api/v1/jobs/{job_id}").json()
            assert job["status"] == "completed" and job["progress"]["pages_crawled"] == 3
    finally:
        e2.close()


# ---- store + migration ---------------------------------------------------------------------
def test_store_progress_never_goes_backwards(tmp_path):
    store = JobStore(tmp_path / "j.db")
    job = store.create("https://a.test/", 5, 1, ["b.test"], True)
    store.mark_running(job.id)
    store.update_progress(job.id, processed=5, crawled=4, failed=1, skipped=0)
    store.update_progress(job.id, processed=3, crawled=2, failed=0, skipped=0)       # stale write
    got = store.get(job.id)
    assert (got.urls_processed, got.pages_crawled, got.pages_failed) == (5, 4, 1) and got.allowed_domains == ["b.test"]
    with pytest.raises(ValueError):
        store.finish(job.id, "running")
    store.close()


def test_old_database_gets_new_columns(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("""CREATE TABLE crawl_jobs (id TEXT PRIMARY KEY, seed_url TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending', max_pages INTEGER NOT NULL, max_depth INTEGER NOT NULL,
        allowed_domains TEXT NOT NULL DEFAULT '[]', pages_crawled INTEGER NOT NULL DEFAULT 0,
        pages_failed INTEGER NOT NULL DEFAULT 0, pages_skipped INTEGER NOT NULL DEFAULT 0, error TEXT,
        created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT);
        INSERT INTO crawl_jobs (id, seed_url, max_pages, max_depth, created_at) VALUES ('j1','https://a.test/',5,1,'t');""")
    conn.commit(); conn.close()
    conn = init_db(path)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(crawl_jobs)")}
    assert {"auto_index", "phase", "urls_processed", "stored", "index_result"} <= cols
    init_db(path).close()                                    # idempotent
    store = JobStore(path)
    assert store.get("j1").phase == "queued" and store.get("j1").auto_index is True
