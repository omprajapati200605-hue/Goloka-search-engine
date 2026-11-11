"""API tests: real FastAPI app, real SQLite/BM25/FAISS, stand-in embedder, no network."""
import threading

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings
from app.engine import SearchEngine
from app.extraction.extractor import extract_page
from app.search.embedder import EmbeddingError
from tests.fake_embedder import HashingEmbedder

CONCEPTS = {"car": "vehicle", "automobile": "vehicle", "truck": "vehicle", "pasta": "food", "noodles": "food"}
PAGES = {
    "cars": ("Fast cars", "A fast car drives down the road. The car engine is loud and quick."),
    "pasta": ("Cooking pasta", "Boil pasta in salted water, then add tomato sauce and basil."),
    "space": ("Astronomy", "Telescopes reveal distant galaxies and bright nebulae far from earth."),
}


def make_settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path / "data", db_path=tmp_path / "data" / "goloka.db",
                    vector_dir=tmp_path / "data" / "vectors")


def store(engine, key, title=None, body=None):
    t, b = PAGES.get(key, (title, body))
    t, b = title or t, body or b
    return engine.repo.upsert_page(extract_page(f"<title>{t}</title><p>{b}</p>", f"https://site.test/{key}")).document


@pytest.fixture
def engine(tmp_path):
    e = SearchEngine(make_settings(tmp_path), embedder=HashingEmbedder(concepts=CONCEPTS))
    yield e
    e.close()


@pytest.fixture
def client(engine):
    with TestClient(create_app(engine=engine)) as c:
        yield c


@pytest.fixture
def loaded(engine, client):
    for key in PAGES:
        store(engine, key)
    assert client.post("/api/v1/index").json()["status"] == "completed"
    return client


# ---- health --------------------------------------------------------------------------------
def test_health_empty(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    b = r.json()
    assert b["status"] == "ok" and b["version"] and b["documents"]["total"] == 0
    assert b["indexes"]["semantic"]["status"] == "not_loaded" and b["last_index"] is None


def test_health_reflects_pending_then_indexed(engine, client):
    store(engine, "cars")
    assert client.get("/api/v1/health").json()["documents"]["pending_indexing"] == 1
    client.post("/api/v1/index")
    b = client.get("/api/v1/health").json()
    assert b["documents"]["pending_indexing"] == 0 and b["documents"]["indexed"] == 1
    assert b["indexes"]["keyword_documents"] == 1 and b["indexes"]["vector_chunks"] >= 1
    assert b["indexes"]["semantic"]["status"] == "ready" and b["last_index"]["status"] == "completed"


# ---- indexing ------------------------------------------------------------------------------
def test_index_is_idempotent_and_does_not_reembed(engine, client):
    for key in PAGES:
        store(engine, key)
    first = client.post("/api/v1/index").json()
    assert first["keyword"]["added"] == 3 and first["vectors"]["added"] == 3 and first["pending"] == 0
    embedded = len(engine.embedder.embedded_texts)
    second = client.post("/api/v1/index").json()
    assert second["keyword"]["unchanged"] == 3 and second["vectors"]["unchanged"] == 3
    assert len(engine.embedder.embedded_texts) == embedded


def test_index_picks_up_updated_and_removed_documents(engine, loaded):
    store(engine, "pasta", body="Now about volcanoes and lava flows.")
    engine.repo.delete(store(engine, "space").id)
    r = loaded.post("/api/v1/index").json()
    assert r["keyword"]["updated"] == 1 and r["vectors"]["updated"] == 1 and r["keyword"]["removed"] == 1
    assert loaded.get("/api/v1/search/keyword", params={"q": "volcanoes"}).json()["total_returned"] == 1
    assert loaded.get("/api/v1/search/keyword", params={"q": "galaxies"}).json()["total_returned"] == 0


def test_embedding_down_gives_partial_but_keyword_works(tmp_path):
    class Down(HashingEmbedder):
        def embed(self, texts):
            raise EmbeddingError("model offline")

    e = SearchEngine(make_settings(tmp_path), embedder=Down())
    try:
        with TestClient(create_app(engine=e)) as c:
            store(e, "pasta")
            r = c.post("/api/v1/index").json()
            assert r["status"] == "partial" and r["vectors"] is None and r["pending"] == 1
            assert any("model offline" in w for w in r["warnings"])
            kw = c.get("/api/v1/search/keyword", params={"q": "pasta"}).json()
            assert kw["total_returned"] == 1
            hybrid = c.get("/api/v1/search", params={"q": "pasta"}).json()
            assert hybrid["results"][0]["url"] == "https://site.test/pasta" and hybrid["results"][0]["semantic_score"] is None
            assert any("not indexed yet" in w for w in hybrid["warnings"])      # still pending: vectors missing
            sem = c.get("/api/v1/search/semantic", params={"q": "pasta"}).json()
            assert sem["results"] == []
            assert c.get("/api/v1/health").json()["status"] == "degraded"
    finally:
        e.close()


def test_concurrent_index_returns_409(engine, client):
    assert engine._index_gate.acquire(blocking=False)
    try:
        r = client.post("/api/v1/index")
    finally:
        engine._index_gate.release()
    assert r.status_code == 409 and r.json()["error"]["code"] == "indexing_in_progress"


# ---- search --------------------------------------------------------------------------------
def test_hybrid_search_result_shape(loaded):
    b = loaded.get("/api/v1/search", params={"q": "telescopes galaxies"}).json()
    assert b["mode"] == "hybrid" and b["method"] == "weighted" and b["total_returned"] == len(b["results"]) >= 1
    top = b["results"][0]
    assert top["title"] == "Astronomy" and top["url"] == "https://site.test/space" and top["domain"] == "site.test"
    assert 0 < top["combined_score"] <= 1 and top["bm25_score"] > 0 and top["semantic_score"] > 0
    assert "galaxies" in top["snippet"] and top["document_id"]
    assert loaded.get(f"/api/v1/documents/{top['document_id']}").json()["url"] == top["url"]


def test_keyword_and_semantic_modes(loaded):
    kw = loaded.get("/api/v1/search/keyword", params={"q": "automobile"}).json()
    assert kw["mode"] == "keyword" and kw["results"] == []                 # BM25 knows only literal words
    sem = loaded.get("/api/v1/search/semantic", params={"q": "automobile"}).json()
    assert sem["mode"] == "semantic" and sem["results"][0]["url"] == "https://site.test/cars"
    assert sem["results"][0]["bm25_score"] is None and sem["results"][0]["semantic_score"] > 0


def test_hybrid_finds_by_meaning_and_rrf_method(loaded):
    b = loaded.get("/api/v1/search", params={"q": "noodles", "method": "rrf"}).json()
    assert b["method"] == "rrf" and b["results"][0]["url"] == "https://site.test/pasta"


def test_k_and_min_score(loaded):
    assert len(loaded.get("/api/v1/search", params={"q": "car pasta galaxies", "k": 1}).json()["results"]) == 1
    assert loaded.get("/api/v1/search", params={"q": "pasta", "min_score": 1.0}).json()["total_returned"] <= 1


def test_no_results_is_empty_list_not_invented(loaded):
    b = loaded.get("/api/v1/search", params={"q": "zzzqqq"}).json()
    assert b["results"] == [] and b["total_returned"] == 0


def test_search_before_indexing_warns(engine, client):
    store(engine, "pasta")
    b = client.get("/api/v1/search/keyword", params={"q": "pasta"}).json()
    assert b["results"] == [] and any("not indexed yet" in w and "/api/v1/index" in w for w in b["warnings"])


@pytest.mark.parametrize("path,params", [
    ("/api/v1/search", {}), ("/api/v1/search", {"q": ""}), ("/api/v1/search", {"q": "   "}),
    ("/api/v1/search", {"q": "x" * 501}), ("/api/v1/search", {"q": "a", "k": 0}),
    ("/api/v1/search", {"q": "a", "k": 51}), ("/api/v1/search", {"q": "a", "method": "bogus"}),
    ("/api/v1/search", {"q": "a", "min_score": 2}), ("/api/v1/search/keyword", {}),
    ("/api/v1/search/semantic", {"q": " "}), ("/api/v1/documents", {"limit": 0}),
    ("/api/v1/documents", {"limit": 201}), ("/api/v1/documents", {"offset": -1}),
])
def test_validation_errors_use_envelope(client, path, params):
    r = client.get(path, params=params)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


# ---- documents -----------------------------------------------------------------------------
def test_documents_list_pagination_and_filters(engine, client):
    for key in PAGES:
        store(engine, key)
    engine.repo.upsert_page(extract_page("<title>Other</title><p>Different site text here.</p>", "https://other.test/x"))
    b = client.get("/api/v1/documents").json()
    assert b["total"] == 4 and len(b["items"]) == 4 and b["limit"] == 50 and b["items"][0]["indexed"] is False
    assert "text" not in b["items"][0] and b["items"][0]["word_count"] > 0
    page = client.get("/api/v1/documents", params={"limit": 3, "offset": 3}).json()
    assert len(page["items"]) == 1 and page["total"] == 4
    assert client.get("/api/v1/documents", params={"domain": "other.test"}).json()["total"] == 1
    engine.repo.mark_status("https://site.test/cars", "gone")
    assert client.get("/api/v1/documents", params={"status": "gone"}).json()["total"] == 1


def test_document_detail_and_404(loaded, engine):
    doc = engine.repo.get_by_url("https://site.test/pasta")
    b = loaded.get(f"/api/v1/documents/{doc.id}").json()
    assert b["title"] == "Cooking pasta" and "tomato sauce" in b["text"] and b["indexed"] is True
    assert b["metadata"]["word_count"] > 0
    r = loaded.get("/api/v1/documents/does-not-exist")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


# ---- errors, docs --------------------------------------------------------------------------
def test_unknown_route_and_wrong_method_use_envelope(client):
    assert client.get("/api/v1/nothing").json()["error"]["code"] == "not_found"
    r = client.post("/api/v1/search", params={"q": "a"})
    assert r.status_code == 405 and r.json()["error"]["code"] == "method_not_allowed"


def test_unexpected_exception_is_500_without_leaking_details(engine):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail")

    engine.search = boom
    with TestClient(create_app(engine=engine), raise_server_exceptions=False) as c:
        r = c.get("/api/v1/search", params={"q": "a"})
    assert r.status_code == 500 and r.json() == {"error": {"code": "internal_error", "message": "Internal server error"}}


def test_openapi_and_docs_available(client):
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]
    for p in ("/api/v1/health", "/api/v1/index", "/api/v1/search", "/api/v1/search/keyword",
              "/api/v1/search/semantic", "/api/v1/documents", "/api/v1/documents/{document_id}"):
        assert p in paths
    for p in ("/api/v1/crawl", "/api/v1/jobs", "/api/v1/jobs/{job_id}", "/api/v1/jobs/{job_id}/urls"):
        assert p in paths
    assert paths["/api/v1/search"]["get"]["summary"] and client.get("/docs").status_code == 200
    assert "text/html" in client.get("/").headers["content-type"]       # the search website (see test_web.py)


def test_state_survives_restart(tmp_path):
    s = make_settings(tmp_path)
    first = SearchEngine(s, embedder=HashingEmbedder(concepts=CONCEPTS))
    for key in PAGES:
        store(first, key)
    with TestClient(create_app(engine=first)) as c:
        c.post("/api/v1/index")
    first.close()
    emb = HashingEmbedder(concepts=CONCEPTS)
    second = SearchEngine(s, embedder=emb)
    try:
        with TestClient(create_app(engine=second)) as c:
            assert c.get("/api/v1/search", params={"q": "tomato sauce"}).json()["results"][0]["title"] == "Cooking pasta"
            assert c.post("/api/v1/index").json()["vectors"]["unchanged"] == 3
            assert emb.embedded_texts == ["tomato sauce"]      # only the query was embedded
    finally:
        second.close()
