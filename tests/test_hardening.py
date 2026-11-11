"""Optional API key, CORS and configuration added in the final stage."""
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings
from app.engine import SearchEngine
from app.search.hybrid import HybridConfig
from tests.fake_embedder import HashingEmbedder


def build(tmp_path, **kw):
    settings = Settings(_env_file=None, data_dir=tmp_path / "d", db_path=tmp_path / "d" / "g.db",
                        vector_dir=tmp_path / "d" / "v", **kw)
    return settings, SearchEngine(settings, embedder=HashingEmbedder())


@pytest.fixture
def secured(tmp_path):
    settings, engine = build(tmp_path, api_key="s3cret-key")
    with TestClient(create_app(engine=engine)) as c:
        yield c
    engine.close()


PROTECTED = [("get", "/api/v1/search", {"q": "a"}), ("get", "/api/v1/search/keyword", {"q": "a"}),
             ("get", "/api/v1/search/semantic", {"q": "a"}), ("get", "/api/v1/documents", {}),
             ("get", "/api/v1/documents/x", {}), ("post", "/api/v1/index", {}), ("get", "/api/v1/jobs", {}),
             ("get", "/api/v1/jobs/x", {}), ("get", "/api/v1/jobs/x/urls", {})]


@pytest.mark.parametrize("method,path,params", PROTECTED)
def test_every_protected_endpoint_requires_the_key(secured, method, path, params):
    r = getattr(secured, method)(path, params=params)
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"
    assert r.headers["www-authenticate"] == "ApiKey"
    assert getattr(secured, method)(path, params=params, headers={"X-API-Key": "wrong"}).status_code == 401


def test_crawl_requires_the_key_and_creates_no_job(secured):
    assert secured.post("/api/v1/crawl", json={"url": "https://site.test/"}).status_code == 401
    ok = secured.get("/api/v1/jobs", headers={"X-API-Key": "s3cret-key"}).json()
    assert ok["total"] == 0


def test_correct_key_works_and_open_endpoints_stay_open(secured):
    h = {"X-API-Key": "s3cret-key"}
    assert secured.get("/api/v1/search", params={"q": "a"}, headers=h).status_code == 200
    assert secured.get("/api/v1/documents", headers=h).status_code == 200
    assert secured.get("/api/v1/health").status_code == 200
    assert secured.get("/docs").status_code == 200 and secured.get("/openapi.json").status_code == 200
    assert "X-API-Key" in str(secured.get("/openapi.json").json()["components"]["securitySchemes"])


def test_no_key_configured_means_open(tmp_path):
    settings, engine = build(tmp_path)
    try:
        with TestClient(create_app(engine=engine)) as c:
            assert c.get("/api/v1/documents").status_code == 200
    finally:
        engine.close()


def test_api_key_never_appears_in_settings_repr():
    assert "s3cret" not in repr(Settings(_env_file=None, api_key="s3cret-key"))


def test_cors_only_for_configured_origins(tmp_path):
    settings, engine = build(tmp_path, cors_origins=["http://localhost:3000"])
    try:
        with TestClient(create_app(engine=engine)) as c:
            ok = c.get("/api/v1/health", headers={"Origin": "http://localhost:3000"})
            assert ok.headers["access-control-allow-origin"] == "http://localhost:3000"
            other = c.get("/api/v1/health", headers={"Origin": "http://evil.example"})
            assert "access-control-allow-origin" not in other.headers
    finally:
        engine.close()


def test_no_cors_headers_by_default(tmp_path):
    settings, engine = build(tmp_path)
    try:
        with TestClient(create_app(engine=engine)) as c:
            assert "access-control-allow-origin" not in c.get("/api/v1/health", headers={"Origin": "http://x.test"}).headers
    finally:
        engine.close()


def test_semantic_threshold_is_configurable():
    assert HybridConfig.from_settings(Settings(_env_file=None)).min_semantic_score == 0.2
    assert HybridConfig.from_settings(Settings(_env_file=None, hybrid_min_semantic_score=0.35)).min_semantic_score == 0.35
    with pytest.raises(ValueError):
        Settings(_env_file=None, hybrid_min_semantic_score=2)
