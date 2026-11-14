"""GOLOKA_SEMANTIC_ENABLED=false: BM25 only, no AI model and no faiss, so the server fits in 512 MB."""
import subprocess
import sys
import textwrap

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import Settings
from app.engine import SearchEngine
from app.extraction.extractor import extract_page


def _settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path / "d", db_path=tmp_path / "d" / "g.db",
                    vector_dir=tmp_path / "d" / "v", semantic_enabled=False)


def test_keyword_only_engine_indexes_and_searches(tmp_path):
    engine = SearchEngine(_settings(tmp_path))
    page = extract_page("<title>Thales of Miletus</title><p>Thales is often called the first mathematician in the "
                        "Greek tradition.</p>", "https://example.org/thales")
    engine.repo.upsert_page(page)
    assert engine.semantic_status == "disabled" and engine.vectors is None
    report = engine.index_pending()
    assert report.vectors is None and report.warnings == [] and report.pending == 0
    result = engine.search("first mathematician")
    assert [r.title for r in result.results] == ["Thales of Miletus"]
    assert result.warnings == []                      # keyword-only is a setting, not a problem
    engine.close()


def test_health_reports_disabled_semantic(tmp_path):
    engine = SearchEngine(_settings(tmp_path))
    with TestClient(create_app(engine=engine)) as c:
        body = c.get("/api/v1/health").json()
        assert body["status"] == "ok" and body["indexes"]["semantic"]["status"] == "disabled"
        assert c.get("/api/v1/search", params={"q": "anything"}).status_code == 200
    engine.close()


def test_keyword_only_never_imports_faiss_or_torch(tmp_path):
    code = textwrap.dedent(f"""
        import sys
        from app.config import Settings
        from app.engine import SearchEngine
        s = Settings(_env_file=None, data_dir=r"{tmp_path}/x", db_path=r"{tmp_path}/x/g.db",
                     vector_dir=r"{tmp_path}/x/v", semantic_enabled=False)
        e = SearchEngine(s); e.index_pending(); e.search("hello"); e.close()
        bad = [m for m in ("faiss", "torch", "sentence_transformers") if m in sys.modules]
        print("LOADED:" + ",".join(bad))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".", timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("LOADED:"), out.stdout
