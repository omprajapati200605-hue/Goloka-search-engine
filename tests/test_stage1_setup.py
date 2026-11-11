import pytest
from pydantic import ValidationError

from app.config import Settings
from app.storage.db import init_db


def test_defaults_load():
    s = Settings(_env_file=None)
    assert s.crawl_default_max_pages <= s.crawl_hard_max_pages
    assert s.embedding_model.endswith("all-MiniLM-L6-v2")


def test_env_override(monkeypatch):
    monkeypatch.setenv("GOLOKA_CRAWL_CONCURRENCY", "8")
    assert Settings(_env_file=None).crawl_concurrency == 8


def test_invalid_overlap_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size_words=50, chunk_overlap_words=60)


def test_default_above_hard_limit_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, crawl_default_max_pages=5000)


def test_init_db_creates_tables_and_is_idempotent(tmp_path):
    db = tmp_path / "t.db"
    conn = init_db(db)
    names = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"documents", "crawl_jobs", "crawl_urls", "chunks", "schema_version"} <= names
    conn.close()
    conn = init_db(db)  # second run must not fail or duplicate version rows
    assert conn.execute("SELECT COUNT(*) c FROM schema_version").fetchone()["c"] == 1
    conn.close()


def test_data_survives_reconnect(tmp_path):
    db = tmp_path / "t.db"
    conn = init_db(db)
    conn.execute(
        "INSERT INTO documents (id,url,domain,content_hash,crawled_at,last_updated_at) "
        "VALUES ('d1','https://a.com/','a.com','h','t','t')")
    conn.commit()
    conn.close()
    conn = init_db(db)
    assert conn.execute("SELECT url FROM documents WHERE id='d1'").fetchone()["url"] == "https://a.com/"