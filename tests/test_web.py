"""The search website: served by the API, safe against hostile page content."""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import WEB_DIR, create_app
from app.config import Settings
from app.engine import SearchEngine
from tests.fake_embedder import HashingEmbedder


@pytest.fixture
def client(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path / "d", db_path=tmp_path / "d" / "g.db", vector_dir=tmp_path / "d" / "v")
    engine = SearchEngine(s, embedder=HashingEmbedder())
    with TestClient(create_app(engine=engine)) as c:
        yield c
    engine.close()


def test_home_page_is_served_with_strict_headers(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"] and "<title>Goloka search</title>" in r.text
    csp = r.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp and "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert r.headers["x-content-type-options"] == "nosniff"


def test_static_files_are_served(client):
    assert "javascript" in client.get("/static/app.js").headers["content-type"]
    assert "text/css" in client.get("/static/styles.css").headers["content-type"]
    assert client.get("/static/../engine.py").status_code in (400, 404)       # no path escape
    assert client.get("/static/nothing.js").status_code == 404
    assert client.get("/static/index.html").status_code == 404                  # the page is only served from / with its headers


def test_page_needs_no_inline_script_style_or_external_resource():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"<script(?![^>]*\bsrc=)", html) and not re.search(r"<style", html) and " style=" not in html
    assert not re.search(r"""(?:src|href)=["']https?://""", html)
    assert re.search(r"\bon\w+=", html) is None                                 # no inline event handlers


@pytest.mark.parametrize("banned", ["innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(",
                                    "new Function", "DOMParser", "createContextualFragment", "setAttribute(\"style\"",
                                    "javascript:"])
def test_script_never_parses_server_data_as_html(banned):
    """Crawled pages are untrusted: titles and snippets may only be inserted as text."""
    js = (WEB_DIR / "static" / "app.js").read_text(encoding="utf-8")
    code = re.sub(r"/\*.*?\*/", "", js, flags=re.S)                              # the header comment names these APIs
    assert banned not in code


def test_links_are_checked_before_use():
    js = (WEB_DIR / "static" / "app.js").read_text(encoding="utf-8")
    assert 'u.protocol === "http:" || u.protocol === "https:"' in js and "noopener noreferrer" in js


def test_page_works_without_api_key_header_and_api_still_protected(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path / "d", db_path=tmp_path / "d" / "g.db", vector_dir=tmp_path / "d" / "v",
                 api_key="k")
    engine = SearchEngine(s, embedder=HashingEmbedder())
    try:
        with TestClient(create_app(engine=engine)) as c:
            assert c.get("/").status_code == 200                                 # the shell itself holds no data
            assert c.get("/api/v1/documents").status_code == 401
    finally:
        engine.close()


def test_add_pages_link_is_hidden_until_the_owner_signs_in():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert re.search(r'<a[^>]*id="nav-add"[^>]*\shidden', html), "the Add pages link must start hidden"
    js = (WEB_DIR / "static" / "app.js").read_text(encoding="utf-8")
    assert "checkOwner" in js and "ownerPrompt" in js
