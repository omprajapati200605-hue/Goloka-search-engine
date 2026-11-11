"""End-to-end (mocked network): Crawler -> extraction -> SQLite, including restart persistence."""
import pytest

from app.crawler.crawler import Crawler
from app.ingest import IngestStats, make_ingest_callback
from app.storage.repository import DocumentRepository


def html(title, body, links=()):
    anchors = "".join(f'<a href="{h}">l</a>' for h in links)
    return (f'<html lang="en"><head><title>{title}</title><meta name="description" content="d {title}"></head>'
            f"<body><nav><a href='/'>NAVJUNK</a></nav><main><h1>{title}</h1><p>{body}</p>{anchors}</main>"
            f"<footer>FOOTERJUNK</footer></body></html>")


def mock_site(mock, pages):
    mock.get("https://site.test/robots.txt").respond(404)
    for path, body in pages.items():
        mock.get(f"https://site.test{path}").respond(200, html=body)


async def crawl_into(repo, settings, resolver, **kwargs):
    stats = IngestStats()
    crawler = Crawler(settings, resolver=resolver, on_page=make_ingest_callback(repo, stats))
    try:
        summary = await crawler.crawl("https://site.test/", **kwargs)
    finally:
        await crawler.aclose()
    return summary, stats


@pytest.fixture
def repo(tmp_path):
    r = DocumentRepository(tmp_path / "e2e.db")
    yield r
    r.close()


async def test_crawl_extract_store(mock, settings, resolver, repo):
    mock_site(mock, {
        "/": html("Home", "Welcome to the home page", ["/a", "/b"]),
        "/a": html("Page A", "Alpha content about tomatoes"),
        "/b": html("Page B", "Beta content about basil"),
    })
    summary, stats = await crawl_into(repo, settings, resolver)
    assert summary.pages_crawled == 3 and stats.created == 3 and stats.stored == 3
    assert repo.count() == 3
    a = repo.get_by_url("https://site.test/a")
    assert a.title == "Page A" and a.domain == "site.test" and a.crawl_status == "ok"
    assert "Alpha content about tomatoes" in a.text
    assert "NAVJUNK" not in a.text and "FOOTERJUNK" not in a.text         # extraction cleaned it
    assert a.metadata["meta_description"] == "d Page A" and a.metadata["language"] == "en"
    assert a.crawled_at and a.last_updated_at and a.needs_indexing        # ready for the indexers


async def test_recrawl_unchanged_does_not_requeue_indexing(mock, settings, resolver, repo):
    pages = {"/": html("Home", "Stable content", ["/a"]), "/a": html("A", "Also stable")}
    mock_site(mock, pages)
    await crawl_into(repo, settings, resolver)
    for d in repo.needs_indexing():
        repo.mark_indexed(d.id, d.content_hash)
    assert repo.needs_indexing() == []
    _, stats = await crawl_into(repo, settings, resolver)
    assert stats.unchanged == 2 and stats.created == 0 and stats.updated == 0
    assert repo.needs_indexing() == [] and repo.count() == 2


async def test_recrawl_changed_page_is_updated_and_requeued(mock, settings, resolver, repo):
    mock_site(mock, {"/": html("Home", "Version one", ["/a"]), "/a": html("A", "Static")})
    await crawl_into(repo, settings, resolver)
    for d in repo.needs_indexing():
        repo.mark_indexed(d.id, d.content_hash)
    mock.clear()
    mock_site(mock, {"/": html("Home", "Version two, edited", ["/a"]), "/a": html("A", "Static")})
    _, stats = await crawl_into(repo, settings, resolver)
    assert stats.updated == 1 and stats.unchanged == 1
    pending = repo.needs_indexing()
    assert [d.url for d in pending] == ["https://site.test/"] and "Version two" in pending[0].text


async def test_duplicate_pages_across_urls_stored_once(mock, settings, resolver, repo):
    same = html("Same", "Identical body")
    mock_site(mock, {"/": html("Home", "Root", ["/x", "/y"]), "/x": same, "/y": same})
    summary, stats = await crawl_into(repo, settings, resolver)
    assert repo.count() == 2 and stats.created == 2


async def test_storage_survives_restart_after_crawl(mock, settings, resolver, tmp_path):
    path = tmp_path / "restart.db"
    mock_site(mock, {"/": html("Home", "Persistent crawl result", ["/a"]), "/a": html("A", "Second page")})
    with DocumentRepository(path) as repo1:
        await crawl_into(repo1, settings, resolver)
    with DocumentRepository(path) as repo2:
        assert repo2.count() == 2
        assert "Persistent crawl result" in repo2.get_by_url("https://site.test/").text


async def test_storage_failure_does_not_stop_crawl(mock, settings, resolver, repo):
    mock_site(mock, {"/": html("Home", "Root", ["/a"]), "/a": html("A", "Alpha")})
    repo.close()  # simulate a broken database
    crawler = Crawler(settings, resolver=resolver, on_page=make_ingest_callback(repo))
    summary = await crawler.crawl("https://site.test/")
    await crawler.aclose()
    assert summary.pages_failed == 2 and all("on_page failed" in o.detail for o in summary.outcomes)
