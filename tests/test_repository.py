import itertools
import sqlite3
import threading

import pytest

from app.extraction.extractor import extract_page
from app.storage.repository import DocumentRepository, document_id_for


def make_page(url="https://a.test/one", title="Title", body="Some readable body text", crawled_at="2026-01-01T00:00:00+00:00",
              final_url=None):
    return extract_page(f"<title>{title}</title><p>{body}</p>", url, final_url=final_url, crawled_at=crawled_at)


@pytest.fixture
def clock():
    counter = itertools.count(1)
    return lambda: f"2026-06-01T00:00:{next(counter):02d}+00:00"


@pytest.fixture
def repo(tmp_path, clock):
    r = DocumentRepository(tmp_path / "db" / "goloka.db", clock=clock)
    yield r
    r.close()


# ---- create ----------------------------------------------------------------------------------
def test_create_stores_all_required_fields(repo):
    res = repo.upsert_page(make_page(title="Café ☕", body="Unicode ünïcode body"))
    assert res.action == "created" and res.stored
    d = res.document
    assert d.id == document_id_for("https://a.test/one")
    assert (d.url, d.title, d.domain, d.crawl_status) == ("https://a.test/one", "Café ☕", "a.test", "ok")
    assert "Unicode ünïcode body" in d.text
    assert d.content_hash and len(d.content_hash) == 64
    assert d.crawled_at == "2026-01-01T00:00:00+00:00" and d.last_updated_at.startswith("2026-06-01")
    assert d.indexed_hash is None and d.needs_indexing
    assert d.metadata["word_count"] == 3 and d.metadata["requested_url"] == "https://a.test/one"


def test_id_is_stable_and_unique_per_url():
    assert document_id_for("https://a.test/x") == document_id_for("https://a.test/x")
    assert document_id_for("https://a.test/x") != document_id_for("https://a.test/y")


def test_document_keyed_by_final_url_original_kept_in_metadata(repo):
    d = repo.upsert_page(make_page(url="https://a.test/old", final_url="https://a.test/new")).document
    assert d.url == "https://a.test/new" and d.metadata["requested_url"] == "https://a.test/old"
    assert repo.get_by_url("https://a.test/new").id == d.id and repo.get_by_url("https://a.test/old") is None


# ---- change detection ------------------------------------------------------------------------
def test_unchanged_only_refreshes_crawl_time(repo):
    first = repo.upsert_page(make_page()).document
    repo.mark_indexed(first.id, first.content_hash)
    res = repo.upsert_page(make_page(crawled_at="2026-02-02T00:00:00+00:00"))
    assert res.action == "unchanged"
    d = res.document
    assert d.id == first.id and d.crawled_at == "2026-02-02T00:00:00+00:00"
    assert d.last_updated_at == first.last_updated_at     # content did not change
    assert d.indexed_hash == first.content_hash and not d.needs_indexing  # no re-indexing needed
    assert repo.count() == 1


def test_changed_content_updates_and_requires_reindex(repo):
    first = repo.upsert_page(make_page(body="old words")).document
    repo.mark_indexed(first.id, first.content_hash)
    res = repo.upsert_page(make_page(body="brand new words", crawled_at="2026-03-03T00:00:00+00:00"))
    assert res.action == "updated"
    d = res.document
    assert d.id == first.id and "brand new words" in d.text and d.content_hash != first.content_hash
    assert d.last_updated_at > first.last_updated_at
    assert d.indexed_hash == first.content_hash and d.needs_indexing   # stale index marker kept
    assert [x.id for x in repo.needs_indexing()] == [d.id]
    assert repo.count() == 1


def test_title_change_counts_as_change(repo):
    repo.upsert_page(make_page(title="One"))
    assert repo.upsert_page(make_page(title="Two")).action == "updated"


# ---- duplicates & empty ----------------------------------------------------------------------
def test_duplicate_content_on_new_url_not_stored(repo):
    a = repo.upsert_page(make_page(url="https://a.test/1")).document
    res = repo.upsert_page(make_page(url="https://a.test/2"))
    assert res.action == "duplicate" and res.duplicate_of == a.id and not res.stored
    assert repo.count() == 1 and repo.get_by_url("https://a.test/2") is None


def test_existing_url_may_change_to_duplicate_content(repo):
    repo.upsert_page(make_page(url="https://a.test/1", body="same"))
    repo.upsert_page(make_page(url="https://a.test/2", body="different"))
    assert repo.upsert_page(make_page(url="https://a.test/2", body="same")).action == "updated"
    assert repo.count() == 2


def test_empty_pages_not_stored(repo):
    page = extract_page("<html><body><script>x()</script></body></html>", "https://a.test/e")
    res = repo.upsert_page(page)
    assert res.action == "empty" and res.document is None and repo.count(None) == 0


# ---- status handling -------------------------------------------------------------------------
def test_mark_status_gone_hides_from_search_set_and_index_queue(repo):
    d = repo.upsert_page(make_page()).document
    assert repo.mark_status("https://a.test/one", "gone")
    assert repo.get(d.id).crawl_status == "gone" and repo.count() == 0 and repo.count("gone") == 1
    assert repo.needs_indexing() == [] and not repo.get(d.id).needs_indexing
    assert not repo.mark_status("https://nope.test/", "gone")


def test_revived_page_is_reindexed_even_if_content_identical(repo):
    d = repo.upsert_page(make_page()).document
    repo.mark_indexed(d.id, d.content_hash)
    repo.mark_status("https://a.test/one", "gone")
    res = repo.upsert_page(make_page())
    assert res.action == "updated" and res.document.crawl_status == "ok"
    assert res.document.indexed_hash is None and res.document.needs_indexing


def test_gone_document_does_not_block_new_url_with_same_content(repo):
    repo.upsert_page(make_page(url="https://a.test/1"))
    repo.mark_status("https://a.test/1", "gone")
    assert repo.upsert_page(make_page(url="https://a.test/2")).action == "created"


def test_mark_indexed_ignores_stale_hash(repo):
    d = repo.upsert_page(make_page(body="v1")).document
    repo.upsert_page(make_page(body="v2"))
    assert repo.mark_indexed(d.id, d.content_hash) is False       # indexer worked on v1; doc is now v2
    assert repo.get(d.id).indexed_hash is None
    new = repo.get(d.id)
    assert repo.mark_indexed(new.id, new.content_hash) is True and not repo.get(d.id).needs_indexing


# ---- reads -----------------------------------------------------------------------------------
def test_get_missing_returns_none(repo):
    assert repo.get("nope") is None and repo.get_by_url("https://x.test/") is None


def test_list_pagination_filters_and_text_option(repo):
    for i in range(5):
        repo.upsert_page(make_page(url=f"https://a.test/{i}", body=f"body {i}", crawled_at=f"2026-01-0{i + 1}T00:00:00+00:00"))
    repo.upsert_page(make_page(url="https://b.test/x", body="other", crawled_at="2026-02-01T00:00:00+00:00"))
    docs, total = repo.list(limit=2, offset=0)
    assert total == 6 and [d.url for d in docs] == ["https://b.test/x", "https://a.test/4"]  # newest crawl first
    assert all(d.text == "" for d in docs)
    page2, _ = repo.list(limit=2, offset=2)
    assert [d.url for d in page2] == ["https://a.test/3", "https://a.test/2"]
    docs, total = repo.list(domain="A.test", limit=100, include_text=True)
    assert total == 5 and all(d.text for d in docs)
    repo.mark_status("https://a.test/0", "gone")
    assert repo.list(status="gone")[1] == 1
    assert repo.list(limit=0)[0] == []


def test_get_many(repo):
    ids = [repo.upsert_page(make_page(url=f"https://a.test/{i}", body=f"b{i}")).document.id for i in range(3)]
    got = repo.get_many(ids + ["missing"])
    assert set(got) == set(ids) and repo.get_many([]) == {}


def test_iter_documents_streams_all_in_batches(repo):
    for i in range(7):
        repo.upsert_page(make_page(url=f"https://a.test/{i}", body=f"b{i}"))
    repo.mark_status("https://a.test/3", "gone")
    ids = [d.id for d in repo.iter_documents(batch_size=3)]
    assert len(ids) == 6 and ids == sorted(ids) and len(set(ids)) == 6
    assert len(list(repo.iter_documents(status=None, batch_size=3))) == 7


def test_stats(repo):
    for i in range(3):
        repo.upsert_page(make_page(url=f"https://a.test/{i}", body=f"b{i}", crawled_at=f"2026-01-0{i + 1}T00:00:00+00:00"))
    d = repo.get_by_url("https://a.test/0")
    repo.mark_indexed(d.id, d.content_hash)
    repo.mark_status("https://a.test/2", "gone")
    s = repo.stats()
    assert s.total == 3 and s.by_status == {"ok": 2, "gone": 1}
    assert s.indexed == 1 and s.pending_indexing == 1 and s.last_crawled_at == "2026-01-03T00:00:00+00:00"


def test_delete_cascades_to_chunks(repo):
    d = repo.upsert_page(make_page()).document
    repo._conn.execute("INSERT INTO chunks (faiss_id, document_id, chunk_index, text) VALUES (1, ?, 0, 'c')", (d.id,))
    repo._conn.commit()
    assert repo.delete(d.id) and not repo.delete(d.id)
    assert repo._conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"] == 0


# ---- persistence & concurrency ---------------------------------------------------------------
def test_data_survives_restart(tmp_path):
    path = tmp_path / "persist.db"
    with DocumentRepository(path) as r1:
        d = r1.upsert_page(make_page(title="Persistent", body="still here after restart")).document
        r1.mark_indexed(d.id, d.content_hash)
    with DocumentRepository(path) as r2:
        got = r2.get(d.id)
        assert got.title == "Persistent" and "still here" in got.text and got.metadata["word_count"] == 4
        assert not got.needs_indexing and r2.count() == 1
        assert r2.upsert_page(make_page(title="Persistent", body="still here after restart")).action == "unchanged"


def test_concurrent_writers_are_safe(repo):
    errors = []

    def work(t):
        try:
            for i in range(20):
                repo.upsert_page(make_page(url=f"https://a.test/{t}/{i}", body=f"unique {t} {i}"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(t,)) for t in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and repo.count() == 160


def test_unique_url_constraint_enforced_at_db_level(repo):
    repo.upsert_page(make_page())
    with pytest.raises(sqlite3.IntegrityError):
        repo._conn.execute(
            "INSERT INTO documents (id,url,domain,content_hash,crawled_at,last_updated_at) "
            "VALUES ('other','https://a.test/one','a.test','h','t','t')")
