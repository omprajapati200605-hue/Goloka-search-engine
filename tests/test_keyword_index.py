import gzip
import json
import threading

import pytest

from app.extraction.extractor import extract_page
from app.search import keyword_index as ki
from app.search.keyword_index import KeywordIndex, default_index_path
from app.storage.repository import DocumentRepository

BACKENDS = ["rank_bm25", "native"]


@pytest.fixture(params=BACKENDS)
def index(request):
    return KeywordIndex(backend=request.param)


def add(index, doc_id, title, text, h=None):
    return index.add_document(doc_id, title, text, h or f"hash-{doc_id}-{title}-{text}")


def fill(index):
    add(index, "tomato", "Growing tomatoes", "Tomatoes need sun and water. Tomato plants love warm soil.")
    add(index, "basil", "Herb garden", "Basil grows next to tomatoes and likes sun.")
    add(index, "engine", "Search engine basics", "A search engine crawls pages and builds an index.")
    add(index, "cars", "Car engines", "Pistons and cylinders power a car engine.")
    return index


# ---- searching -----------------------------------------------------------------------------
def test_basic_ranking(index):
    fill(index)
    hits = index.search("tomato sun")
    assert [h.doc_id for h in hits][:2] == ["tomato", "basil"]
    assert all(h.score > 0 for h in hits)


def test_only_matching_documents_returned(index):
    fill(index)
    assert [h.doc_id for h in index.search("pistons")] == ["cars"]
    assert index.search("nonexistentterm") == []


def test_title_match_outranks_body_only_match(index):
    add(index, "title", "Kubernetes guide", "Container orchestration for beginners and experts alike.")
    add(index, "body", "Cloud notes", "Kubernetes is mentioned here once among other container topics today.")
    add(index, "other", "Cooking", "Pasta sauce with garlic and basil for dinner tonight friends.")
    hits = index.search("kubernetes")
    assert [h.doc_id for h in hits] == ["title", "body"]
    assert hits[0].title_score > 0 and hits[0].content_score == 0
    assert hits[1].title_score == 0 and hits[1].content_score > 0


def test_combined_score_formula(index):
    fill(index)
    h = index.search("search")[0]
    assert h.score == pytest.approx(h.content_score + index.title_boost * h.title_score)


def test_k_limits_results_and_ordering_is_deterministic(index):
    for i in range(12):
        add(index, f"d{i:02d}", "same title", "identical words here")
    hits = index.search("identical", k=5)
    assert len(hits) == 5 and [h.doc_id for h in hits] == [f"d{i:02d}" for i in range(5)]  # ties -> id order
    assert index.search("identical", k=0) == []


def test_results_sorted_by_score(index):
    fill(index)
    scores = [h.score for h in index.search("engine search car")]
    assert scores == sorted(scores, reverse=True)


def test_empty_index_and_empty_queries(index):
    assert index.search("anything") == []
    fill(index)
    assert index.search("") == [] and index.search("   ") == [] and index.search("!!!") == []


def test_stopword_query_still_works(index):
    add(index, "q", "To be or not to be", "That is the question")
    add(index, "z", "Other", "unrelated content entirely")
    assert [h.doc_id for h in index.search("to be or not to be")] == ["q"] or index.search("to be or not to be") == []
    assert [h.doc_id for h in index.search("the question")] == ["q"]


def test_documents_with_only_title_or_only_text(index):
    add(index, "t", "Lonely title words", "")
    add(index, "c", "", "Only body words here")
    assert [h.doc_id for h in index.search("lonely")] == ["t"]
    assert [h.doc_id for h in index.search("body")] == ["c"]


def test_non_english_text_is_searchable(index):
    add(index, "hi", "खोज इंजन", "यह एक हिन्दी खोज इंजन है")
    add(index, "en", "English", "plain english text")
    assert [h.doc_id for h in index.search("खोज")] == ["hi"]
    assert [h.doc_id for h in index.search("हिन्दी")] == ["hi"]


def test_backends_produce_identical_rankings():
    a, b = KeywordIndex(backend="rank_bm25"), KeywordIndex(backend="native")
    for idx in (a, b):
        fill(idx)
    for q in ("tomato", "search engine", "sun water", "car pistons engine"):
        ha, hb = a.search(q), b.search(q)
        assert [h.doc_id for h in ha] == [h.doc_id for h in hb]
        assert [h.score for h in ha] == pytest.approx([h.score for h in hb])


def test_bad_backend_name_fails_fast():
    with pytest.raises(ValueError):
        KeywordIndex(backend="nope")


# ---- incremental updates -------------------------------------------------------------------
def test_add_unchanged_document_is_noop(index):
    assert add(index, "a", "T", "text body", h="h1") is True
    assert add(index, "a", "T", "text body", h="h1") is False
    assert len(index) == 1


def test_documents_added_after_a_search_are_found(index):
    fill(index)
    assert index.search("zebra") == []
    add(index, "zoo", "Zoo", "A zebra lives here")
    assert [h.doc_id for h in index.search("zebra")] == ["zoo"]
    assert {h.doc_id for h in index.search("sun")} == {"tomato", "basil"}   # existing documents not lost


def test_update_replaces_old_terms(index):
    add(index, "a", "Old", "alpha words", h="v1")
    add(index, "b", "Other", "beta words", h="v0")
    assert add(index, "a", "New", "gamma words", h="v2") is True
    assert index.search("alpha") == [] and [h.doc_id for h in index.search("gamma")] == ["a"]
    assert len(index) == 2


def test_remove_document(index):
    fill(index)
    assert index.remove_document("cars") and not index.remove_document("cars")
    assert index.search("pistons") == [] and "cars" not in index and len(index) == 3


def test_concurrent_adds_and_searches(index):
    errors = []

    def writer(t):
        try:
            for i in range(30):
                add(index, f"{t}-{i}", f"title {t}", f"content number {i} unique{t}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def reader():
        try:
            for _ in range(30):
                index.search("content number")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)] + [threading.Thread(target=reader) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(index) == 120


# ---- persistence ---------------------------------------------------------------------------
def test_save_load_roundtrip(tmp_path, index):
    fill(index)
    path = tmp_path / "sub" / "idx.json.gz"
    index.save(path)
    assert path.exists() and not list(path.parent.glob("*.tmp"))
    loaded = KeywordIndex.load(path, backend=index._backend_name)
    assert len(loaded) == 4
    for q in ("tomato", "search engine", "pistons"):
        a, b = index.search(q), loaded.search(q)
        assert [h.doc_id for h in a] == [h.doc_id for h in b]
        assert [h.score for h in a] == pytest.approx([h.score for h in b])
    assert loaded.content_hash_of("cars") == index.content_hash_of("cars")


def test_snapshot_preserves_unicode(tmp_path):
    idx = KeywordIndex()
    add(idx, "hi", "खोज", "इंजन हिन्दी")
    idx.save(tmp_path / "i.json.gz")
    assert [h.doc_id for h in KeywordIndex.load(tmp_path / "i.json.gz").search("हिन्दी")] == ["hi"]


@pytest.mark.parametrize("writer", [
    lambda p: p.write_bytes(b"not gzip at all"),
    lambda p: p.write_bytes(gzip.compress(b"{broken json")),
    lambda p: p.write_bytes(gzip.compress(b'{"version": 1}')),
    lambda p: p.write_bytes(gzip.compress(json.dumps({"version": 1, "tokenizer_version": 1, "docs": []}).encode())),
    lambda p: p.write_bytes(b""),
])
def test_corrupt_snapshot_gives_empty_index_not_crash(tmp_path, writer):
    path = tmp_path / "bad.json.gz"
    writer(path)
    assert len(KeywordIndex.load(path)) == 0


def test_missing_snapshot_gives_empty_index(tmp_path):
    assert len(KeywordIndex.load(tmp_path / "nope.json.gz")) == 0


@pytest.mark.parametrize("field,value", [("version", 999), ("tokenizer_version", 999)])
def test_outdated_snapshot_discarded(tmp_path, index, field, value):
    fill(index)
    path = tmp_path / "i.json.gz"
    index.save(path)
    payload = json.loads(gzip.decompress(path.read_bytes()))
    payload[field] = value
    path.write_bytes(gzip.compress(json.dumps(payload).encode()))
    assert len(KeywordIndex.load(path)) == 0


def test_default_index_path(settings):
    assert default_index_path(settings) == settings.data_dir / "bm25_index.json.gz"


# ---- sync with the document database -------------------------------------------------------
def page(url, title, body):
    return extract_page(f"<title>{title}</title><p>{body}</p>", url)


@pytest.fixture
def repo(tmp_path):
    r = DocumentRepository(tmp_path / "docs.db")
    yield r
    r.close()


def test_sync_added_updated_unchanged_removed(repo):
    for i, w in enumerate(["alpha", "beta", "gamma"]):
        repo.upsert_page(page(f"https://s.test/{i}", f"Doc {w}", f"content about {w}"))
    idx = KeywordIndex()
    assert vars(idx.sync(repo)) == {"added": 3, "updated": 0, "unchanged": 0, "removed": 0}
    assert [h.doc_id for h in idx.search("beta")] == [repo.get_by_url("https://s.test/1").id]

    repo.upsert_page(page("https://s.test/0", "Doc zeta", "content about delta now"))    # changed
    repo.upsert_page(page("https://s.test/3", "Doc eps", "content about epsilon"))        # new
    repo.mark_status("https://s.test/2", "gone")                                          # gone
    assert vars(idx.sync(repo)) == {"added": 1, "updated": 1, "unchanged": 1, "removed": 1}
    assert idx.search("alpha") == [] and idx.search("gamma") == []
    assert len(idx.search("delta")) == 1 and len(idx.search("epsilon")) == 1


def test_restart_does_not_reindex_unchanged_documents(repo, tmp_path, monkeypatch):
    for i in range(5):
        repo.upsert_page(page(f"https://s.test/{i}", f"Title {i}", f"unique content number{i} here"))
    path = tmp_path / "idx.json.gz"
    first = KeywordIndex()
    first.sync(repo)
    first.save(path)
    expected = [h.doc_id for h in first.search("content")]

    calls = []
    original = ki.analyze_document
    monkeypatch.setattr(ki, "analyze_document", lambda t: calls.append(t) or original(t))
    restarted = KeywordIndex.load(path)                 # "application restart"
    stats = restarted.sync(repo)
    assert vars(stats) == {"added": 0, "updated": 0, "unchanged": 5, "removed": 0}
    assert calls == []                                  # nothing was re-tokenized
    assert [h.doc_id for h in restarted.search("content")] == expected


def test_sync_rebuilds_everything_when_snapshot_lost(repo, tmp_path):
    repo.upsert_page(page("https://s.test/a", "Recoverable", "database is the source of truth"))
    idx = KeywordIndex.load(tmp_path / "missing.json.gz")
    assert idx.sync(repo).added == 1 and len(idx.search("truth")) == 1


def test_sync_on_empty_repository(repo):
    idx = KeywordIndex()
    assert vars(idx.sync(repo)) == {"added": 0, "updated": 0, "unchanged": 0, "removed": 0}
    assert idx.search("x") == []
