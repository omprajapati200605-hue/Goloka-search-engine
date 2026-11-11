import pytest

from app.extraction.extractor import extract_page
from app.storage.chunk_store import ChunkStore
from app.storage.repository import DocumentRepository


@pytest.fixture
def repo(tmp_path):
    r = DocumentRepository(tmp_path / "db.sqlite")
    yield r
    r.close()


@pytest.fixture
def store(tmp_path, repo):
    s = ChunkStore(tmp_path / "db.sqlite")
    yield s
    s.close()


def doc(repo, n):
    return repo.upsert_page(extract_page(f"<title>T{n}</title><p>body number {n}</p>", f"https://a.test/{n}")).document


def test_replace_assigns_increasing_ids_and_stores_text(repo, store):
    d = doc(repo, 1)
    old, new = store.replace_document(d.id, "h1", "sig", ["first chunk", "second chunk"])
    assert old == [] and new == [1, 2]
    chunks = store.get_chunks(new)
    assert [(c.chunk_index, c.text, c.document_id) for c in chunks.values()] == [
        (0, "first chunk", d.id), (1, "second chunk", d.id)]
    assert store.document_state(d.id) == ("h1", "sig") and store.count() == 2


def test_ids_are_never_reused_after_replace_or_remove(repo, store):
    a, b = doc(repo, 1), doc(repo, 2)
    _, ids_a = store.replace_document(a.id, "h", "s", ["x", "y", "z"])
    old, ids_a2 = store.replace_document(a.id, "h2", "s", ["x2"])
    assert old == ids_a and ids_a2 == [4]
    store.remove_document(a.id)
    _, ids_b = store.replace_document(b.id, "h", "s", ["q"])
    assert ids_b == [5] and not set(ids_b) & set(ids_a)


def test_replace_removes_old_chunks(repo, store):
    d = doc(repo, 1)
    store.replace_document(d.id, "h1", "s", ["old one", "old two"])
    store.replace_document(d.id, "h2", "s", ["new only"])
    assert [c.text for c in store.get_chunks(store.all_ids()).values()] == ["new only"]
    assert store.document_state(d.id) == ("h2", "s")


def test_remove_document_returns_ids_and_clears_state(repo, store):
    d = doc(repo, 1)
    _, ids = store.replace_document(d.id, "h", "s", ["a", "b"])
    assert store.remove_document(d.id) == ids
    assert store.count() == 0 and store.document_state(d.id) is None and store.remove_document(d.id) == []


def test_indexed_documents_and_documents_for_ids(repo, store):
    a, b = doc(repo, 1), doc(repo, 2)
    _, ids_a = store.replace_document(a.id, "ha", "s1", ["a"])
    _, ids_b = store.replace_document(b.id, "hb", "s2", ["b"])
    assert store.indexed_documents() == {a.id: ("ha", "s1"), b.id: ("hb", "s2")}
    assert store.documents_for_ids(ids_a + ids_b + [999]) == {ids_a[0]: a.id, ids_b[0]: b.id}


def test_deleting_document_cascades_to_chunks_and_state(repo, store):
    d = doc(repo, 1)
    store.replace_document(d.id, "h", "s", ["a", "b"])
    repo.delete(d.id)
    assert store.count() == 0 and store.indexed_documents() == {}


def test_clear_keeps_counter(repo, store):
    d = doc(repo, 1)
    store.replace_document(d.id, "h", "s", ["a", "b"])
    store.clear()
    assert store.count() == 0 and store.replace_document(d.id, "h", "s", ["c"])[1] == [3]


def test_get_chunks_handles_many_ids_and_empty(repo, store):
    d = doc(repo, 1)
    _, ids = store.replace_document(d.id, "h", "s", [f"c{i}" for i in range(1200)])
    assert len(store.get_chunks(ids)) == 1200 and store.get_chunks([]) == {}


def test_persists_across_reopen(tmp_path, repo):
    d = doc(repo, 1)
    s1 = ChunkStore(tmp_path / "db.sqlite")
    s1.replace_document(d.id, "h", "s", ["persisted"])
    s1.close()
    s2 = ChunkStore(tmp_path / "db.sqlite")
    assert [c.text for c in s2.get_chunks(s2.all_ids()).values()] == ["persisted"]
    assert s2.replace_document(d.id, "h2", "s", ["next"])[1] == [2]    # counter survived the restart
    s2.close()
