import threading

import numpy as np
import pytest

from app.extraction.extractor import extract_page
from app.search.embedder import EmbeddingError
from app.search.vector_index import VectorIndex, default_vector_path
from app.storage.chunk_store import ChunkStore
from app.storage.repository import DocumentRepository
from tests.fake_embedder import HashingEmbedder

CONCEPTS = {"car": "vehicle", "automobile": "vehicle", "vehicle": "vehicle", "truck": "vehicle",
            "pasta": "food", "noodles": "food", "spaghetti": "food"}


@pytest.fixture
def db(tmp_path):
    return tmp_path / "goloka.db"


@pytest.fixture
def repo(db):
    r = DocumentRepository(db)
    yield r
    r.close()


@pytest.fixture
def store(db, repo):
    s = ChunkStore(db)
    yield s
    s.close()


@pytest.fixture
def embedder():
    return HashingEmbedder(concepts=CONCEPTS)


@pytest.fixture
def index(embedder, store):
    return VectorIndex(embedder, store, chunk_size_words=40, chunk_overlap_words=8)


def add_doc(repo, n, title, body):
    html = f"<title>{title}</title>" + "".join(f"<p>{p}</p>" for p in body.split("\n"))
    return repo.upsert_page(extract_page(html, f"https://a.test/{n}")).document


def index_doc(index, d):
    return index.add_document(d.id, d.title, d.text, d.content_hash)


def seed(repo, index):
    docs = {
        "car": add_doc(repo, 1, "Cars", "A fast car drives down the road with a loud engine."),
        "pasta": add_doc(repo, 2, "Cooking", "Boil pasta in salted water and add tomato sauce."),
        "space": add_doc(repo, 3, "Astronomy", "Telescopes reveal distant galaxies and bright nebulae."),
    }
    for d in docs.values():
        index_doc(index, d)
    return docs


# ---- searching -----------------------------------------------------------------------------
def test_search_returns_relevant_document_first_with_cosine_scores(repo, index):
    docs = seed(repo, index)
    hits = index.search("fast car engine")
    assert hits[0].doc_id == docs["car"].id and 0.0 < hits[0].score <= 1.0
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_identical_text_has_similarity_near_one(repo, index):
    d = add_doc(repo, 1, "", "unique words alpha beta gamma delta")
    index_doc(index, d)
    assert index.search(d.text)[0].score == pytest.approx(1.0, abs=1e-5)


def test_retrieves_by_meaning_without_shared_words(repo, index):
    docs = seed(repo, index)
    hits = index.search("automobile")            # no literal overlap with the car document
    assert hits[0].doc_id == docs["car"].id


def test_hit_carries_chunk_metadata_for_snippets(repo, index):
    docs = seed(repo, index)
    hit = index.search("telescopes galaxies")[0]
    assert hit.doc_id == docs["space"].id and hit.chunk_index == 0 and "galaxies" in hit.chunk_text
    assert hit.faiss_id >= 1


def test_empty_index_and_blank_queries(repo, index):
    assert index.search("anything") == [] and len(index) == 0
    seed(repo, index)
    assert index.search("") == [] and index.search("   ") == [] and index.search("car", k=0) == []


def test_min_score_filters_weak_matches(repo, index):
    seed(repo, index)
    assert index.search("telescopes galaxies", min_score=0.99) == []
    assert len(index.search("telescopes galaxies", min_score=0.1)) >= 1


def test_one_hit_per_document_even_with_many_chunks(repo, index):
    long_body = "\n".join(f"The car number {i} is a very fast vehicle with a loud engine and shiny paint job." for i in range(30))
    big = add_doc(repo, 1, "Many chunks", long_body)
    other = add_doc(repo, 2, "Other", "A truck is also a vehicle for hauling goods.")
    index_doc(index, big)
    index_doc(index, other)
    assert len(index) > 5
    hits = index.search("fast car vehicle", k=5)
    assert len({h.doc_id for h in hits}) == len(hits) == 2          # widened search found the second document


def test_long_documents_are_split_into_chunks_and_best_chunk_is_returned(repo, index):
    filler = "\n".join(f"Filler sentence number {i} about nothing in particular at all." for i in range(40))
    d = add_doc(repo, 1, "Long page", filler + "\nThe zebra crossed the savanna at dawn.\n" + filler)
    index_doc(index, d)
    assert len(index) > 3
    assert "zebra" in index.search("zebra savanna dawn")[0].chunk_text


def test_title_only_document_is_indexed(repo, index):
    d = repo.upsert_page(extract_page("<title>Quirky title page</title><p>x</p>", "https://a.test/t")).document
    assert index.add_document(d.id, "Quirky title", "", "newhash") == "added"
    assert index.search("quirky title")[0].doc_id == d.id


def test_document_without_text_or_title_is_skipped(repo, index):
    d = add_doc(repo, 1, "T", "body")
    assert index.add_document(d.id, "", "   ", d.content_hash) == "empty" and len(index) == 0


def test_stale_vectors_without_metadata_are_ignored_by_search(repo, index):
    docs = seed(repo, index)
    repo.delete(docs["car"].id)                                      # cascades chunk rows; FAISS still has vectors
    assert all(h.doc_id != docs["car"].id for h in index.search("fast car engine"))


# ---- incremental indexing ------------------------------------------------------------------
def test_unchanged_document_is_not_reembedded(repo, index, embedder):
    d = add_doc(repo, 1, "T", "stable content words")
    assert index_doc(index, d) == "added"
    n = len(embedder.embedded_texts)
    assert index_doc(index, d) == "unchanged" and len(embedder.embedded_texts) == n


def test_update_replaces_old_vectors_only_for_that_document(repo, index):
    docs = seed(repo, index)
    before = len(index)
    new = repo.upsert_page(extract_page("<title>Cars</title><p>Now about volcanoes and lava flows.</p>", "https://a.test/1")).document
    assert index_doc(index, new) == "updated"
    assert len(index) == before                                         # old chunk replaced, none leaked
    assert index.search("volcanoes lava")[0].doc_id == docs["car"].id
    assert index.search("telescopes galaxies")[0].doc_id == docs["space"].id   # others untouched


def test_adding_documents_never_loses_existing_vectors(repo, index):
    docs = seed(repo, index)
    count = len(index)
    for n in range(10, 20):
        index_doc(index, add_doc(repo, n, f"Extra {n}", f"additional unrelated material number {n} words"))
    assert len(index) >= count + 10
    assert index.search("fast car engine")[0].doc_id == docs["car"].id


def test_remove_document(repo, index):
    docs = seed(repo, index)
    before = len(index)
    assert index.remove_document(docs["pasta"].id) and not index.remove_document(docs["pasta"].id)
    assert len(index) == before - 1 and all(h.doc_id != docs["pasta"].id for h in index.search("pasta sauce"))


def test_dimension_mismatch_is_an_error_not_silent_corruption(repo, store):
    a = VectorIndex(HashingEmbedder(dim=64), store, chunk_size_words=40, chunk_overlap_words=8)
    d = add_doc(repo, 1, "T", "some words here")
    index_doc(a, d)
    a._embedder = HashingEmbedder(dim=32)
    with pytest.raises(EmbeddingError):
        a.search("words")


def test_concurrent_adds_and_searches(repo, index):
    docs = [add_doc(repo, n, f"T{n}", f"thread safety document number {n} words") for n in range(24)]
    errors = []

    def writer(part):
        try:
            for d in part:
                index_doc(index, d)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def reader():
        try:
            for _ in range(30):
                index.search("thread safety document")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(docs[i::4],)) for i in range(4)] + [threading.Thread(target=reader) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(index) == 24


# ---- sync with the database ----------------------------------------------------------------
def test_sync_counts(repo, index):
    for n in range(3):
        add_doc(repo, n, f"T{n}", f"document body number {n} unique{n}")
    assert vars(index.sync(repo)) == dict(added=3, updated=0, unchanged=0, removed=0, empty=0, failed=0)
    add_doc(repo, 0, "T0", "completely rewritten content")           # update
    add_doc(repo, 9, "T9", "brand new document here")                 # add
    repo.mark_status("https://a.test/2", "gone")                      # remove
    assert vars(index.sync(repo)) == dict(added=1, updated=1, unchanged=1, removed=1, empty=0, failed=0)


def test_deleted_document_vanishes_from_results_and_orphans_are_cleaned(repo, index):
    d = add_doc(repo, 1, "T", "short lived page")
    index.sync(repo)
    assert len(index.search("short lived")) == 1
    repo.delete(d.id)                                   # cascades the metadata; FAISS still holds the vector
    index.sync(repo)
    assert index.search("short lived") == []            # stale vector is never returned
    assert index.reconcile().orphan_vectors_removed == 1 and len(index) == 0


def test_one_failing_document_does_not_block_the_others(repo, db):
    class Flaky(HashingEmbedder):
        def embed(self, texts):
            if any("poison" in t for t in texts):
                raise ValueError("bad text")
            return super().embed(texts)

    idx = VectorIndex(Flaky(), ChunkStore(db), chunk_size_words=40, chunk_overlap_words=8)
    add_doc(repo, 1, "Good", "perfectly fine document")
    add_doc(repo, 2, "Bad", "poison pill document")
    add_doc(repo, 3, "Good too", "another fine document")
    stats = idx.sync(repo)
    assert stats.added == 2 and stats.failed == 1


def test_unavailable_model_stops_sync_with_clear_error(repo, db):
    class Down(HashingEmbedder):
        def embed(self, texts):
            raise EmbeddingError("model unavailable")

    add_doc(repo, 1, "T", "some content")
    with pytest.raises(EmbeddingError, match="model unavailable"):
        VectorIndex(Down(), ChunkStore(db)).sync(repo)


def test_changing_model_or_chunking_triggers_reembedding(repo, db):
    add_doc(repo, 1, "T", "content that will be indexed twice")
    first = VectorIndex(HashingEmbedder(name="model-a"), ChunkStore(db), chunk_size_words=40, chunk_overlap_words=8)
    assert first.sync(repo).added == 1
    other_model = VectorIndex(HashingEmbedder(name="model-b"), ChunkStore(db), chunk_size_words=40, chunk_overlap_words=8)
    assert other_model.sync(repo).updated == 1
    other_chunks = VectorIndex(HashingEmbedder(name="model-b"), ChunkStore(db), chunk_size_words=60, chunk_overlap_words=8)
    assert other_chunks.sync(repo).updated == 1


# ---- persistence ---------------------------------------------------------------------------
def test_save_load_roundtrip(tmp_path, repo, index, embedder, db):
    docs = seed(repo, index)
    path = default_vector_path(type("S", (), {"vector_dir": tmp_path / "vectors"})())
    index.save(path)
    assert path.exists() and path.with_name("chunks.faiss.meta.json").exists()
    loaded = VectorIndex.load(path, embedder, ChunkStore(db), chunk_size_words=40, chunk_overlap_words=8)
    assert len(loaded) == len(index) and loaded.last_reconcile.documents_reset == 0
    for q in ("fast car engine", "pasta sauce", "telescopes galaxies"):
        a, b = index.search(q), loaded.search(q)
        assert [h.doc_id for h in a] == [h.doc_id for h in b]
        assert [h.score for h in a] == pytest.approx([h.score for h in b])
    assert loaded.search("fast car engine")[0].doc_id == docs["car"].id


def test_restart_does_not_reembed_unchanged_documents(tmp_path, repo, db):
    emb1 = HashingEmbedder(concepts=CONCEPTS)
    first = VectorIndex(emb1, ChunkStore(db), chunk_size_words=40, chunk_overlap_words=8)
    for n in range(5):
        add_doc(repo, n, f"T{n}", f"restart persistence document {n} unique{n}")
    first.sync(repo)
    first.save(tmp_path / "v" / "chunks.faiss")
    emb2 = HashingEmbedder(concepts=CONCEPTS)                                   # "application restart"
    restarted = VectorIndex.load(tmp_path / "v" / "chunks.faiss", emb2, ChunkStore(db), chunk_size_words=40, chunk_overlap_words=8)
    stats = restarted.sync(repo)
    assert stats.unchanged == 5 and stats.added == 0 and emb2.embedded_texts == []
    assert len(restarted.search("restart persistence")) == 5


def test_incremental_add_after_reload_keeps_old_vectors(tmp_path, repo, db, embedder):
    path = tmp_path / "v" / "chunks.faiss"
    a = VectorIndex(embedder, ChunkStore(db), chunk_size_words=40, chunk_overlap_words=8)
    first = add_doc(repo, 1, "First", "alpha bravo charlie first batch material")
    index_doc(a, first)
    a.save(path)
    b = VectorIndex.load(path, embedder, ChunkStore(db), chunk_size_words=40, chunk_overlap_words=8)
    second = add_doc(repo, 2, "Second", "delta echo foxtrot second batch material")
    index_doc(b, second)
    b.save(path)
    c = VectorIndex.load(path, embedder, ChunkStore(db), chunk_size_words=40, chunk_overlap_words=8)
    assert {h.doc_id for h in c.search("batch material", k=5)} == {first.id, second.id}
    assert c.search("alpha bravo")[0].doc_id == first.id


def test_vectors_added_after_last_save_are_recovered_after_crash(tmp_path, repo, db, embedder):
    path = tmp_path / "v" / "chunks.faiss"
    a = VectorIndex(embedder, ChunkStore(db), chunk_size_words=40, chunk_overlap_words=8)
    d1 = add_doc(repo, 1, "One", "saved before the crash")
    index_doc(a, d1)
    a.save(path)
    d2 = add_doc(repo, 2, "Two", "indexed but never saved before the crash")
    index_doc(a, d2)                                                            # DB has it, FAISS file does not
    restarted = VectorIndex.load(path, embedder, ChunkStore(db), chunk_size_words=40, chunk_overlap_words=8)
    assert restarted.last_reconcile.documents_reset == 1
    assert [h.doc_id for h in restarted.search("saved before the crash")][:1] == [d1.id]
    stats = restarted.sync(repo)
    assert stats.added == 1 and stats.unchanged == 1                            # only the lost one re-embedded
    assert {h.doc_id for h in restarted.search("crash", k=5)} == {d1.id, d2.id}


def test_lost_vector_file_is_rebuilt_from_database(tmp_path, repo, db, embedder):
    path = tmp_path / "v" / "chunks.faiss"
    a = VectorIndex(embedder, ChunkStore(db), chunk_size_words=40, chunk_overlap_words=8)
    for n in range(3):
        index_doc(a, add_doc(repo, n, f"T{n}", f"rebuildable document {n} zz{n}"))
    a.save(path)
    path.unlink()
    path.with_name("chunks.faiss.meta.json").unlink()
    b = VectorIndex.load(path, embedder, ChunkStore(db), chunk_size_words=40, chunk_overlap_words=8)
    assert len(b) == 0 and b.last_reconcile.documents_reset == 3
    assert b.sync(repo).added == 3 and len(b.search("rebuildable document", k=5)) == 3


def test_orphan_vectors_are_removed_on_load(tmp_path, repo, db, embedder):
    path = tmp_path / "v" / "chunks.faiss"
    a = VectorIndex(embedder, ChunkStore(db), chunk_size_words=40, chunk_overlap_words=8)
    keep, drop = add_doc(repo, 1, "Keep", "keeper document words"), add_doc(repo, 2, "Drop", "doomed document words")
    index_doc(a, keep)
    index_doc(a, drop)
    a.save(path)
    repo.delete(drop.id)                                                         # metadata gone, vector still in file
    b = VectorIndex.load(path, embedder, ChunkStore(db), chunk_size_words=40, chunk_overlap_words=8)
    assert b.last_reconcile.orphan_vectors_removed == 1 and len(b) == 1


@pytest.mark.parametrize("damage", ["garbage", "truncate", "bad_meta", "other_model"])
def test_damaged_or_incompatible_files_never_crash_load(tmp_path, repo, db, embedder, damage):
    path = tmp_path / "v" / "chunks.faiss"
    a = VectorIndex(embedder, ChunkStore(db), chunk_size_words=40, chunk_overlap_words=8)
    index_doc(a, add_doc(repo, 1, "T", "some document content"))
    a.save(path)
    meta = path.with_name("chunks.faiss.meta.json")
    if damage == "garbage":
        path.write_bytes(b"not a faiss file")
    elif damage == "truncate":
        path.write_bytes(path.read_bytes()[:20])
    elif damage == "bad_meta":
        meta.write_text("{nope")
    loader_embedder = HashingEmbedder(name="different-model") if damage == "other_model" else embedder
    b = VectorIndex.load(path, loader_embedder, ChunkStore(db), chunk_size_words=40, chunk_overlap_words=8)
    assert len(b) == 0
    assert b.sync(repo).added + b.sync(repo).unchanged >= 1 and len(b.search("document content")) == 1


def test_save_empty_index_removes_stale_files(tmp_path, embedder, store):
    path = tmp_path / "v" / "chunks.faiss"
    path.parent.mkdir()
    path.write_bytes(b"old")
    VectorIndex(embedder, store).save(path)
    assert not path.exists()


def test_invalid_chunk_settings_rejected(embedder, store):
    with pytest.raises(ValueError):
        VectorIndex(embedder, store, chunk_size_words=10, chunk_overlap_words=10)


def test_non_normalising_embedder_still_gives_cosine(repo, store):
    class Raw(HashingEmbedder):
        def embed(self, texts):
            return super().embed(texts) * 7.5

    idx = VectorIndex(Raw(), store, chunk_size_words=40, chunk_overlap_words=8)
    d = add_doc(repo, 1, "", "cosine check words alpha beta")
    index_doc(idx, d)
    assert idx.search(d.text)[0].score == pytest.approx(1.0, abs=1e-5)


def test_from_settings_builds_paths_and_persists(tmp_path, repo):
    from app.config import Settings
    settings = Settings(_env_file=None, data_dir=tmp_path / "d", db_path=tmp_path / "goloka.db", vector_dir=tmp_path / "d" / "vec")
    add_doc(repo, 1, "Settings", "built from settings object")
    idx = VectorIndex.from_settings(settings, HashingEmbedder())
    idx.sync(repo)
    idx.save(default_vector_path(settings))
    again = VectorIndex.from_settings(settings, HashingEmbedder())
    assert len(again) == len(idx) > 0 and again.search("settings object")[0].score > 0.3
    assert np.isclose(again.search("settings object")[0].score, idx.search("settings object")[0].score)
