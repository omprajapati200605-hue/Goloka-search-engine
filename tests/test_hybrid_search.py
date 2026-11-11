"""HybridSearcher with real BM25 + vector indexes (deterministic stand-in embedder) and a real SQLite repo."""
import pytest

from app.extraction.extractor import extract_page
from app.search.embedder import EmbeddingError
from app.search.hybrid import HybridConfig, HybridSearcher
from app.search.keyword_index import KeywordIndex
from app.search.vector_index import VectorIndex
from app.storage.chunk_store import ChunkStore
from app.storage.repository import DocumentRepository
from tests.fake_embedder import HashingEmbedder

CONCEPTS = {"car": "vehicle", "automobile": "vehicle", "vehicle": "vehicle", "truck": "vehicle",
            "pasta": "food", "noodles": "food", "spaghetti": "food"}

PAGES = {
    "cars": ("Fast cars", "A fast car drives down the road. The car engine is loud and the car is quick."),
    "trucks": ("Heavy trucks", "A big truck hauls goods across the country on long highways."),
    "pasta": ("Cooking pasta", "Boil pasta in salted water, then add tomato sauce and basil."),
    "garden": ("Garden notes", "Tomatoes need sun and water. Basil grows next to tomatoes in warm soil."),
    "space": ("Astronomy", "Telescopes reveal distant galaxies and bright nebulae far from earth."),
    "autos": ("Automobile history", "The history of the early automobile and its inventors is long."),
}


@pytest.fixture
def env(tmp_path):
    db = tmp_path / "goloka.db"
    repo = DocumentRepository(db)
    embedder = HashingEmbedder(concepts=CONCEPTS)
    kw = KeywordIndex()
    vec = VectorIndex(embedder, ChunkStore(db), chunk_size_words=60, chunk_overlap_words=10)
    ids = {}
    for key, (title, body) in PAGES.items():
        d = repo.upsert_page(extract_page(f"<title>{title}</title><p>{body}</p>", f"https://site.test/{key}")).document
        ids[key] = d.id
    kw.sync(repo)
    vec.sync(repo)
    searcher = HybridSearcher(repo, kw, vec)
    yield type("Env", (), dict(repo=repo, kw=kw, vec=vec, searcher=searcher, ids=ids, embedder=embedder, db=db))
    repo.close()


def top_ids(response, env):
    inverse = {v: k for k, v in env.ids.items()}
    return [inverse[r.doc_id] for r in response.results]


# ---- hybrid behaviour ----------------------------------------------------------------------
def test_keyword_match_is_found_and_ranked_first(env):
    r = env.searcher.search("tomato sauce basil")
    assert top_ids(r, env)[0] == "pasta"
    assert r.mode == "hybrid" and r.method == "weighted" and r.took_ms >= 0 and r.warnings == []


def test_hybrid_finds_documents_by_meaning_that_bm25_misses(env):
    assert env.kw.search("automobile")[0].doc_id == env.ids["autos"]                   # BM25 only knows the literal word
    assert env.ids["cars"] not in {h.doc_id for h in env.kw.search("automobile")}
    ids = top_ids(env.searcher.search("automobile", k=5), env)
    assert "cars" in ids and "trucks" in ids and ids[0] == "autos"                     # literal + semantic neighbours


def test_document_supported_by_both_signals_beats_single_signal_documents(env):
    r = env.searcher.search("automobile", k=5)
    by = {x.doc_id: x for x in r.results}
    autos, cars = by[env.ids["autos"]], by[env.ids["cars"]]
    assert autos.bm25_score is not None and autos.semantic_score is not None
    assert cars.bm25_score is None and cars.semantic_score is not None                 # semantic-only: no invented BM25 score
    assert autos.combined_score > cars.combined_score


def test_result_fields_are_complete_and_truthful(env):
    r = env.searcher.search("telescopes galaxies").results[0]
    assert r.doc_id == env.ids["space"] and r.title == "Astronomy" and r.url == "https://site.test/space"
    assert r.domain == "site.test" and "galaxies" in r.snippet
    assert r.bm25_score > 0 and r.semantic_score > 0 and 0 < r.combined_score <= 1
    assert set(r.components) == {"content", "title", "semantic"} and r.ranks["content"] == 1
    assert env.repo.get(r.doc_id).title == r.title                                      # data comes from storage
    assert set(r.to_dict()) >= {"doc_id", "title", "url", "domain", "snippet", "bm25_score", "semantic_score",
                                "combined_score"}


def test_title_match_contributes(env):
    r = env.searcher.search("astronomy").results[0]
    assert r.doc_id == env.ids["space"] and r.title_score > 0 and r.components["title"] == 1.0


def test_results_sorted_and_bounded(env):
    r = env.searcher.search("car vehicle tomatoes", k=4)
    scores = [x.combined_score for x in r.results]
    assert scores == sorted(scores, reverse=True) and all(0 < s <= 1 for s in scores) and len(scores) <= 4


def test_gibberish_query_returns_nothing(env):
    r = env.searcher.search("zzzqqq xxyyzz")
    assert r.results == [] and r.warnings == []


def test_empty_blank_and_punctuation_queries(env):
    for q in ("", "   ", "!!!", None):
        assert env.searcher.search(q).results == []
    assert env.searcher.search("car", k=0).results == []


def test_k_limits_results(env):
    assert len(env.searcher.search("car vehicle truck automobile", k=2).results) == 2


def test_min_score_filters(env):
    all_ = env.searcher.search("automobile", k=10).results
    cutoff = all_[1].combined_score
    kept = env.searcher.search("automobile", k=10, min_score=cutoff).results
    assert 1 <= len(kept) < len(all_) and all(x.combined_score >= cutoff for x in kept)


# ---- rrf -----------------------------------------------------------------------------------
def test_rrf_alternative_method(env):
    weighted = env.searcher.search("automobile", k=6)
    rrf = env.searcher.search("automobile", k=6, method="rrf")
    assert rrf.method == "rrf" and {r.doc_id for r in rrf.results} == {r.doc_id for r in weighted.results}
    assert rrf.results[0].doc_id == env.ids["autos"] and all(0 < r.combined_score <= 1 for r in rrf.results)
    assert rrf.results[0].combined_score != weighted.results[0].combined_score


def test_default_method_comes_from_config(env):
    s = HybridSearcher(env.repo, env.kw, env.vec, HybridConfig(method="rrf"))
    assert s.search("car").method == "rrf"


def test_unknown_method_rejected(env):
    with pytest.raises(ValueError):
        env.searcher.search("car", method="bogus")


# ---- single-signal modes -------------------------------------------------------------------
def test_keyword_mode(env):
    r = env.searcher.search_keyword("tomatoes basil")
    assert r.mode == "keyword" and top_ids(r, env)[0] == "garden"
    assert r.results[0].semantic_score is None and r.results[0].bm25_score == r.results[0].combined_score > 0
    assert env.searcher.search_keyword("automobile").results[0].doc_id == env.ids["autos"]
    assert env.searcher.search_keyword("").results == []


def test_semantic_mode(env):
    r = env.searcher.search_semantic("automobile")
    assert r.mode == "semantic" and {"cars", "trucks"} <= set(top_ids(r, env))
    assert all(x.bm25_score is None and x.semantic_score == x.combined_score for x in r.results)
    assert env.searcher.search_semantic("zzzqqq").results == []
    assert env.searcher.search_semantic("automobile", min_score=0.99).results == []


def test_semantic_snippet_comes_from_best_matching_chunk(env):
    r = env.searcher.search_semantic("noodles").results[0]
    assert r.doc_id == env.ids["pasta"] and "pasta" in r.snippet


# ---- never return what is not a stored, searchable document -------------------------------
def test_stale_index_entries_are_never_returned(env):
    env.repo.delete(env.ids["pasta"])                       # index files still know the document
    for response in (env.searcher.search("pasta"), env.searcher.search_keyword("pasta"), env.searcher.search_semantic("noodles")):
        assert env.ids["pasta"] not in [r.doc_id for r in response.results]


def test_gone_documents_are_not_returned_but_k_is_still_filled(env):
    env.repo.mark_status("https://site.test/cars", "gone")
    r = env.searcher.search("car vehicle automobile truck", k=3)
    assert env.ids["cars"] not in [x.doc_id for x in r.results] and len(r.results) == 3


# ---- graceful degradation ------------------------------------------------------------------
def test_semantic_failure_falls_back_to_keyword_with_warning(env):
    class Down(HashingEmbedder):
        def embed(self, texts):
            raise EmbeddingError("model offline")

    env.vec._embedder = Down()
    r = env.searcher.search("tomato sauce basil")
    assert top_ids(r, env)[0] == "pasta" and any("model offline" in w for w in r.warnings)
    assert all(x.semantic_score is None for x in r.results)
    assert 0 < r.results[0].combined_score <= 1                       # renormalised over keyword signals only
    assert any("model offline" in w for w in env.searcher.search_semantic("noodles").warnings)


def test_no_vector_index_configured(env):
    s = HybridSearcher(env.repo, env.kw, None)
    r = s.search("tomato basil")
    assert top_ids(r, env)[0] in ("garden", "pasta") and any("not configured" in w for w in r.warnings)
    assert s.search_semantic("noodles").results == [] and s.search_semantic("noodles").warnings


def test_empty_indexes_return_empty_results(tmp_path):
    db = tmp_path / "empty.db"
    repo = DocumentRepository(db)
    s = HybridSearcher(repo, KeywordIndex(), VectorIndex(HashingEmbedder(), ChunkStore(db)))
    r = s.search("anything at all")
    assert r.results == [] and r.warnings == []
    repo.close()


def test_weights_from_config_change_ranking(env):
    kw_only = HybridSearcher(env.repo, env.kw, env.vec, HybridConfig(bm25_weight=1, semantic_weight=0, title_weight=0))
    sem_only = HybridSearcher(env.repo, env.kw, env.vec, HybridConfig(bm25_weight=0, semantic_weight=1, title_weight=0))
    assert top_ids(kw_only.search("automobile"), env)[0] == "autos"
    assert set(top_ids(sem_only.search("automobile", k=3), env)[:2]) <= {"cars", "trucks", "autos"}
    assert [r.doc_id for r in kw_only.search("automobile").results][:1] != []


def test_updated_document_is_searchable_after_resync(env):
    env.repo.upsert_page(extract_page("<title>Cooking pasta</title><p>Now about volcanoes and lava.</p>", "https://site.test/pasta"))
    env.kw.sync(env.repo)
    env.vec.sync(env.repo)
    r = env.searcher.search("volcanoes lava")
    assert top_ids(r, env)[0] == "pasta" and "volcanoes" in r.results[0].snippet
    assert "pasta" not in top_ids(env.searcher.search("boil salted water"), env)      # old content is gone
