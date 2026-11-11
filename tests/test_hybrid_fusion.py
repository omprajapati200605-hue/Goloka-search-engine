"""Pure fusion maths: normalisation, weighted fusion, RRF, configuration. No indexes needed."""
import math

import pytest

from app.config import Settings
from app.search.hybrid import HybridConfig, fuse_scores, normalize_scores

CFG = HybridConfig()  # weights 0.4 / 0.4 / 0.2, rrf_k 60


# ---- normalisation -------------------------------------------------------------------------
def test_max_normalisation():
    assert normalize_scores({"a": 4.0, "b": 2.0, "c": 0.0}, "max") == {"a": 1.0, "b": 0.5, "c": 0.0}
    assert normalize_scores({"a": 0.0}, "max") == {"a": 0.0}


def test_min_max_normalisation():
    assert normalize_scores({"a": 10.0, "b": 6.0, "c": 2.0}, "min_max") == {"a": 1.0, "b": 0.5, "c": 0.0}
    assert normalize_scores({"a": 3.0, "b": 3.0}, "min_max") == {"a": 1.0, "b": 1.0}   # all equal -> all top
    assert normalize_scores({"a": 5.0}, "min_max") == {"a": 1.0}


def test_zscore_normalisation_is_monotonic_and_bounded():
    out = normalize_scores({"a": 1.0, "b": 2.0, "c": 9.0}, "zscore")
    assert 0 < out["a"] < out["b"] < out["c"] < 1
    assert normalize_scores({"a": 2.0, "b": 2.0}, "zscore") == {"a": 0.5, "b": 0.5}


def test_clamp_keeps_absolute_scale():
    assert normalize_scores({"a": 0.35, "b": 1.4, "c": -0.2}, "clamp") == {"a": 0.35, "b": 1.0, "c": 0.0}


def test_empty_and_unknown():
    assert all(normalize_scores({}, m) == {} for m in ("max", "min_max", "zscore", "clamp"))
    with pytest.raises(ValueError):
        normalize_scores({"a": 1.0}, "bogus")


# ---- weighted fusion -----------------------------------------------------------------------
def test_weighted_matches_hand_calculation():
    fused = fuse_scores(content={"d1": 10.0, "d2": 5.0}, title={"d1": 2.0}, semantic={"d2": 0.6, "d3": 0.3}, config=CFG)
    by = {f.doc_id: f for f in fused}
    # d1: content 1.0*0.4 + title 1.0*0.2 + semantic 0 = 0.6
    assert by["d1"].combined == pytest.approx(0.6)
    # d2: content 0.5*0.4 + semantic 0.6*0.4 = 0.44
    assert by["d2"].combined == pytest.approx(0.44)
    # d3: semantic 0.3*0.4 = 0.12
    assert by["d3"].combined == pytest.approx(0.12)
    assert [f.doc_id for f in fused] == ["d1", "d2", "d3"]
    assert by["d2"].components == {"content": 0.5, "title": 0.0, "semantic": 0.6}


def test_document_matching_every_signal_gets_one():
    f = fuse_scores(content={"a": 3.0}, title={"a": 1.0}, semantic={"a": 1.0}, config=CFG)[0]
    assert f.combined == pytest.approx(1.0)


def test_combined_always_within_zero_one():
    f = fuse_scores(content={f"d{i}": float(i + 1) for i in range(20)}, title={"d3": 4.0},
                    semantic={f"d{i}": 0.05 * i for i in range(20)}, config=CFG)
    assert all(0.0 <= x.combined <= 1.0 for x in f)


def test_weights_change_the_order():
    kw = dict(content={"keyword_doc": 9.0}, title={}, semantic={"meaning_doc": 0.9})
    assert fuse_scores(**kw, config=HybridConfig(bm25_weight=1, semantic_weight=0, title_weight=0))[0].doc_id == "keyword_doc"
    sem_cfg = HybridConfig(bm25_weight=0, semantic_weight=1, title_weight=0)
    assert fuse_scores(**kw, config=sem_cfg)[0].doc_id == "meaning_doc"


def test_semantic_unavailable_renormalises_over_remaining_signals():
    f = fuse_scores(content={"a": 5.0}, title={}, semantic={"a": 0.9}, config=CFG, semantic_available=False)
    assert len(f) == 1 and f[0].combined == pytest.approx(1.0 * 0.4 / 0.6)   # semantic ignored entirely
    assert "semantic" not in f[0].ranks


def test_unavailable_signal_with_all_the_weight_falls_back_to_equal_weights():
    cfg = HybridConfig(bm25_weight=0, semantic_weight=1, title_weight=0)
    f = fuse_scores(content={"a": 5.0}, title={}, semantic={}, config=cfg, semantic_available=False)
    assert f[0].combined > 0


def test_semantic_absolute_scale_is_preserved_by_default():
    weak = fuse_scores(content={}, title={}, semantic={"a": 0.22, "b": 0.21}, config=CFG)
    strong = fuse_scores(content={}, title={}, semantic={"a": 0.9}, config=CFG)
    assert weak[0].combined < strong[0].combined / 3          # a weak best-match is not stretched to look strong


def test_empty_inputs_and_deterministic_ties():
    assert fuse_scores(content={}, title={}, semantic={}, config=CFG) == []
    f = fuse_scores(content={"b": 1.0, "a": 1.0, "c": 1.0}, title={}, semantic={}, config=CFG)
    assert [x.doc_id for x in f] == ["a", "b", "c"]


def test_ranks_are_one_based_per_signal():
    f = {x.doc_id: x for x in fuse_scores(content={"a": 9.0, "b": 5.0}, title={"b": 1.0}, semantic={"b": 0.9, "a": 0.4}, config=CFG)}
    assert f["a"].ranks == {"content": 1, "semantic": 2} and f["b"].ranks == {"content": 2, "title": 1, "semantic": 1}


# ---- reciprocal rank fusion ----------------------------------------------------------------
def test_rrf_matches_hand_calculation():
    k = 60
    f = {x.doc_id: x for x in fuse_scores(content={"a": 9.0, "b": 5.0}, title={}, semantic={"b": 0.9, "a": 0.4},
                                         config=CFG, method="rrf")}
    best = 3 / (k + 1)                                    # three signals, all at rank 1
    assert f["a"].combined == pytest.approx((1 / (k + 1) + 1 / (k + 2)) / best)   # content rank 1, semantic rank 2
    assert f["b"].combined == pytest.approx((1 / (k + 2) + 1 / (k + 1)) / best)
    assert f["a"].combined == pytest.approx(f["b"].combined)                     # symmetric -> tie, ordered by id


def test_rrf_ignores_score_magnitude_and_rewards_agreement():
    content = {"both": 0.001, "kw_only": 1000.0}
    semantic = {"both": 0.21, "sem_only": 0.99}
    f = fuse_scores(content=content, title={}, semantic=semantic, config=CFG, method="rrf")
    ranks = {x.doc_id: x for x in f}
    assert ranks["kw_only"].ranks == {"content": 1} and ranks["both"].ranks == {"content": 2, "semantic": 2}
    # raw magnitudes (0.001 vs 1000) are irrelevant; agreement wins: 1/62 + 1/62 > 1/61
    assert f[0].doc_id == "both"
    assert ranks["both"].combined > ranks["kw_only"].combined == pytest.approx(ranks["sem_only"].combined)


def test_rrf_top_is_at_most_one_and_perfect_agreement_scores_one():
    f = fuse_scores(content={"a": 2.0}, title={"a": 1.0}, semantic={"a": 0.8}, config=CFG, method="rrf")[0]
    assert f.combined == pytest.approx(1.0) and f.components == {"content": 1.0, "title": 1.0, "semantic": 1.0}


def test_rrf_k_changes_how_sharply_rank_matters():
    content = {f"d{i}": float(100 - i) for i in range(10)}
    small = fuse_scores(content=content, title={}, semantic={}, config=HybridConfig(rrf_k=1), method="rrf")
    large = fuse_scores(content=content, title={}, semantic={}, config=HybridConfig(rrf_k=1000), method="rrf")
    assert small[0].combined / small[-1].combined > large[0].combined / large[-1].combined


def test_rrf_weighted_option_uses_configured_weights():
    kw = dict(content={"c": 1.0}, title={}, semantic={"s": 0.9})
    classic = {x.doc_id: x.combined for x in fuse_scores(**kw, config=CFG, method="rrf")}
    weighted = {x.doc_id: x.combined for x in fuse_scores(**kw, config=HybridConfig(rrf_weighted=True, bm25_weight=0.9,
                                                                                    semantic_weight=0.1, title_weight=0.0), method="rrf")}
    assert classic["c"] == pytest.approx(classic["s"]) and weighted["c"] > weighted["s"]


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        fuse_scores(content={"a": 1.0}, title={}, semantic={}, config=CFG, method="bogus")


# ---- configuration -------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    {"method": "nope"}, {"bm25_normalization": "nope"}, {"semantic_normalization": "nope"},
    {"bm25_weight": -1}, {"bm25_weight": 0, "semantic_weight": 0, "title_weight": 0},
    {"rrf_k": 0}, {"candidate_pool": 0}, {"snippet_chars": 5},
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        HybridConfig(**kwargs)


def test_config_from_settings_reads_weights_and_rrf_k():
    s = Settings(_env_file=None, hybrid_bm25_weight=0.5, hybrid_semantic_weight=0.3, hybrid_title_weight=0.1, rrf_k=30)
    cfg = HybridConfig.from_settings(s, method="rrf", snippet_chars=100)
    assert (cfg.bm25_weight, cfg.semantic_weight, cfg.title_weight, cfg.rrf_k) == (0.5, 0.3, 0.1, 30)
    assert cfg.method == "rrf" and cfg.snippet_chars == 100
    assert math.isclose(sum(HybridConfig.from_settings(Settings(_env_file=None)).weights().values()), 1.0)
