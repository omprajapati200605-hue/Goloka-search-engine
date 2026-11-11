import math
import random

import pytest

from app.search.bm25 import NativeBM25Backend, RankBM25Backend, idf, make_backend

BACKENDS = [RankBM25Backend, NativeBM25Backend]


def manual_bm25(corpus, query, k1=1.5, b=0.75):
    """Textbook formula, written independently of the implementations."""
    n = len(corpus)
    avgdl = sum(len(d) for d in corpus) / n
    out = {}
    for pos, doc in enumerate(corpus):
        s = 0.0
        for t in query:
            df = sum(1 for d in corpus if t in d)
            tf = doc.count(t)
            if df == 0 or tf == 0:
                continue
            term_idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            s += term_idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avgdl))
        if s > 0:
            out[pos] = s
    return out


CORPUS = [
    ["search", "engine", "crawler", "index"],
    ["search", "search", "ranking", "bm25", "scores"],
    ["cooking", "pasta", "recipe", "sauce", "tomato", "basil", "garlic", "olive"],
    ["engine", "pistons", "cylinders"],
]


@pytest.mark.parametrize("cls", BACKENDS)
def test_matches_textbook_formula(cls):
    backend = cls()
    backend.build(CORPUS)
    for query in (["search"], ["engine"], ["search", "engine"], ["bm25", "pasta"]):
        got, want = backend.score(query), manual_bm25(CORPUS, query)
        assert got.keys() == want.keys()
        for pos in want:
            assert got[pos] == pytest.approx(want[pos], rel=1e-9)


def test_native_and_rank_bm25_agree_on_random_corpus():
    rng = random.Random(42)
    vocab = [f"w{i}" for i in range(60)]
    corpus = [[rng.choice(vocab[: rng.randint(5, 60)]) for _ in range(rng.randint(1, 80))] for _ in range(150)]
    a, b = RankBM25Backend(), NativeBM25Backend()
    a.build(corpus)
    b.build(corpus)
    for _ in range(25):
        query = rng.sample(vocab, rng.randint(1, 5))
        sa, sb = a.score(query), b.score(query)
        assert sa.keys() == sb.keys()
        assert all(sa[p] == pytest.approx(sb[p], rel=1e-9) for p in sa)


@pytest.mark.parametrize("cls", BACKENDS)
def test_more_occurrences_rank_higher_with_saturation(cls):
    backend = cls()
    backend.build([["tomato"] + ["x"] * 9, ["tomato"] * 3 + ["x"] * 7, ["tomato"] * 9 + ["x"]])
    s = backend.score(["tomato"])
    assert s[0] < s[1] < s[2]
    assert (s[2] - s[1]) < (s[1] - s[0]) * 3  # diminishing returns, not linear growth


@pytest.mark.parametrize("cls", BACKENDS)
def test_shorter_document_wins_for_same_term_frequency(cls):
    backend = cls()
    backend.build([["tomato"] + ["filler"] * 3, ["tomato"] + ["filler"] * 50, ["other"] * 10])
    s = backend.score(["tomato"])
    assert s[0] > s[1]


@pytest.mark.parametrize("cls", BACKENDS)
def test_rare_term_outweighs_common_term(cls):
    corpus = [["common", "rare"]] + [["common", f"x{i}"] for i in range(9)]
    backend = cls()
    backend.build(corpus)
    assert backend.score(["rare"])[0] > backend.score(["common"])[0]


@pytest.mark.parametrize("cls", BACKENDS)
def test_term_in_every_document_still_scores_positive(cls):
    backend = cls()
    backend.build([["same", "a"], ["same", "b"], ["same", "c"]])
    assert all(v > 0 for v in backend.score(["same"]).values()) and len(backend.score(["same"])) == 3


@pytest.mark.parametrize("cls", BACKENDS)
def test_unknown_terms_empty_query_and_empty_corpus(cls):
    backend = cls()
    backend.build([])
    assert backend.score(["anything"]) == {}
    backend.build(CORPUS)
    assert backend.score(["zzz"]) == {} and backend.score([]) == {}


@pytest.mark.parametrize("cls", BACKENDS)
def test_parameters_change_scores(cls):
    default, no_length = cls(), cls(b=0.0)
    corpus = [["a"] * 2 + ["f"] * 2, ["a"] * 2 + ["f"] * 40, ["z"]]
    default.build(corpus)
    no_length.build(corpus)
    d, n = default.score(["a"]), no_length.score(["a"])
    assert n[0] == pytest.approx(n[1]) and d[0] > d[1]   # b=0 ignores length


def test_idf_is_positive_and_decreasing():
    assert idf(10, 10) > 0
    assert idf(100, 1) > idf(100, 10) > idf(100, 90) > 0


def test_make_backend():
    assert isinstance(make_backend("native"), NativeBM25Backend)
    assert isinstance(make_backend(), RankBM25Backend)
    with pytest.raises(ValueError):
        make_backend("nope")
