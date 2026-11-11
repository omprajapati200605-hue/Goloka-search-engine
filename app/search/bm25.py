"""BM25 ranking backends.

BM25 formula (Okapi BM25, the ranking function behind Lucene / Elasticsearch):

    score(D, Q) = SUM over query terms t of   IDF(t) * ( f(t,D) * (k1 + 1) )
                                                       ----------------------------------------
                                                       f(t,D) + k1 * (1 - b + b * |D| / avgdl)

    IDF(t) = ln( 1 + (N - n(t) + 0.5) / (n(t) + 0.5) )

    f(t,D)  term frequency: how many times term t occurs in document D
    |D|     document length in tokens;     avgdl  average document length of the collection
    N       number of documents;           n(t)   number of documents containing t
    k1      term-frequency saturation (default 1.5): extra occurrences help less and less
    b       length normalisation (default 0.75): 0 = ignore length, 1 = fully normalise

Intuition: rare terms matter more (IDF), repeating a term helps with diminishing returns (k1),
and a match in a short document beats the same match in a long one (b).

IDF variant: stock `rank-bm25` uses ln((N-n+0.5)/(n+0.5)), which is zero/negative for terms found in
half of the documents or more (tiny collections then return nothing useful). Both backends here use
the always-positive Lucene form above, so they produce identical scores.

Backends are interchangeable (`BM25Backend`): the default wraps `rank-bm25`; `NativeBM25Backend`
is the independent implementation (inverted index, no third-party dependency).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Protocol, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def idf(n_docs: int, doc_freq: int) -> float:
    """Always-positive BM25 inverse document frequency (Lucene variant)."""
    return math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))


class BM25Backend(Protocol):
    def build(self, corpus: Sequence[Sequence[str]]) -> None:
        """Index a corpus: one token list per document (document position = list position)."""

    def score(self, query_terms: Sequence[str]) -> dict[int, float]:
        """Return {document position: BM25 score} for documents with a score > 0."""


class RankBM25Backend:
    """BM25 via the `rank-bm25` package (BM25Okapi) with the positive IDF variant."""

    def __init__(self, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        self.k1, self.b = k1, b
        self._model: BM25Okapi | None = None

    def build(self, corpus: Sequence[Sequence[str]]) -> None:
        if not corpus:
            self._model = None
            return

        class _PositiveIdfOkapi(BM25Okapi):
            def _calc_idf(self, nd):  # noqa: N802 - overrides library hook
                for word, freq in nd.items():
                    self.idf[word] = idf(self.corpus_size, freq)

        self._model = _PositiveIdfOkapi([list(d) for d in corpus], k1=self.k1, b=self.b)

    def score(self, query_terms: Sequence[str]) -> dict[int, float]:
        if self._model is None or not query_terms:
            return {}
        scores = self._model.get_scores(list(query_terms))
        return {int(i): float(scores[i]) for i in np.flatnonzero(scores > 0)}


class NativeBM25Backend:
    """From-scratch BM25 over an inverted index (only documents containing a query term are scored)."""

    def __init__(self, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        self.k1, self.b = k1, b
        self._doc_len: list[int] = []
        self._avgdl = 0.0
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}

    def build(self, corpus: Sequence[Sequence[str]]) -> None:
        n = len(corpus)
        self._doc_len = [len(d) for d in corpus]
        self._avgdl = (sum(self._doc_len) / n) if n else 0.0
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for pos, doc in enumerate(corpus):
            for term, tf in Counter(doc).items():
                postings[term].append((pos, tf))
        self._postings = dict(postings)
        self._idf = {t: idf(n, len(p)) for t, p in self._postings.items()}

    def score(self, query_terms: Sequence[str]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        k1, b, avgdl = self.k1, self.b, self._avgdl
        for term in query_terms:
            term_idf = self._idf.get(term)
            if term_idf is None:
                continue
            for pos, tf in self._postings[term]:
                norm = tf + k1 * (1.0 - b + b * self._doc_len[pos] / avgdl)
                scores[pos] += term_idf * tf * (k1 + 1.0) / norm
        return {p: s for p, s in scores.items() if s > 0}


BACKENDS = {"rank_bm25": RankBM25Backend, "native": NativeBM25Backend}


def make_backend(name: str = "rank_bm25", *, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> BM25Backend:
    try:
        return BACKENDS[name](k1=k1, b=b)
    except KeyError:
        raise ValueError(f"Unknown BM25 backend {name!r}; choose from {sorted(BACKENDS)}") from None
