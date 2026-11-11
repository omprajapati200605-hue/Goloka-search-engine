"""Hybrid retrieval: BM25 (content + title) fused with semantic similarity.

Signals (each scored per candidate document)
    content   BM25 on the page text          weight  hybrid_bm25_weight      (default 0.4)
    semantic  cosine of the best chunk       weight  hybrid_semantic_weight  (default 0.4)
    title     BM25 on the title              weight  hybrid_title_weight     (default 0.2)

Fusion methods
    weighted  combined = sum(weight_i * normalised_score_i) / sum(weights of available signals)
              Each signal is normalised to 0..1 first (see NORMALIZATIONS). Default: BM25 and title by
              their best score ("max"), semantic by absolute cosine ("clamp") so that a weak
              semantic neighbour is not stretched into a strong one.
    rrf       Reciprocal Rank Fusion: combined = sum(1 / (k + rank_i)); ignores raw score magnitudes and
              only uses each signal's ranking (k = hybrid_rrf_k, default 60). Rescaled to 0..1 by the best
              achievable value so both methods report comparable numbers.

Both methods return combined scores in 0..1. Nothing is invented: every result is a stored, searchable
document that at least one signal retrieved. If semantic search is unavailable the search degrades to
keyword-only and says so in `warnings`.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any

from app.config import Settings
from app.search.embedder import EmbeddingError
from app.search.keyword_index import KeywordIndex
from app.search.snippet import build_snippet
from app.search.tokenizer import analyze_query
from app.search.vector_index import VectorIndex
from app.storage.repository import STATUS_OK, DocumentRepository

NORMALIZATIONS = ("max", "min_max", "zscore", "clamp")
METHODS = ("weighted", "rrf")
SIGNALS = ("content", "semantic", "title")


# ------------------------------------------------------------------------------------ configuration
@dataclass(frozen=True)
class HybridConfig:
    method: str = "weighted"
    bm25_weight: float = 0.4
    semantic_weight: float = 0.4
    title_weight: float = 0.2
    rrf_k: int = 60
    rrf_weighted: bool = False          # False = classic RRF (every signal counts equally)
    bm25_normalization: str = "max"
    title_normalization: str = "max"
    semantic_normalization: str = "clamp"
    candidate_pool: int = 100           # candidates fetched per signal
    min_semantic_score: float = 0.2     # cosine below this is not treated as a semantic match (model-dependent)
    snippet_chars: int = 240

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        for name in ("bm25_normalization", "title_normalization", "semantic_normalization"):
            if getattr(self, name) not in NORMALIZATIONS:
                raise ValueError(f"{name} must be one of {NORMALIZATIONS}")
        if min(self.bm25_weight, self.semantic_weight, self.title_weight) < 0:
            raise ValueError("weights must not be negative")
        if self.bm25_weight + self.semantic_weight + self.title_weight <= 0:
            raise ValueError("at least one weight must be positive")
        if self.rrf_k < 1 or self.candidate_pool < 1 or self.snippet_chars < 20:
            raise ValueError("rrf_k and candidate_pool must be >= 1 and snippet_chars >= 20")

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: Any) -> "HybridConfig":
        return replace(cls(
            bm25_weight=settings.hybrid_bm25_weight, semantic_weight=settings.hybrid_semantic_weight,
            title_weight=settings.hybrid_title_weight, rrf_k=settings.rrf_k,
            min_semantic_score=settings.hybrid_min_semantic_score), **overrides)

    def weights(self) -> dict[str, float]:
        return {"content": self.bm25_weight, "semantic": self.semantic_weight, "title": self.title_weight}

    def normalizations(self) -> dict[str, str]:
        return {"content": self.bm25_normalization, "semantic": self.semantic_normalization,
                "title": self.title_normalization}


# ------------------------------------------------------------------------------------ pure fusion maths
def normalize_scores(scores: dict[str, float], method: str) -> dict[str, float]:
    """Map raw scores to 0..1."""
    if not scores:
        return {}
    values = list(scores.values())
    if method == "max":
        top = max(values)
        return {k: (v / top if top > 0 else 0.0) for k, v in scores.items()}
    if method == "min_max":
        lo, hi = min(values), max(values)
        if hi == lo:
            return {k: (1.0 if hi > 0 else 0.0) for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
    if method == "zscore":
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        if std == 0:
            return {k: 0.5 for k in scores}
        return {k: 1.0 / (1.0 + math.exp(-(v - mean) / std)) for k, v in scores.items()}
    if method == "clamp":
        return {k: min(1.0, max(0.0, v)) for k, v in scores.items()}
    raise ValueError(f"unknown normalization {method!r}")


@dataclass
class FusedScore:
    doc_id: str
    combined: float
    components: dict[str, float]          # per-signal contribution on a 0..1 scale
    ranks: dict[str, int] = field(default_factory=dict)   # 1-based rank in each signal that retrieved it


def fuse_scores(*, content: dict[str, float], title: dict[str, float], semantic: dict[str, float],
                config: HybridConfig, method: str | None = None, semantic_available: bool = True) -> list[FusedScore]:
    """Fuse per-signal raw scores into one ranked list (best first, ties broken by document id)."""
    method = method or config.method
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    raw = {"content": content, "title": title, "semantic": semantic if semantic_available else {}}
    available = [s for s in SIGNALS if s != "semantic" or semantic_available]
    weights = config.weights()
    wsum = sum(weights[s] for s in available)
    if wsum <= 0:                                   # e.g. only semantic had weight but it is unavailable
        weights = {s: 1.0 for s in SIGNALS}
        wsum = float(len(available))

    ranks: dict[str, dict[str, int]] = {}
    for signal, scores in raw.items():
        for rank, doc_id in enumerate(sorted(scores, key=lambda d: (-scores[d], d)), start=1):
            ranks.setdefault(doc_id, {})[signal] = rank

    fused: list[FusedScore] = []
    if method == "weighted":
        normalised = {s: normalize_scores(raw[s], config.normalizations()[s]) for s in SIGNALS}
        for doc_id in ranks:
            comps = {s: normalised[s].get(doc_id, 0.0) for s in SIGNALS}
            combined = sum(weights[s] * comps[s] for s in available) / wsum
            fused.append(FusedScore(doc_id, combined, comps, ranks[doc_id]))
    else:
        rrf_w = weights if config.rrf_weighted else {s: 1.0 for s in SIGNALS}
        best = sum(rrf_w[s] / (config.rrf_k + 1) for s in available)
        for doc_id, doc_ranks in ranks.items():
            comps = {s: ((1.0 / (config.rrf_k + doc_ranks[s])) / (1.0 / (config.rrf_k + 1)) if s in doc_ranks else 0.0)
                     for s in SIGNALS}
            total = sum(rrf_w[s] / (config.rrf_k + r) for s, r in doc_ranks.items())
            fused.append(FusedScore(doc_id, total / best, comps, doc_ranks))
    fused = [f for f in fused if f.combined > 0]      # no weighted signal supports it -> not a result
    fused.sort(key=lambda f: (-f.combined, f.doc_id))
    return fused


# ------------------------------------------------------------------------------------ results
@dataclass
class SearchResult:
    doc_id: str
    title: str
    url: str
    domain: str
    snippet: str
    bm25_score: float | None        # raw keyword score (content + boosted title); None if keyword search did not match
    semantic_score: float | None    # raw cosine similarity; None if semantic search did not match
    combined_score: float           # score used for ranking: fused 0..1 (hybrid) or the single signal's raw score
    title_score: float | None = None
    content_score: float | None = None
    components: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class SearchResponse:
    query: str
    mode: str                       # hybrid | keyword | semantic
    method: str | None              # weighted | rrf (hybrid only)
    results: list[SearchResult]
    warnings: list[str] = field(default_factory=list)
    candidates: int = 0
    took_ms: float = 0.0


# ------------------------------------------------------------------------------------ searcher
class HybridSearcher:
    def __init__(self, repo: DocumentRepository, keyword_index: KeywordIndex,
                 vector_index: VectorIndex | None = None, config: HybridConfig | None = None) -> None:
        self._repo = repo
        self._keyword = keyword_index
        self._vector = vector_index
        self.config = config or HybridConfig()

    # -- candidate retrieval
    def _keyword_candidates(self, query: str):
        hits = self._keyword.search(query, k=self.config.candidate_pool)
        return (hits,
                {h.doc_id: h.content_score for h in hits if h.content_score > 0},
                {h.doc_id: h.title_score for h in hits if h.title_score > 0})

    def _semantic_candidates(self, query: str, warnings: list[str]):
        """Returns {doc_id: SemanticHit} or None when semantic search cannot run."""
        if self._vector is None:
            warnings.append("semantic search is not configured; showing keyword results only")
            return None
        try:
            hits = self._vector.search(query, k=self.config.candidate_pool, min_score=self.config.min_semantic_score)
        except EmbeddingError as exc:
            warnings.append(f"semantic search unavailable ({exc}); showing keyword results only")
            return None
        return {h.doc_id: h for h in hits}

    # -- result assembly (drops anything that is no longer a searchable stored document)
    def _assemble(self, ordered: list[str], k: int, query: str, build) -> list[SearchResult]:
        results: list[SearchResult] = []
        batch = max(k * 2, 20)
        pos = 0
        while len(results) < k and pos < len(ordered):
            ids = ordered[pos:pos + batch]
            pos += batch
            docs = self._repo.get_many(ids)
            for doc_id in ids:
                doc = docs.get(doc_id)
                if doc is None or doc.crawl_status != STATUS_OK:
                    continue
                results.append(build(doc))
                if len(results) == k:
                    break
        return results

    def _snippet(self, doc, query: str, chunk: str | None = None, prefer_chunk: bool = False) -> str:
        n = self.config.snippet_chars
        if prefer_chunk and chunk:
            return build_snippet(chunk, query, max_chars=n, fallback=chunk)
        return build_snippet(doc.text, query, max_chars=n, fallback=chunk or doc.metadata.get("meta_description"))

    # -- public API
    def search(self, query: str, k: int = 10, *, method: str | None = None,
               min_score: float | None = None) -> SearchResponse:
        started = time.perf_counter()
        method = method or self.config.method
        warnings: list[str] = []
        response = SearchResponse(query=query, mode="hybrid", method=method, results=[], warnings=warnings)
        if k <= 0 or not analyze_query(query):
            return response

        kw_hits, content, title = self._keyword_candidates(query)
        kw_raw = {h.doc_id: h for h in kw_hits}
        sem_hits = self._semantic_candidates(query, warnings)
        semantic = {d: h.score for d, h in (sem_hits or {}).items()}

        fused = fuse_scores(content=content, title=title, semantic=semantic, config=self.config, method=method,
                            semantic_available=sem_hits is not None)
        if min_score is not None:
            fused = [f for f in fused if f.combined >= min_score]
        by_id = {f.doc_id: f for f in fused}
        response.candidates = len(fused)

        def build(doc) -> SearchResult:
            f, kw, sem = by_id[doc.id], kw_raw.get(doc.id), (sem_hits or {}).get(doc.id)
            keyword_matched = doc.id in content or doc.id in title
            return SearchResult(
                doc_id=doc.id, title=doc.title, url=doc.url, domain=doc.domain,
                snippet=self._snippet(doc, query, sem.chunk_text if sem else None, prefer_chunk=not keyword_matched),
                bm25_score=kw.score if kw and keyword_matched else None,
                semantic_score=sem.score if sem else None, combined_score=f.combined,
                title_score=title.get(doc.id), content_score=content.get(doc.id),
                components=f.components, ranks=f.ranks)

        response.results = self._assemble([f.doc_id for f in fused], k, query, build)
        response.took_ms = (time.perf_counter() - started) * 1000
        return response

    def search_keyword(self, query: str, k: int = 10) -> SearchResponse:
        started = time.perf_counter()
        response = SearchResponse(query=query, mode="keyword", method=None, results=[])
        if k <= 0 or not analyze_query(query):
            return response
        hits = {h.doc_id: h for h in self._keyword.search(query, k=max(k * 2, 20))}
        response.candidates = len(hits)

        def build(doc) -> SearchResult:
            h = hits[doc.id]
            return SearchResult(
                doc_id=doc.id, title=doc.title, url=doc.url, domain=doc.domain, snippet=self._snippet(doc, query),
                bm25_score=h.score, semantic_score=None, combined_score=h.score,
                title_score=h.title_score or None, content_score=h.content_score or None)

        response.results = self._assemble(list(hits), k, query, build)
        response.took_ms = (time.perf_counter() - started) * 1000
        return response

    def search_semantic(self, query: str, k: int = 10, *, min_score: float | None = None) -> SearchResponse:
        started = time.perf_counter()
        response = SearchResponse(query=query, mode="semantic", method=None, results=[])
        if k <= 0 or not (query or "").strip():
            return response
        threshold = self.config.min_semantic_score if min_score is None else min_score
        if self._vector is None:
            response.warnings.append("semantic search is not configured")
            return response
        try:
            hits = {h.doc_id: h for h in self._vector.search(query, k=max(k * 2, 20), min_score=threshold)}
        except EmbeddingError as exc:
            response.warnings.append(f"semantic search unavailable ({exc})")
            return response
        response.candidates = len(hits)

        def build(doc) -> SearchResult:
            h = hits[doc.id]
            return SearchResult(
                doc_id=doc.id, title=doc.title, url=doc.url, domain=doc.domain,
                snippet=self._snippet(doc, query, h.chunk_text, prefer_chunk=True),
                bm25_score=None, semantic_score=h.score, combined_score=h.score)

        response.results = self._assemble(list(hits), k, query, build)
        response.took_ms = (time.perf_counter() - started) * 1000
        return response
