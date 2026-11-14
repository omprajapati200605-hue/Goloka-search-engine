"""Pydantic models: the exact shape of every API response (validated before they leave the server)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str = Field(description="Stable machine-readable error code, e.g. 'not_found'")
    message: str
    details: list[dict[str, Any]] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class ResultItem(BaseModel):
    document_id: str
    title: str
    url: str
    domain: str
    snippet: str
    bm25_score: float | None = Field(None, description="Raw keyword score; null if the keyword search did not match")
    semantic_score: float | None = Field(None, description="Cosine similarity; null if semantic search did not match")
    combined_score: float = Field(description="Score used for ranking (0..1 for hybrid)")
    title_score: float | None = None
    content_score: float | None = None
    components: dict[str, float] = Field(default_factory=dict, description="Normalised per-signal scores (hybrid)")
    ranks: dict[str, int] = Field(default_factory=dict, description="1-based rank per signal (hybrid)")


class SearchResponseOut(BaseModel):
    query: str
    mode: str = Field(description="hybrid | keyword | semantic")
    method: str | None = Field(None, description="weighted | rrf (hybrid only)")
    total_returned: int
    took_ms: float
    results: list[ResultItem]
    warnings: list[str] = Field(default_factory=list)


class DocumentSummary(BaseModel):
    id: str
    url: str
    title: str
    domain: str
    content_hash: str
    crawled_at: str
    last_updated_at: str
    crawl_status: str
    indexed: bool = Field(description="True when the current content is in both the keyword and vector index")
    word_count: int | None = None


class DocumentList(BaseModel):
    items: list[DocumentSummary]
    total: int
    limit: int
    offset: int


class DocumentDetail(DocumentSummary):
    text: str
    metadata: dict[str, Any]


class IndexResponse(BaseModel):
    status: str = Field(description="completed | partial")
    documents: int
    keyword: dict[str, int]
    vectors: dict[str, int] | None
    fully_indexed: int
    pending: int
    warnings: list[str]
    duration_ms: float


class DocumentCounts(BaseModel):
    total: int
    by_status: dict[str, int]
    indexed: int
    pending_indexing: int
    last_crawled_at: str | None


class SemanticStatus(BaseModel):
    model: str
    status: str = Field(description="not_loaded | ready | unavailable | disabled")
    last_error: str | None


class IndexCounts(BaseModel):
    keyword_documents: int
    vector_chunks: int
    semantic: SemanticStatus


class HealthResponse(BaseModel):
    status: str = Field(description="ok | degraded")
    version: str
    uptime_seconds: float
    documents: DocumentCounts
    indexes: IndexCounts
    last_index: IndexResponse | None = None
    jobs: dict[str, int] = Field(default_factory=dict, description="Crawl jobs by status")


# ---- crawl jobs ------------------------------------------------------------------------------
class CrawlRequest(BaseModel):
    model_config = {"extra": "forbid"}

    url: str = Field(min_length=1, max_length=2048, description="Seed URL (http/https, public hosts only)")
    max_pages: int | None = Field(None, ge=1, description="Page limit; defaults to the server setting")
    max_depth: int | None = Field(None, ge=0, description="Link depth limit; defaults to the server setting")
    allowed_domains: list[str] = Field(default_factory=list, max_length=20,
                                       description="Extra domains the crawl may visit (the seed's domain is always allowed)")
    auto_index: bool = Field(True, description="Index the crawled pages automatically when the crawl finishes")


class JobProgress(BaseModel):
    urls_processed: int
    pages_crawled: int
    pages_failed: int
    pages_skipped: int
    max_pages: int
    percent: float = Field(description="urls_processed / max_pages. The crawl may end earlier, when links run out")


class JobOut(BaseModel):
    id: str
    seed_url: str
    status: str = Field(description="pending | running | completed | failed")
    phase: str = Field(description="queued | crawling | indexing | done")
    max_pages: int
    max_depth: int
    allowed_domains: list[str]
    auto_index: bool
    progress: JobProgress
    documents: dict[str, int] = Field(description="Stored-document outcomes: created/updated/unchanged/duplicate/empty")
    index: dict[str, Any] | None = Field(None, description="Result of the automatic indexing step")
    error: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None


class JobList(BaseModel):
    items: list[JobOut]
    total: int
    limit: int
    offset: int


class JobUrl(BaseModel):
    url: str
    depth: int
    status: str = Field(description="fetched | duplicate | blocked | restricted | skipped | error")
    http_status: int | None
    detail: str | None
    fetched_at: str | None


class JobUrlList(BaseModel):
    items: list[JobUrl]
    total: int
    limit: int
    offset: int
