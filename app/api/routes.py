"""HTTP endpoints. Handlers are plain `def` so FastAPI runs them in a thread pool (SQLite/FAISS/BM25 are blocking)."""
from __future__ import annotations

import time
from typing import Literal

from fastapi import APIRouter, Depends, Query, Request, Response

from app import __version__
from app.api.auth import require_api_key
from app.api.errors import ApiError
from app.api.schemas import (CrawlRequest, JobList, JobOut, JobProgress, JobUrl, JobUrlList, DocumentCounts, DocumentDetail, DocumentList, DocumentSummary, ErrorResponse,
                             HealthResponse, IndexCounts, IndexResponse, ResultItem, SearchResponseOut,
                             SemanticStatus)
from app.engine import IndexingInProgress, IndexReport, SearchEngine
from app.jobs.manager import JobManager, JobRejected
from app.jobs.store import Job
from app.search.hybrid import SearchResponse
from app.storage.repository import Document

router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])   # protected when an API key is set
public_router = APIRouter(prefix="/api/v1")                                       # health: always open

Q = Query(..., min_length=1, max_length=500, description="The search query")
K = Query(10, ge=1, le=50, description="Maximum number of results")


def _engine(request: Request) -> SearchEngine:
    return request.app.state.engine


def _query(q: str) -> str:
    if not q.strip():
        raise ApiError(422, "validation_error", "Query must not be blank")
    return q.strip()


def _search_out(r: SearchResponse) -> SearchResponseOut:
    items = [ResultItem(document_id=x.doc_id, **{k: v for k, v in x.to_dict().items() if k != "doc_id"})
             for x in r.results]
    return SearchResponseOut(query=r.query, mode=r.mode, method=r.method, total_returned=len(items),
                             took_ms=round(r.took_ms, 2), results=items, warnings=r.warnings)


def _summary(d: Document) -> dict:
    return dict(id=d.id, url=d.url, title=d.title, domain=d.domain, content_hash=d.content_hash,
                crawled_at=d.crawled_at, last_updated_at=d.last_updated_at, crawl_status=d.crawl_status,
                indexed=d.indexed_hash == d.content_hash, word_count=d.metadata.get("word_count"))


def _index_out(r: IndexReport) -> IndexResponse:
    return IndexResponse(status=r.status, documents=r.documents, keyword=r.keyword, vectors=r.vectors,
                         fully_indexed=r.fully_indexed, pending=r.pending, warnings=r.warnings,
                         duration_ms=round(r.duration_ms, 2))


def _jobs(request: Request) -> JobManager:
    return request.app.state.jobs


def _job_out(j: Job) -> JobOut:
    percent = 100.0 if j.status == "completed" else min(100.0, round(j.urls_processed / j.max_pages * 100, 1))
    return JobOut(
        id=j.id, seed_url=j.seed_url, status=j.status, phase=j.phase, max_pages=j.max_pages, max_depth=j.max_depth,
        allowed_domains=j.allowed_domains, auto_index=j.auto_index,
        progress=JobProgress(urls_processed=j.urls_processed, pages_crawled=j.pages_crawled,
                             pages_failed=j.pages_failed, pages_skipped=j.pages_skipped, max_pages=j.max_pages,
                             percent=percent),
        documents=j.stored, index=j.index_result, error=j.error, created_at=j.created_at,
        started_at=j.started_at, finished_at=j.finished_at)


@public_router.get("/health", response_model=HealthResponse, tags=["system"], summary="Service and index status")
def health(request: Request) -> HealthResponse:
    snap = _engine(request).snapshot()
    stats = snap["stats"]
    degraded = snap["semantic_status"] == "unavailable"
    return HealthResponse(
        status="degraded" if degraded else "ok", version=__version__,
        uptime_seconds=round(time.time() - request.app.state.started_at, 1),
        documents=DocumentCounts(total=stats.total, by_status=stats.by_status, indexed=stats.indexed,
                                 pending_indexing=stats.pending_indexing, last_crawled_at=stats.last_crawled_at),
        indexes=IndexCounts(keyword_documents=snap["keyword_documents"], vector_chunks=snap["vector_chunks"],
                            semantic=SemanticStatus(model=snap["embedding_model"], status=snap["semantic_status"],
                                                    last_error=snap["semantic_error"])),
        last_index=_index_out(snap["last_index"]) if snap["last_index"] else None,
        jobs=request.app.state.jobs_store.counts_by_status())


@router.post("/index", response_model=IndexResponse, tags=["indexing"], summary="Index stored documents",
             description="Builds/updates the BM25 and vector indexes for every stored document that is new or "
                         "changed. Safe to repeat: unchanged documents are skipped and never re-embedded. "
                         "If the embedding model is unavailable the keyword index is still updated and "
                         "`status` is `partial`.",
             responses={409: {"model": ErrorResponse, "description": "An indexing run is already in progress"}})
def index_documents(request: Request) -> IndexResponse:
    try:
        return _index_out(_engine(request).index_pending())
    except IndexingInProgress as exc:
        raise ApiError(409, "indexing_in_progress", str(exc)) from exc


_search_errors = {422: {"model": ErrorResponse, "description": "Invalid parameters"}}


@router.get("/search", response_model=SearchResponseOut, tags=["search"], responses=_search_errors,
            summary="Hybrid search (BM25 + semantic + title)")
def search(request: Request, q: str = Q, k: int = K,
           method: Literal["weighted", "rrf"] = Query("weighted", description="Fusion method"),
           min_score: float | None = Query(None, ge=0, le=1, description="Drop results below this combined score")):
    return _search_out(_engine(request).search(_query(q), k, method=method, min_score=min_score))


@router.get("/search/keyword", response_model=SearchResponseOut, tags=["search"], responses=_search_errors,
            summary="Keyword search (BM25 only)")
def search_keyword(request: Request, q: str = Q, k: int = K):
    return _search_out(_engine(request).search_keyword(_query(q), k))


@router.get("/search/semantic", response_model=SearchResponseOut, tags=["search"], responses=_search_errors,
            summary="Semantic search (embeddings only)")
def search_semantic(request: Request, q: str = Q, k: int = K,
                    min_score: float | None = Query(None, ge=-1, le=1, description="Minimum cosine similarity")):
    return _search_out(_engine(request).search_semantic(_query(q), k, min_score=min_score))


@router.get("/documents", response_model=DocumentList, tags=["documents"], summary="List stored documents")
def list_documents(request: Request, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                   domain: str | None = Query(None, description="Only documents from this domain"),
                   status: str | None = Query(None, description="Filter by crawl status, e.g. ok or gone")):
    docs, total = _engine(request).repo.list(limit=limit, offset=offset, domain=domain, status=status)
    return DocumentList(items=[DocumentSummary(**_summary(d)) for d in docs], total=total, limit=limit, offset=offset)


@router.get("/documents/{document_id}", response_model=DocumentDetail, tags=["documents"],
            summary="Get one document with its full text",
            responses={404: {"model": ErrorResponse, "description": "No such document"}})
def get_document(request: Request, document_id: str):
    doc = _engine(request).repo.get(document_id)
    if doc is None:
        raise ApiError(404, "not_found", f"Document '{document_id}' does not exist")
    return DocumentDetail(**_summary(doc), text=doc.text, metadata=doc.metadata)


# ---- crawl jobs ----------------------------------------------------------------------------
_no_job = {404: {"model": ErrorResponse, "description": "No such job"}}


@router.post("/crawl", response_model=JobOut, status_code=202, tags=["crawl"],
             summary="Start a background crawl job",
             description="Validates the URL (public http/https hosts only; robots.txt is obeyed during the crawl) "
                         "and queues a job. Returns immediately with a job id; poll `GET /api/v1/jobs/{id}` for "
                         "status, progress and errors. Jobs run one at a time, in order.",
             responses={422: {"model": ErrorResponse, "description": "URL or limits rejected"},
                        429: {"model": ErrorResponse, "description": "Too many unfinished jobs"}})
async def start_crawl(request: Request, body: CrawlRequest, response: Response) -> JobOut:
    try:
        job = await _jobs(request).submit(body.url, max_pages=body.max_pages, max_depth=body.max_depth,
                                          allowed_domains=body.allowed_domains, auto_index=body.auto_index)
    except JobRejected as exc:
        raise ApiError(exc.status_code, exc.code, exc.message) from exc
    response.headers["Location"] = f"/api/v1/jobs/{job.id}"
    return _job_out(job)


@router.get("/jobs", response_model=JobList, tags=["crawl"], summary="List crawl jobs (newest first)")
def list_jobs(request: Request, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
              status: Literal["pending", "running", "completed", "failed"] | None = Query(None)):
    jobs, total = request.app.state.jobs_store.list(status=status, limit=limit, offset=offset)
    return JobList(items=[_job_out(j) for j in jobs], total=total, limit=limit, offset=offset)


@router.get("/jobs/{job_id}", response_model=JobOut, tags=["crawl"], summary="Job status, progress and errors",
            responses=_no_job)
def get_job(request: Request, job_id: str):
    job = request.app.state.jobs_store.get(job_id)
    if job is None:
        raise ApiError(404, "not_found", f"Job '{job_id}' does not exist")
    return _job_out(job)


@router.get("/jobs/{job_id}/urls", response_model=JobUrlList, tags=["crawl"],
            summary="Per-URL outcomes of a job (use status=error to see what failed)", responses=_no_job)
def get_job_urls(request: Request, job_id: str, limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
                 status: Literal["fetched", "duplicate", "blocked", "restricted", "skipped", "error"] | None = Query(None)):
    store = request.app.state.jobs_store
    if store.get(job_id) is None:
        raise ApiError(404, "not_found", f"Job '{job_id}' does not exist")
    rows, total = store.list_urls(job_id, status=status, limit=limit, offset=offset)
    return JobUrlList(items=[JobUrl(**r) for r in rows], total=total, limit=limit, offset=offset)
