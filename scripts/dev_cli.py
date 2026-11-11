"""Developer command line for trying GOLOKA before the API exists.

    python -m scripts.dev_cli crawl https://example.com --max-pages 20 --max-depth 2
    python -m scripts.dev_cli search "your query" -k 10          # BM25 keyword search
    python -m scripts.dev_cli semantic "your query" -k 10        # embedding / vector search
    python -m scripts.dev_cli hybrid "your query" --method rrf   # BM25 + semantic + title, fused
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Sequence

from app.config import Settings, get_settings
from app.crawler.crawler import Crawler
from app.crawler.url_utils import Resolver, URLValidationError, system_resolver
from app.ingest import IngestStats, make_ingest_callback
from app.search.embedder import Embedder, EmbeddingError
from app.search.hybrid import HybridConfig, HybridSearcher
from app.search.keyword_index import KeywordIndex, default_index_path
from app.search.vector_index import VectorIndex
from app.storage.repository import DocumentRepository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dev_cli", description="GOLOKA developer CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    crawl = sub.add_parser("crawl", help="crawl a site and store its pages")
    crawl.add_argument("url")
    crawl.add_argument("--max-pages", type=int, default=None)
    crawl.add_argument("--max-depth", type=int, default=None)
    crawl.add_argument("--allow", action="append", default=[], metavar="DOMAIN",
                       help="extra domain the crawler may visit (repeatable)")

    search = sub.add_parser("search", help="BM25 keyword search over stored pages")
    search.add_argument("query")
    search.add_argument("-k", "--top", type=int, default=10)
    search.add_argument("--backend", choices=["rank_bm25", "native"], default="rank_bm25")

    semantic = sub.add_parser("semantic", help="semantic (embedding) search over stored pages")
    semantic.add_argument("query")
    semantic.add_argument("-k", "--top", type=int, default=10)

    hybrid = sub.add_parser("hybrid", help="hybrid search: BM25 + semantic + title, fused")
    hybrid.add_argument("query")
    hybrid.add_argument("-k", "--top", type=int, default=10)
    hybrid.add_argument("--method", choices=["weighted", "rrf"], default="weighted")
    return parser


def cmd_crawl(settings: Settings, args: argparse.Namespace, resolver: Resolver = system_resolver) -> int:
    settings.ensure_dirs()
    repo = DocumentRepository(settings.db_path)
    stats = IngestStats()

    async def run():
        crawler = Crawler(settings, resolver=resolver, on_page=make_ingest_callback(repo, stats))
        try:
            return await crawler.crawl(args.url, max_pages=args.max_pages, max_depth=args.max_depth,
                                       allowed_domains=args.allow)
        finally:
            await crawler.aclose()

    try:
        summary = asyncio.run(run())
    except (URLValidationError, ValueError) as exc:
        print(f"Cannot crawl: {exc}", file=sys.stderr)
        repo.close()
        return 2
    print(f"Crawled {summary.pages_crawled} page(s), {summary.pages_failed} failed, "
          f"{summary.pages_skipped} skipped (limits: {summary.max_pages} pages, depth {summary.max_depth}).")
    print(f"Stored: {stats.as_dict()}")
    for o in [o for o in summary.outcomes if o.status in ("error", "blocked", "restricted")][:10]:
        print(f"  {o.status}: {o.url} -> {o.detail}")
    print(f"Database now holds {repo.count()} searchable document(s).")
    repo.close()
    return 0


def cmd_search(settings: Settings, args: argparse.Namespace) -> int:
    settings.ensure_dirs()
    repo = DocumentRepository(settings.db_path)
    if repo.count() == 0:
        print("No documents stored yet. Run:  python -m scripts.dev_cli crawl <url>")
        repo.close()
        return 1
    path = default_index_path(settings)
    index = KeywordIndex.load(path, backend=args.backend)
    stats = index.sync(repo)
    index.save(path)
    print(f"Index: {len(index)} documents (added {stats.added}, updated {stats.updated}, "
          f"unchanged {stats.unchanged}, removed {stats.removed})")
    hits = index.search(args.query, k=args.top)
    if not hits:
        print("No results.")
    docs = repo.get_many([h.doc_id for h in hits])
    for rank, hit in enumerate(hits, 1):
        doc = docs[hit.doc_id]
        print(f"{rank}. [{hit.score:.3f}] {doc.title or '(no title)'}")
        print(f"   {doc.url}   (title {hit.title_score:.3f}, content {hit.content_score:.3f})")
    repo.close()
    return 0


def cmd_semantic(settings: Settings, args: argparse.Namespace, embedder: Embedder | None = None) -> int:
    settings.ensure_dirs()
    repo = DocumentRepository(settings.db_path)
    if repo.count() == 0:
        print("No documents stored yet. Run:  python -m scripts.dev_cli crawl <url>")
        repo.close()
        return 1
    print(f"Loading embedding model {settings.embedding_model} (the first run downloads ~90 MB)...", flush=True)
    try:
        index = VectorIndex.from_settings(settings, embedder)
        stats = index.sync(repo)
        index.save(settings.vector_dir / "chunks.faiss")
        hits = index.search(args.query, k=args.top)
    except EmbeddingError as exc:
        print(f"Embedding problem: {exc}", file=sys.stderr)
        repo.close()
        return 3
    print(f"Vector index: {len(index)} chunks (added {stats.added}, updated {stats.updated}, "
          f"unchanged {stats.unchanged}, removed {stats.removed}, failed {stats.failed})")
    if not hits:
        print("No results.")
    docs = repo.get_many([h.doc_id for h in hits])
    for rank, hit in enumerate(hits, 1):
        doc = docs[hit.doc_id]
        print(f"{rank}. [similarity {hit.score:.3f}] {doc.title or '(no title)'}")
        print(f"   {doc.url}")
        print(f"   ...{hit.chunk_text[:160]}...")
    repo.close()
    return 0


def cmd_hybrid(settings: Settings, args: argparse.Namespace, embedder: Embedder | None = None) -> int:
    settings.ensure_dirs()
    repo = DocumentRepository(settings.db_path)
    if repo.count() == 0:
        print("No documents stored yet. Run:  python -m scripts.dev_cli crawl <url>")
        repo.close()
        return 1
    keyword_path = default_index_path(settings)
    keyword = KeywordIndex.load(keyword_path)
    keyword.sync(repo)
    keyword.save(keyword_path)

    vectors: VectorIndex | None = None
    notes: list[str] = []
    try:
        print(f"Preparing semantic index with {settings.embedding_model} (first run downloads ~90 MB)...", flush=True)
        vectors = VectorIndex.from_settings(settings, embedder)
        vectors.sync(repo)
        vectors.save(settings.vector_dir / "chunks.faiss")
    except EmbeddingError as exc:
        notes.append(f"semantic search unavailable ({exc}); showing keyword results only")
        vectors = None

    searcher = HybridSearcher(repo, keyword, vectors, HybridConfig.from_settings(settings, method=args.method))
    response = searcher.search(args.query, k=args.top)
    for warning in notes + response.warnings:
        print(f"WARNING: {warning}")
    print(f"Hybrid search ({response.method}) - {len(response.results)} result(s) in {response.took_ms:.0f} ms")
    if not response.results:
        print("No results.")
    for rank, r in enumerate(response.results, 1):
        bm25 = "-" if r.bm25_score is None else f"{r.bm25_score:.2f}"
        sem = "-" if r.semantic_score is None else f"{r.semantic_score:.2f}"
        print(f"{rank}. [{r.combined_score:.3f}] {r.title or '(no title)'}   (bm25 {bm25}, semantic {sem})")
        print(f"   {r.url}")
        print(f"   {r.snippet}")
    repo.close()
    return 0


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = settings or get_settings()
    handlers = {"crawl": cmd_crawl, "search": cmd_search, "semantic": cmd_semantic, "hybrid": cmd_hybrid}
    return handlers[args.command](settings, args)


if __name__ == "__main__":
    raise SystemExit(main())
