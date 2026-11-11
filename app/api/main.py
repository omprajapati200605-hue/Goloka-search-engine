"""FastAPI application factory.   Run with:  python -m scripts.run_api   (docs at /docs)"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.errors import register_error_handlers
from app.api.routes import public_router, router
from app.config import Settings, get_settings
from app.crawler.url_utils import Resolver, system_resolver
from app.engine import SearchEngine
from app.jobs.manager import JobManager
from app.jobs.store import JobStore

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

DESCRIPTION = """GOLOKA search engine API: stored web documents searched by BM25 keywords, semantic embeddings, or both.

Only documents that were actually crawled and indexed are ever returned."""

TAGS = [
    {"name": "search", "description": "Keyword, semantic and hybrid search"},
    {"name": "documents", "description": "Browse stored documents"},
    {"name": "crawl", "description": "Background crawl jobs: start, track progress, see errors"},
    {"name": "indexing", "description": "Build and update the search indexes"},
    {"name": "system", "description": "Health and status"},
]


# The page loads only its own files and talks only to its own server; crawled text can never run as script.
# Applied to the page itself, not to /docs, which loads its viewer from a CDN.
PAGE_HEADERS = {
    "Cache-Control": "no-cache",
    "Content-Security-Policy": ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
                                "img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def create_app(settings: Settings | None = None, engine: SearchEngine | None = None, *,
               resolver: Resolver = system_resolver) -> FastAPI:
    """`engine` and `resolver` let tests inject a stand-in embedder and fake DNS; otherwise real ones are used."""
    settings = settings or (engine.settings if engine else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=settings.log_level.upper())
        owned = engine is None
        app.state.engine = engine or SearchEngine(settings)
        app.state.started_at = time.time()
        app.state.jobs_store = JobStore(settings.db_path)
        app.state.jobs = JobManager(settings, app.state.engine.repo, app.state.jobs_store,
                                    app.state.engine.index_pending, resolver=resolver)
        await app.state.jobs.start()
        try:
            yield
        finally:
            await app.state.jobs.stop()
            app.state.jobs_store.close()
            if owned:
                app.state.engine.close()

    app = FastAPI(title="GOLOKA", version=__version__, description=DESCRIPTION, openapi_tags=TAGS, lifespan=lifespan)
    app.state.settings = settings
    register_error_handlers(app)
    if settings.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                           allow_headers=["X-API-Key", "Content-Type"])
    app.include_router(public_router)
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html", media_type="text/html; charset=utf-8", headers=PAGE_HEADERS)

    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
    return app


app = create_app()
