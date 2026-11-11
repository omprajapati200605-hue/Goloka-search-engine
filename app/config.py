"""Central configuration, loaded from environment variables (prefix GOLOKA_) or .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GOLOKA_", env_file=".env", extra="ignore")

    # Storage
    data_dir: Path = Path("./data")
    db_path: Path = Path("./data/goloka.db")
    vector_dir: Path = Path("./data/vectors")

    # Crawler
    user_agent: str = "GOLOKABot/0.1 (+https://example.com/bot)"
    crawl_timeout_seconds: float = Field(10.0, gt=0)
    crawl_max_redirects: int = Field(5, ge=0, le=20)
    crawl_max_page_bytes: int = Field(2_000_000, gt=0)
    crawl_per_domain_delay_seconds: float = Field(1.0, ge=0)
    crawl_concurrency: int = Field(4, ge=1, le=64)
    crawl_default_max_pages: int = Field(50, ge=1)
    crawl_default_max_depth: int = Field(2, ge=0)
    crawl_hard_max_pages: int = Field(1000, ge=1)
    crawl_hard_max_depth: int = Field(6, ge=0)

    # Background jobs
    jobs_max_pending: int = Field(20, ge=1)      # queued + running crawl jobs accepted at once

    # Semantic search
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size_words: int = Field(200, ge=20)
    chunk_overlap_words: int = Field(40, ge=0)

    # Hybrid search
    hybrid_bm25_weight: float = Field(0.4, ge=0)
    hybrid_semantic_weight: float = Field(0.4, ge=0)
    hybrid_title_weight: float = Field(0.2, ge=0)
    rrf_k: int = Field(60, ge=1)
    hybrid_min_semantic_score: float = Field(0.2, ge=-1, le=1)   # cosine below this is not a semantic match

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"
    api_key: SecretStr | None = None              # when set, every endpoint except /health needs X-API-Key
    public_read: bool = False                     # with an API key set: let anyone search/read; writes still need the key
    cors_origins: list[str] = Field(default_factory=list)   # JSON list, e.g. ["http://localhost:3000"]

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")
        if self.crawl_default_max_pages > self.crawl_hard_max_pages:
            raise ValueError("crawl_default_max_pages exceeds crawl_hard_max_pages")
        if self.crawl_default_max_depth > self.crawl_hard_max_depth:
            raise ValueError("crawl_default_max_depth exceeds crawl_hard_max_depth")
        if self.hybrid_bm25_weight + self.hybrid_semantic_weight + self.hybrid_title_weight <= 0:
            raise ValueError("at least one hybrid weight must be positive")
        return self

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
