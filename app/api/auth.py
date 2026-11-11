"""Optional API-key protection. Off unless GOLOKA_API_KEY is set. /health, /docs and /openapi.json stay open."""
from __future__ import annotations

import hmac

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException

_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="Required only when the server sets GOLOKA_API_KEY")


def _check(request: Request, supplied: str | None) -> None:
    configured = request.app.state.settings.api_key
    if configured is None:
        return
    expected = configured.get_secret_value().encode()
    if not supplied or not hmac.compare_digest(supplied.encode(), expected):   # constant-time comparison
        raise HTTPException(401, "Missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"})


def require_read_access(request: Request, supplied: str | None = Depends(_header)) -> None:
    """Search and document reads: open when GOLOKA_PUBLIC_READ is true, otherwise they need the key too."""
    if request.app.state.settings.public_read:
        return
    _check(request, supplied)


def require_api_key(request: Request, supplied: str | None = Depends(_header)) -> None:
    """Anything that changes state (indexing, crawling, jobs): always needs the key when one is set."""
    _check(request, supplied)
