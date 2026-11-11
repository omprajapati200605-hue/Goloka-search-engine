"""Optional API-key protection. Off unless GOLOKA_API_KEY is set. /health, /docs and /openapi.json stay open."""
from __future__ import annotations

import hmac

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException

_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="Required only when the server sets GOLOKA_API_KEY")


def require_api_key(request: Request, supplied: str | None = Depends(_header)) -> None:
    configured = request.app.state.settings.api_key
    if configured is None:
        return
    expected = configured.get_secret_value().encode()
    if not supplied or not hmac.compare_digest(supplied.encode(), expected):   # constant-time comparison
        raise HTTPException(401, "Missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"})
