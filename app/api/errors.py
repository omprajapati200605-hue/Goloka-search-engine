"""One error format for the whole API:  {"error": {"code": ..., "message": ..., "details": [...]}}"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("goloka.api")

_CODES = {400: "bad_request", 401: "unauthorized", 404: "not_found", 405: "method_not_allowed", 409: "conflict", 422: "validation_error"}


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code, self.code, self.message = status_code, code, message


def _body(code: str, message: str, details: list | None = None) -> dict:
    err: dict = {"code": code, "message": message}
    if details:
        err["details"] = details
    return {"error": err}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return JSONResponse(_body(exc.code, exc.message), status_code=exc.status_code)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        code = _CODES.get(exc.status_code, "http_error")
        return JSONResponse(_body(code, str(exc.detail)), status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        details = [{"location": ".".join(str(p) for p in e["loc"]), "message": e["msg"]} for e in exc.errors()]
        return JSONResponse(_body("validation_error", "Invalid request parameters", details), status_code=422)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(_body("internal_error", "Internal server error"), status_code=500)
