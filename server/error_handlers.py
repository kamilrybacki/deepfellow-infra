# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Global exception handlers producing OpenAI-style error responses."""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

OPENAI_ERROR_TYPE_BY_STATUS: dict[int, str] = {
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
}


def _error_type_for_status(status_code: int) -> str:
    if status_code >= 500:
        return "api_error"
    return OPENAI_ERROR_TYPE_BY_STATUS.get(status_code, "invalid_request_error")


def build_error_body(message: str, type_: str, param: str | None = None, code: str | None = None) -> dict[str, Any]:
    """Build an OpenAI-style `{"error": {...}}` response body."""
    return {"error": {"message": message, "type": type_, "param": param, "code": code}}


def _message_and_param_from_detail(detail: Any) -> tuple[str, str | None]:  # noqa: ANN401
    if isinstance(detail, str):
        return detail, None
    if isinstance(detail, list) and detail and all(isinstance(item, dict) and "message" in item for item in detail):
        message = "; ".join(f"{item.get('field')}: {item['message']}" if item.get("field") else item["message"] for item in detail)
        param = detail[0].get("field")
        return message, param
    return str(detail), None


async def handle_http_exception(_request: Request, exc: Exception) -> JSONResponse:
    """Translate any `HTTPException` into an OpenAI-style error response."""
    assert isinstance(exc, HTTPException)
    message, param = _message_and_param_from_detail(exc.detail)
    body = build_error_body(message, _error_type_for_status(exc.status_code), param)
    return JSONResponse(status_code=exc.status_code, content=body, headers=exc.headers)


async def handle_validation_error(_request: Request, exc: Exception) -> JSONResponse:
    """Translate FastAPI's `RequestValidationError` into an OpenAI-style error response."""
    assert isinstance(exc, RequestValidationError)
    errors = exc.errors()
    error = errors[0]
    param = ".".join(str(loc) for loc in error["loc"] if loc not in ("body", "query", "path", "header")) or None
    body = build_error_body(error["msg"], "invalid_request_error", param)
    return JSONResponse(status_code=400, content=body)


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers producing OpenAI-style error responses."""
    app.add_exception_handler(HTTPException, handle_http_exception)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
