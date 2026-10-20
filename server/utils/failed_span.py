# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Carry the OTel span of a failed request from the tracer to the global exception handler."""

from fastapi import Request
from opentelemetry.trace import SpanContext


def remember_failed_span(request: Request | None, span_context: SpanContext) -> None:
    """Record the span a request failed in, so the exception handler can correlate it; no-op without a request.

    `InfraTracer.trace_request` never attaches its span to the OTel context, so by the time the
    exception handler runs, `get_current_span()` doesn't see it. `request.state` is backed by the
    ASGI scope shared by every layer, so the `Request` the handler gets still reads the value
    stored here.
    """
    if request is not None:
        request.state.failed_span_context = span_context


def get_failed_span_context(request: Request) -> SpanContext | None:
    """Return the span context recorded by `remember_failed_span`, or None if no traced endpoint failed."""
    return getattr(request.state, "failed_span_context", None)
