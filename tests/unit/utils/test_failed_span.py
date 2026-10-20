# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

from typing import Any

import pytest
from fastapi import Request
from opentelemetry.trace import SpanContext

from server.utils.failed_span import get_failed_span_context, remember_failed_span


@pytest.fixture
def scope() -> dict[str, Any]:
    return {"type": "http", "method": "POST", "path": "/path", "query_string": b"", "headers": []}


@pytest.fixture
def span_context() -> SpanContext:
    return SpanContext(trace_id=0x0AF7651916CD43DD8448EB211C80319C, span_id=0x00F067AA0BA902B7, is_remote=False)


def test_get_failed_span_context_returns_none_when_nothing_recorded(scope: dict[str, Any]):
    request = Request(scope)

    result = get_failed_span_context(request)

    assert result is None


def test_get_failed_span_context_returns_recorded_context(scope: dict[str, Any], span_context: SpanContext):
    request = Request(scope)

    remember_failed_span(request, span_context)

    assert get_failed_span_context(request) is span_context


def test_remember_failed_span_does_not_raise_without_request(span_context: SpanContext):
    remember_failed_span(None, span_context)


def test_get_failed_span_context_is_visible_to_another_request_of_the_same_scope(scope: dict[str, Any], span_context: SpanContext):
    remember_failed_span(Request(scope), span_context)

    result = get_failed_span_context(Request(scope))

    assert result is span_context
