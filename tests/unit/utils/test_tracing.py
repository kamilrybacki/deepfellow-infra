# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import inspect
import logging
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry.trace.status import StatusCode
from pydantic import BaseModel

from server.utils.tracing import FuncArgs, InfraTracer, OtlpLoggingManager


def _make_config(enabled: bool = True, endpoint: str = "http://localhost:4317") -> MagicMock:
    cfg = MagicMock()
    cfg.otel_tracing_enabled = enabled
    cfg.otel_exporter_otlp_endpoint = endpoint
    return cfg


def _make_span() -> MagicMock:
    span = MagicMock()
    span.set_attribute = MagicMock()
    span.set_status = MagicMock()
    span.record_exception = MagicMock()
    span.end = MagicMock()
    return span


def _make_request(
    method: str = "POST",
    path: str = "/path",
    headers: dict[str, str] | None = None,
) -> Request:
    if headers is None:
        headers = {
            "content-type": "application/json",
            "content-length": "42",
            "user-agent": "test-agent",
            "accept": "*/*",
            "host": "test.com",
        }
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "server": ("test.com", 80),
    }
    return Request(scope)


def _make_tracer_with_config(enabled: bool) -> InfraTracer:
    t = InfraTracer("svc")
    t.config = _make_config(enabled=enabled)
    return t


def _bind(func: Any, *args: Any, **kwargs: Any) -> inspect.BoundArguments:
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound


def _patch_otel_context_managers():
    return [
        patch("server.utils.tracing.TracerProvider"),
        patch("server.utils.tracing.OTLPSpanExporter"),
        patch("server.utils.tracing.BatchSpanProcessor"),
    ]


def _patch_otlp_logging_context_managers():
    return [
        patch("server.utils.tracing.LoggerProvider"),
        patch("server.utils.tracing.OTLPLogExporter"),
        patch("server.utils.tracing.BatchLogRecordProcessor"),
        patch("server.utils.tracing.set_logger_provider"),
        patch("server.utils.tracing.LoggingHandler"),
    ]


def test_setup_otlp_logging_creates_provider_with_service_name() -> None:
    cfg = _make_config(endpoint="http://otel:4317")
    patches = _patch_otlp_logging_context_managers()

    with patches[0] as mock_provider_cls, patches[1], patches[2], patches[3], patches[4], patch("server.utils.tracing.logging.getLogger"):
        OtlpLoggingManager().setup(cfg)

    resource_arg = mock_provider_cls.call_args[1]["resource"]
    assert resource_arg.attributes["service.name"] == "llm-audit"


def test_setup_otlp_logging_uses_endpoint_from_config() -> None:
    cfg = _make_config(endpoint="http://otel:4317")
    patches = _patch_otlp_logging_context_managers()

    with patches[0], patches[1] as mock_exporter_cls, patches[2], patches[3], patches[4], patch("server.utils.tracing.logging.getLogger"):
        OtlpLoggingManager().setup(cfg)

    assert mock_exporter_cls.call_args == call(endpoint="http://otel:4317", insecure=True)


def test_setup_otlp_logging_registers_processor_and_sets_global_provider() -> None:
    cfg = _make_config(endpoint="http://otel:4317")
    patches = _patch_otlp_logging_context_managers()

    with (
        patches[0] as mock_provider_cls,
        patches[1],
        patches[2] as mock_processor_cls,
        patches[3] as mock_set,
        patches[4],
        patch("server.utils.tracing.logging.getLogger"),
    ):
        OtlpLoggingManager().setup(cfg)

    provider = mock_provider_cls.return_value
    assert provider.add_log_record_processor.call_args == call(mock_processor_cls.return_value)
    assert mock_set.call_args == call(provider)


def test_setup_otlp_logging_attaches_handler_to_root_and_uvicorn_loggers() -> None:
    cfg = _make_config(endpoint="http://otel:4317")
    patches = _patch_otlp_logging_context_managers()

    with (
        patches[0] as mock_provider_cls,
        patches[1],
        patches[2],
        patches[3],
        patches[4] as mock_handler_cls,
        patch("server.utils.tracing.logging.getLogger") as mock_get_logger,
    ):
        OtlpLoggingManager().setup(cfg)

    handler = mock_handler_cls.return_value
    assert mock_handler_cls.call_args == call(level=logging.DEBUG, logger_provider=mock_provider_cls.return_value)

    logger_names = [c[0][0] if c[0] else "" for c in mock_get_logger.call_args_list]
    assert "" in logger_names
    assert "uvicorn" in logger_names
    assert "uvicorn.error" in logger_names
    assert "uvicorn.access" in logger_names

    for logger_call in mock_get_logger.return_value.addHandler.call_args_list:
        assert logger_call == call(handler)


def test_teardown_otlp_logging_noop_when_no_handler() -> None:
    manager = OtlpLoggingManager()

    with patch("server.utils.tracing.logging.getLogger") as mock_get_logger:
        manager.teardown()

    assert mock_get_logger.call_count == 0


def test_teardown_otlp_logging_removes_handler_from_loggers() -> None:
    manager = OtlpLoggingManager()
    handler = MagicMock()
    manager._handler = handler  # pyright: ignore[reportPrivateUsage]

    with patch("server.utils.tracing.logging.getLogger") as mock_get_logger:
        manager.teardown()

        logger_names = [c[0][0] if c[0] else "" for c in mock_get_logger.call_args_list]
        assert logger_names == ["", "uvicorn", "uvicorn.error", "uvicorn.access"]
        for logger_call in mock_get_logger.return_value.removeHandler.call_args_list:
            assert logger_call == call(handler)

        assert manager._handler is None  # pyright: ignore[reportPrivateUsage]


def test_teardown_otlp_logging_shuts_down_old_provider() -> None:
    manager = OtlpLoggingManager()
    manager._handler = MagicMock()  # pyright: ignore[reportPrivateUsage]
    provider = MagicMock()
    manager._provider = provider  # pyright: ignore[reportPrivateUsage]

    with patch("server.utils.tracing.logging.getLogger"):
        manager.teardown()

    assert provider.shutdown.call_count == 1
    assert manager._provider is None  # pyright: ignore[reportPrivateUsage]


def test_reconfigure_otlp_logging_shuts_down_previous_provider_before_rebuilding() -> None:
    cfg = _make_config()
    cfg.otel_logging_enabled = True
    manager = OtlpLoggingManager()
    patches = _patch_otlp_logging_context_managers()

    with patches[0] as mock_provider_cls, patches[1], patches[2], patches[3], patches[4], patch("server.utils.tracing.logging.getLogger"):
        first_provider = mock_provider_cls.return_value
        manager.setup(cfg)
        manager.reconfigure(cfg)

    assert first_provider.shutdown.call_count == 1


def test_reconfigure_otlp_logging_sets_up_when_enabled() -> None:
    cfg = _make_config()
    cfg.otel_logging_enabled = True
    manager = OtlpLoggingManager()

    with patch.object(manager, "teardown") as mock_teardown, patch.object(manager, "setup") as mock_setup:
        manager.reconfigure(cfg)

    assert mock_teardown.call_count == 1
    assert mock_setup.call_args == call(cfg)


def test_reconfigure_otlp_logging_skips_setup_when_disabled() -> None:
    cfg = _make_config()
    cfg.otel_logging_enabled = False
    manager = OtlpLoggingManager()

    with patch.object(manager, "teardown") as mock_teardown, patch.object(manager, "setup") as mock_setup:
        manager.reconfigure(cfg)

    assert mock_teardown.call_count == 1
    assert mock_setup.call_count == 0


def test_func_args_defaults_are_none() -> None:
    args = FuncArgs()

    assert args.request is None
    assert args.query is None
    assert args.model is None


def test_infra_tracer_service_name_stored() -> None:
    t = InfraTracer("my-service")

    assert t.service_name == "my-service"


def test_infra_tracer_config_and_tracer_are_none() -> None:
    t = InfraTracer("svc")

    assert t.config is None
    assert t.tracer is None


def test_get_config_returns_config_when_set() -> None:
    t = InfraTracer("svc")
    cfg = _make_config()
    t.config = cfg

    assert t._get_config() is cfg  # pyright: ignore[reportPrivateUsage]


def test_get_config_raises_when_not_set() -> None:
    t = InfraTracer("svc")

    with pytest.raises(RuntimeError, match=r"InfraTracer\.config was not set"):
        t._get_config()  # pyright: ignore[reportPrivateUsage]


def test_get_tracer_returns_tracer() -> None:
    t = InfraTracer("svc")
    t.config = _make_config(endpoint="http://otel:4317")
    patches = _patch_otel_context_managers()

    with patches[0] as mock_provider_cls, patches[1], patches[2]:
        fake_tracer = mock_provider_cls.return_value.get_tracer.return_value
        result = t._get_tracer()  # pyright: ignore[reportPrivateUsage]

    assert result is fake_tracer


def test_get_tracer_caches_tracer() -> None:
    t = InfraTracer("svc")
    t.config = _make_config()
    patches = _patch_otel_context_managers()

    with patches[0] as mock_provider_cls, patches[1], patches[2]:
        fake_tracer = mock_provider_cls.return_value.get_tracer.return_value
        t._get_tracer()  # pyright: ignore[reportPrivateUsage]
        t._get_tracer()  # pyright: ignore[reportPrivateUsage]

    assert t.tracer is fake_tracer
    assert mock_provider_cls.return_value.get_tracer.call_count == 1


def test_get_tracer_rebuilds_from_new_providers_tracer_when_endpoint_changes() -> None:
    """The tracer must come from the freshly built provider, not a stale global one.

    `trace.set_tracer_provider()` is set-once in the OTel SDK — a second call is a silent no-op —
    so getting the tracer via `provider.get_tracer(...)` directly (bypassing the global registry)
    is the only way a changed endpoint actually takes effect without a restart.
    """
    t = InfraTracer("svc")
    t.config = _make_config(endpoint="http://old:4317")
    patches = _patch_otel_context_managers()

    with patches[0] as mock_provider_cls, patches[1], patches[2]:
        first_provider = MagicMock()
        second_provider = MagicMock()
        mock_provider_cls.side_effect = [first_provider, second_provider]

        first_tracer = t._get_tracer()  # pyright: ignore[reportPrivateUsage]
        t.config.otel_exporter_otlp_endpoint = "http://new:4317"
        second_tracer = t._get_tracer()  # pyright: ignore[reportPrivateUsage]

    assert first_tracer is first_provider.get_tracer.return_value
    assert second_tracer is second_provider.get_tracer.return_value
    assert first_tracer is not second_tracer
    assert first_provider.shutdown.call_count == 1
    assert second_provider.shutdown.call_count == 0


def test_set_success_attributes_sets_execution_time_and_ok_status() -> None:
    t = InfraTracer("svc")
    span = _make_span()

    t._set_success_attributes(span, 0.5)  # pyright: ignore[reportPrivateUsage]

    assert span.set_attribute.call_args == call("execution.time_ms", 500.0)
    assert span.set_status.call_count == 1
    status_arg = span.set_status.call_args[0][0]
    assert status_arg.status_code == StatusCode.OK


def test_set_success_attributes_rounds_execution_time() -> None:
    t = InfraTracer("svc")
    span = _make_span()

    t._set_success_attributes(span, 1.0 / 3)  # pyright: ignore[reportPrivateUsage]

    call = span.set_attribute.call_args_list[0]
    assert call[0][0] == "execution.time_ms"
    assert call[0][1] == round((1.0 / 3) * 1000, 2)


def test_bind_args_binds_correctly() -> None:
    t = InfraTracer("svc")

    def func(a: int, b: str = "x") -> None:
        pass

    bound = t._bind_args(func, (1,), {})  # pyright: ignore[reportPrivateUsage]

    assert bound is not None
    assert bound.arguments["a"] == 1
    assert bound.arguments["b"] == "x"


def test_bind_args_returns_none_on_binding_failure() -> None:
    t = InfraTracer("svc")

    def func(a: int) -> None:
        pass

    result = t._bind_args(func, (), {})  # pyright: ignore[reportPrivateUsage]

    assert result is None


def test_extract_arguments_extracts_request() -> None:
    t = InfraTracer("svc")
    req = _make_request()

    def func(request: Request) -> None:
        pass

    bound = _bind(func, req)

    result = t._extract_arguments(bound)  # pyright: ignore[reportPrivateUsage]

    assert result.request is req


def test_extract_arguments_extracts_model() -> None:
    t = InfraTracer("svc")

    class M(BaseModel):
        x: int

    def func(model: M) -> None:
        pass

    m = M(x=1)
    bound = _bind(func, m)

    result = t._extract_arguments(bound)  # pyright: ignore[reportPrivateUsage]

    assert result.model is m


def test_extract_arguments_extracts_query() -> None:
    t = InfraTracer("svc")

    class Q(BaseModel):
        q: str

    def func(query: Q) -> None:
        pass

    q = Q(q="hello")
    bound = _bind(func, q)

    result = t._extract_arguments(bound)  # pyright: ignore[reportPrivateUsage]

    assert result.query is q


def test_extract_arguments_ignores_non_request_type() -> None:
    t = InfraTracer("svc")

    def func(request: str) -> None:
        pass

    bound = _bind(func, "not-a-request")

    result = t._extract_arguments(bound)  # pyright: ignore[reportPrivateUsage]

    assert result.request is None


def test_extract_arguments_empty_when_no_matching_params() -> None:
    t = InfraTracer("svc")

    def func(x: int) -> None:
        pass

    bound = _bind(func, 42)

    result = t._extract_arguments(bound)  # pyright: ignore[reportPrivateUsage]

    assert result.request is None
    assert result.model is None
    assert result.query is None


def test_add_attributes_to_span_adds_request_attributes() -> None:
    t = InfraTracer("svc")
    span = _make_span()
    req = _make_request()
    args = FuncArgs()
    args.request = req

    t._add_attributes_to_span(args, span)  # pyright: ignore[reportPrivateUsage]

    keys = {c[0][0] for c in span.set_attribute.call_args_list}
    assert "request.method" in keys
    assert "request.url" in keys
    assert "request.path" in keys
    assert "request.content_type" in keys


def test_add_attributes_to_span_adds_model_attributes() -> None:
    t = InfraTracer("svc")
    span = _make_span()

    class M(BaseModel):
        value: int

    args = FuncArgs()
    args.model = M(value=7)

    t._add_attributes_to_span(args, span)  # pyright: ignore[reportPrivateUsage]

    keys = {c[0][0] for c in span.set_attribute.call_args_list}
    assert "model.value" in keys


def test_add_attributes_to_span_adds_query_attributes() -> None:
    t = InfraTracer("svc")
    span = _make_span()

    class Q(BaseModel):
        term: str

    args = FuncArgs()
    args.query = Q(term="hello")

    t._add_attributes_to_span(args, span)  # pyright: ignore[reportPrivateUsage]

    keys = {c[0][0] for c in span.set_attribute.call_args_list}
    assert "query.term" in keys


def test_add_attributes_to_span_silently_ignores_exceptions() -> None:
    t = InfraTracer("svc")
    span = _make_span()
    span.set_attribute.side_effect = Exception("unexpected")
    args = FuncArgs()
    req = _make_request()
    args.request = req

    t._add_attributes_to_span(args, span)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_trace_request_skips_tracing_when_disabled() -> None:
    t = _make_tracer_with_config(enabled=False)

    @t.trace_request()
    async def handler():
        return "result"

    result = await handler()

    assert result == "result"


@pytest.mark.asyncio
async def test_trace_request_returns_result_when_tracing_enabled() -> None:
    t = _make_tracer_with_config(enabled=True)
    span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = span
    t.tracer = fake_tracer
    t._tracer_endpoint = t.config.otel_exporter_otlp_endpoint  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]

    with patch("server.utils.tracing.trace.set_span_in_context"):

        @t.trace_request()
        async def handler():
            return "ok"

        result = await handler()

    assert result == "ok"
    assert span.end.call_count == 1


@pytest.mark.asyncio
async def test_trace_request_re_raises_exception_and_ends_span() -> None:
    t = _make_tracer_with_config(enabled=True)
    span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = span
    t.tracer = fake_tracer
    t._tracer_endpoint = t.config.otel_exporter_otlp_endpoint  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]

    with patch("server.utils.tracing.trace.set_span_in_context"):

        @t.trace_request()
        async def handler():
            raise ValueError("fail")

        with pytest.raises(ValueError, match="fail"):
            await handler()

    assert span.record_exception.call_count == 1
    assert span.end.call_count == 1


@pytest.mark.asyncio
async def test_trace_request_streaming_response_sets_attributes() -> None:
    t = _make_tracer_with_config(enabled=True)
    span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = span
    t.tracer = fake_tracer
    t._tracer_endpoint = t.config.otel_exporter_otlp_endpoint  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]

    async def body_gen():
        yield b"data"

    with patch("server.utils.tracing.trace.set_span_in_context"):

        @t.trace_request()
        async def handler():
            return StreamingResponse(body_gen(), media_type="text/plain")

        await handler()

    keys = {c[0][0] for c in span.set_attribute.call_args_list}
    assert "response.status" in keys
    assert "response.content_type" in keys


@pytest.mark.asyncio
async def test_trace_request_json_response_sets_attributes() -> None:
    t = _make_tracer_with_config(enabled=True)
    span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = span
    t.tracer = fake_tracer
    t._tracer_endpoint = t.config.otel_exporter_otlp_endpoint  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]

    with patch("server.utils.tracing.trace.set_span_in_context"):

        @t.trace_request()
        async def handler():
            return JSONResponse(content={"ok": True})

        await handler()

    attr_map = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
    assert attr_map.get("response.content_type") == "application/json"


@pytest.mark.asyncio
async def test_trace_request_base_model_response_sets_attributes() -> None:
    t = _make_tracer_with_config(enabled=True)
    span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = span
    t.tracer = fake_tracer
    t._tracer_endpoint = t.config.otel_exporter_otlp_endpoint  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]

    class Resp(BaseModel):
        value: int

    with patch("server.utils.tracing.trace.set_span_in_context"):

        @t.trace_request()
        async def handler():
            return Resp(value=1)

        await handler()

    attr_map = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
    assert attr_map.get("response.status") == 200
    assert attr_map.get("response.content_type") == "application/json"


@pytest.mark.asyncio
async def test_trace_request_unknown_response_type_sets_unknown_attributes() -> None:
    t = _make_tracer_with_config(enabled=True)
    span = _make_span()
    fake_tracer = MagicMock()
    fake_tracer.start_span.return_value = span
    t.tracer = fake_tracer
    t._tracer_endpoint = t.config.otel_exporter_otlp_endpoint  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]
    with patch("server.utils.tracing.trace.set_span_in_context"):

        @t.trace_request()
        async def handler():
            return 42

        await handler()

    attr_map = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
    assert attr_map.get("response.status") == "<unknown>"
    assert attr_map.get("response.content_type") == "<unknown>"


@pytest.mark.asyncio
async def test_trace_request_preserves_function_signature() -> None:
    t = _make_tracer_with_config(enabled=False)

    async def my_handler(x: int, y: str = "a") -> str:
        return f"{x}{y}"

    wrapped = t.trace_request()(my_handler)

    assert wrapped.__name__ == "my_handler"
