# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenTelemetry tracing utilities for request and streaming response tracing."""

import functools
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Status, StatusCode, TracerProvider  # type: ignore
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Tracer
from pydantic import BaseModel

from server.config import AppSettings

uvicorn_logger = logging.getLogger("uvicorn")


class OtlpLoggingManager:
    """Owns the OTLP log handler lifecycle for one app instance.

    Lives on `app.state.otlp_logging` (see `lifecycle.py`) rather than as module state, so each
    `FastAPI` app instance owns its own handler reference instead of sharing one across instances.
    """

    def __init__(self) -> None:
        self._handler: LoggingHandler | None = None
        self._provider: LoggerProvider | None = None

    def setup(self, config: AppSettings) -> None:
        """Attach an OTLP log handler to the root Python logger and non-propagating uvicorn loggers.

        Idempotent: calling this again after `teardown()` rebuilds the exporter with the current
        `config.otel_exporter_otlp_endpoint`, so it can be re-applied when that value changes via
        `/admin/config` without a process restart.
        """
        resource = Resource(attributes={"service.name": "llm-audit"})
        provider = self._provider = LoggerProvider(resource=resource)
        processor = BatchLogRecordProcessor(OTLPLogExporter(endpoint=config.otel_exporter_otlp_endpoint, insecure=True))
        provider.add_log_record_processor(processor)
        set_logger_provider(provider)

        handler = self._handler = LoggingHandler(level=logging.DEBUG, logger_provider=provider)
        logging.getLogger().addHandler(handler)

        # uvicorn loggers have propagate=false in logging_config.yaml so they won't reach root
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            logging.getLogger(name).addHandler(handler)

    def teardown(self) -> None:
        """Detach the OTLP log handler previously attached by `setup()`, if any."""
        if self._handler is None:
            return
        for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access"):
            logging.getLogger(name).removeHandler(self._handler)
        self._handler = None

        # Stop the old BatchLogRecordProcessor's background export thread — otherwise every
        # reconfigure() leaks one more thread that outlives its now-detached handler forever.
        if self._provider is not None:
            self._provider.shutdown()
            self._provider = None

    def reconfigure(self, config: AppSettings) -> None:
        """Apply an `otel_logging_enabled`/`otel_exporter_otlp_endpoint` change without a restart."""
        self.teardown()
        if config.otel_logging_enabled:
            self.setup(config)


class FuncArgs:
    request: Request | None = None
    query: BaseModel | None = None
    model: BaseModel | None = None


class InfraTracer:
    service_name: str
    config: AppSettings | None
    tracer: Tracer | None

    def __init__(self, service_name: str) -> None:
        self.service_name = service_name
        self.config: AppSettings | None = None
        self.tracer = None
        self._tracer_endpoint: str | None = None
        self._provider: TracerProvider | None = None

    def _get_config(self) -> AppSettings:
        if self.config is None:
            msg = "InfraTracer.config was not set. Ensure lifecycle.py assigns `tracer.config` at startup."
            raise RuntimeError(msg)
        return self.config

    def _get_tracer(self) -> Tracer:
        otlp_endpoint = self._get_config().otel_exporter_otlp_endpoint
        # Rebuild the provider if the endpoint changed via /admin/config, so the exporter target
        # updates without a process restart. Get the tracer directly from this provider instance
        # (not `trace.set_tracer_provider()` + `trace.get_tracer()`) — the OTel SDK's global
        # tracer provider is set-once; a second call is a silent no-op, which would leave every
        # rebuilt provider/exporter constructed-then-discarded and spans stuck on the old endpoint.
        if self.tracer is None or self._tracer_endpoint != otlp_endpoint:
            # Stop the old BatchSpanProcessor's background export thread before discarding its
            # provider — otherwise every endpoint change via /admin/config leaks one more thread.
            if self._provider is not None:
                self._provider.shutdown()

            resource = Resource(attributes={"service.name": self.service_name})
            provider = self._provider = TracerProvider(resource=resource)
            processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True))
            provider.add_span_processor(processor)

            self.tracer = provider.get_tracer(__name__)
            self._tracer_endpoint = otlp_endpoint
        return self.tracer

    def _set_success_attributes(self, span: Span, execution_time: float) -> None:
        """Set success attributes on the span."""
        attributes = {"execution.time_ms": round(execution_time * 1000, 2)}

        for key, value in attributes.items():
            span.set_attribute(key, value)
        span.set_status(Status(StatusCode.OK))

    def _bind_args(self, func: Callable[..., Any], args: Any, kwargs: Any) -> inspect.BoundArguments | None:  # noqa: ANN401
        try:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            return bound_args  # noqa: TRY300
        except Exception:
            return None

    def _extract_arguments(self, bound_args: inspect.BoundArguments) -> FuncArgs:
        res = FuncArgs()
        if "request" in bound_args.arguments and isinstance(request := bound_args.arguments["request"], Request):
            res.request = request
        if "model" in bound_args.arguments and isinstance(model := bound_args.arguments["model"], BaseModel):
            res.model = model
        if "query" in bound_args.arguments and isinstance(query := bound_args.arguments["query"], BaseModel):
            res.query = query
        return res

    def _add_attributes_to_span(self, args: FuncArgs, span: Span) -> None:
        """Extract span attributes from function arguments."""
        try:
            if args.request:
                span.set_attribute("request.method", str(args.request.method))
                span.set_attribute("request.url", str(args.request.url))
                span.set_attribute("request.path", args.request.url.path)
                span.set_attribute("request.content_type", args.request.headers.get("content-type") or "<unknown>")
                span.set_attribute("request.content_length", args.request.headers.get("content-length") or "<unknown>")
                span.set_attribute("request.user-agent", args.request.headers.get("user-agent") or "<unknown>")
                span.set_attribute("request.accept", args.request.headers.get("accept") or "<unknown>")
                span.set_attribute("request.host", args.request.headers.get("host") or "<unknown>")

            if args.model:
                for key, value in args.model.model_dump().items():
                    span.set_attribute(f"model.{key}", value)
            if args.query:
                for key, value in args.query.model_dump().items():
                    span.set_attribute(f"query.{key}", value)

        except Exception:
            pass

    def trace_request(self) -> Callable[..., Any]:
        """Create a tracing decorator for async function."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
                if not self._get_config().otel_tracing_enabled:
                    return await func(*args, **kwargs)
                span_name = "test"
                span = self._get_tracer().start_span(span_name)
                trace.set_span_in_context(span)
                start_time = time.time()
                # Extract and set attributes from function arguments
                bound_args = self._bind_args(func, args, kwargs)
                func_args = self._extract_arguments(bound_args) if bound_args else FuncArgs()
                self._add_attributes_to_span(func_args, span)

                def span_end() -> None:
                    execution_time = time.time() - start_time
                    self._set_success_attributes(span, execution_time)
                    span.end()

                try:
                    res = await func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span_end()
                    raise
                else:
                    if isinstance(res, StreamingResponse):
                        span.set_attribute("response.status", res.status_code)
                        span.set_attribute("response.content_type", res.media_type or "<unknown>")
                        span_end()
                    if isinstance(res, JSONResponse):
                        span.set_attribute("response.status", res.status_code)
                        span.set_attribute("response.content_type", "application/json")
                        span_end()
                    elif isinstance(res, BaseModel):
                        span.set_attribute("response.status", 200)
                        span.set_attribute("response.content_type", "application/json")
                        span_end()
                    else:
                        span.set_attribute("response.status", "<unknown>")
                        span.set_attribute("response.content_type", "<unknown>")
                        span_end()
                    return res

            return async_wrapper

        return decorator


tracer = InfraTracer(service_name="llm-audit")
