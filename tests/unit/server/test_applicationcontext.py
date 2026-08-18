# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/applicationcontext.py."""

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from server.applicationcontext import ApplicationContext, get_base_url


def _make_context(
    services: dict[str, object] | None = None,
    known_service_ids: set[str] | None = None,
) -> tuple[ApplicationContext, MagicMock, MagicMock]:
    endpoint_registry = MagicMock()
    config = MagicMock()
    service_provider = MagicMock()
    service_provider.load = AsyncMock(return_value={"services": services or {}})
    service_provider.add_warning = AsyncMock()
    service_provider.dismiss_warnings_matching = AsyncMock()
    services_manager = MagicMock()
    services_manager.load_service = AsyncMock(return_value=None)
    services_manager.services = dict.fromkeys(known_service_ids if known_service_ids is not None else (services or {}))

    ctx = ApplicationContext(endpoint_registry, config, service_provider, services_manager)

    return ctx, service_provider, services_manager


def test_stores_endpoint_registry() -> None:
    registry = MagicMock()

    ctx = ApplicationContext(registry, MagicMock(), MagicMock(), MagicMock())

    assert ctx.endpoint_registry is registry


def test_stores_config() -> None:
    config = MagicMock()

    ctx = ApplicationContext(MagicMock(), config, MagicMock(), MagicMock())

    assert ctx.config is config


def test_stores_service_provider() -> None:
    sp = MagicMock()

    ctx = ApplicationContext(MagicMock(), MagicMock(), sp, MagicMock())

    assert ctx.service_provider is sp


def test_stores_services_manager() -> None:
    sm = MagicMock()

    ctx = ApplicationContext(MagicMock(), MagicMock(), MagicMock(), sm)

    assert ctx.services_manager is sm


def test_allocated_ports_is_empty_set() -> None:
    ctx, _, _ = _make_context()

    assert ctx.allocated_ports == set()


@pytest.mark.asyncio
async def test_calls_services_manager_load_service() -> None:
    ctx, _, sm = _make_context(known_service_ids={"my-service"})
    cfg = MagicMock()

    await ctx._load_service("my-service", cfg)  # pyright: ignore[reportPrivateUsage]

    assert sm.load_service.await_count == 1
    assert sm.load_service.await_args == call("my-service", cfg)


@pytest.mark.asyncio
async def test_dismisses_warnings_on_successful_load() -> None:
    ctx, sp, _ = _make_context(known_service_ids={"my-service"})
    cfg = MagicMock()

    await ctx._load_service("my-service", cfg)  # pyright: ignore[reportPrivateUsage]

    sp.dismiss_warnings_matching.assert_awaited_once_with("my-service")


@pytest.mark.asyncio
async def test_does_not_raise_on_exception() -> None:
    ctx, _, sm = _make_context(known_service_ids={"svc"})
    sm.load_service = AsyncMock(side_effect=RuntimeError("boom"))

    await ctx._load_service("svc", MagicMock())  # pyright: ignore[reportPrivateUsage] # should not raise


@pytest.mark.asyncio
async def test_records_warning_on_load_exception() -> None:
    ctx, sp, sm = _make_context(known_service_ids={"svc"})
    sm.load_service = AsyncMock(side_effect=RuntimeError("boom"))

    await ctx._load_service("svc", MagicMock())  # pyright: ignore[reportPrivateUsage]

    sp.add_warning.assert_awaited_once()
    assert sp.add_warning.await_args.args[0] == "svc"


@pytest.mark.asyncio
async def test_records_warning_for_unregistered_service() -> None:
    ctx, sp, sm = _make_context(known_service_ids=set())

    await ctx._load_service("mystery-service", MagicMock())  # pyright: ignore[reportPrivateUsage]

    assert sm.load_service.await_count == 0
    sp.add_warning.assert_awaited_once()
    assert sp.add_warning.await_args.args[0] == "mystery-service"


@pytest.mark.asyncio
async def test_does_not_raise_when_recording_warning_fails() -> None:
    ctx, sp, sm = _make_context(known_service_ids=set())
    sp.add_warning = AsyncMock(side_effect=RuntimeError("disk full"))

    await ctx._load_service("mystery-service", MagicMock())  # pyright: ignore[reportPrivateUsage] # should not raise

    sm.load_service.assert_not_awaited()


@pytest.mark.asyncio
async def test_does_not_raise_when_dismissing_warnings_fails() -> None:
    ctx, sp, _ = _make_context(known_service_ids={"my-service"})
    sp.dismiss_warnings_matching = AsyncMock(side_effect=RuntimeError("disk full"))

    await ctx._load_service("my-service", MagicMock())  # pyright: ignore[reportPrivateUsage] # should not raise


@pytest.mark.asyncio
async def test_calls_load_for_each_service() -> None:
    cfg_a = MagicMock()
    cfg_b = MagicMock()
    ctx, _, sm = _make_context(services={"svc-a": cfg_a, "svc-b": cfg_b})

    await ctx.load_services()

    assert sm.load_service.await_count == 2


@pytest.mark.asyncio
async def test_calls_service_provider_load() -> None:
    ctx, sp, _ = _make_context()

    await ctx.load_services()

    assert sp.load.call_count == 1


@pytest.mark.asyncio
async def test_handles_empty_services() -> None:
    ctx, _, sm = _make_context(services={})

    await ctx.load_services()

    assert sm.load_service.await_count == 0


@pytest.mark.parametrize(
    ("host", "port", "result"), [("localhost", 8080, "http://localhost:8080"), ("192.168.1.1", 9000, "http://192.168.1.1:9000")]
)
def test_builds_http_url(host: str, port: int, result: str) -> None:
    assert get_base_url(host, port) == result
