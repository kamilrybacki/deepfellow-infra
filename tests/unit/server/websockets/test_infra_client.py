# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/websockets/infra_client.py."""

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from server.websockets.infra_client import InfraClient
from server.websockets.models import (
    InitRequest,
    InitResponse,
    TopologyUpdateRequest,
    UpdateModelsRequest,
    UsageChangeRequest,
    WarmChangeRequest,
)


def _make_rpc_client(return_value: object = "OK") -> MagicMock:
    client = MagicMock()
    client.request = AsyncMock(return_value=return_value)
    return client


@pytest.mark.asyncio
async def test_init_calls_init_method() -> None:
    rpc = _make_rpc_client()
    infra = InfraClient(rpc)
    params = MagicMock(spec=InitRequest)

    await infra.init(params)

    assert rpc.request.await_count == 1
    assert rpc.request.await_args == call("init", params)


@pytest.mark.asyncio
async def test_init_returns_ok() -> None:
    rpc = _make_rpc_client("OK")
    infra = InfraClient(rpc)

    result = await infra.init(MagicMock(spec=InitRequest))

    assert result == InitResponse(ancestors=[])


@pytest.mark.asyncio
async def test_usage_change_calls_usage_change_method() -> None:
    rpc = _make_rpc_client()
    infra = InfraClient(rpc)
    params = MagicMock(spec=UsageChangeRequest)

    await infra.usage_change(params)

    assert rpc.request.await_count == 1
    assert rpc.request.await_args == call("usage_change", params)


@pytest.mark.asyncio
async def test_usage_change_returns_ok() -> None:
    rpc = _make_rpc_client("OK")
    infra = InfraClient(rpc)

    result = await infra.usage_change(MagicMock(spec=UsageChangeRequest))

    assert result == "OK"


@pytest.mark.asyncio
async def test_warm_change_calls_warm_change_method() -> None:
    rpc = _make_rpc_client()
    infra = InfraClient(rpc)
    params = MagicMock(spec=WarmChangeRequest)

    await infra.warm_change(params)

    assert rpc.request.await_count == 1
    assert rpc.request.await_args == call("warm_change", params)


@pytest.mark.asyncio
async def test_warm_change_returns_ok() -> None:
    rpc = _make_rpc_client("OK")
    infra = InfraClient(rpc)

    result = await infra.warm_change(MagicMock(spec=WarmChangeRequest))

    assert result == "OK"


@pytest.mark.asyncio
async def test_update_models_calls_update_models_method() -> None:
    rpc = _make_rpc_client()
    infra = InfraClient(rpc)
    params = MagicMock(spec=UpdateModelsRequest)

    await infra.update_models(params)

    assert rpc.request.await_count == 1
    assert rpc.request.await_args == call("update_models", params)


@pytest.mark.asyncio
async def test_update_models_returns_ok() -> None:
    rpc = _make_rpc_client("OK")
    infra = InfraClient(rpc)

    result = await infra.update_models(MagicMock(spec=UpdateModelsRequest))

    assert result == "OK"


@pytest.mark.asyncio
async def test_topology_update_calls_topology_update_method() -> None:
    rpc = _make_rpc_client()
    infra = InfraClient(rpc)
    params = MagicMock(spec=TopologyUpdateRequest)

    await infra.topology_update(params)

    assert rpc.request.await_count == 1
    assert rpc.request.await_args == call("topology_update", params)


@pytest.mark.asyncio
async def test_topology_update_returns_ok() -> None:
    rpc = _make_rpc_client("OK")
    infra = InfraClient(rpc)

    result = await infra.topology_update(MagicMock(spec=TopologyUpdateRequest))

    assert result == "OK"
