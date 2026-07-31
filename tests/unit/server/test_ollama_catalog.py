# SPDX-License-Identifier: MIT

"""Unit tests for server/utils/ollama_catalog.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.utils.ollama_catalog import OllamaCatalogClient, bytes_to_human


def _make_response(status: int, json_data: object) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _make_session(response: MagicMock) -> MagicMock:
    session = MagicMock()
    session.get = MagicMock(return_value=response)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


def test_bytes_to_human_gigabytes() -> None:
    assert bytes_to_human(8_600_000_000) == "8.6 GB"


def test_bytes_to_human_small_value() -> None:
    assert bytes_to_human(1_000_000_000) == "1.0 GB"


def test_bytes_to_human_large_value() -> None:
    assert bytes_to_human(70_000_000_000) == "70.0 GB"


@pytest.mark.asyncio
async def test_fetch_trending_returns_models() -> None:
    resp = _make_response(
        200,
        {
            "models": [
                {"name": "gemma3:12b", "size": 8_600_000_000, "digest": "abc123"},
                {"name": "llama3.2:3b", "size": 2_000_000_000, "digest": "def456"},
            ]
        },
    )
    with patch("aiohttp.ClientSession", return_value=_make_session(resp)):
        client = OllamaCatalogClient()
        models = await client.fetch_trending()

    assert len(models) == 2
    assert models[0]["id"] == "gemma3:12b"
    assert models[0]["size"] == "8.6 GB"
    assert models[0]["hash"] == "abc123"


@pytest.mark.asyncio
async def test_fetch_trending_skips_zero_size() -> None:
    resp = _make_response(
        200,
        {
            "models": [
                {"name": "cloud-model:latest", "size": 0, "digest": "cloud"},
                {"name": "real-model:7b", "size": 4_000_000_000, "digest": "local"},
            ]
        },
    )
    with patch("aiohttp.ClientSession", return_value=_make_session(resp)):
        client = OllamaCatalogClient()
        models = await client.fetch_trending()

    assert len(models) == 1
    assert models[0]["id"] == "real-model:7b"


@pytest.mark.asyncio
async def test_fetch_trending_skips_empty_name() -> None:
    resp = _make_response(
        200,
        {
            "models": [
                {"name": "", "size": 4_000_000_000, "digest": "x"},
                {"name": "valid:7b", "size": 4_000_000_000, "digest": "y"},
            ]
        },
    )
    with patch("aiohttp.ClientSession", return_value=_make_session(resp)):
        client = OllamaCatalogClient()
        models = await client.fetch_trending()

    assert len(models) == 1
    assert models[0]["id"] == "valid:7b"


@pytest.mark.asyncio
async def test_fetch_trending_non_200_returns_empty() -> None:
    resp = _make_response(503, {})
    with patch("aiohttp.ClientSession", return_value=_make_session(resp)):
        client = OllamaCatalogClient()
        models = await client.fetch_trending()

    assert models == []


@pytest.mark.asyncio
async def test_fetch_trending_exception_returns_empty() -> None:
    session = MagicMock()
    session.__aenter__ = AsyncMock(side_effect=OSError("connection refused"))
    session.__aexit__ = AsyncMock(return_value=False)
    with patch("aiohttp.ClientSession", return_value=session):
        client = OllamaCatalogClient()
        models = await client.fetch_trending()

    assert models == []


@pytest.mark.asyncio
async def test_fetch_trending_empty_models_list() -> None:
    resp = _make_response(200, {"models": []})
    with patch("aiohttp.ClientSession", return_value=_make_session(resp)):
        client = OllamaCatalogClient()
        models = await client.fetch_trending()

    assert models == []
