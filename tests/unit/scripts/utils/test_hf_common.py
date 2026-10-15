# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for scripts/utils/hf_common.py."""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.utils.hf_common import get_model_detail_cached


def _make_mock_session(json_data: object) -> MagicMock:
    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = AsyncMock(return_value=json_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    return mock_session


@pytest.mark.asyncio
async def test_no_cache_dir_always_hits_network() -> None:
    session = _make_mock_session({"siblings": ["a"]})

    await get_model_detail_cached(session, "org/model")
    await get_model_detail_cached(session, "org/model")

    assert session.get.call_count == 2


@pytest.mark.asyncio
async def test_cache_miss_writes_cache_file(tmp_path: Path) -> None:
    session = _make_mock_session({"siblings": ["a"]})

    result = await get_model_detail_cached(session, "org/model", cache_dir=tmp_path)

    assert result == {"siblings": ["a"]}
    cached = json.loads((tmp_path / "org--model.json").read_text())
    assert cached["data"] == {"siblings": ["a"]}


@pytest.mark.asyncio
async def test_cache_hit_avoids_second_network_call(tmp_path: Path) -> None:
    session = _make_mock_session({"siblings": ["a"]})

    first = await get_model_detail_cached(session, "org/model", cache_dir=tmp_path)
    second = await get_model_detail_cached(session, "org/model", cache_dir=tmp_path)

    assert first == second
    assert session.get.call_count == 1


@pytest.mark.asyncio
async def test_cache_shared_across_independent_sessions(tmp_path: Path) -> None:
    """The whole point: a second script invocation (fresh session, no in-memory state) reuses the first's fetch."""
    first_session = _make_mock_session({"siblings": ["a"]})
    second_session = _make_mock_session({"siblings": ["a"]})

    await get_model_detail_cached(first_session, "org/model", cache_dir=tmp_path)
    result = await get_model_detail_cached(second_session, "org/model", cache_dir=tmp_path)

    assert result == {"siblings": ["a"]}
    second_session.get.assert_not_called()


@pytest.mark.asyncio
async def test_expired_cache_entry_refetches(tmp_path: Path) -> None:
    session = _make_mock_session({"siblings": ["fresh"]})
    cache_file = tmp_path / "org--model.json"
    cache_file.write_text(json.dumps({"fetched_at": 0.0, "data": {"siblings": ["stale"]}}))

    result = await get_model_detail_cached(session, "org/model", cache_dir=tmp_path, ttl_seconds=900.0)

    assert result == {"siblings": ["fresh"]}
    session.get.assert_called_once()


@pytest.mark.asyncio
async def test_corrupt_cache_file_refetches(tmp_path: Path) -> None:
    session = _make_mock_session({"siblings": ["fresh"]})
    cache_file = tmp_path / "org--model.json"
    cache_file.write_text("not valid json")

    result = await get_model_detail_cached(session, "org/model", cache_dir=tmp_path)

    assert result == {"siblings": ["fresh"]}


@pytest.mark.asyncio
async def test_model_id_with_slash_sanitized_to_flat_filename(tmp_path: Path) -> None:
    session = _make_mock_session({"siblings": []})

    await get_model_detail_cached(session, "org/sub/model", cache_dir=tmp_path)

    assert (tmp_path / "org--sub--model.json").exists()


@pytest.mark.asyncio
async def test_different_model_ids_do_not_share_cache(tmp_path: Path) -> None:
    session = _make_mock_session({"siblings": ["a"]})

    await get_model_detail_cached(session, "org/model-a", cache_dir=tmp_path)
    await get_model_detail_cached(session, "org/model-b", cache_dir=tmp_path)

    assert session.get.call_count == 2


@pytest.mark.asyncio
async def test_wrong_shape_cache_file_refetches(tmp_path: Path) -> None:
    session = _make_mock_session({"siblings": ["fresh"]})
    (tmp_path / "org--model.json").write_text(json.dumps(["not", "a", "dict"]))

    result = await get_model_detail_cached(session, "org/model", cache_dir=tmp_path)

    assert result == {"siblings": ["fresh"]}


@pytest.mark.asyncio
async def test_fresh_cache_entry_within_ttl_is_used(tmp_path: Path) -> None:
    session = _make_mock_session({"siblings": ["fresh"]})
    (tmp_path / "org--model.json").write_text(json.dumps({"fetched_at": time.time() - 10, "data": {"siblings": ["cached"]}}))

    result = await get_model_detail_cached(session, "org/model", cache_dir=tmp_path, ttl_seconds=900.0)

    assert result == {"siblings": ["cached"]}
    session.get.assert_not_called()


@pytest.mark.asyncio
async def test_cache_write_failure_still_returns_data(tmp_path: Path) -> None:
    session = _make_mock_session({"siblings": ["a"]})
    blocker = tmp_path / "blocked"
    blocker.write_text("a file, so mkdir of a child dir fails")

    result = await get_model_detail_cached(session, "org/model", cache_dir=blocker / "cache")

    assert result == {"siblings": ["a"]}
