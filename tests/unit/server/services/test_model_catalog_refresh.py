# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

from pathlib import Path

import pytest
from fastapi import HTTPException

from server.services import model_catalog_refresh
from server.services.model_catalog_refresh import (
    HuggingFaceRefreshGuard,
    probe_static_dir_writable,
    set_catalog_refresh_supported,
)


def test_probe_static_dir_writable_returns_true_for_writable_dir(tmp_path: Path) -> None:
    assert probe_static_dir_writable(tmp_path) is True
    assert not (tmp_path / ".catalog_refresh_write_probe").exists()


def test_probe_static_dir_writable_returns_false_for_missing_dir(tmp_path: Path) -> None:
    assert probe_static_dir_writable(tmp_path / "does-not-exist") is False


def test_set_catalog_refresh_supported_updates_module_flag() -> None:
    try:
        set_catalog_refresh_supported(False)
        assert model_catalog_refresh.catalog_refresh_supported is False
        set_catalog_refresh_supported(True)
        assert model_catalog_refresh.catalog_refresh_supported is True
    finally:
        set_catalog_refresh_supported(True)


def test_guard_acquire_allows_first_caller() -> None:
    guard = HuggingFaceRefreshGuard()
    guard.acquire("vllm")
    guard.release()


def test_guard_acquire_rejects_concurrent_caller() -> None:
    guard = HuggingFaceRefreshGuard()
    guard.acquire("vllm")
    with pytest.raises(HTTPException) as exc_info:
        guard.acquire("llamacpp")
    assert exc_info.value.status_code == 429


def test_guard_release_allows_next_caller() -> None:
    guard = HuggingFaceRefreshGuard()
    guard.acquire("vllm")
    guard.release()
    guard.acquire("llamacpp")
    guard.release()


def test_guard_acquire_rejects_concurrent_caller_with_same_service_id() -> None:
    guard = HuggingFaceRefreshGuard()
    guard.acquire("vllm")
    with pytest.raises(HTTPException) as exc_info:
        guard.acquire("vllm")
    assert exc_info.value.status_code == 429
