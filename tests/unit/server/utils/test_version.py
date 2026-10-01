# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/utils/version.py."""

import os
from collections.abc import Iterator
from unittest import mock

import pytest

from server.utils.version import get_build_commit


@pytest.fixture(autouse=True)
def clear_build_commit_cache() -> Iterator[None]:
    # The reader is `lru_cache`d, so a value read under one test's environment would otherwise be
    # served to every later test in the process, here and in other files.
    get_build_commit.cache_clear()
    yield
    get_build_commit.cache_clear()


@mock.patch.dict(os.environ, {"DF_BUILD_COMMIT": "9c6e4f0a1b2c3d4e5f60718293a4b5c6d7e8f900"})
def test_get_build_commit_reads_the_baked_in_value() -> None:
    assert get_build_commit() == "9c6e4f0a1b2c3d4e5f60718293a4b5c6d7e8f900"


@mock.patch.dict(os.environ, {}, clear=True)
def test_get_build_commit_is_none_when_not_baked_in() -> None:
    assert get_build_commit() is None


@mock.patch.dict(os.environ, {"DF_BUILD_COMMIT": ""})
def test_get_build_commit_treats_an_empty_value_as_absent() -> None:
    # The Dockerfile defaults the build argument to an empty string, so an image built without it
    # carries the variable set to "" rather than unset.
    assert get_build_commit() is None
