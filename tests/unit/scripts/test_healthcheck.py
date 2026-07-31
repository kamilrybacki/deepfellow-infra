# SPDX-License-Identifier: MIT

"""Tests for scripts/healthcheck.py."""

from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from scripts.healthcheck import check_health


def _make_response(status: int) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


@pytest.mark.parametrize(("status_code", "result"), [(200, 0), (503, 1)])
def test_check_health_200(status_code: int, result: int) -> None:
    with patch("scripts.healthcheck.urlopen", return_value=_make_response(status_code)):
        assert check_health() == result


@pytest.mark.parametrize(
    ("side_effect"),
    [
        HTTPError(
            url="http://localhost:8086/health",
            code=500,
            msg="Internal Server Error",
            hdrs=None,  # pyright: ignore[reportArgumentType]
            fp=None,
        ),
        URLError("Connection refused"),
        RuntimeError("boom"),
    ],
)
def test_check_health_on_http_error(side_effect: Exception) -> None:
    with patch("scripts.healthcheck.urlopen", side_effect=side_effect):
        assert check_health() == 1
