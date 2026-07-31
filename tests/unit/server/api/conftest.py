# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _disable_tracing():  # pyright: ignore[reportUnusedFunction]
    config = MagicMock()
    config.otel_tracing_enabled = False
    with patch("server.utils.tracing.InfraTracer._get_config", return_value=config):
        yield


@pytest.fixture
def auth_header() -> dict[str, str]:
    return {"Authorization": "Bearer test-key"}
