# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import pytest

from server.services.remote_service import DefaultRemoteServiceOptions


@pytest.fixture
def default_remote_service_options() -> DefaultRemoteServiceOptions:
    return DefaultRemoteServiceOptions(api_url="https://api.openai.com", api_key="test-api-key")


def test_default_remote_service_options_headers_provides_dict(default_remote_service_options: DefaultRemoteServiceOptions):
    assert default_remote_service_options.headers == {"Authorization": "Bearer test-api-key"}
