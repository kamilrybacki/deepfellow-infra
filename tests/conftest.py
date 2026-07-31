# SPDX-License-Identifier: MIT

import pytest

from server.services_manager import ServicesManager


@pytest.fixture
def services_manager() -> ServicesManager:
    return ServicesManager()
