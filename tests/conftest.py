# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import pytest

from server.services_manager import ServicesManager


@pytest.fixture
def services_manager() -> ServicesManager:
    return ServicesManager()
