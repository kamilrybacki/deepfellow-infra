# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for POST /admin/services/{id}/catalog/refresh endpoint."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from starlette.testclient import TestClient

import server.api.services as services_module
from server.api.services import router
from server.core.dependencies import auth_admin, get_services_manager

SERVICE_ID = "ollama"


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    services_module.catalog_refresh_cache.clear()


@pytest.fixture
def services_manager() -> MagicMock:
    manager = MagicMock()
    manager.refresh_catalog = AsyncMock(return_value=(10, 700))
    return manager


@pytest.fixture
def client(services_manager: MagicMock) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_services_manager] = lambda: services_manager
    app.dependency_overrides[auth_admin] = lambda: "admin"
    return TestClient(app)


def test_refresh_catalog_success_returns_counts(client: TestClient, services_manager: MagicMock) -> None:
    response = client.post(f"/admin/services/{SERVICE_ID}/catalog/refresh")
    assert response.status_code == 200
    data = response.json()
    assert data["added"] == 10
    assert data["total"] == 700


def test_refresh_catalog_cache_hit_skips_fetch(client: TestClient, services_manager: MagicMock) -> None:
    client.post(f"/admin/services/{SERVICE_ID}/catalog/refresh")
    client.post(f"/admin/services/{SERVICE_ID}/catalog/refresh")
    assert services_manager.refresh_catalog.call_count == 1


def test_refresh_catalog_cache_hit_returns_same_counts(client: TestClient, services_manager: MagicMock) -> None:
    client.post(f"/admin/services/{SERVICE_ID}/catalog/refresh")
    response = client.post(f"/admin/services/{SERVICE_ID}/catalog/refresh")
    data = response.json()
    assert data["added"] == 10
    assert data["total"] == 700


def test_refresh_catalog_force_bypasses_cache(client: TestClient, services_manager: MagicMock) -> None:
    client.post(f"/admin/services/{SERVICE_ID}/catalog/refresh")
    client.post(f"/admin/services/{SERVICE_ID}/catalog/refresh?force=true")
    assert services_manager.refresh_catalog.call_count == 2


def test_refresh_catalog_cache_expires(client: TestClient, services_manager: MagicMock) -> None:
    client.post(f"/admin/services/{SERVICE_ID}/catalog/refresh")
    added, total, _ = services_module.catalog_refresh_cache[SERVICE_ID]
    services_module.catalog_refresh_cache[SERVICE_ID] = (added, total, time.monotonic() - (6 * 3600 + 1))

    client.post(f"/admin/services/{SERVICE_ID}/catalog/refresh")
    assert services_manager.refresh_catalog.call_count == 2


def test_refresh_catalog_upstream_error_returns_502(client: TestClient, services_manager: MagicMock) -> None:
    services_manager.refresh_catalog = AsyncMock(side_effect=HTTPException(502, "Upstream unavailable"))
    response = client.post(f"/admin/services/{SERVICE_ID}/catalog/refresh")
    assert response.status_code == 502


def test_refresh_catalog_unsupported_service_returns_405(client: TestClient, services_manager: MagicMock) -> None:
    services_manager.refresh_catalog = AsyncMock(side_effect=HTTPException(405, "Not supported"))
    response = client.post("/admin/services/vllm/catalog/refresh")
    assert response.status_code == 405
