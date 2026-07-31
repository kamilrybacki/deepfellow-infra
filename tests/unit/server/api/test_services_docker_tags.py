# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for GET /admin/services/{id}/docker-tags endpoint."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from starlette.testclient import TestClient

from server.api.services import router
from server.core.dependencies import auth_admin, get_services_manager
from server.models.services import RetrieveServiceOut, ServiceField, ServiceSpecification
from server.utils.registry_client import RegistryUnavailableError

SERVICE_ID = "ollama"
TAGS = ["0.20.4", "0.20.3", "0.19.0"]


@pytest.fixture
def service_out() -> RetrieveServiceOut:
    return RetrieveServiceOut(
        id=SERVICE_ID,
        type=SERVICE_ID,
        instance="default",
        description="Ollama",
        installed=False,
        downloaded=False,
        spec=ServiceSpecification(
            fields=[
                ServiceField(
                    type="docker-tags",
                    name="image_version",
                    description="Docker image version",
                    docker_image="ollama/ollama",
                    required=False,
                )
            ]
        ),
        size="6.2 GB",
        custom_model_spec=None,
        has_docker=True,
    )


@pytest.fixture
def services_manager(service_out: RetrieveServiceOut) -> MagicMock:
    manager = MagicMock()
    manager.docker_tags_cache = {}
    manager.get_docker_tags_for_service = AsyncMock(return_value=TAGS)
    manager.get_service = AsyncMock(return_value=service_out)
    manager.get_default_docker_tag_for_service = MagicMock(return_value=TAGS[0])
    manager.get_docker_image_repo_for_service = MagicMock(return_value=None)
    return manager


@pytest.fixture
def client(services_manager: MagicMock) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_services_manager] = lambda: services_manager
    app.dependency_overrides[auth_admin] = lambda: "admin"
    return TestClient(app)


def test_get_docker_tags_returns_tags(client: TestClient, services_manager: MagicMock) -> None:
    response = client.get(f"/admin/services/{SERVICE_ID}/docker-tags")
    assert response.status_code == 200
    data = response.json()
    assert data["image"] == "ollama/ollama"
    assert data["tags"] == TAGS
    assert data["default"] == TAGS[0]


def test_get_docker_tags_forwards_hardware_to_default_tag(client: TestClient, services_manager: MagicMock) -> None:
    client.get(f"/admin/services/{SERVICE_ID}/docker-tags?hardware=gpu")
    services_manager.get_default_docker_tag_for_service.assert_called_once_with(SERVICE_ID, "gpu")


def test_get_docker_tags_forwards_hardware_param(client: TestClient, services_manager: MagicMock) -> None:
    client.get(f"/admin/services/{SERVICE_ID}/docker-tags?hardware=gpu")
    services_manager.get_docker_tags_for_service.assert_called_once_with(SERVICE_ID, "gpu")


def test_get_docker_tags_cache_hit(client: TestClient, services_manager: MagicMock) -> None:
    client.get(f"/admin/services/{SERVICE_ID}/docker-tags")
    client.get(f"/admin/services/{SERVICE_ID}/docker-tags")
    # Second call should hit cache; get_docker_tags_for_service called only once
    assert services_manager.get_docker_tags_for_service.call_count == 1


def test_get_docker_tags_cache_hit_returns_correct_image(client: TestClient) -> None:
    first = client.get(f"/admin/services/{SERVICE_ID}/docker-tags").json()
    second = client.get(f"/admin/services/{SERVICE_ID}/docker-tags").json()
    assert second["image"] == first["image"]
    assert second["tags"] == first["tags"]


def test_get_docker_tags_cache_miss_different_hardware(client: TestClient, services_manager: MagicMock) -> None:
    client.get(f"/admin/services/{SERVICE_ID}/docker-tags?hardware=gpu")
    client.get(f"/admin/services/{SERVICE_ID}/docker-tags?hardware=cpu")
    assert services_manager.get_docker_tags_for_service.call_count == 2


def test_get_docker_tags_cache_expires(client: TestClient, services_manager: MagicMock) -> None:
    client.get(f"/admin/services/{SERVICE_ID}/docker-tags")
    # Manually expire the cache entry
    key = (SERVICE_ID, None)
    tags, image, _ = services_manager.docker_tags_cache[key]
    services_manager.docker_tags_cache[key] = (tags, image, time.monotonic() - 7201)

    client.get(f"/admin/services/{SERVICE_ID}/docker-tags")
    assert services_manager.get_docker_tags_for_service.call_count == 2


def test_get_docker_tags_image_prefers_hardware_aware_repo_over_spec_field(client: TestClient, services_manager: MagicMock) -> None:
    services_manager.get_docker_image_repo_for_service = MagicMock(return_value="public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo")
    response = client.get(f"/admin/services/{SERVICE_ID}/docker-tags?hardware=cpu")
    assert response.status_code == 200
    assert response.json()["image"] == "public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo"


def test_get_docker_tags_image_fallback_to_service_id_when_no_docker_tags_field(services_manager: MagicMock) -> None:
    service_no_field = RetrieveServiceOut(
        id=SERVICE_ID,
        type=SERVICE_ID,
        instance="default",
        description="Ollama",
        installed=False,
        downloaded=False,
        spec=ServiceSpecification(fields=[]),
        size="6.2 GB",
        custom_model_spec=None,
        has_docker=True,
    )
    services_manager.get_service = AsyncMock(return_value=service_no_field)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_services_manager] = lambda: services_manager
    app.dependency_overrides[auth_admin] = lambda: "admin"
    client = TestClient(app)

    response = client.get(f"/admin/services/{SERVICE_ID}/docker-tags")
    assert response.status_code == 200
    assert response.json()["image"] == SERVICE_ID


def test_get_docker_tags_empty_result_not_cached(client: TestClient, services_manager: MagicMock) -> None:
    services_manager.get_docker_tags_for_service = AsyncMock(return_value=[])
    client.get(f"/admin/services/{SERVICE_ID}/docker-tags")
    assert (SERVICE_ID, None) not in services_manager.docker_tags_cache


def test_get_docker_tags_service_not_found_returns_404(client: TestClient, services_manager: MagicMock) -> None:
    services_manager.get_service = AsyncMock(side_effect=HTTPException(status_code=404, detail="Service not found"))
    response = client.get("/admin/services/nonexistent/docker-tags")
    assert response.status_code == 404


def test_get_docker_tags_registry_unavailable_returns_empty_tags(client: TestClient, services_manager: MagicMock) -> None:
    services_manager.get_docker_tags_for_service = AsyncMock(side_effect=RegistryUnavailableError("unreachable"))
    response = client.get(f"/admin/services/{SERVICE_ID}/docker-tags")
    assert response.status_code == 200
    assert response.json()["tags"] == []
    assert (SERVICE_ID, None) not in services_manager.docker_tags_cache
