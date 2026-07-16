# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for server/api/utils.py endpoints."""

from unittest import mock

from fastapi import FastAPI
from starlette.testclient import TestClient

from server.api.utils import router
from server.core.dependencies import get_config


def _make_config(api_key: str) -> mock.MagicMock:
    cfg = mock.MagicMock()
    cfg.infra_api_key.get_secret_value.return_value = api_key
    return cfg


def _make_app(api_key: str | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    if api_key is not None:
        app.dependency_overrides[get_config] = lambda: _make_config(api_key)
    return app


def test_health() -> None:
    with TestClient(_make_app()) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == "OK"


@mock.patch("server.api.utils.get_infra_version", return_value="1.2.3")
def test_info_returns_version_with_valid_server_key(mock_get_infra_version: mock.MagicMock) -> None:
    with TestClient(_make_app(api_key="server-key")) as client:
        resp = client.get("/info", headers={"Authorization": "Bearer server-key"})

    assert resp.status_code == 200
    assert resp.json() == {"version": "1.2.3"}
    assert mock_get_infra_version.call_count == 1


def test_info_rejects_invalid_server_key() -> None:
    with TestClient(_make_app(api_key="server-key")) as client:
        resp = client.get("/info", headers={"Authorization": "Bearer wrong-key"})

    assert resp.status_code == 401


def test_info_rejects_missing_token() -> None:
    with TestClient(_make_app(api_key="server-key"), raise_server_exceptions=False) as client:
        resp = client.get("/info")

    assert resp.status_code == 401


def test_info_rejects_empty_bearer_when_key_unset() -> None:
    with TestClient(_make_app(api_key=""), raise_server_exceptions=False) as client:
        resp = client.get("/info", headers={"Authorization": "Bearer "})

    assert resp.status_code == 401
