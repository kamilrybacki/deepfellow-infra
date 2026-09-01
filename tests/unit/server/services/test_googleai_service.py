# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.models.models import ListModelsFilters
from server.models.services import InstallServiceIn
from server.services.googleai_service import _MAX_PAGES, GoogleAIService  # pyright: ignore[reportPrivateUsage]
from server.services.remote_service import DefaultRemoteServiceOptions, InstalledInfo


@pytest.fixture
def deps() -> dict[str, Any]:
    return {
        "config": MagicMock(),
        "endpoint_registry": MagicMock(),
        "service_provider": MagicMock(),
        "model_downloader": MagicMock(),
        "docker_service": MagicMock(),
        "hardware": MagicMock(gpus=[]),
    }


@pytest.fixture
def svc(deps: dict[str, Any]) -> GoogleAIService:
    return GoogleAIService(**deps)


def _make_installed(api_key: str = "test-key") -> InstalledInfo[DefaultRemoteServiceOptions]:
    return InstalledInfo(
        models={},
        options=InstallServiceIn(spec={"api_url": "https://generativelanguage.googleapis.com", "api_key": api_key}),
        parsed_options=DefaultRemoteServiceOptions(api_url="https://generativelanguage.googleapis.com", api_key=api_key),
    )


def _mock_models_response(body: object, status: int = 200) -> MagicMock:
    text = body if isinstance(body, str) else json.dumps(body)
    mock_resp = AsyncMock()
    mock_resp.status = status
    mock_resp.text = AsyncMock(return_value=text)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    return mock_resp


def _mock_session(*responses: MagicMock) -> MagicMock:
    mock_session = AsyncMock()
    mock_session.get = MagicMock(side_effect=list(responses))
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


@pytest.mark.asyncio
async def test_fetch_live_models_returns_entries_on_success(svc: GoogleAIService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    body = {"models": [{"name": "models/gemini-1.5-flash"}, {"name": "models/some-new-model"}]}
    mock_session = _mock_session(_mock_models_response(body))

    with patch("server.services.googleai_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    ids = {entry.id for entry in result}
    assert ids == {"gemini-1.5-flash", "some-new-model"}
    for entry in result:
        assert entry.type is None
        assert entry.messages is None
        assert entry.responses is None
        assert entry.completions is None
        assert entry.legacy_completions is None
        assert entry.context_length is None
        assert entry.max_context_length is None


@pytest.mark.asyncio
async def test_fetch_live_models_resolves_type_from_supported_generation_methods(svc: GoogleAIService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    body = {
        "models": [
            {"name": "models/gemini-2.5-pro", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/imagen-4.0-generate-001", "supportedGenerationMethods": ["predict"]},
            {"name": "models/aqa", "supportedGenerationMethods": ["generateAnswer"]},
        ]
    }
    mock_session = _mock_session(_mock_models_response(body))

    with patch("server.services.googleai_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    by_id = {entry.id: entry.type for entry in result}
    assert by_id["gemini-2.5-pro"] == "llm"
    assert by_id["text-embedding-004"] == "embedding"
    assert by_id["imagen-4.0-generate-001"] == "txt2img"
    assert by_id["aqa"] is None


@pytest.mark.asyncio
async def test_fetch_live_models_uses_goog_api_key_header(svc: GoogleAIService) -> None:
    svc.instances_info["default"].installed = _make_installed(api_key="secret-key")
    mock_session = _mock_session(_mock_models_response({"models": []}))

    with patch("server.services.googleai_service.aiohttp.ClientSession", return_value=mock_session):
        await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    call_kwargs = mock_session.get.call_args_list[0].kwargs
    assert call_kwargs["headers"] == {"x-goog-api-key": "secret-key"}


@pytest.mark.asyncio
async def test_fetch_live_models_follows_pagination(svc: GoogleAIService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    first_body = {"models": [{"name": "models/gemini-1.5-flash"}], "nextPageToken": "page-2"}
    second_body = {"models": [{"name": "models/gemini-2.5-pro"}]}
    mock_session = _mock_session(_mock_models_response(first_body), _mock_models_response(second_body))

    with patch("server.services.googleai_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    ids = {entry.id for entry in result}
    assert ids == {"gemini-1.5-flash", "gemini-2.5-pro"}

    assert mock_session.get.call_count == 2
    second_call_kwargs = mock_session.get.call_args_list[1].kwargs
    assert second_call_kwargs["params"] == {"pageToken": "page-2"}


@pytest.mark.asyncio
async def test_fetch_live_models_returns_none_on_non_200_first_page(svc: GoogleAIService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    mock_session = _mock_session(_mock_models_response({}, status=500))

    with patch("server.services.googleai_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_fetch_live_models_returns_none_on_non_200_later_page(svc: GoogleAIService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    first_body = {"models": [{"name": "models/gemini-1.5-flash"}], "nextPageToken": "page-2"}
    mock_session = _mock_session(_mock_models_response(first_body), _mock_models_response({}, status=500))

    with patch("server.services.googleai_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_fetch_live_models_returns_none_on_request_exception(svc: GoogleAIService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(side_effect=ConnectionError("network error"))

    with patch("server.services.googleai_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        "not json",
        {},
        {"models": "not-a-list"},
        {"models": [{"baseModelId": "gemini-1.5-flash"}]},
    ],
)
async def test_fetch_live_models_returns_none_on_malformed_body(svc: GoogleAIService, body: object) -> None:
    svc.instances_info["default"].installed = _make_installed()
    mock_session = _mock_session(_mock_models_response(body))

    with patch("server.services.googleai_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_fetch_live_models_returns_none_on_malformed_later_page(svc: GoogleAIService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    first_body = {"models": [{"name": "models/gemini-1.5-flash"}], "nextPageToken": "page-2"}
    second_body = {"models": "not-a-list"}
    mock_session = _mock_session(_mock_models_response(first_body), _mock_models_response(second_body))

    with patch("server.services.googleai_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_fetch_live_models_returns_none_on_unexpected_name_format(svc: GoogleAIService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    body = {"models": [{"name": "gemini-1.5-flash"}]}  # missing the "models/" prefix
    mock_session = _mock_session(_mock_models_response(body))

    with patch("server.services.googleai_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_fetch_live_models_returns_collected_entries_when_pages_exhausted(svc: GoogleAIService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    pages = [
        _mock_models_response({"models": [{"name": f"models/model-{i}"}], "nextPageToken": f"page-{i + 1}"}) for i in range(_MAX_PAGES)
    ]
    mock_session = _mock_session(*pages)

    with patch("server.services.googleai_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    assert {entry.id for entry in result} == {f"model-{i}" for i in range(_MAX_PAGES)}


@pytest.mark.asyncio
async def test_fetch_live_models_returns_none_when_instance_not_installed(svc: GoogleAIService) -> None:
    result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_refresh_catalog_resolves_known_id_and_flags_unknown_id(svc: GoogleAIService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    body = {"models": [{"name": "models/gemini-2.5-pro"}, {"name": "models/some-brand-new-model"}]}
    mock_session = _mock_session(_mock_models_response(body))

    with patch("server.services.googleai_service.aiohttp.ClientSession", return_value=mock_session):
        await svc.refresh_catalog()

    out = await svc.list_models("default", ListModelsFilters())
    by_id = {model.id: model for model in out.list}

    assert by_id["gemini-2.5-pro"].capabilities_resolved is True
    assert by_id["some-brand-new-model"].capabilities_resolved is False
