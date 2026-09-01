# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from server.models.models import ListModelsFilters
from server.models.services import InstallServiceIn
from server.services.claude_service import _MAX_PAGES, ClaudeService, ClaudeServiceOptions  # pyright: ignore[reportPrivateUsage]
from server.services.remote_service import InstalledInfo


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
def svc(deps: dict[str, Any]) -> ClaudeService:
    return ClaudeService(**deps)


def _make_installed(api_key: str = "test-key") -> InstalledInfo[ClaudeServiceOptions]:
    return InstalledInfo(
        models={},
        options=InstallServiceIn(spec={"api_url": "https://api.anthropic.com", "api_key": api_key, "anthropic_version": "2023-06-01"}),
        parsed_options=ClaudeServiceOptions(api_url="https://api.anthropic.com", api_key=api_key, anthropic_version="2023-06-01"),
    )


def test_claude_service_options_require_api_key() -> None:
    """The Claude service requires an API key, unlike the OpenAI-compatible default (empty-string allowed) -
    it's needed to query the live catalog, not just to authenticate proxied requests."""
    with pytest.raises(ValidationError):
        ClaudeServiceOptions(api_url="https://api.anthropic.com", anthropic_version="2023-06-01")  # pyright: ignore[reportCallIssue]


@pytest.mark.asyncio
async def test_install_instance_rejects_missing_api_key(svc: ClaudeService) -> None:
    with pytest.raises(HTTPException) as exc:
        await svc.install_instance("default", InstallServiceIn(spec={"anthropic_version": "2023-06-01"}))

    assert exc.value.status_code == 400


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
async def test_fetch_live_models_returns_entries_on_success(svc: ClaudeService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    body = {"data": [{"id": "claude-sonnet-5", "type": "model"}, {"id": "some-new-model", "type": "model"}], "has_more": False}
    mock_session = _mock_session(_mock_models_response(body))

    with patch("server.services.claude_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    ids = {entry.id for entry in result}
    assert ids == {"claude-sonnet-5", "some-new-model"}
    for entry in result:
        # Anthropic's listing only ever contains LLMs, so the service sets this itself.
        assert entry.type == "llm"
        assert entry.messages is None
        assert entry.responses is None
        assert entry.completions is None
        assert entry.legacy_completions is None
        assert entry.context_length is None
        assert entry.max_context_length is None


@pytest.mark.asyncio
async def test_fetch_live_models_follows_pagination(svc: ClaudeService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    first_body = {"data": [{"id": "claude-sonnet-5"}], "has_more": True, "last_id": "claude-sonnet-5"}
    second_body = {"data": [{"id": "claude-opus-4-8"}], "has_more": False}
    mock_session = _mock_session(_mock_models_response(first_body), _mock_models_response(second_body))

    with patch("server.services.claude_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    ids = {entry.id for entry in result}
    assert ids == {"claude-sonnet-5", "claude-opus-4-8"}

    assert mock_session.get.call_count == 2
    second_call_kwargs = mock_session.get.call_args_list[1].kwargs
    assert second_call_kwargs["params"] == {"after_id": "claude-sonnet-5"}


@pytest.mark.asyncio
async def test_fetch_live_models_returns_none_on_non_200_first_page(svc: ClaudeService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    mock_session = _mock_session(_mock_models_response({}, status=500))

    with patch("server.services.claude_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_fetch_live_models_returns_none_on_non_200_later_page(svc: ClaudeService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    first_body = {"data": [{"id": "claude-sonnet-5"}], "has_more": True, "last_id": "claude-sonnet-5"}
    mock_session = _mock_session(_mock_models_response(first_body), _mock_models_response({}, status=500))

    with patch("server.services.claude_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_fetch_live_models_returns_none_on_request_exception(svc: ClaudeService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(side_effect=ConnectionError("network error"))

    with patch("server.services.claude_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        "not json",
        {"has_more": False},
        {"data": "not-a-list", "has_more": False},
        {"data": [{"type": "model"}], "has_more": False},
    ],
)
async def test_fetch_live_models_returns_none_on_malformed_body(svc: ClaudeService, body: object) -> None:
    svc.instances_info["default"].installed = _make_installed()
    mock_session = _mock_session(_mock_models_response(body))

    with patch("server.services.claude_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_fetch_live_models_returns_none_when_instance_not_installed(svc: ClaudeService) -> None:
    result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_fetch_live_models_treats_null_data_as_empty_page(svc: ClaudeService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    body = {"data": None, "has_more": False}
    mock_session = _mock_session(_mock_models_response(body))

    with patch("server.services.claude_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result == []


@pytest.mark.asyncio
async def test_fetch_live_models_requests_models_endpoint_regardless_of_v1_suffix(svc: ClaudeService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(
        models={},
        options=InstallServiceIn(
            spec={"api_url": "https://api.anthropic.com/v1", "api_key": "test-key", "anthropic_version": "2023-06-01"}
        ),
        parsed_options=ClaudeServiceOptions(api_url="https://api.anthropic.com/v1", api_key="test-key", anthropic_version="2023-06-01"),
    )
    body = {"data": [{"id": "claude-sonnet-5"}], "has_more": False}
    mock_session = _mock_session(_mock_models_response(body))

    with patch("server.services.claude_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    called_url = mock_session.get.call_args_list[0].args[0]
    assert called_url == "https://api.anthropic.com/v1/models"


@pytest.mark.asyncio
async def test_fetch_live_models_returns_collected_entries_when_pages_exhausted(svc: ClaudeService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    pages = [_mock_models_response({"data": [{"id": f"model-{i}"}], "has_more": True, "last_id": f"model-{i}"}) for i in range(_MAX_PAGES)]
    mock_session = _mock_session(*pages)

    with patch("server.services.claude_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is not None
    assert {entry.id for entry in result} == {f"model-{i}" for i in range(_MAX_PAGES)}


@pytest.mark.asyncio
async def test_fetch_live_models_returns_none_on_malformed_later_page(svc: ClaudeService) -> None:
    svc.instances_info["default"].installed = _make_installed()
    first_body = {"data": [{"id": "claude-sonnet-5"}], "has_more": True, "last_id": "claude-sonnet-5"}
    second_body = {"data": "not-a-list", "has_more": False}
    mock_session = _mock_session(_mock_models_response(first_body), _mock_models_response(second_body))

    with patch("server.services.claude_service.aiohttp.ClientSession", return_value=mock_session):
        result = await svc._fetch_live_models("default")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_refresh_catalog_resolves_known_id_and_new_model_via_forced_llm_type(svc: ClaudeService) -> None:
    """Anthropic's listing is LLM-only, so even an id absent from the RemoteConst overlay resolves as `llm`
    (unlike OpenAI/Google, which have no such provider-wide type guarantee and can produce unresolved entries).
    """
    svc.instances_info["default"].installed = _make_installed()
    body = {"data": [{"id": "claude-sonnet-5"}, {"id": "some-brand-new-model"}], "has_more": False}
    mock_session = _mock_session(_mock_models_response(body))

    with patch("server.services.claude_service.aiohttp.ClientSession", return_value=mock_session):
        await svc.refresh_catalog()

    out = await svc.list_models("default", ListModelsFilters())
    by_id = {model.id: model for model in out.list}

    assert by_id["claude-sonnet-5"].capabilities_resolved is True
    assert by_id["claude-sonnet-5"].type == "llm"
    assert by_id["some-brand-new-model"].capabilities_resolved is True
    assert by_id["some-brand-new-model"].type == "llm"
