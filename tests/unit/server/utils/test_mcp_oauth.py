# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import base64
import hashlib
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlparse

import pytest
from pydantic import ValidationError

from server.utils.mcp_oauth import (
    AuthorizationServerMetadata,
    DcrResponse,
    McpOAuthConfig,
    McpOAuthError,
    McpOAuthStateStore,
    PendingOAuthFlow,
    build_authorize_url,
    discover_authorization_server_for_resource,
    discover_authorization_server_metadata,
    discover_protected_resource_metadata,
    exchange_code_for_token,
    generate_pkce_pair,
    has_valid_access_token,
    refresh_access_token,
    register_dynamic_client,
)


def _make_acm(value: Any) -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=value)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _make_mock_resp(status: int = 200, json_data: Any = None) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data)
    return resp


def _make_flow(instance: str = "default", model_id: str = "my-server") -> PendingOAuthFlow:
    return PendingOAuthFlow(
        instance=instance,
        model_id=model_id,
        code_verifier="verifier",
        redirect_uri="http://infra.example/mcp-oauth/callback",
        resource="http://mcp.example/mcp",
    )


def test_has_valid_access_token_no_token_returns_false() -> None:
    oauth = McpOAuthConfig(access_token=None)

    result = has_valid_access_token(oauth)

    assert result is False


def test_has_valid_access_token_no_expiry_returns_true() -> None:
    oauth = McpOAuthConfig(access_token="tok", token_expires_at=None)

    result = has_valid_access_token(oauth)

    assert result is True


def test_has_valid_access_token_future_expiry_beyond_margin_returns_true() -> None:
    oauth = McpOAuthConfig(access_token="tok", token_expires_at=1_000.0)

    result = has_valid_access_token(oauth, now=100.0)

    assert result is True


def test_has_valid_access_token_within_safety_margin_returns_false() -> None:
    oauth = McpOAuthConfig(access_token="tok", token_expires_at=140.0)

    result = has_valid_access_token(oauth, now=100.0)

    assert result is False


def test_has_valid_access_token_already_expired_returns_false() -> None:
    oauth = McpOAuthConfig(access_token="tok", token_expires_at=50.0)

    result = has_valid_access_token(oauth, now=100.0)

    assert result is False


def test_mcp_oauth_state_store_add_and_pop_round_trips() -> None:
    store = McpOAuthStateStore()
    flow = _make_flow()

    store.add("state-1", flow)
    result = store.pop("state-1")

    assert result == flow


def test_mcp_oauth_state_store_pop_is_one_shot() -> None:
    store = McpOAuthStateStore()
    store.add("state-1", _make_flow())

    store.pop("state-1")
    second_result = store.pop("state-1")

    assert second_result is None


def test_mcp_oauth_state_store_pop_unknown_state_returns_none() -> None:
    store = McpOAuthStateStore()

    result = store.pop("unknown")

    assert result is None


def test_mcp_oauth_state_store_pop_expired_returns_none() -> None:
    store = McpOAuthStateStore(ttl_seconds=0)
    store.add("state-1", _make_flow())

    result = store.pop("state-1")

    assert result is None


def test_mcp_oauth_state_store_add_evicts_oldest_at_capacity() -> None:
    store = McpOAuthStateStore(max_flows=2)
    store.add("state-1", _make_flow(model_id="first"))
    store.add("state-2", _make_flow(model_id="second"))

    store.add("state-3", _make_flow(model_id="third"))

    assert store.pop("state-1") is None
    assert store.pop("state-2") is not None
    assert store.pop("state-3") is not None


def test_mcp_oauth_state_store_find_for_model_true_while_flow_live() -> None:
    store = McpOAuthStateStore()
    store.add("state-1", _make_flow(instance="default", model_id="my-server"))

    result = store.find_for_model("default", "my-server")

    assert result is True


def test_mcp_oauth_state_store_find_for_model_false_after_pop() -> None:
    store = McpOAuthStateStore()
    store.add("state-1", _make_flow(instance="default", model_id="my-server"))
    store.pop("state-1")

    result = store.find_for_model("default", "my-server")

    assert result is False


def test_mcp_oauth_state_store_find_for_model_false_after_expiry() -> None:
    store = McpOAuthStateStore(ttl_seconds=0)
    store.add("state-1", _make_flow(instance="default", model_id="my-server"))

    result = store.find_for_model("default", "my-server")

    assert result is False


def test_mcp_oauth_state_store_find_for_model_false_for_other_model() -> None:
    store = McpOAuthStateStore()
    store.add("state-1", _make_flow(instance="default", model_id="my-server"))

    result = store.find_for_model("default", "other-server")

    assert result is False


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_discover_protected_resource_metadata_parses_response(mock_client_session: MagicMock) -> None:
    resp = _make_mock_resp(status=200, json_data={"resource": "http://mcp.example/mcp", "authorization_servers": ["http://as.example"]})
    session = MagicMock()
    session.get.return_value = _make_acm(resp)
    mock_client_session.return_value = _make_acm(session)

    result = await discover_protected_resource_metadata("http://mcp.example/mcp")

    assert result is not None
    assert result.resource == "http://mcp.example/mcp"
    assert result.authorization_servers == ["http://as.example"]
    assert session.get.call_count == 1
    called_url = session.get.call_args.args[0]
    assert called_url == "http://mcp.example/.well-known/oauth-protected-resource"


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_discover_protected_resource_metadata_non_200_returns_none(mock_client_session: MagicMock) -> None:
    resp = _make_mock_resp(status=404, json_data=None)
    session = MagicMock()
    session.get.return_value = _make_acm(resp)
    mock_client_session.return_value = _make_acm(session)

    result = await discover_protected_resource_metadata("http://mcp.example/mcp")

    assert result is None


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_discover_protected_resource_metadata_invalid_shape_returns_none(mock_client_session: MagicMock) -> None:
    resp = _make_mock_resp(status=200, json_data={"authorization_servers": ["http://as.example"]})
    session = MagicMock()
    session.get.return_value = _make_acm(resp)
    mock_client_session.return_value = _make_acm(session)

    result = await discover_protected_resource_metadata("http://mcp.example/mcp")

    assert result is None


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_discover_authorization_server_metadata_parses_response(mock_client_session: MagicMock) -> None:
    resp = _make_mock_resp(
        status=200,
        json_data={
            "issuer": "http://as.example",
            "authorization_endpoint": "http://as.example/authorize",
            "token_endpoint": "http://as.example/token",
        },
    )
    session = MagicMock()
    session.get.return_value = _make_acm(resp)
    mock_client_session.return_value = _make_acm(session)

    result = await discover_authorization_server_metadata("http://as.example")

    assert result is not None
    assert result.authorization_endpoint == "http://as.example/authorize"
    assert result.token_endpoint == "http://as.example/token"
    called_url = session.get.call_args.args[0]
    assert called_url == "http://as.example/.well-known/oauth-authorization-server"


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_discover_authorization_server_metadata_falls_back_to_openid_configuration(mock_client_session: MagicMock) -> None:
    not_found = _make_mock_resp(status=404, json_data=None)
    found = _make_mock_resp(
        status=200,
        json_data={
            "issuer": "http://as.example",
            "authorization_endpoint": "http://as.example/authorize",
            "token_endpoint": "http://as.example/token",
        },
    )
    session = MagicMock()
    session.get.side_effect = [_make_acm(not_found), _make_acm(found)]
    mock_client_session.return_value = _make_acm(session)

    result = await discover_authorization_server_metadata("http://as.example")

    assert result is not None
    assert result.token_endpoint == "http://as.example/token"
    assert session.get.call_count == 2
    second_url = session.get.call_args_list[1].args[0]
    assert second_url == "http://as.example/.well-known/openid-configuration"


@pytest.mark.parametrize("bad_url", ["javascript:alert(1)", "data:text/html,<script>alert(1)</script>", "file:///etc/passwd"])
def test_authorization_server_metadata_rejects_non_http_scheme(bad_url: str) -> None:
    with pytest.raises(ValidationError, match="http or https"):
        AuthorizationServerMetadata(
            issuer="http://as.example",
            authorization_endpoint=bad_url,
            token_endpoint="http://as.example/token",
        )


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_discover_authorization_server_metadata_rejects_malicious_scheme(mock_client_session: MagicMock) -> None:
    resp = _make_mock_resp(
        status=200,
        json_data={
            "issuer": "http://as.example",
            "authorization_endpoint": "javascript:alert(1)",
            "token_endpoint": "http://as.example/token",
        },
    )
    session = MagicMock()
    session.get.return_value = _make_acm(resp)
    mock_client_session.return_value = _make_acm(session)

    result = await discover_authorization_server_metadata("http://as.example")

    assert result is None


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_discover_authorization_server_metadata_both_missing_returns_none(mock_client_session: MagicMock) -> None:
    not_found = _make_mock_resp(status=404, json_data=None)
    session = MagicMock()
    session.get.side_effect = [_make_acm(not_found), _make_acm(not_found)]
    mock_client_session.return_value = _make_acm(session)

    result = await discover_authorization_server_metadata("http://as.example")

    assert result is None


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_discover_protected_resource_metadata_network_exception_returns_none(mock_client_session: MagicMock) -> None:
    mock_client_session.side_effect = ConnectionError("boom")

    result = await discover_protected_resource_metadata("http://mcp.example/mcp")

    assert result is None


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_discover_authorization_server_metadata_invalid_first_response_falls_back(mock_client_session: MagicMock) -> None:
    invalid = _make_mock_resp(status=200, json_data={"issuer": "http://as.example"})
    valid = _make_mock_resp(
        status=200,
        json_data={
            "issuer": "http://as.example",
            "authorization_endpoint": "http://as.example/authorize",
            "token_endpoint": "http://as.example/token",
        },
    )
    session = MagicMock()
    session.get.side_effect = [_make_acm(invalid), _make_acm(valid)]
    mock_client_session.return_value = _make_acm(session)

    result = await discover_authorization_server_metadata("http://as.example")

    assert result is not None
    assert result.token_endpoint == "http://as.example/token"
    assert session.get.call_count == 2


@mock.patch("server.utils.mcp_oauth.discover_authorization_server_metadata")
@mock.patch("server.utils.mcp_oauth.discover_protected_resource_metadata")
@pytest.mark.asyncio
async def test_discover_authorization_server_for_resource_two_hop(
    mock_discover_prm: AsyncMock,
    mock_discover_as: AsyncMock,
) -> None:
    metadata = AuthorizationServerMetadata(
        issuer="http://as.example",
        authorization_endpoint="http://as.example/authorize",
        token_endpoint="http://as.example/token",
    )
    mock_discover_prm.return_value = MagicMock(authorization_servers=["http://as.example"])
    mock_discover_as.return_value = metadata

    result = await discover_authorization_server_for_resource("http://mcp.example/mcp")

    assert result == metadata
    assert mock_discover_prm.call_args == mock.call("http://mcp.example/mcp")
    assert mock_discover_as.call_args == mock.call("http://as.example")


@mock.patch("server.utils.mcp_oauth.discover_authorization_server_metadata")
@mock.patch("server.utils.mcp_oauth.discover_protected_resource_metadata")
@pytest.mark.asyncio
async def test_discover_authorization_server_for_resource_falls_back_to_direct_issuer(
    mock_discover_prm: AsyncMock,
    mock_discover_as: AsyncMock,
) -> None:
    metadata = AuthorizationServerMetadata(
        issuer="http://mcp.example",
        authorization_endpoint="http://mcp.example/authorize",
        token_endpoint="http://mcp.example/token",
    )
    mock_discover_prm.return_value = None
    mock_discover_as.return_value = metadata

    result = await discover_authorization_server_for_resource("http://mcp.example/mcp")

    assert result == metadata
    assert mock_discover_as.call_args == mock.call("http://mcp.example/mcp")


@mock.patch("server.utils.mcp_oauth.discover_authorization_server_metadata")
@mock.patch("server.utils.mcp_oauth.discover_protected_resource_metadata")
@pytest.mark.asyncio
async def test_discover_authorization_server_for_resource_all_issuers_fail_returns_none(
    mock_discover_prm: AsyncMock,
    mock_discover_as: AsyncMock,
) -> None:
    mock_discover_prm.return_value = MagicMock(authorization_servers=["http://as1.example", "http://as2.example"])
    mock_discover_as.side_effect = [None, None]

    result = await discover_authorization_server_for_resource("http://mcp.example/mcp")

    assert result is None
    assert mock_discover_as.await_count == 2


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_register_dynamic_client_success_returns_response(mock_client_session: MagicMock) -> None:
    resp = _make_mock_resp(status=201, json_data={"client_id": "abc123", "client_secret": "shh"})
    session = MagicMock()
    session.post.return_value = _make_acm(resp)
    mock_client_session.return_value = _make_acm(session)

    result = await register_dynamic_client("http://as.example/register", "http://infra.example/mcp-oauth/callback", "DeepFellow")

    assert result == DcrResponse(client_id="abc123", client_secret="shh")


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_register_dynamic_client_non_2xx_returns_none(mock_client_session: MagicMock) -> None:
    resp = _make_mock_resp(status=400, json_data={"error": "invalid_request"})
    session = MagicMock()
    session.post.return_value = _make_acm(resp)
    mock_client_session.return_value = _make_acm(session)

    result = await register_dynamic_client("http://as.example/register", "http://infra.example/mcp-oauth/callback", "DeepFellow")

    assert result is None


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_register_dynamic_client_network_exception_returns_none(mock_client_session: MagicMock) -> None:
    mock_client_session.side_effect = ConnectionError("boom")

    result = await register_dynamic_client("http://as.example/register", "http://infra.example/mcp-oauth/callback", "DeepFellow")

    assert result is None


def test_generate_pkce_pair_verifier_length_is_spec_valid() -> None:
    code_verifier, _ = generate_pkce_pair()

    assert 43 <= len(code_verifier) <= 128


def test_generate_pkce_pair_challenge_matches_s256_of_verifier() -> None:
    code_verifier, code_challenge = generate_pkce_pair()

    expected_digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    expected_challenge = base64.urlsafe_b64encode(expected_digest).rstrip(b"=").decode("ascii")
    assert code_challenge == expected_challenge
    assert "=" not in code_challenge


def test_build_authorize_url_contains_required_params() -> None:
    url = build_authorize_url(
        "http://as.example/authorize",
        "client-1",
        "http://infra.example/mcp-oauth/callback",
        "state-1",
        "challenge-1",
        "http://mcp.example/mcp",
    )

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    assert parsed.scheme == "http"
    assert parsed.netloc == "as.example"
    assert parsed.path == "/authorize"
    assert query["response_type"] == ["code"]
    assert query["client_id"] == ["client-1"]
    assert query["redirect_uri"] == ["http://infra.example/mcp-oauth/callback"]
    assert query["state"] == ["state-1"]
    assert query["code_challenge"] == ["challenge-1"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["resource"] == ["http://mcp.example/mcp"]
    assert "scope" not in query


def test_build_authorize_url_includes_scope_when_given() -> None:
    url = build_authorize_url(
        "http://as.example/authorize",
        "client-1",
        "http://infra.example/mcp-oauth/callback",
        "state-1",
        "challenge-1",
        "http://mcp.example/mcp",
        scope="tools:read",
    )

    query = parse_qs(urlparse(url).query)
    assert query["scope"] == ["tools:read"]


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_exchange_code_for_token_success_returns_token_response(mock_client_session: MagicMock) -> None:
    resp = _make_mock_resp(status=200, json_data={"access_token": "tok", "refresh_token": "rtok", "expires_in": 3600})
    session = MagicMock()
    session.post.return_value = _make_acm(resp)
    mock_client_session.return_value = _make_acm(session)

    result = await exchange_code_for_token(
        "http://as.example/token", "code-1", "http://infra.example/mcp-oauth/callback", "client-1", None, "verifier-1", "http://mcp.example"
    )

    assert result.access_token == "tok"
    assert result.refresh_token == "rtok"
    assert result.expires_in == 3600


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_exchange_code_for_token_non_200_raises(mock_client_session: MagicMock) -> None:
    resp = _make_mock_resp(status=400, json_data={"error": "invalid_grant"})
    session = MagicMock()
    session.post.return_value = _make_acm(resp)
    mock_client_session.return_value = _make_acm(session)

    with pytest.raises(McpOAuthError):
        await exchange_code_for_token(
            "http://as.example/token",
            "code-1",
            "http://infra.example/mcp-oauth/callback",
            "client-1",
            None,
            "verifier-1",
            "http://mcp.example",
        )


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_exchange_code_for_token_omits_client_secret_when_not_provided(mock_client_session: MagicMock) -> None:
    resp = _make_mock_resp(status=200, json_data={"access_token": "tok"})
    session = MagicMock()
    session.post.return_value = _make_acm(resp)
    mock_client_session.return_value = _make_acm(session)

    await exchange_code_for_token(
        "http://as.example/token", "code-1", "http://infra.example/mcp-oauth/callback", "client-1", None, "verifier-1", "http://mcp.example"
    )

    sent_form = session.post.call_args.kwargs["data"]
    assert "client_secret" not in sent_form


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_exchange_code_for_token_includes_client_secret_when_provided(mock_client_session: MagicMock) -> None:
    resp = _make_mock_resp(status=200, json_data={"access_token": "tok"})
    session = MagicMock()
    session.post.return_value = _make_acm(resp)
    mock_client_session.return_value = _make_acm(session)

    await exchange_code_for_token(
        "http://as.example/token",
        "code-1",
        "http://infra.example/mcp-oauth/callback",
        "client-1",
        "secret-1",
        "verifier-1",
        "http://mcp.example",
    )

    sent_form = session.post.call_args.kwargs["data"]
    assert sent_form["client_secret"] == "secret-1"


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_exchange_code_for_token_network_exception_raises_mcpoautherror(mock_client_session: MagicMock) -> None:
    mock_client_session.side_effect = ConnectionError("boom")

    with pytest.raises(McpOAuthError):
        await exchange_code_for_token(
            "http://as.example/token",
            "code-1",
            "http://infra.example/mcp-oauth/callback",
            "client-1",
            None,
            "verifier-1",
            "http://mcp.example",
        )


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_refresh_access_token_success_returns_token_response(mock_client_session: MagicMock) -> None:
    resp = _make_mock_resp(status=200, json_data={"access_token": "new-tok", "expires_in": 60})
    session = MagicMock()
    session.post.return_value = _make_acm(resp)
    mock_client_session.return_value = _make_acm(session)

    result = await refresh_access_token("http://as.example/token", "rtok", "client-1", None, "http://mcp.example")

    assert result.access_token == "new-tok"
    assert result.expires_in == 60


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_refresh_access_token_error_field_raises(mock_client_session: MagicMock) -> None:
    resp = _make_mock_resp(status=200, json_data={"error": "invalid_grant"})
    resp.status = 400
    session = MagicMock()
    session.post.return_value = _make_acm(resp)
    mock_client_session.return_value = _make_acm(session)

    with pytest.raises(McpOAuthError):
        await refresh_access_token("http://as.example/token", "rtok", "client-1", None, "http://mcp.example")


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_refresh_access_token_omits_client_secret_when_not_provided(mock_client_session: MagicMock) -> None:
    resp = _make_mock_resp(status=200, json_data={"access_token": "new-tok"})
    session = MagicMock()
    session.post.return_value = _make_acm(resp)
    mock_client_session.return_value = _make_acm(session)

    await refresh_access_token("http://as.example/token", "rtok", "client-1", None, "http://mcp.example")

    sent_form = session.post.call_args.kwargs["data"]
    assert "client_secret" not in sent_form


@mock.patch("server.utils.mcp_oauth.aiohttp.ClientSession")
@pytest.mark.asyncio
async def test_refresh_access_token_includes_client_secret_when_provided(mock_client_session: MagicMock) -> None:
    resp = _make_mock_resp(status=200, json_data={"access_token": "new-tok"})
    session = MagicMock()
    session.post.return_value = _make_acm(resp)
    mock_client_session.return_value = _make_acm(session)

    await refresh_access_token("http://as.example/token", "rtok", "client-1", "secret-1", "http://mcp.example")

    sent_form = session.post.call_args.kwargs["data"]
    assert sent_form["client_secret"] == "secret-1"
