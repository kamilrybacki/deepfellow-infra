# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""OAuth 2.1 client logic for authenticating to remote (proxy) MCP servers.

Pure OAuth-client functions and models, with no FastAPI/McpService coupling, so this
module is fully unit-testable against a mocked authorization server. Implements the
relevant subset of the MCP Authorization spec (2025-06-18): protected-resource discovery
(RFC 9728), authorization-server metadata discovery (RFC 8414), dynamic client
registration (RFC 7591), and the PKCE (S256-only) authorization-code flow.
"""

import base64
import hashlib
import logging
import secrets
import time
from typing import Any, NoReturn
from urllib.parse import urlencode, urljoin, urlparse

import aiohttp
from pydantic import BaseModel, field_validator

logger = logging.getLogger("uvicorn.error")

_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=10, connect=5)
_TOKEN_EXPIRY_SAFETY_MARGIN_SECONDS = 60


class McpOAuthError(Exception):
    """Raised when an OAuth-client operation (token exchange, refresh) fails."""


class ProtectedResourceMetadata(BaseModel):
    """RFC 9728 OAuth 2.0 Protected Resource Metadata."""

    resource: str
    authorization_servers: list[str] = []
    bearer_methods_supported: list[str] | None = None


class AuthorizationServerMetadata(BaseModel):
    """RFC 8414 OAuth 2.0 Authorization Server Metadata (subset used by this client)."""

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str | None = None
    scopes_supported: list[str] | None = None
    code_challenge_methods_supported: list[str] | None = None

    @field_validator("authorization_endpoint", "token_endpoint", "registration_endpoint")
    @classmethod
    def _require_http_scheme(cls, value: str | None) -> str | None:
        """Reject non-http(s) schemes (e.g. `javascript:`, `data:`, `file:`) from a remote server's own metadata."""
        if value is not None and urlparse(value).scheme not in ("http", "https"):
            msg = f"endpoint must use http or https, got {value!r}"
            raise ValueError(msg)
        return value


class DcrRequest(BaseModel):
    """RFC 7591 Dynamic Client Registration request."""

    client_name: str
    redirect_uris: list[str]
    grant_types: list[str] = ["authorization_code", "refresh_token"]
    response_types: list[str] = ["code"]


class DcrResponse(BaseModel):
    """RFC 7591 Dynamic Client Registration response (subset used by this client)."""

    client_id: str
    client_secret: str | None = None


class TokenResponse(BaseModel):
    """OAuth 2.1 token endpoint response (subset used by this client)."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int | None = None
    refresh_token: str | None = None
    scope: str | None = None


class PendingOAuthFlow(BaseModel):
    """State correlated between `start_oauth_flow` and the unauthenticated callback."""

    instance: str
    model_id: str
    code_verifier: str
    redirect_uri: str
    resource: str


class McpOAuthConfig(BaseModel):
    """Persisted OAuth client config + tokens for a single proxy MCP model.

    Stored plaintext in ``services.json``, consistent with every other secret in this
    codebase today (see ``SrvMcpProxyModel.headers``). Never included in an API response
    sent to the WebUI — only non-secret status booleans/timestamps are ever serialized
    for that purpose (see ``mcp_service.McpOAuthStatusOut``).
    """

    enabled: bool = False
    client_id: str | None = None
    client_secret: str | None = None
    scope: str | None = None
    resource: str | None = None
    authorization_endpoint: str | None = None
    token_endpoint: str | None = None
    registration_endpoint: str | None = None
    access_token: str | None = None
    refresh_token: str | None = None
    token_expires_at: float | None = None
    last_error: str | None = None


def has_valid_access_token(oauth: McpOAuthConfig, now: float | None = None) -> bool:
    """Return True if `oauth` carries an access token that isn't expired (or expiry-less)."""
    if not oauth.access_token:
        return False
    if oauth.token_expires_at is None:
        return True
    return (now if now is not None else time.time()) < oauth.token_expires_at - _TOKEN_EXPIRY_SAFETY_MARGIN_SECONDS


class McpOAuthStateStore:
    """Maps a one-shot `state` value to the pending OAuth flow it was issued for.

    Copies the exact shape of ``endpointregistry.McpSseSessionStore``: plain dict, lazy
    TTL eviction, capacity-bounded, no lock (all mutations are synchronous dict
    operations with no await points, so this is safe under asyncio's single-threaded
    event loop). Unlike that store, `pop()` is used for the callback lookup so a `state`
    is consumed exactly once.
    """

    def __init__(self, ttl_seconds: int = 600, max_flows: int = 128) -> None:
        self._flows: dict[str, tuple[PendingOAuthFlow, float]] = {}
        self._ttl = ttl_seconds
        self._max = max_flows

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [state for state, (_, ts) in self._flows.items() if now - ts >= self._ttl]
        for state in expired:
            del self._flows[state]

    def add(self, state: str, flow: PendingOAuthFlow) -> None:
        """Store a pending flow under a fresh `state`, evicting expired/oldest if at capacity."""
        self._evict_expired()
        if len(self._flows) >= self._max:
            oldest = next(iter(self._flows))
            del self._flows[oldest]
        self._flows[state] = (flow, time.monotonic())

    def pop(self, state: str) -> PendingOAuthFlow | None:
        """Consume and return the pending flow for `state`, or None if unknown/expired."""
        entry = self._flows.pop(state, None)
        if entry is None:
            return None
        flow, ts = entry
        if time.monotonic() - ts >= self._ttl:
            return None
        return flow

    def find_for_model(self, instance: str, model_id: str) -> bool:
        """Return True if a still-live pending flow exists for this model (used for status display)."""
        self._evict_expired()
        return any(f.instance == instance and f.model_id == model_id for f, _ in self._flows.values())


def _well_known_url(base_url: str, well_known_path: str) -> str:
    parsed = urlparse(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}/"
    return urljoin(origin, well_known_path)


async def _get_json(url: str) -> dict[str, Any] | None:
    try:
        async with aiohttp.ClientSession() as session, session.get(url, timeout=_HTTP_TIMEOUT) as resp:
            if resp.status != 200:
                return None
            return await resp.json(content_type=None)
    except Exception:
        logger.debug("OAuth metadata fetch failed for %s", url, exc_info=True)
        return None


async def discover_protected_resource_metadata(resource_url: str) -> ProtectedResourceMetadata | None:
    """RFC 9728: discover the authorization server(s) protecting `resource_url`."""
    data = await _get_json(_well_known_url(resource_url, ".well-known/oauth-protected-resource"))
    if data is None:
        return None
    try:
        return ProtectedResourceMetadata.model_validate(data)
    except Exception:
        logger.debug("Invalid protected resource metadata from %s", resource_url, exc_info=True)
        return None


async def discover_authorization_server_metadata(issuer_url: str) -> AuthorizationServerMetadata | None:
    """RFC 8414: discover AS metadata at `issuer_url`, falling back to OIDC discovery."""
    for well_known_path in (".well-known/oauth-authorization-server", ".well-known/openid-configuration"):
        data = await _get_json(_well_known_url(issuer_url, well_known_path))
        if data is None:
            continue
        try:
            return AuthorizationServerMetadata.model_validate(data)
        except Exception:
            logger.debug("Invalid authorization server metadata from %s%s", issuer_url, well_known_path, exc_info=True)
    return None


async def discover_authorization_server_for_resource(resource_url: str) -> AuthorizationServerMetadata | None:
    """Combine protected-resource + AS discovery: find the AS metadata that protects `resource_url`.

    Falls back to treating `resource_url`'s own origin as the issuer when the resource
    doesn't publish RFC 9728 protected-resource metadata (some MCP servers host AS
    metadata directly without the extra indirection).
    """
    prm = await discover_protected_resource_metadata(resource_url)
    if prm and prm.authorization_servers:
        for issuer in prm.authorization_servers:
            metadata = await discover_authorization_server_metadata(issuer)
            if metadata:
                return metadata
        return None
    return await discover_authorization_server_metadata(resource_url)


async def register_dynamic_client(registration_endpoint: str, redirect_uri: str, client_name: str) -> DcrResponse | None:
    """RFC 7591: attempt dynamic client registration. Returns None on any failure."""
    request = DcrRequest(client_name=client_name, redirect_uris=[redirect_uri])
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(registration_endpoint, json=request.model_dump(), timeout=_HTTP_TIMEOUT) as resp,
        ):
            if resp.status not in (200, 201):
                return None
            data = await resp.json(content_type=None)
            return DcrResponse.model_validate(data)
    except Exception:
        logger.debug("Dynamic client registration failed at %s", registration_endpoint, exc_info=True)
        return None


def generate_pkce_pair() -> tuple[str, str]:
    """Generate a PKCE (code_verifier, code_challenge) pair using S256 only (no `plain`)."""
    code_verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


def build_authorize_url(
    authorization_endpoint: str,
    client_id: str,
    redirect_uri: str,
    state: str,
    code_challenge: str,
    resource: str,
    scope: str | None = None,
) -> str:
    """Build the authorization request URL to open in the admin's browser."""
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "resource": resource,
    }
    if scope:
        params["scope"] = scope
    return f"{authorization_endpoint}?{urlencode(params)}"


def _raise_token_error(detail: str) -> NoReturn:
    msg = f"Token request failed: {detail}"
    raise McpOAuthError(msg)


async def _post_token_request(token_endpoint: str, form: dict[str, str]) -> TokenResponse:
    try:
        async with aiohttp.ClientSession() as session, session.post(token_endpoint, data=form, timeout=_HTTP_TIMEOUT) as resp:
            body = await resp.json(content_type=None)
            if resp.status != 200:
                error = (body or {}).get("error", f"HTTP {resp.status}") if isinstance(body, dict) else f"HTTP {resp.status}"
                _raise_token_error(error)
            return TokenResponse.model_validate(body)
    except McpOAuthError:
        raise
    except Exception as exc:
        _raise_token_error(str(exc))


async def exchange_code_for_token(
    token_endpoint: str,
    code: str,
    redirect_uri: str,
    client_id: str,
    client_secret: str | None,
    code_verifier: str,
    resource: str,
) -> TokenResponse:
    """Exchange an authorization code for tokens."""
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": code_verifier,
        "resource": resource,
    }
    if client_secret:
        form["client_secret"] = client_secret
    return await _post_token_request(token_endpoint, form)


async def refresh_access_token(
    token_endpoint: str,
    refresh_token: str,
    client_id: str,
    client_secret: str | None,
    resource: str,
) -> TokenResponse:
    """Exchange a refresh token for a new access token."""
    form = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "resource": resource,
    }
    if client_secret:
        form["client_secret"] = client_secret
    return await _post_token_request(token_endpoint, form)
