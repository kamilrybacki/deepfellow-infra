# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""MCP OAuth API.

Two separate routers, kept deliberately apart:

- `router`: admin-only endpoints (`Depends(auth_admin)`) to start an OAuth flow and poll its status.
- `callback_router`: the OAuth provider's browser redirect target. It carries **no auth dependency
  at all** — the authorization server calls it directly from the admin's browser, which never has a
  DeepFellow admin API key. This is completely separate from (and must never be conflated with)
  DeepFellow's own `auth_admin`/`auth_server` bearer-token gate on its own API.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from server.core.dependencies import auth_admin, get_services_manager
from server.services.mcp_service import McpOAuthStatusOut, McpService, get_mcp_service_instance, render_oauth_callback_page
from server.services_manager import ServicesManager

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/admin/services/{service_id}/models/{model_id}/oauth", tags=["MCP OAuth"])
callback_router = APIRouter(prefix="/mcp-oauth", tags=["MCP OAuth"])


class McpOAuthStartOut(BaseModel):
    authorize_url: str


@router.post("/start", summary="Begin an OAuth authorization flow for a proxy MCP model.")
async def start_mcp_oauth(
    _: Annotated[str, Depends(auth_admin)],
    model_id: Annotated[str, Path(description="The ID of the proxy MCP model.")],
    mcp: Annotated[tuple[McpService, str], Depends(get_mcp_service_instance)],
) -> McpOAuthStartOut:
    """Discover the OAuth server, register a client if needed, and return the URL to open in a browser."""
    service, instance = mcp
    authorize_url = await service.start_oauth_flow(instance, model_id)
    return McpOAuthStartOut(authorize_url=authorize_url)


@router.get("/status", summary="Get the OAuth status for a proxy MCP model.")
async def get_mcp_oauth_status(
    _: Annotated[str, Depends(auth_admin)],
    model_id: Annotated[str, Path(description="The ID of the proxy MCP model.")],
    mcp: Annotated[tuple[McpService, str], Depends(get_mcp_service_instance)],
) -> McpOAuthStatusOut:
    """Return non-secret OAuth status (never a client_secret, access_token, or refresh_token)."""
    service, instance = mcp
    return service.get_oauth_status(instance, model_id)


@callback_router.get(
    "/callback",
    summary="OAuth redirect callback (unauthenticated by design — see module docstring).",
    include_in_schema=False,
)
async def mcp_oauth_callback(
    services_manager: Annotated[ServicesManager, Depends(get_services_manager)],
    state: Annotated[str | None, Query()] = None,
    code: Annotated[str | None, Query()] = None,
    error: Annotated[str | None, Query()] = None,
) -> HTMLResponse:
    """Handle the authorization server's redirect back to DeepFellow."""
    service = services_manager.services.get("mcp")
    if not isinstance(service, McpService) or not state:
        return HTMLResponse(render_oauth_callback_page("Invalid OAuth callback: missing or unrecognized state."))
    message = await service.complete_oauth_callback(state, code, error)
    return HTMLResponse(render_oauth_callback_page(message))
