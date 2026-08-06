# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""MCP API."""

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel

from server.core.dependencies import auth_admin
from server.utils.mcp_json_converter import ConvertedMcpConfig, McpJsonConvertError, parse_mcp_json_config

router = APIRouter(prefix="/admin/mcp", tags=["MCP"])


class ConvertMcpJsonIn(BaseModel):
    config: dict[str, Any]


@router.post("/convert-config", summary="Convert a standard MCP client JSON config into DeepFellow's custom-model parameters.")
async def convert_mcp_config(
    body: Annotated[ConvertMcpJsonIn, Body()],
    _: Annotated[str, Depends(auth_admin)],
) -> ConvertedMcpConfig:
    """Parse a standard `{"mcpServers": {...}}` config and return the parameters needed to set up a custom MCP server."""
    try:
        return parse_mcp_json_config(body.config)
    except McpJsonConvertError as e:
        raise HTTPException(400, str(e)) from e
