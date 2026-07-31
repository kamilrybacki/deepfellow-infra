# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Others API."""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from server.core.dependencies import auth_server
from server.dynamic_config import get_infra_version

router = APIRouter(tags=["Utils"])


class InfoResponse(BaseModel):
    """Response of the `/info` endpoint."""

    version: str


@router.get("/health", summary="Checks if the server is running.")
async def health() -> str:
    """Check if the server is running."""
    return "OK"


@router.get("/info", summary="Returns information about the running Infra instance.")
async def info(_: Annotated[str, Depends(auth_server)]) -> InfoResponse:
    """Return the running Infra version."""
    return InfoResponse(version=get_infra_version())
