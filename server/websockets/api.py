# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Webscoket endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket

from server.websockets.dependencies import get_infra_websocket_server
from server.websockets.infra_websocket_server import InfraWebsocketServer

router = APIRouter()


@router.websocket("/ws")
async def ws(
    ws: WebSocket,
    infra_websocket_server: Annotated[InfraWebsocketServer, Depends(get_infra_websocket_server)],
) -> None:
    """Websocket endpoint."""
    await infra_websocket_server.connect(ws)
