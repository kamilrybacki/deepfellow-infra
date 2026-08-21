# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Infra client."""

from typing import Literal

from pydantic import BaseModel, Field

from server.models.api import Model, RegistrationId


class AncestorInfo(BaseModel):
    url: str
    name: str
    api_key: str = ""
    models: list[Model] = []


class InitResponse(BaseModel):
    ancestors: list[AncestorInfo]


class AncestorsNotification(BaseModel):
    """Unsolicited push from a parent to an already-connected child.

    Tells the child that the set of ancestor-exposed models (parent's own + parent's own
    ancestors) has changed. Sent as a bare JSON object over the same websocket used for the
    request/response JSON-RPC
    traffic, distinguished from a JSON-RPC response by the `type` discriminator (a JSON-RPC
    response never has a `type` field).
    """

    type: Literal["ancestors_update"] = "ancestors_update"
    ancestors: list[AncestorInfo] = []


class TopologyUpdateRequest(BaseModel):
    action: Literal["join", "leave"]
    url: str
    name: str = ""
    models: list[Model] = []
    children: dict[str, "TopologyUpdateRequest"] = {}


class InitRequest(BaseModel):
    auth: str
    name: str
    url: str
    api_key: str
    models: list[Model]
    children: dict[str, TopologyUpdateRequest] = {}
    check_key: str = Field(min_length=1)


class UsageChangeRequest(BaseModel):
    id: RegistrationId
    usage: int


class WarmChangeRequest(BaseModel):
    id: RegistrationId
    warm: bool


class UpdateModelsRequest(BaseModel):
    models: list[Model]
