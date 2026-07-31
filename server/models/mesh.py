# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Models for mesh api."""

from pydantic import BaseModel


class MeshInfoModel(BaseModel):
    name: str
    type: str


class MeshInfoInfra(BaseModel):
    name: str
    url: str
    models: list[MeshInfoModel]


class MeshInfo(BaseModel):
    connections: list[MeshInfoInfra]


class ShowMeshInfoOut(BaseModel):
    info: MeshInfo


class CheckMeshConnection(BaseModel):
    infra_api_key: str
    connection_verifier: str


class MeshTopologyNode(BaseModel):
    url: str
    name: str
    models: list[MeshInfoModel]
    you_are_here: bool = False
    children: list["MeshTopologyNode"]
