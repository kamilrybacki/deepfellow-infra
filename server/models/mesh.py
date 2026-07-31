# SPDX-License-Identifier: MIT

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
