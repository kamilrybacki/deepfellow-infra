# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Models for models api."""

from typing import Any, Literal

from pydantic import BaseModel

from server.models.api import McpToolInfo, RegistrationId

type InstallModelOptions = dict[str, Any]
type CustomModelDefiniton = dict[str, Any]
type CustomModelId = str


class ModelIdQuery(BaseModel):
    model_id: str


class OneOfOption(BaseModel):
    value: str
    label: str


class ModelField(BaseModel):
    type: str
    name: str
    description: str
    default: str | None = None
    placeholder: str | None = None
    required: bool = True
    values: list[OneOfOption | str] | None = None
    required_keys: list[str] | None = None
    docker_image: str | None = None
    depends_on: str | None = None


class ModelSpecification(BaseModel):
    fields: list[ModelField]


class CustomModelField(BaseModel):
    type: str
    name: str
    description: str
    default: str | None = None
    placeholder: str | None = None
    required: bool = True
    values: list[str] | None = None
    display: str | None = None


class CustomModelSpecification(BaseModel):
    fields: list[CustomModelField]


class InstallModelIn(BaseModel):
    stream: bool = False
    ignore_warnings: bool = False
    spec: InstallModelOptions | None = None


class ModelInfo(BaseModel):
    spec: InstallModelOptions | None
    registration_id: RegistrationId


class InstallModelOut(BaseModel):
    status: Literal["OK"]
    details: str
    requires_oauth: bool = False


class UninstallModelIn(BaseModel):
    purge: bool = True


class UninstallModelOut(BaseModel):
    status: Literal["OK"]


class InstallModelProgress(BaseModel):
    stage: str
    value: float


class RetrieveModelOut(BaseModel):
    id: str
    service: str
    type: str
    installed: bool | InstallModelProgress | ModelInfo
    downloaded: bool
    custom: CustomModelId | None = None
    default_prefix: str | None = None
    # The prefix this model is actually reachable on - the live install-time prefix when installed,
    # else `default_prefix`. `edit_model_install_options` can move a model's install-time prefix
    # independently of `default_prefix`, so the two can disagree; collision checks must compare
    # against this field, not `default_prefix`, which only describes the declared definition.
    effective_prefix: str | None = None
    size: str
    spec: ModelSpecification
    has_docker: bool
    vram_estimate_gb: float | None = None
    is_loaded: bool | None = None
    variant: str | None = None
    command: str | None = None
    base_image: str | None = None
    custom_spec: dict[str, Any] | None = None
    description: str | None = None
    repository_url: str | None = None


class ListModelsFilters(BaseModel):
    installed: bool | None = None


class ListModelsOut(BaseModel):
    list: list[RetrieveModelOut]


class AddCustomModelIn(BaseModel):
    spec: CustomModelDefiniton


class AddCustomModelOut(BaseModel):
    custom_model_id: CustomModelId


class RemoveCustomModelOut(BaseModel):
    status: Literal["OK"]


class UpdateCustomModelOut(BaseModel):
    status: Literal["OK"]


class EditModelOut(BaseModel):
    status: Literal["OK"]
    reinstalled: bool


class DuplicateSpecOut(BaseModel):
    spec: dict[str, Any]


class SyncModelsOut(BaseModel):
    status: Literal["OK"]


class CancelModelInstallOut(BaseModel):
    status: Literal["OK"]


class McpHealthCheckResult(BaseModel):
    healthy: bool
    transport: Literal["streamable_http", "sse"] | None = None
    tools: list[McpToolInfo] = []
    error: str | None = None
    requires_oauth: bool = False
