# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Base service."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import HTTPException

from server.models.models import (
    AddCustomModelIn,
    CustomModelId,
    CustomModelSpecification,
    InstallModelIn,
    InstallModelOut,
    ListModelsFilters,
    ListModelsOut,
    RetrieveModelOut,
    UninstallModelIn,
)
from server.models.services import (
    CatalogRefreshOut,
    InstallServiceIn,
    InstallServiceOut,
    InstallServiceProgress,
    RetrieveServiceOut,
    ServiceOptions,
    ServiceSize,
    ServiceSpecification,
    UninstallServiceIn,
)
from server.serviceprovider import ServiceRawConfig
from server.utils.core import PromiseWithProgress, StreamChunk
from server.utils.registry_client import RegistryUnavailableError, image_without_registry_prefix, registry_for

logger = logging.getLogger("uvicorn.error")


class BaseService(ABC):
    instances_info: dict[str, Any]
    is_cloud: bool = False

    def get_id(self, instance: str) -> str:
        """Return the service id."""
        if instance == "default":
            return self.get_type()
        return f"{self.get_type()}|{instance}"

    def get_service_id(self, instance: str) -> str:
        """Return the service id in form usable in a filesystem path name."""
        if instance == "default":
            return self.get_type()
        return f"{self.get_type()}-{instance}"

    @abstractmethod
    def get_type(self) -> str:
        """Return the service type."""

    @abstractmethod
    def get_description(self) -> str:
        """Return the service description."""

    @abstractmethod
    def get_size(self) -> ServiceSize:
        """Return the service size."""

    @abstractmethod
    def get_spec(self) -> ServiceSpecification:
        """Return the service specification."""

    @abstractmethod
    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        """Return the custom model specification or None if custom model is not supported."""

    def get_custom_model_definition(self, custom_model_id: CustomModelId) -> dict[str, Any] | None:  # noqa: ARG002
        """Return the stored spec a custom model was created from, or None if unavailable."""
        return None

    def get_info(self, instance: str) -> RetrieveServiceOut:
        """Return the service info."""
        return RetrieveServiceOut(
            id=self.get_id(instance),
            type=self.get_type(),
            instance=instance,
            description=self.get_description(),
            installed=self.get_installed_info(instance),
            downloaded=self.get_downloaded(),
            spec=self.get_spec(),
            size=self.get_size(),
            custom_model_spec=self.get_custom_model_spec(),
            has_docker=self.service_has_docker(),
            is_cloud=self.is_cloud_service(),
            disabled_reason=self.get_hardware_unsupported_reason(),
            catalog_refresh_unavailable_reason=self.get_catalog_refresh_unavailable_reason(),
        )

    def get_catalog_refresh_unavailable_reason(self) -> str | None:
        """Return why this service's catalog refresh can't run right now, or None if it can.

        None by default (no catalog refresh concept, or nothing blocking it). Services backing a
        catalog refresh with the static-directory writability capability flag (vLLM, llama.cpp,
        SGLang) override this to surface that flag to the WebUI.
        """
        return None

    @abstractmethod
    def get_instance_install_progress(self, instance: str) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Return actually installing instance."""

    @abstractmethod
    async def get_model_install_progress(self, instance: str, model: str) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Return actually installing models."""

    async def cancel_model_install(self, instance: str, model_id: str) -> None:  # noqa: ARG002
        """Cancel an in-progress model install. Services without install progress cannot cancel."""
        raise HTTPException(405, f"Service for model {model_id} does not support cancelling an installation.")

    async def refresh_catalog(self) -> PromiseWithProgress[CatalogRefreshOut, StreamChunk]:
        """Refresh the model catalog from an external library API. Services without a catalog cannot refresh."""
        raise HTTPException(405, "This service does not support catalog refresh")

    @abstractmethod
    def is_installed(self, instance: str) -> bool:
        """Check whether instance is installed."""

    @abstractmethod
    def get_installed_info(self, instance: str) -> bool | InstallServiceProgress | ServiceOptions:
        """Get service installed info."""

    @abstractmethod
    def get_downloaded(self) -> bool:
        """Get service downloaded info."""

    def service_has_docker(self) -> bool:
        """Return true when docker is started when service is installed."""
        return False

    def is_cloud_service(self) -> bool:
        """Return true when this service uses external cloud APIs."""
        return self.is_cloud

    def get_hardware_unsupported_reason(self) -> str | None:
        """Return why this service can't run on the current hardware, or None if it can. None by default."""
        return None

    @abstractmethod
    async def load_service(self, config: ServiceRawConfig) -> None:
        """Load service using the config."""

    @abstractmethod
    async def install_instance(self, instance: str, options: InstallServiceIn) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Install the service."""

    @abstractmethod
    async def update_instance(self, instance: str, options: InstallServiceIn) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Update the installed service configuration."""

    @abstractmethod
    async def uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        """Uninstall the instance."""

    @abstractmethod
    async def list_models(self, input_instance: str | list[str] | None, filters: ListModelsFilters) -> ListModelsOut:
        """List models."""

    @abstractmethod
    async def get_model(self, instance: str, model_id: str) -> RetrieveModelOut:
        """Get the model."""

    @abstractmethod
    async def install_model(
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Install the model."""

    @abstractmethod
    async def uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        """Uninstall the model."""

    @abstractmethod
    async def add_custom_model(self, instance: str, options: AddCustomModelIn) -> CustomModelId:
        """Add custom model."""

    @abstractmethod
    async def remove_custom_model(self, instance: str, custom_model_id: CustomModelId) -> None:
        """Remove custom model."""

    async def update_custom_model(self, instance: str, custom_model_id: CustomModelId, options: AddCustomModelIn) -> None:  # noqa: B027
        """Update custom model. No-op by default; override to support."""

    async def edit_model(
        self,
        instance: str,  # noqa: ARG002
        custom_model_id: CustomModelId,  # noqa: ARG002
        new_definition: AddCustomModelIn,  # noqa: ARG002
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk] | None:
        """Edit a custom-backed model's definition, uninstalling and reinstalling it if installed.

        Services without custom model support raise 405 by default.
        """
        raise HTTPException(405, "This service does not support editing custom models.")

    async def edit_model_install_options(
        self,
        instance: str,  # noqa: ARG002
        model_id: str,  # noqa: ARG002
        new_options: InstallModelIn,  # noqa: ARG002
    ) -> tuple[bool, PromiseWithProgress[InstallModelOut, StreamChunk]]:
        """Edit a model's install-time options only (no persisted definition change), reinstalling it.

        Services without this concept raise 405 by default.
        """
        raise HTTPException(405, "This service does not support editing model install options.")

    async def get_duplicate_spec(self, instance: str, model_id: str) -> dict[str, Any]:  # noqa: ARG002
        """Return a full add-model spec synthesized from an existing model, for duplication.

        Works for both custom-backed models (their stored definition) and catalog models (synthesized
        from the live registered model object). Services without this concept raise 405 by default.
        """
        raise HTTPException(405, "This service does not support duplicating models.")

    async def sync_models(self, instance: str) -> None:  # noqa: B027
        """Trigger an immediate model sync. No-op for services without sync support."""

    async def get_docker_tags(self, hardware: str | None) -> list[str]:  # noqa: ARG002
        """Return available Docker image tags for this service. Empty list by default."""
        return []

    def get_default_docker_tag(self, hardware: str | None) -> str | None:  # noqa: ARG002
        """Return the pinned default Docker image tag used when no version is selected. None by default."""
        return None

    def get_docker_image_repo(self, hardware: str | None) -> str | None:  # noqa: ARG002
        """Return the Docker repo (no tag) that get_docker_tags fetches from for the given hardware.

        None by default, meaning the repo is the same regardless of hardware (falls back to the
        spec field's static `docker_image`). Override when different hardware variants live in
        different repos, so the reported repo actually matches where the tags came from.
        """
        return None

    def filter_docker_tags(self, tags: list[str], hardware: str | None) -> list[str]:  # noqa: ARG002
        """Filter tags for the given hardware variant. Passthrough by default."""
        return tags

    async def get_docker_tags_for_model(self, model_id: str | None, hardware: str | None) -> list[str]:  # noqa: ARG002
        """Return available Docker image tags, optionally scoped to a specific model_id.

        Delegates to get_docker_tags() by default; override for services (like CustomService) that
        install several distinct models — each backed by a different image — under one service id.
        """
        return await self.get_docker_tags(hardware)

    def get_default_docker_tag_for_model(self, model_id: str | None, hardware: str | None) -> str | None:  # noqa: ARG002
        """Return the pinned default Docker image tag, optionally scoped to a specific model_id."""
        return self.get_default_docker_tag(hardware)

    def get_docker_image_repo_for_model(self, model_id: str | None, hardware: str | None) -> str | None:  # noqa: ARG002
        """Return the Docker repo (no tag), optionally scoped to a specific model_id."""
        return self.get_docker_image_repo(hardware)

    def _resolve_docker_image_repo(self, hardware: str | None) -> str | None:
        """Return the repo (no tag) backing get_docker_tags(), falling back to the spec field's static docker_image."""
        return self.get_docker_image_repo(hardware) or next(
            (f.docker_image for f in self.get_spec().fields if f.type == "docker-tags" and f.docker_image), None
        )

    async def validate_docker_image_version(self, image_version: str | None, hardware: str | bool | None) -> None:
        """Raise 400 if image_version is set but isn't among the available tags for the given hardware.

        If the registry can't be reached, validation is skipped (fail-open) rather than blocking
        the install — but a successfully fetched, genuinely empty tag list still rejects any
        explicit image_version, since it means no tag is known to be valid.

        get_docker_tags() only returns the most recent tags (capped by the registry client), so a
        tag missing from that list isn't necessarily invalid — e.g. installing an older tag via the
        CLI. Before rejecting, check the registry directly for that specific tag's existence.
        """
        await self._validate_docker_image_version(
            image_version,
            hardware,
            get_tags=self.get_docker_tags,
            resolve_repo=self._resolve_docker_image_repo,
        )

    async def validate_docker_image_version_for_model(
        self, model_id: str | None, image_version: str | None, hardware: str | bool | None
    ) -> None:
        """Validate the same way as validate_docker_image_version(), but scoped to a specific model_id.

        For services (like CustomService) that install several distinct models — each backed by a
        different image — under one service id, get_docker_tags()/get_docker_image_repo() alone
        can't validate a model's image_version, since they aren't model-aware.
        """
        await self._validate_docker_image_version(
            image_version,
            hardware,
            get_tags=lambda hw: self.get_docker_tags_for_model(model_id, hw),
            resolve_repo=lambda hw: self.get_docker_image_repo_for_model(model_id, hw),
        )

    async def _validate_docker_image_version(
        self,
        image_version: str | None,
        hardware: str | bool | None,
        *,
        get_tags: Callable[[str | None], Awaitable[list[str]]],
        resolve_repo: Callable[[str | None], str | None],
    ) -> None:
        if not image_version:
            return
        if isinstance(hardware, bool):
            hardware = "GPU" if hardware else "CPU"
        try:
            tags = await get_tags(hardware)
        except RegistryUnavailableError:
            logger.warning("Skipping docker image tag validation for %r: registry unavailable", image_version)
            return
        if image_version in tags:
            return
        docker_image = resolve_repo(hardware)
        if docker_image:
            try:
                client = registry_for(docker_image)
                if await client.tag_exists(image_without_registry_prefix(docker_image), image_version):
                    return
            except RegistryUnavailableError:
                logger.warning("Skipping docker image tag existence check for %r: registry unavailable", image_version)
                return
        raise HTTPException(400, f"Docker image tag '{image_version}' is not available for this service")

    @abstractmethod
    async def get_docker_logs(self, instance: str, model_id: str | None) -> str:
        """Get docker logs."""

    @abstractmethod
    async def get_docker_compose_file(self, instance: str, model_id: str | None) -> str:
        """Get docker compose file."""

    @abstractmethod
    async def restart_docker(self, instance: str, model_id: str | None) -> None:
        """Get docker compose file."""

    async def get_loaded_model_info(self, instance: str) -> dict[str, int] | None:  # noqa: ARG002
        """Return {model_name: context_length} for models currently in VRAM. None if not applicable."""
        return None

    @abstractmethod
    async def stop_instance(self, instance: str) -> None:
        """Stop the service gracefully.

        This method is called during application shutdown when DF_STOP_CONTAINERS_ON_SHUTDOWN is enabled.
        Services with Docker containers should stop their containers here.
        Services without Docker (proxies, external services) can leave this as a no-op.
        """
