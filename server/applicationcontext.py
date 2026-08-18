# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Application Content module."""

import asyncio
import logging
import time

from server.config import AppSettings
from server.services_manager import ServicesManager

from .endpointregistry import EndpointRegistry
from .serviceprovider import ServiceProvider, ServiceRawConfig

logger = logging.getLogger("uvicorn.error")


class ApplicationContext:
    def __init__(
        self,
        endpoint_registry: EndpointRegistry,
        config: AppSettings,
        service_provider: ServiceProvider,
        services_manager: ServicesManager,
    ):
        self.endpoint_registry = endpoint_registry
        self.config = config
        self.service_provider = service_provider
        self.services_manager = services_manager
        self.allocated_ports = set[int]()

    async def _load_service(self, service_id: str, service_cfg: ServiceRawConfig) -> None:
        """Load single service."""
        start = time.time()
        logger.info(f"{service_id} loading...")  # noqa: G004
        if service_id not in self.services_manager.services:
            logger.warning(f"{service_id} is present in the persisted config but is not registered in this build")  # noqa: G004
            await self._record_warning_safely(
                service_id, f"Service '{service_id}' is present in the persisted config but is not registered in this build."
            )
            return
        try:
            await self.services_manager.load_service(service_id, service_cfg)
            logger.info(f"{service_id} fully loaded in {round(time.time() - start, 1)}s")  # noqa: G004
        except Exception as exc:
            logger.exception(f"{service_id} error occurs during loading {round(time.time() - start, 1)}s")  # noqa: G004
            await self._record_warning_safely(service_id, f"Service '{service_id}' failed to load: {exc}")
            return
        await self._dismiss_warnings_safely(service_id)

    async def _record_warning_safely(self, service_id: str, message: str) -> None:
        """Record a warning, but never let a failure to record one abort startup or masquerade as a load failure."""
        try:
            await self.service_provider.add_warning(service_id, message)
        except Exception:
            logger.exception(f"{service_id} failed to record warning")  # noqa: G004

    async def _dismiss_warnings_safely(self, service_id: str) -> None:
        """Dismiss recovered warnings, but never let a failure here be reported as a load failure."""
        try:
            await self.service_provider.dismiss_warnings_matching(service_id)
        except Exception:
            logger.exception(f"{service_id} failed to dismiss warnings")  # noqa: G004

    async def load_services(self) -> None:
        """Load all service from bootstrap."""
        info = await self.service_provider.load()
        tasks = [asyncio.create_task(self._load_service(service_id, service_cfg)) for service_id, service_cfg in info["services"].items()]
        await asyncio.gather(*tasks)


def get_base_url(host: str, port: int) -> str:
    """Get base url."""
    return f"http://{host}:{port}"
