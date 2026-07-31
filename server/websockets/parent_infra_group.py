# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Group of ParentInfra connections for multi-parent mesh topology."""

import asyncio
from typing import TYPE_CHECKING, Literal

from server.models.api import Model
from server.models.mesh import CheckMeshConnection
from server.websockets.models import AncestorInfo, TopologyUpdateRequest, UsageChangeRequest
from server.websockets.parent_infra import ParentInfra

if TYPE_CHECKING:
    from collections.abc import Callable

    from server.config import AppSettings
    from server.endpointregistry import EndpointRegistry
    from server.task_manager import TaskManager


class ParentInfraGroup:
    def __init__(self, parents: list[ParentInfra]):
        self.parents = parents
        self._endpoint_registry: EndpointRegistry | None = None
        self._get_children: Callable[[], dict[str, TopologyUpdateRequest]] = dict

    @property
    def enabled(self) -> bool:
        """Return True if any parent connection is enabled."""
        return any(parent.enabled for parent in self.parents)

    @property
    def ancestors(self) -> list[AncestorInfo]:
        """Deduplicated union of all ancestor chains (parent first)."""
        seen: set[str] = set()
        result: list[AncestorInfo] = []
        for parent in self.parents:
            if not parent.enabled:
                continue
            for ancestor in parent.ancestors:
                if ancestor.url not in seen:
                    seen.add(ancestor.url)
                    result.append(ancestor)
        return result

    @property
    def endpoint_registry(self) -> "EndpointRegistry | None":
        """Return the endpoint registry, if one has been assigned."""
        return self._endpoint_registry

    @endpoint_registry.setter
    def endpoint_registry(self, value: "EndpointRegistry") -> None:
        self._endpoint_registry = value
        for parent in self.parents:
            parent.endpoint_registry = value

    @property
    def get_children(self) -> "Callable[[], dict[str, TopologyUpdateRequest]]":
        """Return the callback used to fetch this group's children, for propagation to new parents."""
        return self._get_children

    @get_children.setter
    def get_children(self, value: "Callable[[], dict[str, TopologyUpdateRequest]]") -> None:
        self._get_children = value
        for parent in self.parents:
            parent.get_children = value

    def send_models_list(self) -> None:
        """Broadcast the models list to all parents."""
        for parent in self.parents:
            parent.send_models_list()

    def send_usage(self, usage: UsageChangeRequest) -> None:
        """Broadcast a usage change to all parents."""
        for parent in self.parents:
            parent.send_usage(usage)

    def send_topology_update(
        self,
        action: Literal["join", "leave"],
        url: str,
        name: str,
        models: list[Model],
        children: "dict[str, TopologyUpdateRequest] | None" = None,
    ) -> None:
        """Send a topology update to all enabled parents."""
        for parent in self.parents:
            if not parent.enabled or not parent.ws:
                # Dropped: on reconnect, on_start sends the full sub-tree atomically via InitRequest.children.
                continue
            parent.task_manager.add_task_safe(
                parent.infra_client.topology_update(
                    TopologyUpdateRequest(action=action, url=url, name=name, models=models, children=children or {})
                ),
                "infra_websocket_server.topology_update",
            )

    def check_subinfra_connection(self, model: CheckMeshConnection) -> bool:
        """Return True if any parent can reach the given subinfra model."""
        return any(parent.check_subinfra_connection(model) for parent in self.parents)

    async def run(self) -> None:
        """Run all parent connections concurrently."""
        if not self.parents:
            return
        await asyncio.gather(*[parent.run() for parent in self.parents])

    async def reconfigure(self, config: "AppSettings", task_manager: "TaskManager") -> None:
        """Rebuild the parent connection(s) from the current `config.connect_to_mesh_url`/key.

        Called by the `/admin/config` handler when those fields change, so the mesh websocket
        client reconnects to the new parent (or disconnects, if cleared) without a process restart.
        """
        await asyncio.gather(*[parent.stop() for parent in self.parents])

        new_parents = [ParentInfra(config, task_manager, config.connect_to_mesh_url)] if config.connect_to_mesh_url else []
        for parent in new_parents:
            parent.get_children = self._get_children
            if self._endpoint_registry is not None:
                parent.endpoint_registry = self._endpoint_registry

        self.parents = new_parents
        for parent in new_parents:
            task_manager.add_task_safe(parent.run(), "parent_infra_group.reconfigure")
