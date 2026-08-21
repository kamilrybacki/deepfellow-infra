# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Websocket Manager for subinfras."""

import json
import logging
from typing import TYPE_CHECKING

from server.config import AppSettings
from server.models.mesh import CheckMeshConnection
from server.task_manager import TaskManager
from server.utils.core import OneTimeKey
from server.utils.exceptions import ApiError
from server.utils.json_rpc_client import JsonRpcClient
from server.websockets.infra_client import InfraClient
from server.websockets.models import (
    AncestorInfo,
    AncestorsNotification,
    InitRequest,
    TopologyUpdateRequest,
    UpdateModelsRequest,
    UsageChangeRequest,
    WarmChangeRequest,
)
from server.websockets.websocket_client import WebSocketClient

if TYPE_CHECKING:
    from collections.abc import Callable

    from server.endpointregistry import EndpointRegistry
    from server.models.api import Model

logger = logging.getLogger("uvicorn.error")


class ParentInfra(WebSocketClient):
    endpoint_registry: "EndpointRegistry"

    def __init__(self, config: AppSettings, task_manager: TaskManager, url: str = ""):
        self.config = config
        self.task_manager = task_manager
        self.client = JsonRpcClient(send=lambda x: self._send(x), timeout=30)
        self.infra_client = InfraClient(self.client)
        self.parent_url = url or config.connect_to_mesh_url
        self.enabled = self.parent_url != ""
        self.one_time_key = OneTimeKey()
        self._ancestors: list[AncestorInfo] = []
        self._registered_ancestor_models: dict[str, list[Model]] = {}
        self.on_ancestors_changed: Callable[[], None] = lambda: None
        self._warm_change_unsupported = False
        self.get_children: Callable[[], dict[str, TopologyUpdateRequest]] = dict
        uri = f"{self.parent_url}/ws" if self.enabled else ""
        super().__init__(uri)

    @property
    def ancestors(self) -> list[AncestorInfo]:
        """Ordered list of ancestors received during init handshake (parent first)."""
        return self._ancestors

    async def _send(self, data: str) -> None:
        if not self.enabled or not self.ws:
            raise RuntimeError("Not connected")
        self.send(data)

    def on_message(self, msg: str | bytes) -> None:
        """Perform action on new message.

        An unsolicited `ancestors_update` push (see `AncestorsNotification`) is handled directly;
        everything else is assumed to be a JSON-RPC response to one of our own pending requests
        and handed to `JsonRpcClient.resolve`, which does its own parsing/validation.
        """
        try:
            obj = json.loads(msg)
        except Exception:
            self.client.resolve(msg)
            return
        if isinstance(obj, dict) and obj.get("type") == "ancestors_update":
            self._apply_ancestors(AncestorsNotification.model_validate(obj).ancestors)
            return
        self.client.resolve(msg)

    def on_disconnect(self) -> None:
        """Perform action on disconnect."""
        self.client.clear()
        self._apply_ancestors([])

    def _apply_ancestors(self, new_ancestors: list[AncestorInfo]) -> None:
        """Register/unregister proxy endpoints for models exposed by ancestors, and notify children.

        Ancestor models are registered with `owned_by="mesh-ancestor"` so `EndpointRegistry.list_models()`
        excludes them from what gets reported back upward (to the very ancestor that sent them) or
        re-offered downward as if they were this node's own.
        """
        self._ancestors = new_ancestors
        new_by_url = {a.url: a for a in new_ancestors}
        for url, prev_models in list(self._registered_ancestor_models.items()):
            if url not in new_by_url:
                self.endpoint_registry.update_models(prev_models, [], url, "", owned_by="mesh-ancestor")
                del self._registered_ancestor_models[url]
        for ancestor in new_ancestors:
            prev_models = self._registered_ancestor_models.get(ancestor.url, [])
            self.endpoint_registry.update_models(prev_models, ancestor.models, ancestor.url, ancestor.api_key, owned_by="mesh-ancestor")
            self._registered_ancestor_models[ancestor.url] = ancestor.models
        self.on_ancestors_changed()

    async def before_loop(self) -> bool:
        """Load models."""
        return self.enabled

    async def on_start(self) -> None:
        """On start functions."""
        self._warm_change_unsupported = False  # re-probe on each (re)connect in case the peer was upgraded
        try:
            response = await self.infra_client.init(
                InitRequest(
                    auth=self.config.connect_to_mesh_key.get_secret_value(),
                    name=self.config.name,
                    url=self.config.infra_url,
                    api_key=self.config.infra_api_key.get_secret_value(),
                    models=self.endpoint_registry.list_models(),
                    children=self.get_children(),
                    check_key=self.one_time_key.key,
                )
            )
            self._apply_ancestors(response.ancestors)
        except ApiError as e:
            if e.code == 2 and e.message == "Invalid api key":
                self.process_loop = False
            raise

    def send_usage(self, usage: UsageChangeRequest) -> None:
        """Send usage."""
        if not self.enabled or not self.ws:
            return

        self.task_manager.add_task_safe(self.infra_client.usage_change(usage), "parent_infra.usage_change")

    def send_warm(self, warm: WarmChangeRequest) -> None:
        """Send warm-state change.

        Older peers that predate `warm_change` reply with a JSON-RPC `-32601 Method not found`
        error for every call; once that's seen, stop calling `warm_change` on that peer until
        the next reconnect instead of logging a fresh error on every ~15s warmth poll.
        """
        if not self.enabled or not self.ws or self._warm_change_unsupported:
            return

        async def _send_warm() -> None:
            try:
                await self.infra_client.warm_change(warm)
            except ApiError as e:
                if e.code != -32601:
                    raise
                self._warm_change_unsupported = True
                logger.warning(
                    "Mesh peer %r does not support warm_change (older DeepFellow version); "
                    "no longer sending warm-state updates to it until reconnect.",
                    self.parent_url,
                )

        self.task_manager.add_task_safe(_send_warm(), "parent_infra.warm_change")

    def send_models_list(self) -> None:
        """Send usage."""
        if not self.enabled or not self.ws:
            return

        self.task_manager.add_task_safe(
            self.infra_client.update_models(UpdateModelsRequest(models=self.endpoint_registry.list_models())),
            "parent_infra.update_models",
        )

    def check_subinfra_connection(self, model: CheckMeshConnection) -> bool:
        """Check if given sub infra connection data is valid."""
        api_key = self.config.infra_api_key.get_secret_value()
        return self.one_time_key.check(model.connection_verifier) and api_key == model.infra_api_key
