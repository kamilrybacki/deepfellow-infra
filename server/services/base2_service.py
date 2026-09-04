# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Base2 service."""

import asyncio
import logging
import shutil
import time
import uuid
from abc import abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, cast

from fastapi import HTTPException
from pydantic import BaseModel

from server.config import AppSettings
from server.docker import (
    DockerImage,
    DockerOptions,
    DockerService,
)
from server.endpointregistry import EndpointRegistry
from server.models.models import (
    AddCustomModelIn,
    CustomModelDefiniton,
    CustomModelId,
    InstallModelIn,
    InstallModelOut,
    InstallModelProgress,
    ModelField,
    UninstallModelIn,
)
from server.models.services import (
    InstallServiceIn,
    InstallServiceOut,
    InstallServiceProgress,
    OneOfOption,
    ServiceField,
    UninstallServiceIn,
)
from server.serviceprovider import ServiceProvider, ServiceRawConfig
from server.services.base_service import BaseService
from server.utils.core import (
    PromiseWithProgress,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    Utils,
    convert_size_to_bytes,
    is_own_cancellation,
)
from server.utils.hardware import GpuInfo, Hardware, HardwarePartInfo, NvidiaGpuInfo
from server.utils.model_downloader import ModelDownloader


class ModelConfig(BaseModel):
    model_id: str
    options: InstallModelIn
    definition: dict[str, Any] | None = None
    capacity: int | None = None
    capacity_known: bool = False
    gpu_memory_utilization: float | None = None


class CustomModel(BaseModel):
    id: CustomModelId
    data: CustomModelDefiniton


_CUSTOM_MODEL_NON_SERVING_FIELDS = {"id", "size"}


def _custom_model_serving_fields_changed(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """Return True if any field affecting whether the repo can be served differs between specs.

    `id` (display name) and `size` are cosmetic/derived and don't affect serviceability, so an
    update that only touches those shouldn't force re-validation of a model that already passed
    it - which would otherwise be able to permanently strand a model added before validation
    existed, or during a transient HuggingFace outage.

    Values are compared with falsy values normalized to `None` first, so a field missing from an
    older spec (e.g. `revision`, or `skip_validation` on a model added before that field existed)
    doesn't register as "changed" just because the current caller submits it explicitly as `""` or
    `False` - which would otherwise defeat the point of this check for exactly the legacy models it
    exists to protect.
    """
    keys = (set(old) | set(new)) - _CUSTOM_MODEL_NON_SERVING_FIELDS
    return any((old.get(k) or None) != (new.get(k) or None) for k in keys)


class InstanceConfig(BaseModel):
    options: InstallServiceIn | None = None
    models: list[ModelConfig] | None = None
    custom: list[CustomModel] | None = None


class ServiceConfig(BaseModel):
    instances: dict[str, InstanceConfig] | None = None
    downloaded: dict[str, Any] | None = None
    service_downloaded: bool | None = True


InstalledInfoType = TypeVar("InstalledInfoType")
DownloadInfoType = TypeVar("DownloadInfoType")


class _HasModels(Protocol):
    """Structural view of every service's ``InstalledInfo``: it always carries a ``models`` mapping."""

    models: dict[str, Any]


logger = logging.getLogger("uvicorn.error")


class InstallingModel:
    """Tracks one model install.

    Reserved as soon as `install_model()` decides to start one, resolved once the service's
    `_install_model()` actually produces the real promise - so a concurrent caller can find the
    reservation and wait for the real promise instead of racing `_install_model()` a second time.
    """

    last_chunk: StreamChunk | None = None

    def __init__(self) -> None:
        self.promise: PromiseWithProgress[InstallModelOut, StreamChunk] | None = None
        self.chained_promise: PromiseWithProgress[InstallModelOut, StreamChunk] | None = None
        self.task: asyncio.Task[None] | None = None
        self.pending_task: asyncio.Task[PromiseWithProgress[InstallModelOut, StreamChunk]] | None = None
        self._ready = asyncio.Event()
        self._error: BaseException | None = None

    def resolve(
        self,
        promise: PromiseWithProgress[InstallModelOut, StreamChunk],
        on_success: Callable[[InstallModelOut], Awaitable[InstallModelOut]],
        on_error: Callable[[Exception], None],
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Attach the real install promise once it exists, unblocking any concurrent waiters.

        Chains the one-time post-install bookkeeping (`on_success`/`on_error`) onto the real promise
        exactly once here, rather than letting each caller of `install_model()` chain its own copy -
        every caller (the original one and any that joined this reservation) shares the single
        resulting promise, so bookkeeping like persisting config or recording a warning runs once per
        real install, not once per concurrent caller.
        """
        self.promise = promise
        self.chained_promise = promise.next(on_success, on_error)

        async def the_func() -> None:
            async for chunk in promise.progress.as_generator():
                self.last_chunk = chunk

        self.task = asyncio.create_task(the_func())
        self._ready.set()
        return self.chained_promise

    def reject(self, error: BaseException) -> None:
        """Unblock any concurrent waiters with a failure - no promise will ever exist for this reservation.

        A raw `CancelledError` is converted to a plain `HTTPException` before being stored: it's
        only meaningful as a real cancellation to whichever task actually owned it (that task sees
        the original error via its own `raise`, untouched by this) - an unrelated waiter (a
        progress-watcher, a concurrent joiner) reattaching later just needs a clean, ordinary error
        to turn into a response, not a `BaseException` most frameworks won't handle gracefully.
        """
        if isinstance(error, asyncio.CancelledError):
            error = HTTPException(409, "Model install was cancelled")
        self._error = error
        self._ready.set()

    async def wait_ready(self) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Wait until the real install promise is available, then return it.

        Re-raises the original failure if the reservation was rejected instead of resolved, so a
        concurrent caller learns the real outcome instead of hanging forever.
        """
        await self._ready.wait()
        if self._error is not None:
            raise self._error
        assert self.promise is not None
        return self.promise

    async def wait_chained(self) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Wait until the reservation resolves, then return the single shared post-bookkeeping promise."""
        await self._ready.wait()
        if self._error is not None:
            raise self._error
        assert self.chained_promise is not None
        return self.chained_promise


class InstallingInstance:
    """Tracks one service-instance install/update.

    Reserved synchronously as soon as `install_instance()`/`update_instance()` decide to proceed,
    resolved once the service's `_install_instance()` actually produces the real promise - so the
    "already installing" guard, and any concurrent reader of the install progress, see "installing"
    for the whole duration instead of a false "not installing" while `_install_instance()` is still
    doing its (potentially slow) pre-promise setup work.

    `timeout_seconds` bounds how long a progress reader waits on `wait_ready()` before giving up -
    it's set by whichever caller (`install_instance()`/`update_instance()`) created the reservation,
    since they don't do the same amount of pre-promise work (see the two `_INSTANCE_*_STATUS_WAIT_TIMEOUT_SECONDS`
    constants below).
    """

    last_chunk: StreamChunk | None = None

    def __init__(self, timeout_seconds: float) -> None:
        self.promise: PromiseWithProgress[InstallServiceOut, StreamChunk] | None = None
        self.task: asyncio.Task[None] | None = None
        self.timeout_seconds = timeout_seconds
        self._ready = asyncio.Event()
        self._error: BaseException | None = None

    def resolve(self, promise: PromiseWithProgress[InstallServiceOut, StreamChunk]) -> None:
        """Attach the real install promise once it exists, unblocking any concurrent waiters."""
        self.promise = promise

        async def the_func() -> None:
            async for chunk in promise.progress.as_generator():
                self.last_chunk = chunk

        self.task = asyncio.create_task(the_func())
        self._ready.set()

    def reject(self, error: BaseException) -> None:
        """Unblock any concurrent waiters with a failure - no promise will ever exist for this reservation.

        A raw `CancelledError` is converted to a plain `HTTPException` before being stored: it's only
        meaningful as a real cancellation to whichever task actually owned it (that task sees the
        original error via its own `raise`, untouched by this) - an unrelated waiter (a progress
        reader) reattaching later just needs a clean, ordinary error to turn into a response, not a
        `BaseException` most frameworks won't handle gracefully.
        """
        if isinstance(error, asyncio.CancelledError):
            error = HTTPException(409, "Service instance install was cancelled")
        self._error = error
        self._ready.set()

    async def wait_ready(self) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Wait until the real install promise is available, then return it.

        Re-raises the original failure if the reservation was rejected instead of resolved, so a
        concurrent reader learns the real outcome instead of hanging forever.
        """
        await self._ready.wait()
        if self._error is not None:
            raise self._error
        if self.promise is None:
            raise RuntimeError("InstallingInstance became ready without a promise or an error - resolve()/reject() were never called")
        return self.promise


@dataclass
class Instance[InstalledInfoType]:
    installed: InstalledInfoType | None
    installing: InstallingInstance | None
    installing_model_progress: dict[str, InstallingModel]
    config: InstanceConfig


_LOG_CACHE_TTL = 8.0
_FAILED_LOG_CACHE_TTL = 1.0
_DOCKER_LOGS_TAIL_LINES = 1000  # generous for a startup capacity line; bounds memory on a long-running container
_RECONCILE_INTERVAL_SECONDS = 90
_RECONCILE_DEAD_THRESHOLD = 3
# Bounds how long a caller waits for a model install's pre-promise setup work (GPU/image/registry
# checks) to settle - not the install itself, which can legitimately run far longer than this and is
# awaited separately once a promise exists. A generous bound for setup work specifically, so a
# genuinely hung service doesn't leave every caller waiting forever with no way to notice or recover.
_INSTALL_STATUS_WAIT_TIMEOUT_SECONDS = 30
_CANCEL_WAIT_TIMEOUT_SECONDS = 30
# Bounds how long a caller waits for install_instance()'s pre-promise setup work (Docker
# image/registry checks) to settle - not the install itself, which can legitimately run far longer
# than this and is awaited separately once a promise exists. A generous bound for setup work
# specifically, so a genuinely hung service doesn't leave every poller waiting forever with no way
# to notice or recover.
_INSTANCE_INSTALL_STATUS_WAIT_TIMEOUT_SECONDS = 30
# update_instance()'s reservation doesn't resolve until _validate_update_options(), _uninstall_instance(),
# AND _install_instance() all complete - unlike install_instance(), its pre-promise phase includes real
# teardown work (stopping/removing containers, unregistering models), whose duration scales with what
# was installed rather than indicating a hang. Budgeted more generously so tearing down a busy instance
# doesn't trip a spurious timeout.
_INSTANCE_UPDATE_STATUS_WAIT_TIMEOUT_SECONDS = 120


class Base2Service(Generic[InstalledInfoType, DownloadInfoType], BaseService):  # noqa: UP046
    config: AppSettings
    endpoint_registry: EndpointRegistry
    service_provider: ServiceProvider
    model_downloader: ModelDownloader
    docker_service: DockerService
    models: dict[str, dict[str, Any]]
    models_downloaded: dict[str, DownloadInfoType]
    service_downloaded: bool
    hardware: Hardware
    instances_info: dict[str, Instance[InstalledInfoType]]
    images_download_progress: dict[str, Stream[StreamChunk]]
    models_download_progress: dict[str, Stream[StreamChunk]]
    _log_cache: dict[str, tuple[float, str, float]]
    _installing: set[tuple[str, str]]
    _reconciliation_tasks: dict[str, "asyncio.Task[None]"]
    _crash_poll_state: dict[tuple[str, str], int]
    _failed_models: dict[str, set[str]]
    _warning_tasks: set["asyncio.Task[None]"]

    _persists_model_definitions: bool = True
    """Override to False for services whose ``_generate_instance_config`` never populates ``ModelConfig.definition``
    (e.g. it has no registry snapshot to persist). Otherwise ``load_service`` treats every model as needing a
    backfill save on every startup, forever."""
    _edit_locks: dict[tuple[str, str], asyncio.Lock]

    def __init__(
        self,
        config: AppSettings,
        endpoint_registry: EndpointRegistry,
        service_provider: ServiceProvider,
        model_downloader: ModelDownloader,
        docker_service: DockerService,
        hardware: Hardware,
    ):
        super().__init__()
        self.config = config
        self.endpoint_registry = endpoint_registry
        self.service_provider = service_provider
        self.model_downloader = model_downloader
        self.docker_service = docker_service
        self.hardware = hardware
        self.models = {}
        self.models_downloaded = {}
        self.service_downloaded = False
        self.instances_info = {"default": Instance(None, None, {}, InstanceConfig())}
        self.models_download_progress = {}
        self.images_download_progress = {}
        self._log_cache = {}
        self._failed_models = {}
        self._warning_tasks = set()
        self._edit_locks = {}
        self._after_init()

    def _after_init(self) -> None:
        """Do some custom initialization."""

    def _init_reconciliation(self) -> None:
        self._reconciliation_tasks = {}
        self._crash_poll_state = {}

    def _start_reconciliation_task(self, instance: str) -> None:
        """Start (or restart) the background task that detects crashed/stopped model containers and releases their bookkeeping."""
        if instance in self._reconciliation_tasks:
            self._reconciliation_tasks[instance].cancel()

        async def _reconcile_loop() -> None:
            while True:
                await asyncio.sleep(_RECONCILE_INTERVAL_SECONDS)
                info = self.instances_info.get(instance)
                if not info or not info.installed:
                    break
                try:
                    await self._reconcile_instance_models(instance, info.installed)
                except Exception:
                    logger.exception(f"{self.get_id(instance)} reconciliation tick failed")  # noqa: G004

        self._reconciliation_tasks[instance] = asyncio.create_task(_reconcile_loop())

    def _stop_reconciliation_task(self, instance: str) -> None:
        if instance in self._reconciliation_tasks:
            self._reconciliation_tasks[instance].cancel()
            del self._reconciliation_tasks[instance]

    def _reconcile_container_name(self, instance: str, info: InstalledInfoType, model_info: Any) -> str | None:  # noqa: ANN401
        """Return the container name to inspect for a given model (None => skip this model).

        Per-model-container services (vLLM, llama.cpp) return the model's own container; Ollama returns the
        single shared instance container so a dead server releases every model loaded into it. Services that
        never call ``_start_reconciliation_task`` inherit this default, which is not reached in practice.
        """
        raise NotImplementedError

    async def _release_dead_model(self, instance: str, info: InstalledInfoType, model_id: str, model_info: Any) -> None:  # noqa: ANN401
        """Release the bookkeeping of a model whose container was confirmed dead.

        Implementations tear down whatever the service reserved (docker container, VRAM counter/cache),
        drop the model from ``info.models`` and unregister its endpoint. Raising signals a failed teardown:
        the caller keeps the crash-poll state and retries on the next tick instead of releasing prematurely.
        """
        raise NotImplementedError

    async def _reconcile_instance_models(self, instance: str, info: InstalledInfoType) -> None:
        """Detect models whose container is no longer running for good and release their VRAM bookkeeping."""
        for model_id, model_info in cast("_HasModels", info).models.copy().items():
            key = (instance, model_id)
            if key in self._installing:
                self._crash_poll_state.pop(key, None)
                continue

            container_name = self._reconcile_container_name(instance, info, model_info)
            if not container_name:
                self._crash_poll_state.pop(key, None)
                continue

            status = await self.docker_service.get_container_status(container_name)
            is_bad = not status.exists or status.state != "running" or status.health == "unhealthy"

            if not is_bad:
                self._crash_poll_state.pop(key, None)
                continue

            bad_count = self._crash_poll_state.get(key, 0) + 1
            self._crash_poll_state[key] = bad_count
            if bad_count < _RECONCILE_DEAD_THRESHOLD:
                continue

            if key in self._installing or cast("_HasModels", info).models.get(model_id) is not model_info:
                self._crash_poll_state.pop(key, None)
                continue  # uninstalled or reinstalled while we were polling

            try:
                await self._release_dead_model(instance, info, model_id, model_info)
            except Exception:
                # Keep bookkeeping intact and retry teardown on the next tick rather than releasing VRAM
                # for a container we couldn't confirm is actually gone.
                logger.exception(f"{self.get_id(instance)} failed to tear down dead container for model {model_id}, will retry")  # noqa: G004
                continue

            self._crash_poll_state.pop(key, None)
            logger.warning(f"{self.get_id(instance)} model {model_id} container is dead, released VRAM bookkeeping")  # noqa: G004

    async def _get_docker_logs(self, container_name: str) -> str:
        """Return raw docker logs output, with TTL caching per container.

        Tailed to the last `_DOCKER_LOGS_TAIL_LINES` lines: this is only ever used to find a
        single startup-time capacity line (see `_get_max_concurrency_from_logs`), and an
        untailed `docker logs` on a long-running container can read hundreds of MB into memory
        on every restart. A failed or empty read (non-zero exit code, or no output at all —
        e.g. a race right after the container restarts, before the backend has logged anything)
        is cached only briefly (`_FAILED_LOG_CACHE_TTL`) rather than for the full TTL, so a
        container stuck in that state doesn't spawn a `docker logs` subprocess on every poll
        tick, while a real startup log that lands moments later is still picked up quickly.
        """
        cached = self._log_cache.get(container_name)
        if cached and (time.monotonic() - cached[0]) < cached[2]:
            return cached[1]
        result = await Utils.run_command(["docker", "logs", "--tail", str(_DOCKER_LOGS_TAIL_LINES), container_name])
        raw = result.stdout + result.stderr
        if result.exit_code != 0:
            logger.warning("docker logs failed for %r (exit %d): %s", container_name, result.exit_code, result.stderr)
        ttl = _LOG_CACHE_TTL if result.exit_code == 0 and raw else _FAILED_LOG_CACHE_TTL
        self._log_cache[container_name] = (time.monotonic(), raw, ttl)
        return raw

    async def _get_max_concurrency_from_logs(
        self, container_name: str, backend_name: str, parse: Callable[[str], int | None]
    ) -> int | None:
        """Return a backend's reported max concurrency for a running container, or None if unavailable."""
        try:
            raw = await self._get_docker_logs(container_name)
        except Exception:
            logger.exception("Failed to read docker logs for %r; %s capacity will be unknown", container_name, backend_name)
            return None
        return parse(raw)

    async def _resolve_model_capacity(
        self,
        instance: str,
        model_id: str,
        backend_name: str,
        container_name: str,
        restarted: bool,
        parse: Callable[[str], int | None],
    ) -> tuple[int | None, bool]:
        """Determine a model's concurrency capacity, re-parsing logs only if the container was actually (re)started.

        A container that's still running from before this process started never rewrote its
        startup capacity line, so re-reading its logs can't recover it (see `_get_docker_logs`);
        the last persisted value is reused instead, which is safe because an unchanged config
        implies an unchanged real capacity. If there's no persisted value either - e.g. an
        orphan-adopted container (see `_try_adopt_orphaned_container`) that this process never
        started and has no prior config entry for - the startup line may still be sitting in the
        container's tailed log, so it's worth a best-effort read before giving up.
        """
        if restarted:
            self._log_cache.pop(container_name, None)
            capacity = await self._get_max_concurrency_from_logs(container_name, backend_name, parse)
            capacity_known = capacity is not None
            if capacity is None:
                logger.warning(
                    "Could not determine %s concurrency for %r from its startup logs after a real (re)start; "
                    "capacity will be treated as unknown and this instance will be deprioritized for routing.",
                    backend_name,
                    model_id,
                )
            return capacity, capacity_known

        capacity, capacity_known = self._get_persisted_model_capacity(instance, model_id)
        if capacity is not None or capacity_known:
            logger.debug("Container for %r wasn't (re)started; reusing last known %s concurrency %s.", model_id, backend_name, capacity)
            return capacity, capacity_known

        recovered = await self._get_max_concurrency_from_logs(container_name, backend_name, parse)
        if recovered is not None:
            logger.info(
                "Recovered %s concurrency %s for %r from its still-running container's logs (no prior known value).",
                backend_name,
                recovered,
                model_id,
            )
            return recovered, True

        logger.warning(
            "Could not determine %s concurrency for %r: container already running, no prior known value, "
            "and its startup capacity line is no longer in the tailed logs.",
            backend_name,
            model_id,
        )
        return capacity, capacity_known

    def load_default_models(self, instance: str) -> None:
        """Load default models to instance."""

    @abstractmethod
    def get_type(self) -> str:
        """Return the type."""

    @abstractmethod
    def get_description(self) -> str:
        """Return the service description."""

    def check_instance_exists(self, instance: str) -> None:
        """Check is instance exists."""
        if not self.instances_info.get(instance):
            raise HTTPException(404, "Instance doesn't exist.")

    def is_model_installed_in_other_instance(self, instance: str, model: str) -> bool:
        """Check is model installed in other instance."""
        return any(model in getattr(self.instances_info[i].installed, "models", []) for i in self.instances_info if i != instance)

    def get_instance_info(self, instance: str) -> Instance[InstalledInfoType]:
        """Return instance info."""
        instance_info = self.instances_info.get(instance)
        if not instance_info:
            raise HTTPException(404, "Instance doesn't exist.")
        return instance_info

    def _pop_installing_if_current(self, instance: str, model_id: str, reservation: InstallingModel) -> None:
        """Remove model_id's reservation if it's still this exact one and the instance still exists.

        No-ops otherwise instead of raising, since callers use this before reject()/resolve() and
        must not be blocked by either a superseded reservation or an already-uninstalled instance.
        """
        instance_info = self.instances_info.get(instance)
        if instance_info is None:
            return
        progress = instance_info.installing_model_progress
        if progress.get(model_id) is reservation:
            del progress[model_id]

    @staticmethod
    async def _wait_for_install_setup(
        model_id: str, awaitable: Awaitable[PromiseWithProgress[InstallModelOut, StreamChunk]]
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Await a reservation's pre-promise setup, converting a timeout into a clean 504."""
        try:
            return await asyncio.wait_for(awaitable, timeout=_INSTALL_STATUS_WAIT_TIMEOUT_SECONDS)
        except TimeoutError:
            raise HTTPException(504, f"Model {model_id} is still preparing to install; try again shortly.") from None

    async def get_model_install_progress(self, instance: str, model: str) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Return actually installing models."""
        installing = self.get_instance_info(instance).installing_model_progress.get(model)

        if not installing:
            raise HTTPException(404, "This model is not installing now.")

        return await self._wait_for_install_setup(model, installing.wait_ready())

    async def cancel_model_install(self, instance: str, model_id: str) -> None:
        """Cancel an in-progress model install, stopping the underlying Docker image pull."""
        installing = self.get_instance_info(instance).installing_model_progress.get(model_id)

        if not installing:
            raise HTTPException(404, f"Model {model_id} is not installing now.")

        # Always attempt to cancel the pending setup call first, unconditionally - cancelling an
        # already-finished task is a harmless no-op. Deciding which of "still setting up" vs
        # "real promise exists" applies *before* attempting cancellation would race: the setup
        # work can finish (and install_model() resolve the reservation) in the gap between that
        # check and acting on it, silently leaving the real install running untouched while this
        # reports success. Cancelling first, then waiting for the reservation to settle, means we
        # always act on the real outcome instead of a stale snapshot of it.
        assert installing.pending_task is not None
        installing.pending_task.cancel()

        try:
            promise = await asyncio.wait_for(installing.wait_ready(), timeout=_CANCEL_WAIT_TIMEOUT_SECONDS)
        except TimeoutError:
            # We genuinely don't know whether the cancel above landed or the pending setup is just
            # slow - leaving the reservation untouched lets a later call (retry cancel, or simply
            # waiting) observe the real eventual outcome instead of us guessing and potentially
            # hiding a still-running install by clearing its tracking prematurely.
            raise HTTPException(504, f"Timed out waiting to confirm cancellation for model {model_id}.") from None
        except Exception as e:
            # wait_ready() re-raising here means the pending setup work was cancelled (reject()
            # converts any CancelledError into a clean HTTPException(409) before storing it, so this
            # never surfaces as a raw one from that) or failed on its own - either way, nothing
            # further to cancel. A raw CancelledError reaching this point instead can only mean
            # this call's own task is being cancelled (e.g. its request disconnected); `except
            # Exception` doesn't catch that, so it propagates naturally instead of being treated
            # as if the cancel we asked for had completed.
            #
            # Log the "failed on its own" case specifically - it's otherwise indistinguishable from
            # an ordinary successful cancellation, and a real setup failure that happens to coincide
            # with someone clicking cancel would vanish without a trace.
            if not (isinstance(e, HTTPException) and e.status_code == 409):
                logger.exception(
                    f"{self.get_id(instance)} pending setup for model {model_id} failed independently of the cancel request"  # noqa: G004
                )
            self._pop_installing_if_current(instance, model_id, installing)
            return

        # Cancel the real pull work and every chained promise (cancel() propagates down the chain),
        # plus the chunk reader, then await them all so no task is left pending and GC'd mid-run.
        # promise.tasks() also returns the chained promise's task (it is registered as a child).
        promise.cancel()
        assert installing.task is not None
        installing.task.cancel()
        tasks = [*promise.tasks(), installing.task]
        await asyncio.gather(*tasks, return_exceptions=True)

        promise.progress.close()

        self._pop_installing_if_current(instance, model_id, installing)

    async def get_instance_install_progress(self, instance: str) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Return actually installing service."""
        installing = self.get_instance_info(instance).installing

        if not installing:
            raise HTTPException(404, "This service is not installing now.")

        try:
            return await asyncio.wait_for(installing.wait_ready(), timeout=installing.timeout_seconds)
        except TimeoutError:
            raise HTTPException(504, f"Service {self.get_id(instance)} is still preparing to install; try again shortly.") from None

    def is_installed(self, instance: str) -> bool:
        """Check whether service is installed."""
        return self.get_instance_info(instance).installed is not None

    def get_downloaded(self) -> bool:
        """Get service downloaded info."""
        return self.service_downloaded

    def _restore_model_definition(self, instance: str, model_id: str, definition: dict[str, Any]) -> None:
        """Re-register a model definition missing from the live registry, from a persisted snapshot. No-op by default."""

    def _get_persisted_model_definition(self, instance: str, model_id: str) -> dict[str, Any] | None:
        """Look up the last-persisted definition snapshot for a model no longer in the live catalog."""
        persisted = self.instances_info.get(instance)
        if not persisted:
            return None
        return next((m.definition for m in persisted.config.models or [] if m.model_id == model_id), None)

    def _get_persisted_model_capacity(self, instance: str, model_id: str) -> tuple[int | None, bool]:
        """Look up the last-persisted capacity for a model, from a prior successful detection."""
        persisted = self.instances_info.get(instance)
        if not persisted:
            return None, False
        match = next((m for m in persisted.config.models or [] if m.model_id == model_id), None)
        return (match.capacity, match.capacity_known) if match else (None, False)

    def _get_persisted_model_gpu_memory_utilization(self, instance: str, model_id: str) -> float | None:
        """Look up the last-persisted auto-computed GPU utilization for a model, from a prior successful install.

        Reusing this avoids spurious docker-compose diffs on reload: the auto-tuned default is sized off
        currently-free VRAM (see `VllmService._get_default_gpu_memory_utilization`), which keeps shrinking
        once the model's own container is already running and holding memory, making every reload look like
        a config change and forcing an unnecessary container restart.
        """
        persisted = self.instances_info.get(instance)
        if not persisted:
            return None
        match = next((m for m in persisted.config.models or [] if m.model_id == model_id), None)
        return match.gpu_memory_utilization if match else None

    def _record_warning_in_background(self, message: str, *, instance: str | None = None, model_id: str | None = None) -> None:
        """Fire-and-forget a warning write from a synchronous callback, e.g. a promise's on_error."""
        task = asyncio.create_task(self.service_provider.add_warning(self.get_type(), message, instance=instance, model_id=model_id))
        self._warning_tasks.add(task)
        task.add_done_callback(self._on_warning_task_done)

    def _on_warning_task_done(self, task: "asyncio.Task[None]") -> None:
        self._warning_tasks.discard(task)
        if not task.cancelled() and (exc := task.exception()) is not None:
            logger.exception("Failed to record warning in background", exc_info=exc)

    async def drain_warning_tasks(self) -> None:
        """Await pending fire-and-forget warning writes so shutdown doesn't drop them."""
        if self._warning_tasks:
            await asyncio.gather(*self._warning_tasks, return_exceptions=True)

    async def load_model(self, instance: str, model: ModelConfig) -> None:
        """Load single model."""
        logger.info(f"{self.get_id(instance)} loading model {model.model_id}")  # noqa: G004
        try:
            if model.model_id not in self.models.get(instance, {}) and model.definition:
                logger.warning(
                    f"{self.get_id(instance)} model {model.model_id} missing from current registry, restoring from persisted definition"  # noqa: G004
                )
                self._restore_model_definition(instance, model.model_id, model.definition)

            async def on_success(data: InstallModelOut) -> InstallModelOut:
                return data

            # Reserve like install_model() does, so a concurrent POST /install for this model_id
            # joins this load instead of racing it or getting a fabricated "already installing"
            # response - but with no-op bookkeeping callbacks, since this method already has its own
            # separate success/failure handling below. Routing through install_model() directly
            # would run both, double-recording the warning on failure.
            promise = await self._reserve_and_install(instance, model.model_id, model.options, on_success, lambda _e: None)
            await promise.wait()
        except Exception as exc:
            logger.exception(f"{self.get_id(instance)} get error while loading model {model.model_id}")  # noqa: G004
            self._failed_models.setdefault(instance, set()).add(model.model_id)
            try:
                await self.service_provider.add_warning(
                    self.get_type(),
                    f"Model '{model.model_id}' failed to load on instance '{instance}': {exc}",
                    instance=instance,
                    model_id=model.model_id,
                )
            except Exception:
                logger.exception(f"{self.get_id(instance)} failed to record warning for model {model.model_id}")  # noqa: G004
            return

        if instance_failures := self._failed_models.get(instance):
            instance_failures.discard(model.model_id)
        try:
            await self.service_provider.dismiss_warnings_matching(self.get_type(), instance=instance, model_id=model.model_id)
        except Exception:
            logger.exception(f"{self.get_id(instance)} failed to dismiss warnings for model {model.model_id} after successful load")  # noqa: G004

    async def load_instance(self, instance: str, instance_data: InstanceConfig) -> None:
        """Load instance of service."""
        if not instance_data.options:
            return

        self.load_default_models(instance)

        if instance_data.custom:
            for custom in instance_data.custom:
                model_id = custom.data.get("id", custom.id)
                try:
                    self._add_custom_model(instance, custom)
                except Exception as exc:
                    # Registering one custom model must not take the whole instance - and, since
                    # load_service gathers all of a service's instances without return_exceptions=True,
                    # every other instance of this service too - down with it. Matches load_model's
                    # per-model warning-and-continue pattern just above.
                    logger.exception(f"{self.get_id(instance)} failed to register custom model {model_id} while loading")  # noqa: G004
                    try:
                        await self.service_provider.add_warning(
                            self.get_type(),
                            f"Custom model '{model_id}' failed to load on instance '{instance}': {exc}",
                            instance=instance,
                            model_id=model_id,
                        )
                    except Exception:
                        logger.exception(f"{self.get_id(instance)} failed to record warning for custom model {model_id}")  # noqa: G004

        promise = await self.install_instance(instance, instance_data.options, instance_data, save=False, run_after_install=False)
        await promise.wait()
        logger.info(f"{self.get_id(instance)} service checked")  # noqa: G004
        tasks = [asyncio.create_task(self.load_model(instance, model)) for model in instance_data.models or []]
        await asyncio.gather(*tasks)
        await self._after_install(instance)

    async def load_service(self, config: ServiceRawConfig) -> None:
        """Load service using the config."""
        cfg = ServiceConfig(**config)
        self.models_downloaded = ({key: self._load_download_info(value) for key, value in cfg.downloaded.items()}) if cfg.downloaded else {}
        self.service_downloaded = cfg.service_downloaded or False

        msg = f"{self.get_type()} service installed."
        logger.info(msg)
        if cfg.instances:
            needs_backfill = self._persists_model_definitions and any(
                m.definition is None for instance in cfg.instances.values() for m in instance.models or []
            )
            tasks = [asyncio.create_task(self.load_instance(name, instance)) for name, instance in cfg.instances.items()]
            await asyncio.gather(*tasks)
            if needs_backfill:
                logger.info(f"{self.get_type()} backfilling model definitions into persisted config")  # noqa: G004
                await self._save()

    @abstractmethod
    def _load_download_info(self, data: dict[str, Any]) -> DownloadInfoType:
        pass

    async def _save(self) -> None:
        instances_config = {}
        for instance_name, instance in self.instances_info.items():
            generated = self._generate_instance_config(instance_name, instance.installed, instance.config.custom)
            generated.models = self._preserve_failed_models(instance_name, generated.models or [])
            instance.config.models = generated.models
            instances_config[instance_name] = generated
        cfg = self.service_config(instances_config)
        await self.service_provider.save_service_config(self.get_type(), cfg.model_dump())

    def _preserve_failed_models(self, instance: str, live_models: list[ModelConfig]) -> list[ModelConfig]:
        """Keep the persisted entry for a model that failed to (re)load.

        Otherwise it would get dropped the moment anything else triggers a save. Explicit uninstalls
        clear the failure marker, so they are never resurrected here.
        """
        failed_ids = self._failed_models.get(instance)
        if not failed_ids:
            return live_models
        live_ids = {m.model_id for m in live_models}
        persisted_by_id = {m.model_id: m for m in self.instances_info[instance].config.models or []}
        preserved = [persisted_by_id[model_id] for model_id in failed_ids if model_id not in live_ids and model_id in persisted_by_id]
        return [*live_models, *preserved]

    @abstractmethod
    def _generate_instance_config(self, instance: str, info: InstalledInfoType | None, custom: list[CustomModel] | None) -> InstanceConfig:
        """Generate instance config."""

    def service_config(self, instances_config: dict[str, InstanceConfig]) -> ServiceConfig:
        """Generate service config."""
        return ServiceConfig(instances=instances_config, downloaded=self.models_downloaded, service_downloaded=self.service_downloaded)

    async def install_instance(
        self,
        instance: str,
        options: InstallServiceIn,
        instance_config: InstanceConfig | None = None,
        save: bool = True,
        run_after_install: bool = True,
    ) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Install the service.

        `run_after_install=False` lets a caller that still needs to load persisted models onto the
        fresh instance (see `load_instance`) defer `_after_install` until after that happens - running
        it here for a restore would merge a live catalog against an empty `installed.models`, causing
        every not-yet-reloaded model to look uninstalled and get dropped instead of preserved as stale.
        """
        if self.instances_info and self.instances_info.get(instance):
            if self.instances_info[instance].installed:
                raise HTTPException(status_code=400, detail=f"Service {self.get_id(instance)} on {instance} instance already installed")
            if self.instances_info[instance].installing:
                raise HTTPException(status_code=400, detail=f"Service {self.get_id(instance)} on {instance} instance already installing")

        # Reserve the slot synchronously (no `await` before this point) so a concurrent call can
        # never observe "not installing" once this one has started, no matter how long the
        # service's own `_install_instance()` takes to actually produce the real promise.
        reservation = InstallingInstance(_INSTANCE_INSTALL_STATUS_WAIT_TIMEOUT_SECONDS)
        self.instances_info[instance] = Instance(None, reservation, {}, instance_config or InstanceConfig())

        async def func(data: InstalledInfoType) -> InstallServiceOut:
            self.instances_info[instance].installed = data
            if run_after_install:
                await self._after_install(instance)
            if save:
                await self._save()
            self.instances_info[instance].installing = None
            try:
                await self.service_provider.dismiss_warnings_matching_any(self.get_type(), [(None, None), (instance, None)])
            except Exception:
                logger.exception(f"{self.get_id(instance)} failed to dismiss warnings after successful install")  # noqa: G004
            return InstallServiceOut(status="OK")

        def on_error(e: Exception) -> None:
            self.instances_info[instance].installing = None
            self._record_warning_in_background(f"Service instance '{instance}' failed to install: {e}", instance=instance)

        try:
            promise = await self._install_instance(instance, options)
        except BaseException as e:
            self.instances_info[instance].installing = None
            reservation.reject(e)
            raise

        next_promise = promise.next(func, on_error)
        reservation.resolve(next_promise)
        return next_promise

    async def update_instance(self, instance: str, options: InstallServiceIn) -> PromiseWithProgress[InstallServiceOut, StreamChunk]:
        """Update an installed service by re-applying the install flow with new options.

        The underlying container is recreated to pick up the new options; already-installed models are
        preserved (their weights stay on disk) and re-registered against the recreated container.
        """
        info = self.get_instance_info(instance)
        if info.installing:
            raise HTTPException(status_code=400, detail=f"Service {self.get_id(instance)} on {instance} instance is currently installing")
        if not info.installed:
            raise HTTPException(status_code=400, detail=f"Service {self.get_id(instance)} on {instance} instance is not installed")

        preserved_models = (self._generate_instance_config(instance, info.installed, info.config.custom).models) or []
        preserved_models = self._preserve_failed_models(instance, preserved_models)

        # Reserve the slot synchronously (no `await` before this point) so a concurrent call can
        # never observe "not installing" once this one has started, no matter how long validation,
        # the uninstall, or the service's own `_install_instance()` take to run.
        reservation = InstallingInstance(_INSTANCE_UPDATE_STATUS_WAIT_TIMEOUT_SECONDS)
        info.installing = reservation

        async def func(data: InstalledInfoType) -> InstallServiceOut:
            self.instances_info[instance].installed = data
            for model in preserved_models:
                await self.load_model(instance, model)
            await self._after_install(instance)
            await self._save()
            self.instances_info[instance].installing = None
            try:
                await self.service_provider.dismiss_warnings_matching_any(self.get_type(), [(None, None), (instance, None)])
            except Exception:
                logger.exception(f"{self.get_id(instance)} failed to dismiss warnings after successful update")  # noqa: G004
            return InstallServiceOut(status="OK")

        def on_error(e: Exception) -> None:
            self.instances_info[instance].installing = None
            self._record_warning_in_background(f"Service instance '{instance}' failed to update: {e}", instance=instance)

        try:
            await self._validate_update_options(options)
            await self._uninstall_instance(instance, UninstallServiceIn(purge=False))
            self._failed_models.pop(instance, None)
            promise = await self._install_instance(instance, options)
        except BaseException as e:
            self.instances_info[instance].installing = None
            reservation.reject(e)
            raise

        next_promise = promise.next(func, on_error)
        reservation.resolve(next_promise)
        return next_promise

    async def _validate_update_options(self, options: InstallServiceIn) -> None:
        """Validate new options before update_instance tears down the existing installation.

        No-op by default. Override for services whose `_install_instance` can reject the new
        options (e.g. an unavailable Docker image tag) — without this, update_instance's
        uninstall-then-reinstall ordering would destroy a working installation before the
        rejection surfaces, since `_install_instance` only validates after being invoked.
        """

    async def _after_install(self, instance: str) -> None:
        """Run right after `instance` becomes installed, on both fresh install and update.

        No-op by default. Override for services that should refresh derived state (e.g. a live
        model catalog) as soon as connection details are known, instead of waiting for a manual
        refresh.
        """

    @abstractmethod
    async def _install_instance(self, instance: str, options: InstallServiceIn) -> PromiseWithProgress[InstalledInfoType, StreamChunk]:
        """Install service."""

    async def uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        """Uninstall the service."""
        await self._uninstall_instance(instance, options)
        self._failed_models.pop(instance, None)
        await self.service_provider.dismiss_warnings_for_instance(self.get_type(), instance)
        await self._save()

    @abstractmethod
    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        """Uninstall instance."""

    async def _resolve_custom_model_size(self, spec: dict[str, Any], instance: str = "") -> str | None:  # noqa: ARG002
        """Return auto-fetched size string for the model, or None if not supported."""
        return None

    def get_custom_model_definition(self, custom_model_id: CustomModelId) -> dict[str, Any] | None:
        """Return the stored spec a custom model was created from, searching all instances by id."""
        for instance in self.instances_info.values():
            for custom in instance.config.custom or []:
                if custom.id == custom_model_id:
                    return custom.data
        return None

    async def _validate_custom_model(self, spec: dict[str, Any], instance: str = "") -> None:
        """Raise `HTTPException(400, ...)` if the custom model spec can't be served; no-op by default.

        Runs before `_resolve_custom_model_size`, so an override may also fill in derived fields
        (e.g. `size`) it already fetched, sparing a redundant lookup - callers must not reorder the
        two.
        """

    async def add_custom_model(self, instance: str, options: AddCustomModelIn) -> CustomModelId:
        """Add custom model."""
        spec = dict(options.spec)
        await self._validate_custom_model(spec, instance)
        if not spec.get("size"):
            resolved = await self._resolve_custom_model_size(spec, instance)
            spec["size"] = resolved if resolved else "unknown"
        model = CustomModel(id=str(uuid.uuid4()), data=spec)
        config = self.get_instance_info(instance).config
        self._add_custom_model(instance, model)
        if config.custom is None:
            config.custom = []

        config.custom.append(model)
        await self._save()
        return model.id

    def _add_custom_model(self, instance: str, model: CustomModel) -> None:  # noqa: ARG002
        """Add custom model."""
        raise HTTPException(400, "This service does not support custom models.")

    async def remove_custom_model(self, instance: str, custom_model_id: CustomModelId) -> None:
        """Remove custom model."""
        config = self.get_instance_info(instance).config
        model = next(x for x in config.custom or {} if x.id == custom_model_id)
        self._remove_custom_model(instance, model)
        config.custom = [x for x in config.custom or {} if x.id != custom_model_id]
        model_id = model.data.get("id")
        if model_id:
            self._edit_locks.pop((instance, model_id), None)
        await self._save()

    def _remove_custom_model(self, instance: str, model: CustomModel) -> None:  # noqa: ARG002
        """Remove custom model."""
        raise HTTPException(400, "This service does not support custom models.")

    async def update_custom_model(self, instance: str, custom_model_id: CustomModelId, options: AddCustomModelIn) -> None:
        """Update custom model."""
        config = self.get_instance_info(instance).config
        model = next((x for x in config.custom or [] if x.id == custom_model_id), None)
        if model is None:
            raise HTTPException(404, f"Custom model {custom_model_id} not found.")
        new_data: dict[str, Any] = dict(options.spec)
        if _custom_model_serving_fields_changed(model.data, new_data):
            await self._validate_custom_model(new_data, instance)
        if not new_data.get("size"):
            resolved = await self._resolve_custom_model_size(new_data, instance)
            new_data["size"] = resolved or model.data.get("size") or "unknown"
        await self._update_custom_model(instance, model, new_data)
        model.data = new_data
        await self._save()

    async def _update_custom_model(self, instance: str, model: CustomModel, new_data: dict[str, Any]) -> None:
        """Update custom model by rebuilding it: remove the old, add the new, restoring the old on failure."""
        self._remove_custom_model(instance, model)
        try:
            self._add_custom_model(instance, CustomModel(id=model.id, data=new_data))
        except Exception:
            self._add_custom_model(instance, model)
            raise

    async def _install_model_and_wait(
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Install the model and wait for the underlying work to actually finish before returning.

        `install_model` returns a `PromiseWithProgress` whose real work runs in a background task -
        awaiting the call itself only awaits kicking it off, not its completion. Edit orchestration needs
        to know whether the (re)install genuinely succeeded before deciding to roll back, so this awaits
        the promise fully and lets any failure propagate.
        """
        promise = await self.install_model(instance, model_id, options)
        await promise.wait()
        return promise

    def _get_edit_lock(self, instance: str, model_id: str) -> asyncio.Lock:
        """Return the lock serializing `edit_model`/`edit_model_install_options` calls for one model.

        Scoped to edit orchestration only - `install_model`/`uninstall_model`/`update_custom_model`
        already guard themselves against concurrent calls to themselves (see design.md), so they don't
        need to acquire this lock.
        """
        key = (instance, model_id)
        if key not in self._edit_locks:
            self._edit_locks[key] = asyncio.Lock()
        return self._edit_locks[key]

    def _validate_edit(self, instance: str, model_id: str) -> None:
        """Reject an edit upfront, before any uninstall happens. Default no-op.

        Override for service-specific preconditions that would otherwise only be caught deep inside
        `update_custom_model`, after `edit_model` has already uninstalled the model for nothing (see
        MCP's OAuth-pending check).
        """

    def _check_prefix_collision(self, instance: str, prefix: str, exclude_model_id: str | None) -> None:
        """Reject `prefix` if another model in `instance` is already registered under it.

        Compares against each model's *effective* prefix - the live install-time prefix when the
        model is installed, falling back to `default_prefix` otherwise - not `default_prefix` alone.
        `edit_model_install_options` can move a model's install-time prefix independently of its
        `default_prefix`, so the two can disagree; the effective prefix is whichever one actually
        reflects what a model is doing right now (serving traffic on it, or reserving it unstalled).
        """
        # Looked up directly rather than via `get_instance_info` (which 404s): this can run for an
        # instance that isn't registered in `instances_info` yet, e.g. while adding its first model.
        info = self.instances_info.get(instance)
        installed_models = cast("_HasModels", info.installed).models if info and info.installed else {}
        for mid, m in self.models.get(instance, {}).items():
            if mid == exclude_model_id:
                continue
            effective_prefix = installed_models[mid].prefix if mid in installed_models else m.default_prefix
            if effective_prefix == prefix:
                raise HTTPException(400, f"Prefix '{prefix}' is already in use by model '{mid}'.")

    async def edit_model(  # noqa: C901
        self, instance: str, custom_model_id: CustomModelId, new_definition: AddCustomModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk] | None:
        """Edit a custom-backed model's definition without requiring a manual uninstall first.

        If the model is currently installed, uninstalls it, applies the new definition, then reinstalls
        it with its previous install-time options (prefix/envs/headers), returning the reinstall promise.
        If it isn't installed, this is a plain definition update (same as calling `update_custom_model`
        directly) and returns None. On failure applying the new definition or reinstalling, restores the
        model to its previous definition and installed state on a best-effort basis.
        """
        config = self.get_instance_info(instance).config
        model = next((x for x in config.custom or [] if x.id == custom_model_id), None)
        if model is None:
            raise HTTPException(404, f"Custom model {custom_model_id} not found.")
        model_id = model.data.get("id")
        if not model_id:
            raise HTTPException(400, "Custom model definition has no id.")
        new_id = new_definition.spec.get("id")
        if new_id is not None and new_id != model_id:
            # `edit_model` uninstalls/reinstalls under this one captured `model_id` throughout - an id
            # change would rename the entry out from under that, so the reinstall step would fail
            # looking for the old id. MCP's own per-kind `_update_custom_model` branches already reject
            # this, but only *after* uninstalling; checking here first avoids that pointless disruption,
            # and closes the same gap for services (e.g. CustomService) whose generic update path never
            # enforced id immutability at all.
            raise HTTPException(400, "Cannot change the model's id while editing. Duplicate it instead to use a new id.")
        self._validate_edit(instance, model_id)

        async with self._get_edit_lock(instance, model_id):
            info = self.get_instance_info(instance)
            installed_models = cast("_HasModels", info.installed).models if info.installed else {}
            was_installed = model_id in installed_models
            install_options: InstallModelIn | None = installed_models[model_id].options if was_installed else None
            old_data = dict(model.data)

            if was_installed:
                await self.uninstall_model(instance, model_id, UninstallModelIn(purge=False))

            try:
                await self.update_custom_model(instance, custom_model_id, new_definition)
            except Exception:
                logger.exception(f"{self.get_id(instance)} failed to apply new definition for model {model_id} while editing")  # noqa: G004
                if was_installed:
                    assert install_options is not None
                    try:
                        await self._install_model_and_wait(instance, model_id, install_options)
                    except Exception:
                        logger.exception(f"{self.get_id(instance)} failed to roll back model {model_id} after a failed edit")  # noqa: G004
                raise

            if not was_installed:
                return None

            assert install_options is not None
            # Read the prefix `update_custom_model` actually computed for the model now sitting in the
            # registry - not `new_definition.spec.get("default_prefix")`. `default_prefix` is optional
            # for MCP `user`/`proxy` models and the WebUI sends it as absent whenever the field is
            # cleared; `_update_custom_model` still falls back to `normalize_name(id)` in that case and
            # writes that as the model's new `default_prefix`, which the raw request body never reflects.
            updated_model = self.models.get(instance, {}).get(model_id)
            new_default_prefix = updated_model.default_prefix if updated_model else None
            reinstall_options = install_options
            if (
                install_options.spec is not None
                and new_default_prefix is not None
                and install_options.spec.get("prefix") != new_default_prefix
            ):
                # The install-time `prefix` starts out equal to the definition's `default_prefix` (set by
                # `_install_model` when "prefix" is absent from the install spec) and there's no user-facing
                # way to diverge a custom-backed model's `prefix` from that - `edit_model_install_options`
                # (the only path that edits `prefix` directly) is only reachable for catalog models with no
                # `CustomModel` definition, mutually exclusive with this one. So `prefix` must always follow
                # `default_prefix` here, or editing it would silently do nothing to the registered endpoint.
                # `install_options` itself is left untouched so a rollback below reinstalls with the prefix
                # that actually matches the restored (old) definition.
                reinstall_options = install_options.model_copy(update={"spec": {**install_options.spec, "prefix": new_default_prefix}})
            try:
                return await self._install_model_and_wait(instance, model_id, reinstall_options)
            except Exception:
                logger.exception(f"{self.get_id(instance)} failed to reinstall model {model_id} while editing")  # noqa: G004
                try:
                    await self.update_custom_model(instance, custom_model_id, AddCustomModelIn(spec=old_data))
                    await self._install_model_and_wait(instance, model_id, install_options)
                except Exception:
                    logger.exception(f"{self.get_id(instance)} failed to roll back model {model_id} after a failed reinstall")  # noqa: G004
                raise

    async def edit_model_install_options(
        self, instance: str, model_id: str, new_options: InstallModelIn
    ) -> tuple[bool, PromiseWithProgress[InstallModelOut, StreamChunk]]:
        """Edit a model's install-time options only (no persisted `CustomModel` definition involved).

        For catalog models (no `CustomModel` backing) whose only editable state is install-time options
        like prefix/envs/headers. Uninstalls the model if installed, then reinstalls it with the new
        options, restoring the previous options on a best-effort basis if the reinstall fails. Always
        installs, even if the model wasn't installed to begin with - the returned bool tells the caller
        which case it was, so it can report whether this was actually a reinstall.
        """
        model = self.models.get(instance, {}).get(model_id)
        new_prefix = (new_options.spec or {}).get("prefix") or (model.default_prefix if model else None)
        if new_prefix is not None:
            # Checked upfront, before any uninstall, so a doomed edit doesn't disrupt the running
            # model for nothing - mirrors `edit_model`'s id-immutability check above.
            self._check_prefix_collision(instance, new_prefix, exclude_model_id=model_id)

        async with self._get_edit_lock(instance, model_id):
            info = self.get_instance_info(instance)
            installed_models = cast("_HasModels", info.installed).models if info.installed else {}
            was_installed = model_id in installed_models
            old_options: InstallModelIn | None = installed_models[model_id].options if was_installed else None

            if was_installed:
                await self.uninstall_model(instance, model_id, UninstallModelIn(purge=False))

            try:
                promise = await self._install_model_and_wait(instance, model_id, new_options)
            except Exception:
                logger.exception(f"{self.get_id(instance)} failed to reinstall model {model_id} with new install options")  # noqa: G004
                if was_installed:
                    assert old_options is not None
                    try:
                        await self._install_model_and_wait(instance, model_id, old_options)
                    except Exception:
                        logger.exception(
                            f"{self.get_id(instance)} failed to roll back model {model_id} after a failed install-options edit"  # noqa: G004
                        )
                raise
            return was_installed, promise

    async def _reserve_and_install(
        self,
        instance: str,
        model_id: str,
        options: InstallModelIn,
        on_success: Callable[[InstallModelOut], Awaitable[InstallModelOut]],
        on_error: Callable[[Exception], None],
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Reserve model_id for install, or join an already-in-progress reservation for it.

        Shared by install_model() (the public entry point) and load_model() (which starts an
        install directly at startup, bypassing the public API) so a concurrent call through either
        one sees the same reservation, instead of racing a duplicate install or fabricating a
        result. on_success/on_error carry each caller's own post-install bookkeeping; reservation
        cleanup itself always happens here regardless of what they do.
        """
        installing_model_progress = self.get_instance_info(instance).installing_model_progress

        async def wrapped_on_success(data: InstallModelOut) -> InstallModelOut:
            self._pop_installing_if_current(instance, model_id, reservation)
            return await on_success(data)

        def wrapped_on_error(e: Exception) -> None:
            self._pop_installing_if_current(instance, model_id, reservation)
            on_error(e)

        existing = installing_model_progress.get(model_id)
        if existing is not None:
            # Already installing (or reserved a moment ago by a concurrent call): join the real
            # install instead of asking the service to start a second one or fabricating a result.
            # The bookkeeping continuation is shared (see resolve()), not re-chained per caller.
            # wait_chained() only waits for the (short) pre-promise setup to settle, not the whole
            # install - the real download is awaited separately, by whoever calls promise.wait()
            # on what this returns - so a bounded wait here doesn't cut a legitimately long install
            # short, only a genuinely hung setup phase.
            return await self._wait_for_install_setup(model_id, existing.wait_chained())

        # Reserve the slot synchronously (no `await` before this point) so a concurrent call can
        # never observe "not reserved yet" once this one has started, no matter how long the
        # service's own `_install_model()` takes to actually produce the real promise.
        reservation = InstallingModel()
        installing_model_progress[model_id] = reservation
        # Run _install_model() as its own task (rather than awaiting it inline) so cancel_model_install(),
        # called from a completely different request, has something real to cancel even before this
        # produces a promise - not just something to wait on.
        reservation.pending_task = asyncio.create_task(self._install_model(instance, model_id, options))
        try:
            promise = await reservation.pending_task
        except BaseException as e:
            self._pop_installing_if_current(instance, model_id, reservation)
            reservation.reject(e)
            # A CancelledError here can mean two different things: this call's own task was
            # cancelled (e.g. its request disconnected) - which must propagate as-is - or someone
            # else cancelled the pending setup task (cancel_model_install(), from a different
            # request) while this call was merely awaiting it - which isn't this caller's own
            # cancellation and shouldn't surface as a raw CancelledError FastAPI can't handle.
            if isinstance(e, asyncio.CancelledError) and not is_own_cancellation():
                raise HTTPException(409, f"Model {model_id} install was cancelled") from None
            raise

        return reservation.resolve(promise, wrapped_on_success, wrapped_on_error)

    async def install_model(
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Install the model."""

        async def on_success(data: InstallModelOut) -> InstallModelOut:
            if instance_failures := self._failed_models.get(instance):
                instance_failures.discard(model_id)
            await self._save()
            try:
                await self.service_provider.dismiss_warnings_matching(self.get_type(), instance=instance, model_id=model_id)
            except Exception:
                logger.exception(f"{self.get_id(instance)} failed to dismiss warnings for model {model_id} after successful install")  # noqa: G004
            msg = f"{model_id} model installed."
            logger.debug(msg)
            return data

        def on_error(e: Exception) -> None:
            error = str(e)
            self._record_warning_in_background(error, instance=instance, model_id=model_id)

        return await self._reserve_and_install(instance, model_id, options, on_success, on_error)

    @abstractmethod
    async def _install_model(
        self, instance: str, model_id: str, options: InstallModelIn
    ) -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        """Install the model."""

    async def uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        """Uninstall the model."""
        await self._uninstall_model(instance, model_id, options)
        if instance_failures := self._failed_models.get(instance):
            instance_failures.discard(model_id)
        try:
            await self.service_provider.dismiss_warnings_matching(self.get_type(), instance=instance, model_id=model_id)
        except Exception:
            logger.exception(f"{self.get_id(instance)} failed to dismiss warnings for model {model_id} after uninstall")  # noqa: G004
        await self._save()

    @abstractmethod
    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        """Uninstall the model."""

    async def get_docker_logs(self, instance: str, model_id: str | None) -> str:
        """Get docker logs."""
        docker_compose_file_path = self.get_docker_compose_file_path(instance, model_id)
        return await self.docker_service.get_docker_compose_logs(docker_compose_file_path)

    def _get_model_installed_info(self, instance: str, model_id: str) -> bool | InstallModelProgress:
        installing_model_progress = self.get_instance_info(instance).installing_model_progress

        if model_id in installing_model_progress:
            installing = installing_model_progress[model_id]
            if installing.last_chunk and installing.last_chunk["type"] == "progress":
                return InstallModelProgress(stage=installing.last_chunk["stage"], value=installing.last_chunk["value"])
            if installing.last_chunk and installing.last_chunk["type"] == "finish":
                return InstallModelProgress(stage="install", value=1)
            return InstallModelProgress(stage="download", value=0)
        return False

    def _get_service_installed_info(self, instance: str) -> bool | InstallServiceProgress:
        installing = self.get_instance_info(instance).installing

        if not installing:
            return False
        if installing.last_chunk and installing.last_chunk["type"] == "progress":
            return InstallServiceProgress(stage=installing.last_chunk["stage"], value=installing.last_chunk["value"])
        if installing.last_chunk and installing.last_chunk["type"] == "finish":
            return InstallServiceProgress(stage="install", value=1)
        return InstallServiceProgress(stage="download", value=0)

    async def get_docker_compose_file(self, instance: str, model_id: str | None) -> str:
        """Get docker compose file."""
        docker_compose_file_path = self.get_docker_compose_file_path(instance, model_id)
        return await Utils.read_file(docker_compose_file_path)

    async def restart_docker(self, instance: str, model_id: str | None) -> None:
        """Restart the instance/model's docker-backed service, resolving its DockerOptions first.

        Handles the no-compose-file case (e.g. an adopted container) via DockerService.restart_docker_compose.
        """
        options = self.get_docker_options(instance, model_id)
        await self.docker_service.restart_docker_compose(options)

    def get_docker_compose_file_path(self, instance: str, model_id: str | None) -> Path:
        """Get docker compose file path."""
        options = self.get_docker_options(instance, model_id)
        return self.docker_service.get_docker_compose_file_path(options.name)

    def get_docker_options(self, instance: str, model_id: str | None) -> DockerOptions:  # noqa: ARG002
        """Return the resolved DockerOptions for this instance/model."""
        if not self.is_installed(instance):
            raise HTTPException(400, "Instance not installed")
        raise HTTPException(400, "Docker is not bound with this object")

    def _get_working_dir(self) -> Path:
        return self._get_service_dir(self.get_type())

    def _get_service_dir(self, service: str) -> Path:
        """Get service dir."""
        dir = self.config.get_storage_services_dir() / f"./{service}"
        if not dir.is_dir():
            dir.mkdir(parents=True)
        return dir

    async def _clear_working_dir(self) -> None:
        working_dir = self._get_working_dir()
        if working_dir.exists():
            shutil.rmtree(working_dir)

    def get_instance_installed_info(self, instance: str) -> InstalledInfoType:
        """Get instance installed info or return exception."""
        installed = self.get_instance_info(instance).installed
        if installed is None:
            raise HTTPException(status_code=400, detail=f"Service {self.get_id(instance)} not installed")
        return installed

    def get_hugging_face_token(self) -> str:
        """Return Hugging Face Key."""
        return self.config.hugging_face_token.get_secret_value()

    def get_civitai_token(self) -> str:
        """Return Civitai Face Key."""
        return self.config.civitai_token.get_secret_value()

    def _has_gpu_for_spec(self) -> str:
        return "true" if self.docker_service.has_gpu_support else "false"

    async def _download_image_or_set_progress(self, stream: Stream[StreamChunk], image: DockerImage) -> None:
        if image.name not in self.images_download_progress:
            self.images_download_progress[image.name] = stream
            try:
                await self._docker_pull(image, stream)
            finally:
                self.images_download_progress.pop(image.name, None)
        else:
            chunk: StreamChunk
            async for chunk in self.images_download_progress[image.name].as_generator():
                if chunk.get("type") == "progress" and chunk.get("stage") == "download":
                    stream.emit(chunk)
                else:
                    break

    async def _docker_pull(
        self,
        image: DockerImage,
        stream: Stream[StreamChunk],
    ) -> None:
        """Docker pull only if image does not exist."""
        stream.emit(StreamChunkProgress(type="progress", stage="download", value=0, data={}))
        if not await self.docker_service.is_docker_image_pulled(image.name):
            size = await self.docker_service.get_docker_image_size(image.name)
            async for progress in self.docker_service.docker_pull(image.name, size or convert_size_to_bytes(image.size) or 0):
                stream.emit(StreamChunkProgress(type="progress", stage="download", value=progress, data={}))
        stream.emit(StreamChunkProgress(type="progress", stage="download", value=1, data={}))

    async def _stop_docker(self, docker_options: DockerOptions) -> None:
        """Stop docker and log error if it occurs."""
        try:
            await self.docker_service.stop_docker(docker_options)
        except Exception:
            logger.exception("Error during stopping docker compose %s", docker_options.name)

    async def _rollback_failed_install_docker(self, docker_options: DockerOptions, *, adopted: bool) -> None:
        """Undo `install_and_run_docker` after a later step of the same install attempt fails.

        Skips touching the container when `adopted` is True: an adopted container (DFINFRA-281) pre-dates
        this install attempt — it wasn't created by it — so a failed install must leave it exactly as it
        found it instead of tearing it down. Only a container this attempt itself started gets stopped.
        """
        if adopted:
            logger.info(
                "Not stopping %r on install rollback: it was adopted from a pre-existing container, not created by this install attempt",
                docker_options.name,
            )
            return
        await self._stop_docker(docker_options)

    async def _stop_dockers_parallel(self, docker_options_list: list[DockerOptions]) -> None:
        """Stop docker and log error if it occurs."""
        tasks = [asyncio.create_task(self._stop_docker(docker_options)) for docker_options in docker_options_list]
        await asyncio.gather(*tasks)

    async def _verify_docker_image(self, docker_image: str, ignore_warning: bool) -> None:
        warnings = await self.docker_service.get_image_warnings(docker_image)
        if len(warnings) > 0 and not ignore_warning:
            raise HTTPException(400, {"warnings": warnings})

    @property
    def _supported_gpus(self) -> list[GpuInfo]:
        """Return GPUs supported by this service. Override to include non-NVIDIA GPUs."""
        return [gpu for gpu in self.hardware.gpus if isinstance(gpu, NvidiaGpuInfo)]

    def is_given_hardware_support_gpu(self, hardware_specification: str | bool | None) -> bool:
        """Return is gpu will be used."""
        if hardware_specification is None:
            return bool(self._supported_gpus)
        if isinstance(hardware_specification, str):
            has_gpu_support = hardware_specification.startswith("GPU")
            if has_gpu_support and not self._supported_gpus:
                raise HTTPException(400, "Given hardware specification is not supported")
            return has_gpu_support
        return hardware_specification

    def get_specified_hardware_parts(self, hardware_specification: str | bool | None) -> Sequence[HardwarePartInfo]:
        """Get specified hardware parts."""
        if hardware_specification is None:
            return self._supported_gpus if self._supported_gpus else [self.hardware.cpu]
        if (hardware_specification is False) or (isinstance(hardware_specification, str) and hardware_specification == "CPU"):
            return [self.hardware.cpu]

        if (hardware_specification is True) or (hardware_specification == "GPUs") or (hardware_specification == "GPU"):
            return self._supported_gpus

        gpus: list[GpuInfo] = []

        for gpu_name in hardware_specification.removeprefix("GPU | ").split(","):
            for gpu in self._supported_gpus:
                if gpu_name == gpu.long_name:
                    gpus.append(gpu)

        return gpus

    def canonicalize_hardware_spec(self, hardware_specification: str | bool | None) -> str:
        """Convert a hardware spec into the descriptive string used by the UI.

        A CLI install may leave `hardware` as a bare bool (or None), which the UI renders literally
        as "true"/"false". This resolves the value to an option produced by `add_hardware_field_to_spec`
        ("CPU", "GPU | <name>" for a single GPU, or "GPUs" for several) so it displays correctly and
        round-trips through the settings form. Descriptive strings are returned as-is.
        """
        if isinstance(hardware_specification, str):
            return hardware_specification

        gpus = [part for part in self.get_specified_hardware_parts(hardware_specification) if isinstance(part, GpuInfo)]
        if not gpus:
            return "CPU"
        if len(gpus) == 1:
            return f"GPU | {gpus[0].long_name}"
        return "GPUs"

    def add_hardware_field_to_spec(
        self,
        fields: list[ServiceField] | None = None,
        add_cpu_option_only_on_avx512_support: bool = False,
    ) -> list[ServiceField]:
        """Add hardware (CPU/GPU/GPUs) field to specification."""
        fields = fields or []
        options: list[str | OneOfOption] = []
        default: str | None = None
        gpus = self._supported_gpus
        if len(gpus) == 1:
            options.append(OneOfOption(value="GPU", label="Default GPU"))
            default = "GPU"
        elif gpus:
            options.append(OneOfOption(value="GPUs", label="All GPUs"))
            default = "GPUs"
        options.extend([f"GPU | {gpu.long_name}" for gpu in gpus])
        if not add_cpu_option_only_on_avx512_support or self.hardware.cpu.avx512:
            options.append("CPU")
            if default is None:
                default = "CPU"

        gpus_select_field = ServiceField(type="oneof", name="hardware", description="Choose hardware:", values=options, default=default)
        fields.append(gpus_select_field)

        return fields

    def add_hardware_field_to_model_spec(
        self,
        fields: list[ModelField] | None = None,
        add_cpu_option_only_on_avx512_support: bool = False,
    ) -> list[ModelField]:
        """Add hardware (CPU/GPU/GPUs) field to specification."""
        fields = fields or []
        options: list[str | OneOfOption] = []
        default: str | None = None
        gpus = self._supported_gpus
        if len(gpus) == 1:
            options.append(OneOfOption(value="GPU", label="Default GPU"))
            default = "GPU"
        elif gpus:
            options.append(OneOfOption(value="GPUs", label="All GPUs"))
            default = "GPUs"
        options.extend([f"GPU | {gpu.long_name}" for gpu in gpus])
        if not add_cpu_option_only_on_avx512_support or self.hardware.cpu.avx512:
            options.append("CPU")
            if default is None:
                default = "CPU"

        gpus_select_field = ModelField(type="oneof", name="hardware", description="Choose hardware:", values=options, default=default)
        fields.append(gpus_select_field)

        return fields
