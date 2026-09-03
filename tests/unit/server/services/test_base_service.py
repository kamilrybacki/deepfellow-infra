# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import asyncio
import contextlib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from server.docker import DockerImage
from server.models.models import (
    AddCustomModelIn,
    CustomModelSpecification,
    InstallModelIn,
    InstallModelOut,
    ListModelsFilters,
    ListModelsOut,
    RetrieveModelOut,
    UninstallModelIn,
)
from server.models.services import (
    InstallServiceIn,
    InstallServiceOut,
    RetrieveServiceOut,
    ServiceField,
    ServiceSpecification,
    UninstallServiceIn,
)
from server.serviceprovider import ServiceRawConfig
from server.services.base2_service import (
    Base2Service,
    CustomModel,
    InstallingInstance,
    InstallingModel,
    InstanceConfig,
    ModelConfig,
)
from server.services.base_service import BaseService
from server.utils.core import PromiseWithProgress, Stream, StreamChunk, StreamChunkProgress
from server.utils.hardware import NvidiaGpuInfo
from server.utils.registry_client import RegistryUnavailableError


class _BaseImpl(BaseService):
    """Minimal concrete BaseService for testing abstract method contracts."""

    instances_info: dict[str, Any] = {}

    def get_type(self) -> str:
        return "test-service"

    def get_description(self) -> str:
        return "A test service."

    def get_size(self) -> str:
        return "small"

    def get_spec(self) -> ServiceSpecification:
        return ServiceSpecification(fields=[])

    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        return None

    def get_instance_install_progress(self, instance: str) -> Any:
        raise NotImplementedError

    def get_model_install_progress(self, instance: str, model: str) -> Any:
        raise NotImplementedError

    def is_installed(self, instance: str) -> bool:
        return False

    def get_installed_info(self, instance: str) -> bool:
        return False

    def get_downloaded(self) -> bool:
        return False

    async def load_service(self, config: ServiceRawConfig) -> None:
        pass

    async def install_instance(self, instance: str, options: InstallServiceIn) -> Any:
        raise NotImplementedError

    async def update_instance(self, instance: str, options: InstallServiceIn) -> Any:
        raise NotImplementedError

    async def uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        pass

    async def list_models(self, input_instance: Any, filters: ListModelsFilters) -> ListModelsOut:
        return ListModelsOut(list=[])

    async def get_model(self, instance: str, model_id: str) -> RetrieveModelOut:
        raise NotImplementedError

    async def install_model(self, instance: str, model_id: str, options: InstallModelIn) -> Any:
        raise NotImplementedError

    async def uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        pass

    async def add_custom_model(self, instance: str, options: Any) -> Any:
        raise NotImplementedError

    async def remove_custom_model(self, instance: str, custom_model_id: Any) -> None:
        pass

    async def get_docker_logs(self, instance: str, model_id: str | None) -> str:
        return ""

    async def get_docker_compose_file(self, instance: str, model_id: str | None) -> str:
        return ""

    async def restart_docker(self, instance: str, model_id: str | None) -> None:
        pass

    async def stop_instance(self, instance: str) -> None:
        pass


class _Base2ImplWithCustom(Base2Service):  # pyright: ignore[reportMissingTypeArgument]
    """Base2Service that supports custom models."""

    _custom_store: dict[str, CustomModel]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._custom_store = {}

    def get_type(self) -> str:
        return "test-b2-custom"

    def get_description(self) -> str:
        return "Test base2 service with custom models."

    def get_size(self) -> str:
        return ""

    def get_spec(self) -> ServiceSpecification:
        return ServiceSpecification(fields=[])

    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        return None

    def get_installed_info(self, instance: str) -> bool:
        return self.is_installed(instance)

    def _load_download_info(self, data: dict[str, Any]) -> dict[str, Any]:
        return data

    def _generate_instance_config(self, instance: Any, info: Any, custom: Any) -> InstanceConfig:
        return InstanceConfig()

    async def _install_instance(self, instance: str, options: InstallServiceIn) -> Any:
        raise NotImplementedError

    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        pass

    async def _install_model(self, instance: str, model_id: str, options: InstallModelIn) -> Any:
        raise NotImplementedError

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        pass

    async def list_models(self, input_instance: Any, filters: ListModelsFilters) -> ListModelsOut:
        return ListModelsOut(list=[])

    async def get_model(self, instance: str, model_id: str) -> RetrieveModelOut:
        raise NotImplementedError

    async def stop_instance(self, instance: str) -> None:
        pass

    def _add_custom_model(self, instance: str, model: CustomModel) -> None:
        self._custom_store[model.id] = model

    def _remove_custom_model(self, instance: str, model: CustomModel) -> None:
        del self._custom_store[model.id]


class _Base2Impl(Base2Service):  # pyright: ignore[reportMissingTypeArgument]
    """Minimal concrete Base2Service for testing."""

    def get_type(self) -> str:
        return "test-b2"

    def get_description(self) -> str:
        return "Test base2 service."

    def get_size(self) -> str:
        return ""

    def get_spec(self) -> ServiceSpecification:
        return ServiceSpecification(fields=[])

    def get_custom_model_spec(self) -> CustomModelSpecification | None:
        return None

    def get_installed_info(self, instance: str) -> bool:
        return self.is_installed(instance)

    def _load_download_info(self, data: dict[str, Any]) -> dict[str, Any]:
        return data

    def _generate_instance_config(self, instance: Any, info: Any, custom: Any) -> InstanceConfig:
        return InstanceConfig()

    async def _install_instance(self, instance: str, options: InstallServiceIn) -> Any:
        raise NotImplementedError

    async def _uninstall_instance(self, instance: str, options: UninstallServiceIn) -> None:
        pass

    async def _install_model(self, instance: str, model_id: str, options: InstallModelIn) -> Any:
        raise NotImplementedError

    async def _uninstall_model(self, instance: str, model_id: str, options: UninstallModelIn) -> None:
        pass

    async def list_models(self, input_instance: Any, filters: ListModelsFilters) -> ListModelsOut:
        return ListModelsOut(list=[])

    async def get_model(self, instance: str, model_id: str) -> RetrieveModelOut:
        raise NotImplementedError

    async def stop_instance(self, instance: str) -> None:
        pass


@pytest.fixture
def base_svc() -> _BaseImpl:
    return _BaseImpl()


@pytest.fixture
def base2_deps() -> dict[str, Any]:
    service_provider = MagicMock()
    service_provider.add_warning = AsyncMock()
    service_provider.dismiss_warnings_matching = AsyncMock()
    service_provider.dismiss_warnings_matching_any = AsyncMock()
    service_provider.dismiss_warnings_for_instance = AsyncMock()
    return {
        "config": MagicMock(),
        "endpoint_registry": MagicMock(),
        "service_provider": service_provider,
        "model_downloader": MagicMock(),
        "docker_service": MagicMock(),
        "hardware": MagicMock(),
    }


@pytest.fixture
def base2_svc(base2_deps: dict[str, Any]) -> _Base2Impl:
    return _Base2Impl(**base2_deps)


@pytest.fixture
def custom_svc(base2_deps: dict[str, Any]) -> _Base2ImplWithCustom:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    return _Base2ImplWithCustom(**base2_deps)


@pytest.mark.parametrize(
    ("instance", "expected"),
    [
        ("default", "test-service"),
        ("gpu-1", "test-service|gpu-1"),
    ],
)
def test_get_id(base_svc: _BaseImpl, instance: str, expected: str) -> None:
    assert base_svc.get_id(instance) == expected


@pytest.mark.parametrize(
    ("instance", "expected"),
    [
        ("default", "test-service"),
        ("gpu-1", "test-service-gpu-1"),
    ],
)
def test_get_service_id(base_svc: _BaseImpl, instance: str, expected: str) -> None:
    assert base_svc.get_service_id(instance) == expected


def test_service_has_docker_default_false(base_svc: _BaseImpl) -> None:
    assert base_svc.service_has_docker() is False


def test_is_cloud_service_default_false(base_svc: _BaseImpl) -> None:
    assert base_svc.is_cloud_service() is False


@pytest.mark.asyncio
async def test_get_docker_tags_default_returns_empty(base_svc: _BaseImpl) -> None:
    result = await base_svc.get_docker_tags(None)
    assert result == []


def test_filter_docker_tags_default_passthrough(base_svc: _BaseImpl) -> None:
    tags = ["v1.0", "v2.0", "latest"]
    assert base_svc.filter_docker_tags(tags, "gpu") == tags


def test_get_default_docker_tag_default_returns_none(base_svc: _BaseImpl) -> None:
    assert base_svc.get_default_docker_tag(None) is None


def test_get_docker_image_repo_default_returns_none(base_svc: _BaseImpl) -> None:
    assert base_svc.get_docker_image_repo(None) is None


@pytest.mark.asyncio
async def test_validate_docker_image_version_noop_when_not_set(base_svc: _BaseImpl) -> None:
    await base_svc.validate_docker_image_version(None, "gpu")


@pytest.mark.asyncio
async def test_validate_docker_image_version_passes_when_tag_available(base_svc: _BaseImpl) -> None:
    with patch.object(base_svc, "get_docker_tags", AsyncMock(return_value=["v1.0", "v2.0"])):
        await base_svc.validate_docker_image_version("v1.0", "gpu")


@pytest.mark.asyncio
async def test_validate_docker_image_version_raises_when_tag_unavailable(base_svc: _BaseImpl) -> None:
    with (
        patch.object(base_svc, "get_docker_tags", AsyncMock(return_value=["v1.0", "v2.0"])),
        pytest.raises(HTTPException) as exc_info,
    ):
        await base_svc.validate_docker_image_version("v9.9", "gpu")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_validate_docker_image_version_rejects_when_tag_list_genuinely_empty(base_svc: _BaseImpl) -> None:
    """A successfully fetched but empty tag list means no tag is known to be valid — reject."""
    with (
        patch.object(base_svc, "get_docker_tags", AsyncMock(return_value=[])),
        pytest.raises(HTTPException) as exc_info,
    ):
        await base_svc.validate_docker_image_version("v9.9", "gpu")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_validate_docker_image_version_skips_when_registry_unavailable(base_svc: _BaseImpl) -> None:
    with patch.object(base_svc, "get_docker_tags", AsyncMock(side_effect=RegistryUnavailableError("unreachable"))):
        await base_svc.validate_docker_image_version("v9.9", "gpu")


@pytest.mark.parametrize(
    ("hardware", "expected"),
    [
        (True, "GPU"),
        (False, "CPU"),
    ],
)
@pytest.mark.asyncio
async def test_validate_docker_image_version_normalizes_bool_hardware(base_svc: _BaseImpl, hardware: bool, expected: str) -> None:
    """A bool hardware default (has_gpu_support) must not be str()-ed into 'True'/'False'."""
    get_docker_tags = AsyncMock(return_value=["v1.0"])
    with patch.object(base_svc, "get_docker_tags", get_docker_tags):
        await base_svc.validate_docker_image_version("v1.0", hardware)

    get_docker_tags.assert_awaited_once_with(expected)


@pytest.mark.asyncio
async def test_validate_docker_image_version_passes_when_tag_exists_outside_capped_list(base_svc: _BaseImpl) -> None:
    """A tag older than the registry's capped get_docker_tags() list can still be installed via CLI."""
    mock_client = MagicMock()
    mock_client.tag_exists = AsyncMock(return_value=True)
    with (
        patch.object(base_svc, "get_docker_tags", AsyncMock(return_value=["v2.0"])),
        patch.object(base_svc, "get_docker_image_repo", return_value="ollama/ollama"),
        patch("server.services.base_service.registry_for", return_value=mock_client) as mock_registry_for,
    ):
        await base_svc.validate_docker_image_version("v0.1-old", "gpu")

    mock_registry_for.assert_called_once_with("ollama/ollama")
    mock_client.tag_exists.assert_awaited_once_with("ollama/ollama", "v0.1-old")


@pytest.mark.asyncio
async def test_validate_docker_image_version_raises_when_tag_exists_check_returns_false(base_svc: _BaseImpl) -> None:
    mock_client = MagicMock()
    mock_client.tag_exists = AsyncMock(return_value=False)
    with (
        patch.object(base_svc, "get_docker_tags", AsyncMock(return_value=["v2.0"])),
        patch.object(base_svc, "get_docker_image_repo", return_value="ollama/ollama"),
        patch("server.services.base_service.registry_for", return_value=mock_client),
        pytest.raises(HTTPException) as exc_info,
    ):
        await base_svc.validate_docker_image_version("v9.9", "gpu")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_validate_docker_image_version_skips_when_tag_exists_check_registry_unavailable(base_svc: _BaseImpl) -> None:
    mock_client = MagicMock()
    mock_client.tag_exists = AsyncMock(side_effect=RegistryUnavailableError("unreachable"))
    with (
        patch.object(base_svc, "get_docker_tags", AsyncMock(return_value=["v2.0"])),
        patch.object(base_svc, "get_docker_image_repo", return_value="ollama/ollama"),
        patch("server.services.base_service.registry_for", return_value=mock_client),
    ):
        await base_svc.validate_docker_image_version("v0.1-old", "gpu")


@pytest.mark.asyncio
async def test_validate_docker_image_version_raises_without_falling_back_when_repo_unknown(base_svc: _BaseImpl) -> None:
    """No get_docker_image_repo override and no docker-tags spec field means no repo to check against."""
    with (
        patch.object(base_svc, "get_docker_tags", AsyncMock(return_value=["v2.0"])),
        pytest.raises(HTTPException) as exc_info,
    ):
        await base_svc.validate_docker_image_version("v9.9", "gpu")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_validate_docker_image_version_resolves_repo_from_spec_field_fallback(base_svc: _BaseImpl) -> None:
    """When get_docker_image_repo() returns None, fall back to the docker-tags spec field's docker_image."""
    mock_client = MagicMock()
    mock_client.tag_exists = AsyncMock(return_value=True)
    spec = ServiceSpecification(
        fields=[ServiceField(type="docker-tags", name="image_version", description="", docker_image="ollama/ollama")]
    )
    with (
        patch.object(base_svc, "get_docker_tags", AsyncMock(return_value=["v2.0"])),
        patch.object(base_svc, "get_spec", return_value=spec),
        patch("server.services.base_service.registry_for", return_value=mock_client) as mock_registry_for,
    ):
        await base_svc.validate_docker_image_version("v0.1-old", "gpu")

    mock_registry_for.assert_called_once_with("ollama/ollama")


@pytest.mark.asyncio
async def test_cancel_model_install_default_raises_405(base_svc: _BaseImpl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await base_svc.cancel_model_install("default", "m1")

    assert exc_info.value.status_code == 405


@pytest.mark.asyncio
async def test_edit_model_default_raises_405(base_svc: _BaseImpl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await base_svc.edit_model("default", "custom-1", AddCustomModelIn(spec={}))

    assert exc_info.value.status_code == 405


@pytest.mark.asyncio
async def test_edit_model_install_options_default_raises_405(base_svc: _BaseImpl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await base_svc.edit_model_install_options("default", "m1", InstallModelIn())

    assert exc_info.value.status_code == 405


@pytest.mark.asyncio
async def test_get_duplicate_spec_default_raises_405(base_svc: _BaseImpl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await base_svc.get_duplicate_spec("default", "m1")

    assert exc_info.value.status_code == 405


def test_is_cloud_service_true_when_class_attr_set() -> None:
    class CloudSvc(_BaseImpl):
        is_cloud = True

    assert CloudSvc().is_cloud_service() is True


def test_get_info_default_instance(base_svc: _BaseImpl) -> None:
    result = base_svc.get_info("default")

    assert isinstance(result, RetrieveServiceOut)
    assert result.id == "test-service"
    assert result.type == "test-service"
    assert result.instance == "default"
    assert result.description == "A test service."
    assert result.size == "small"
    assert result.has_docker is False
    assert result.is_cloud is False
    assert result.custom_model_spec is None


def test_get_info_non_default_instance(base_svc: _BaseImpl) -> None:
    result = base_svc.get_info("gpu-1")

    assert result.id == "test-service|gpu-1"
    assert result.instance == "gpu-1"


def test_get_info_cloud_service() -> None:
    class CloudSvc(_BaseImpl):
        is_cloud = True

    result = CloudSvc().get_info("default")
    assert result.is_cloud is True


def test_base2_init_creates_default_instance(base2_svc: _Base2Impl) -> None:
    assert "default" in base2_svc.instances_info


def test_base2_init_service_downloaded_false(base2_svc: _Base2Impl) -> None:
    assert base2_svc.service_downloaded is False


def test_base2_init_models_downloaded_empty(base2_svc: _Base2Impl) -> None:
    assert base2_svc.models_downloaded == {}


def test_base2_is_installed_false_when_not_installed(base2_svc: _Base2Impl) -> None:
    assert base2_svc.is_installed("default") is False


def test_base2_is_installed_true_when_installed(base2_svc: _Base2Impl) -> None:
    base2_svc.instances_info["default"].installed = object()

    assert base2_svc.is_installed("default") is True


def test_base2_get_downloaded_reflects_flag(base2_svc: _Base2Impl) -> None:
    assert base2_svc.get_downloaded() is False

    base2_svc.service_downloaded = True

    assert base2_svc.get_downloaded() is True


def test_base2_check_instance_exists_raises_for_missing(base2_svc: _Base2Impl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        base2_svc.check_instance_exists("nonexistent")

    assert exc_info.value.status_code == 404


def test_base2_check_instance_exists_passes_for_default(base2_svc: _Base2Impl) -> None:
    base2_svc.check_instance_exists("default")


@pytest.mark.parametrize(
    ("hardware_support", "expected"),
    [
        (False, False),
        ("CPU", False),
        (True, True),
    ],
)
def test_is_given_hardware_support_gpu(base2_svc: _Base2Impl, hardware_support: Any, expected: bool) -> None:
    assert base2_svc.is_given_hardware_support_gpu(hardware_support) is expected


def test_is_given_hardware_support_gpu_none_no_gpus_returns_false(base2_deps: dict[str, Any]) -> None:
    base2_deps["hardware"].gpus = []

    svc = _Base2Impl(**base2_deps)

    assert svc.is_given_hardware_support_gpu(None) is False


def test_is_given_hardware_support_gpu_string_gpu_raises_when_no_gpus(base2_deps: dict[str, Any]) -> None:
    base2_deps["hardware"].gpus = []
    svc = _Base2Impl(**base2_deps)

    with pytest.raises(HTTPException) as exc_info:
        svc.is_given_hardware_support_gpu("GPU")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_installing_model_init_creates_task() -> None:
    value = InstallModelOut(status="OK", details="done")
    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(value=value)
    promise.progress.close()

    async def on_success(data: InstallModelOut) -> InstallModelOut:
        return data

    installing = InstallingModel()
    installing.resolve(promise, on_success, lambda _e: None)

    assert installing.promise is promise
    assert installing.task is not None
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_installing_instance_init_creates_task() -> None:
    value = InstallServiceOut(status="OK")
    promise: PromiseWithProgress[InstallServiceOut, StreamChunk] = PromiseWithProgress(value=value)
    promise.progress.close()

    installing = InstallingInstance(promise=promise)

    assert installing.promise is promise
    assert installing.task is not None
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_installing_model_stores_last_chunk_from_progress() -> None:
    chunk: StreamChunk = {"type": "progress", "stage": "download", "percentage": 0.5}  # pyright: ignore[reportAssignmentType]
    value = InstallModelOut(status="OK", details="done")
    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(value=value)
    promise.progress.emit(chunk)
    promise.progress.close()

    async def on_success(data: InstallModelOut) -> InstallModelOut:
        return data

    installing = InstallingModel()
    installing.resolve(promise, on_success, lambda _e: None)
    await asyncio.sleep(0)

    assert installing.last_chunk == chunk


@pytest.mark.asyncio
async def test_installing_instance_stores_last_chunk_from_progress() -> None:
    chunk: StreamChunk = {"type": "progress", "stage": "download", "percentage": 0.8}  # pyright: ignore[reportAssignmentType]
    value = InstallServiceOut(status="OK")
    promise: PromiseWithProgress[InstallServiceOut, StreamChunk] = PromiseWithProgress(value=value)
    promise.progress.emit(chunk)
    promise.progress.close()

    installing = InstallingInstance(promise=promise)
    await asyncio.sleep(0)

    assert installing.last_chunk == chunk


def test_get_instance_info_raises_404_for_missing(base2_svc: _Base2Impl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        base2_svc.get_instance_info("nonexistent")

    assert exc_info.value.status_code == 404


def test_get_instance_info_returns_instance(base2_svc: _Base2Impl) -> None:
    result = base2_svc.get_instance_info("default")

    assert result is base2_svc.instances_info["default"]


@pytest.mark.asyncio
async def test_get_model_install_progress_raises_when_not_installing(base2_svc: _Base2Impl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await base2_svc.get_model_install_progress("default", "unknown-model")

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_model_install_progress_returns_promise(base2_svc: _Base2Impl) -> None:
    mock_promise = MagicMock()
    mock_installing = MagicMock()
    mock_installing.wait_ready = AsyncMock(return_value=mock_promise)

    base2_svc.instances_info["default"].installing_model_progress["m1"] = mock_installing

    assert await base2_svc.get_model_install_progress("default", "m1") is mock_promise


@pytest.mark.asyncio
async def test_cancel_model_install_raises_when_not_installing(base2_svc: _Base2Impl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await base2_svc.cancel_model_install("default", "unknown-model")

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_cancel_model_install_cancels_tasks_and_clears_tracking(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    started = asyncio.Event()
    release = asyncio.Event()

    async def func(_stream: Stream[StreamChunk]) -> InstallModelOut:
        started.set()
        await release.wait()
        return InstallModelOut(status="OK", details="done")

    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(func=func)

    with patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=promise)):
        returned_promise = await base2_svc.install_model("default", "m1", InstallModelIn())

    await started.wait()
    installing = base2_svc.instances_info["default"].installing_model_progress["m1"]

    await base2_svc.cancel_model_install("default", "m1")

    assert "m1" not in base2_svc.instances_info["default"].installing_model_progress
    assert promise.task.cancelled()
    assert installing.task is not None
    assert installing.task.cancelled()
    assert promise.progress._closed  # pyright: ignore[reportPrivateUsage]
    assert promise._future.cancelled()  # pyright: ignore[reportPrivateUsage]
    assert returned_promise.task.done()
    assert returned_promise._future.cancelled()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_cancel_model_install_during_pre_promise_setup_interrupts_it(base2_svc: _Base2Impl) -> None:
    """Cancelling while `_install_model()` is still doing pre-promise setup must actually interrupt
    it, not just wait around for it to finish on its own.
    """
    reached_slow_point = asyncio.Event()
    was_cancelled = False

    async def slow_install_model(_instance: str, _model_id: str, _options: InstallModelIn) -> Any:
        nonlocal was_cancelled
        reached_slow_point.set()
        try:
            await asyncio.Event().wait()  # never completes on its own
        except asyncio.CancelledError:
            was_cancelled = True
            raise

    with patch.object(base2_svc, "_install_model", new=slow_install_model):
        task = asyncio.create_task(base2_svc.install_model("default", "m1", InstallModelIn()))
        await reached_slow_point.wait()

        await asyncio.wait_for(base2_svc.cancel_model_install("default", "m1"), timeout=1)

        # install_model()'s own task was never asked to cancel - only the pending setup task it was
        # awaiting was, by a different request. A raw CancelledError here would be unfixable by
        # FastAPI (a bare 500); this caller should see a clean, ordinary "it got cancelled" error.
        with pytest.raises(HTTPException) as exc_info:
            await task
        assert exc_info.value.status_code == 409

    assert was_cancelled
    assert "m1" not in base2_svc.instances_info["default"].installing_model_progress


@pytest.mark.asyncio
async def test_cancel_model_install_propagates_its_own_cancellation_instead_of_swallowing_it(
    base2_svc: _Base2Impl,
) -> None:
    """If cancel_model_install()'s own caller is cancelled (e.g. its request disconnects) while it's
    waiting to learn the pending setup's outcome, that cancellation must propagate - not be treated
    as if the cancel it requested had simply completed.
    """
    reached_slow_point = asyncio.Event()

    async def slow_install_model(_instance: str, _model_id: str, _options: InstallModelIn) -> Any:
        reached_slow_point.set()
        await asyncio.Event().wait()  # never completes on its own

    with patch.object(base2_svc, "_install_model", new=slow_install_model):
        install_task = asyncio.create_task(base2_svc.install_model("default", "m1", InstallModelIn()))
        await reached_slow_point.wait()

        cancel_task = asyncio.create_task(base2_svc.cancel_model_install("default", "m1"))
        await asyncio.sleep(0)  # let cancel_model_install() cancel the pending task and start waiting

        cancel_task.cancel()  # cancel cancel_model_install()'s OWN task, not a second cancel request

        with pytest.raises(asyncio.CancelledError):
            await cancel_task

        # install_model()'s own task was never cancelled here either - only its pending setup task
        # was, as a side effect of cancel_model_install() cancelling it. That's not this caller's own
        # cancellation, so it should see a clean HTTPException, not a raw CancelledError.
        with pytest.raises(HTTPException) as exc_info:
            await install_task
        assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_cancel_model_install_cleans_up_when_pending_task_already_failed_with_ordinary_exception(
    base2_svc: _Base2Impl,
) -> None:
    """If the pending setup already finished with an ordinary exception (unrelated to the cancel
    request) by the time cancel_model_install() is waiting on it, cancel must clean up gracefully
    instead of propagating that unrelated failure as if cancelling itself had failed.
    """
    reservation = InstallingModel()
    base2_svc.instances_info["default"].installing_model_progress["m1"] = reservation

    async def already_failed() -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        raise RuntimeError("setup blew up")

    reservation.pending_task = asyncio.create_task(already_failed())
    with pytest.raises(RuntimeError):
        await reservation.pending_task  # pending_task is done (failed)...

    cancel_task = asyncio.create_task(base2_svc.cancel_model_install("default", "m1"))
    await asyncio.sleep(0)
    assert not cancel_task.done()  # correctly waiting - reject() hasn't been called yet

    reservation.reject(RuntimeError("setup blew up"))  # install_model() "catches up" now

    await cancel_task  # must not raise - a stale cancel on an already-failed install is a no-op

    assert "m1" not in base2_svc.instances_info["default"].installing_model_progress


@pytest.mark.asyncio
async def test_cancel_model_install_logs_when_pending_setup_fails_independently(
    base2_svc: _Base2Impl, caplog: pytest.LogCaptureFixture
) -> None:
    """A pending setup failure that's unrelated to the cancel request (e.g. a real Docker error that
    happens to coincide with someone clicking cancel) must leave a trace in the logs - otherwise it's
    indistinguishable from an ordinary successful cancellation and vanishes without explanation.
    """
    reservation = InstallingModel()
    base2_svc.instances_info["default"].installing_model_progress["m1"] = reservation

    async def already_failed() -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        raise RuntimeError("setup blew up")

    reservation.pending_task = asyncio.create_task(already_failed())
    with pytest.raises(RuntimeError):
        await reservation.pending_task

    with caplog.at_level("ERROR", logger="uvicorn.error"):
        cancel_task = asyncio.create_task(base2_svc.cancel_model_install("default", "m1"))
        await asyncio.sleep(0)

        reservation.reject(RuntimeError("setup blew up"))

        await cancel_task

    assert "m1" in caplog.text
    assert "independently of the cancel request" in caplog.text


@pytest.mark.asyncio
async def test_cancel_model_install_does_not_log_on_genuine_cancellation(base2_svc: _Base2Impl, caplog: pytest.LogCaptureFixture) -> None:
    """A genuine, successful cancellation must not be logged as if it were an unrelated failure."""
    reached_slow_point = asyncio.Event()

    async def slow_install_model(_instance: str, _model_id: str, _options: InstallModelIn) -> Any:
        reached_slow_point.set()
        await asyncio.Event().wait()  # never completes on its own

    with caplog.at_level("ERROR", logger="uvicorn.error"), patch.object(base2_svc, "_install_model", new=slow_install_model):
        task = asyncio.create_task(base2_svc.install_model("default", "m1", InstallModelIn()))
        await reached_slow_point.wait()

        await asyncio.wait_for(base2_svc.cancel_model_install("default", "m1"), timeout=1)

        with pytest.raises(HTTPException):
            await task

    assert "independently of the cancel request" not in caplog.text


@pytest.mark.asyncio
async def test_cancel_model_install_waits_for_resolve_when_pending_task_already_done(base2_svc: _Base2Impl) -> None:
    """Reproduces the race the fix closes: `pending_task` can finish before `install_model()` gets
    around to calling `resolve()` on the reservation. Cancelling in that exact window must not
    conclude "nothing to cancel" just because the pending task looks done - it must wait for the
    real promise to actually attach, then cancel it for real, instead of walking away while the
    real install keeps running untouched.
    """
    started = asyncio.Event()
    release = asyncio.Event()

    async def func(_stream: Stream[StreamChunk]) -> InstallModelOut:
        started.set()
        await release.wait()
        return InstallModelOut(status="OK", details="done")

    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(func=func)
    await started.wait()  # the real background work is genuinely already running

    reservation = InstallingModel()
    base2_svc.instances_info["default"].installing_model_progress["m1"] = reservation

    async def already_finished() -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        return promise

    reservation.pending_task = asyncio.create_task(already_finished())
    await reservation.pending_task  # pending_task is done...
    assert reservation.promise is None  # ...but resolve() hasn't been called yet - the race window

    cancel_task = asyncio.create_task(base2_svc.cancel_model_install("default", "m1"))
    await asyncio.sleep(0)
    assert not cancel_task.done()  # correctly waiting, not giving up because pending_task looked done

    async def on_success(data: InstallModelOut) -> InstallModelOut:
        return data

    reservation.resolve(promise, on_success, lambda _e: None)  # install_model() "catches up" now

    await cancel_task

    assert "m1" not in base2_svc.instances_info["default"].installing_model_progress
    assert promise.task.cancelled()


@pytest.mark.asyncio
async def test_cancel_model_install_does_not_clobber_a_newer_reservation(base2_svc: _Base2Impl) -> None:
    """cancel_model_install() must only ever remove *its own* reservation - not whatever happens to
    be sitting at installing_model_progress[model_id] once it's done waiting. If a brand new install
    for the same model_id gets reserved (e.g. a retry) while this call was awaiting confirmation, that
    new reservation must survive untouched instead of getting silently deleted out from under it.
    """

    async def install_func(_stream: Stream[StreamChunk]) -> InstallModelOut:
        await asyncio.Event().wait()  # never completes on its own
        return InstallModelOut(status="OK", details="done")

    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(func=install_func)

    installing = InstallingModel()
    base2_svc.instances_info["default"].installing_model_progress["m1"] = installing

    async def already_finished() -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        return promise

    installing.pending_task = asyncio.create_task(already_finished())
    await installing.pending_task

    async def on_success(data: InstallModelOut) -> InstallModelOut:
        return data

    installing.resolve(promise, on_success, lambda _e: None)

    newer_reservation = InstallingModel()
    original_wait_ready = installing.wait_ready

    async def wait_ready_then_get_replaced() -> PromiseWithProgress[InstallModelOut, StreamChunk]:
        result = await original_wait_ready()
        # Simulates a retry reserving the same model_id while cancel_model_install() was waiting.
        base2_svc.instances_info["default"].installing_model_progress["m1"] = newer_reservation
        return result

    installing.wait_ready = wait_ready_then_get_replaced  # type: ignore[method-assign]

    await base2_svc.cancel_model_install("default", "m1")

    assert base2_svc.instances_info["default"].installing_model_progress["m1"] is newer_reservation


@pytest.mark.asyncio
async def test_cancel_model_install_unblocks_concurrent_progress_watcher(base2_svc: _Base2Impl) -> None:
    """A caller blocked in `get_model_install_progress()` during pre-promise setup must be unblocked
    when someone else cancels the install, instead of hanging forever - with a clean error, not the
    raw `CancelledError` (an unrelated caller's own request handling shouldn't have to deal with a
    `BaseException` it never asked to be cancelled by).
    """
    reached_slow_point = asyncio.Event()

    async def slow_install_model(_instance: str, _model_id: str, _options: InstallModelIn) -> Any:
        reached_slow_point.set()
        await asyncio.Event().wait()  # never completes on its own

    with patch.object(base2_svc, "_install_model", new=slow_install_model):
        install_task = asyncio.create_task(base2_svc.install_model("default", "m1", InstallModelIn()))
        await reached_slow_point.wait()

        watcher_task = asyncio.create_task(base2_svc.get_model_install_progress("default", "m1"))
        await asyncio.sleep(0)  # let the watcher reach wait_ready()

        await asyncio.wait_for(base2_svc.cancel_model_install("default", "m1"), timeout=1)

        with pytest.raises(HTTPException) as exc_info:
            await asyncio.wait_for(watcher_task, timeout=1)
        assert exc_info.value.status_code == 409
        # install_model()'s own task was never cancelled either - only its pending setup task was.
        with pytest.raises(HTTPException) as install_exc_info:
            await install_task
        assert install_exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_download_image_progress_cleared_on_cancel(base2_svc: _Base2Impl) -> None:
    image = DockerImage(name="example/image:latest", size="1GB")
    stream: Stream[StreamChunk] = Stream()

    async def fake_pull(_image: DockerImage, _stream: Stream[StreamChunk]) -> None:
        raise asyncio.CancelledError

    with patch.object(base2_svc, "_docker_pull", new=fake_pull), pytest.raises(asyncio.CancelledError):
        await base2_svc._download_image_or_set_progress(stream, image)  # pyright: ignore[reportPrivateUsage]

    assert image.name not in base2_svc.images_download_progress


def test_get_instance_install_progress_raises_when_not_installing(base2_svc: _Base2Impl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        base2_svc.get_instance_install_progress("default")

    assert exc_info.value.status_code == 404


def test_get_instance_install_progress_returns_promise(base2_svc: _Base2Impl) -> None:
    mock_promise = MagicMock()
    mock_installing = MagicMock()
    mock_installing.promise = mock_promise

    base2_svc.instances_info["default"].installing = mock_installing

    assert base2_svc.get_instance_install_progress("default") is mock_promise


@pytest.mark.asyncio
async def test_load_model_happy_path(base2_svc: _Base2Impl) -> None:
    value = InstallModelOut(status="OK", details="done")
    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(value=value)
    model = ModelConfig(model_id="m1", options=InstallModelIn())

    with patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=promise)):
        await base2_svc.load_model("default", model)


@pytest.mark.asyncio
async def test_load_model_exception_is_caught(base2_svc: _Base2Impl) -> None:
    model = ModelConfig(model_id="m1", options=InstallModelIn())

    with patch.object(base2_svc, "_install_model", new=AsyncMock(side_effect=RuntimeError("fail"))):
        await base2_svc.load_model("default", model)


@pytest.mark.asyncio
async def test_load_model_exception_records_warning(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    model = ModelConfig(model_id="m1", options=InstallModelIn())

    with patch.object(base2_svc, "_install_model", new=AsyncMock(side_effect=RuntimeError("fail"))):
        await base2_svc.load_model("default", model)

    base2_deps["service_provider"].add_warning.assert_awaited_once()
    call_kwargs = base2_deps["service_provider"].add_warning.await_args.kwargs
    assert call_kwargs["model_id"] == "m1"
    assert call_kwargs["instance"] == "default"


@pytest.mark.asyncio
async def test_load_model_success_with_failing_dismiss_is_not_marked_failed(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    """A successful install must not be marked as failed just because clearing its warning afterwards errors."""
    value = InstallModelOut(status="OK", details="done")
    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(value=value)
    model = ModelConfig(model_id="m1", options=InstallModelIn())
    base2_deps["service_provider"].dismiss_warnings_matching = AsyncMock(side_effect=RuntimeError("disk full"))

    with patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=promise)):
        await base2_svc.load_model("default", model)

    assert "m1" not in base2_svc._failed_models.get("default", set())  # pyright: ignore[reportPrivateUsage]
    base2_deps["service_provider"].add_warning.assert_not_awaited()


@pytest.mark.asyncio
async def test_load_model_failing_add_warning_does_not_propagate(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    """A failure while recording the warning itself must not escape load_model and cancel sibling loads."""
    model = ModelConfig(model_id="m1", options=InstallModelIn())
    base2_deps["service_provider"].add_warning = AsyncMock(side_effect=RuntimeError("disk full"))

    with patch.object(base2_svc, "_install_model", new=AsyncMock(side_effect=RuntimeError("fail"))):
        await base2_svc.load_model("default", model)

    assert "m1" in base2_svc._failed_models.get("default", set())  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_load_model_restores_missing_definition_from_snapshot(base2_svc: _Base2Impl) -> None:
    value = InstallModelOut(status="OK", details="done")
    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(value=value)
    base2_svc.models = {"default": {}}
    model = ModelConfig(model_id="m1", options=InstallModelIn(), definition={"hf_id": "m1"})

    with (
        patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=promise)),
        patch.object(base2_svc, "_restore_model_definition") as restore_mock,
    ):
        await base2_svc.load_model("default", model)

    restore_mock.assert_called_once_with("default", "m1", {"hf_id": "m1"})


@pytest.mark.asyncio
async def test_load_model_skips_restore_when_already_in_registry(base2_svc: _Base2Impl) -> None:
    value = InstallModelOut(status="OK", details="done")
    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(value=value)
    base2_svc.models = {"default": {"m1": object()}}
    model = ModelConfig(model_id="m1", options=InstallModelIn(), definition={"hf_id": "m1"})

    with (
        patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=promise)),
        patch.object(base2_svc, "_restore_model_definition") as restore_mock,
    ):
        await base2_svc.load_model("default", model)

    restore_mock.assert_not_called()


@pytest.mark.asyncio
async def test_load_instance_skips_when_no_options(base2_svc: _Base2Impl) -> None:
    await base2_svc.load_instance("default", InstanceConfig(options=None))

    assert base2_svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_load_instance_installs_service(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    installed_value: dict[str, Any] = {}
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value=installed_value)
    instance_data = InstanceConfig(options=InstallServiceIn(spec={}), models=[], custom=None)

    with patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=promise)):
        await base2_svc.load_instance("default", instance_data)

    assert base2_svc.instances_info["default"].installed is installed_value


@pytest.mark.asyncio
async def test_load_instance_with_custom_models(custom_svc: _Base2ImplWithCustom, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    installed_value: dict[str, Any] = {}
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value=installed_value)
    custom = CustomModel(id="cm-test", data={"name": "my-model"})
    instance_data = InstanceConfig(options=InstallServiceIn(spec={}), models=[], custom=[custom])

    with patch.object(custom_svc, "_install_instance", new=AsyncMock(return_value=promise)):
        await custom_svc.load_instance("default", instance_data)

    assert "cm-test" in custom_svc._custom_store  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_load_instance_custom_model_failure_does_not_block_instance(
    custom_svc: _Base2ImplWithCustom, base2_deps: dict[str, Any]
) -> None:
    """A single custom model raising while loading (e.g. a stale prefix collision from before that
    check existed) must not stop the rest of the instance from loading - previously any exception
    from _add_custom_model propagated out of load_instance uncaught, and since load_service gathers
    every instance of a service without return_exceptions=True, one bad model took the whole service
    down on startup, not just itself."""
    base2_deps["service_provider"].save_service_config = AsyncMock()
    installed_value: dict[str, Any] = {}
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value=installed_value)
    bad = CustomModel(id="cm-bad", data={"id": "bad-model"})
    good = CustomModel(id="cm-good", data={"name": "my-model"})
    instance_data = InstanceConfig(options=InstallServiceIn(spec={}), models=[], custom=[bad, good])

    def add_custom_model(instance: str, model: CustomModel) -> None:
        if model.id == "cm-bad":
            raise HTTPException(400, "Prefix 'x' is already in use by model 'other'.")
        custom_svc._custom_store[model.id] = model  # pyright: ignore[reportPrivateUsage]

    with (
        patch.object(custom_svc, "_add_custom_model", side_effect=add_custom_model),
        patch.object(custom_svc, "_install_instance", new=AsyncMock(return_value=promise)),
    ):
        await custom_svc.load_instance("default", instance_data)

    assert "cm-good" in custom_svc._custom_store  # pyright: ignore[reportPrivateUsage]
    assert "cm-bad" not in custom_svc._custom_store  # pyright: ignore[reportPrivateUsage]
    assert custom_svc.instances_info["default"].installed is installed_value
    base2_deps["service_provider"].add_warning.assert_awaited_once()
    call_kwargs = base2_deps["service_provider"].add_warning.await_args.kwargs
    assert call_kwargs["model_id"] == "bad-model"
    assert call_kwargs["instance"] == "default"


@pytest.mark.asyncio
async def test_load_instance_custom_model_failure_with_failing_add_warning_does_not_propagate(
    custom_svc: _Base2ImplWithCustom, base2_deps: dict[str, Any]
) -> None:
    """A failure while recording the warning itself must not escape load_instance either - same
    guarantee test_load_model_failing_add_warning_does_not_propagate already gives load_model."""
    base2_deps["service_provider"].save_service_config = AsyncMock()
    base2_deps["service_provider"].add_warning = AsyncMock(side_effect=RuntimeError("disk full"))
    installed_value: dict[str, Any] = {}
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value=installed_value)
    bad = CustomModel(id="cm-bad", data={"id": "bad-model"})
    instance_data = InstanceConfig(options=InstallServiceIn(spec={}), models=[], custom=[bad])

    def add_custom_model(instance: str, model: CustomModel) -> None:
        raise HTTPException(400, "Prefix 'x' is already in use by model 'other'.")

    with (
        patch.object(custom_svc, "_add_custom_model", side_effect=add_custom_model),
        patch.object(custom_svc, "_install_instance", new=AsyncMock(return_value=promise)),
    ):
        await custom_svc.load_instance("default", instance_data)

    assert custom_svc.instances_info["default"].installed is installed_value


@pytest.mark.asyncio
async def test_install_instance_on_error_clears_installing(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    async def fail_func(stream: Any) -> Any:
        raise RuntimeError("install failed")

    fail_promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(func=fail_func)

    with patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=fail_promise)):
        result_promise = await base2_svc.install_instance("default", InstallServiceIn(spec={}))
        with pytest.raises(RuntimeError):
            await result_promise.wait()

    assert base2_svc.instances_info["default"].installing is None


@pytest.mark.asyncio
async def test_install_instance_on_error_records_warning(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    async def fail_func(stream: Any) -> Any:
        raise RuntimeError("install failed")

    fail_promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(func=fail_func)

    with patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=fail_promise)):
        result_promise = await base2_svc.install_instance("default", InstallServiceIn(spec={}))
        with pytest.raises(RuntimeError):
            await result_promise.wait()
        await base2_svc.drain_warning_tasks()

    base2_deps["service_provider"].add_warning.assert_awaited_once()
    call_kwargs = base2_deps["service_provider"].add_warning.await_args.kwargs
    assert call_kwargs["instance"] == "default"


@pytest.mark.asyncio
async def test_drain_warning_tasks_noop_when_no_pending_tasks(base2_svc: _Base2Impl) -> None:
    await base2_svc.drain_warning_tasks()


@pytest.mark.asyncio
async def test_record_warning_in_background_logs_when_add_warning_fails(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    """A background warning write that itself fails must be logged, not left to become an unretrieved exception."""
    base2_deps["service_provider"].add_warning = AsyncMock(side_effect=RuntimeError("disk full"))

    base2_svc._record_warning_in_background("boom", instance="default")  # pyright: ignore[reportPrivateUsage]
    await base2_svc.drain_warning_tasks()

    assert not base2_svc._warning_tasks  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_instance_success_dismisses_instance_warnings(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    installed_value: dict[str, Any] = {}
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value=installed_value)

    with patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=promise)):
        result_promise = await base2_svc.install_instance("default", InstallServiceIn(spec={}))
        await result_promise.wait()

    base2_deps["service_provider"].dismiss_warnings_matching_any.assert_awaited_once_with(
        base2_svc.get_type(), [(None, None), ("default", None)]
    )


@pytest.mark.asyncio
async def test_install_instance_success_with_failing_dismiss_still_completes(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    """A successful install must not fail just because clearing its warnings afterwards errors."""
    base2_deps["service_provider"].save_service_config = AsyncMock()
    base2_deps["service_provider"].dismiss_warnings_matching_any = AsyncMock(side_effect=RuntimeError("disk full"))
    installed_value: dict[str, Any] = {}
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value=installed_value)

    with patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=promise)):
        result_promise = await base2_svc.install_instance("default", InstallServiceIn(spec={}))
        result = await result_promise.wait()

    assert result.status == "OK"


@pytest.mark.asyncio
async def test_load_service_sets_downloads_and_flag(base2_svc: _Base2Impl) -> None:
    config: ServiceRawConfig = {"downloaded": {"m1": {"key": "val"}}, "service_downloaded": True}

    await base2_svc.load_service(config)

    assert base2_svc.service_downloaded is True
    assert "m1" in base2_svc.models_downloaded


@pytest.mark.asyncio
async def test_load_service_loads_instances(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    installed_value: dict[str, Any] = {}
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value=installed_value)
    config: ServiceRawConfig = {
        "downloaded": None,
        "service_downloaded": False,
        "instances": {"default": {"options": {"stream": False, "ignore_warnings": False, "spec": {}}, "models": [], "custom": None}},
    }

    with patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=promise)):
        await base2_svc.load_service(config)


@pytest.mark.asyncio
async def test_load_service_backfills_when_definition_missing(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    installed_value: dict[str, Any] = {}
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value=installed_value)
    config: ServiceRawConfig = {
        "downloaded": None,
        "service_downloaded": False,
        "instances": {
            "default": {
                "options": {"stream": False, "ignore_warnings": False, "spec": {}},
                "models": [{"model_id": "m1", "options": {}, "definition": None}],
                "custom": None,
            }
        },
    }

    with (
        patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=promise)),
        patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=promise)),
        patch.object(base2_svc, "_save", new=AsyncMock()) as save_mock,
    ):
        await base2_svc.load_service(config)

    save_mock.assert_called_once()


@pytest.mark.asyncio
async def test_load_service_skips_backfill_when_definitions_present(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    installed_value: dict[str, Any] = {}
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value=installed_value)
    config: ServiceRawConfig = {
        "downloaded": None,
        "service_downloaded": False,
        "instances": {
            "default": {
                "options": {"stream": False, "ignore_warnings": False, "spec": {}},
                "models": [{"model_id": "m1", "options": {}, "definition": {"hf_id": "m1"}}],
                "custom": None,
            }
        },
    }

    with (
        patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=promise)),
        patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=promise)),
        patch.object(base2_svc, "_save", new=AsyncMock()) as save_mock,
    ):
        await base2_svc.load_service(config)

    save_mock.assert_not_called()

    assert base2_svc.instances_info["default"].installed is installed_value


@pytest.mark.asyncio
async def test_save_calls_service_provider(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()

    await base2_svc._save()  # pyright: ignore[reportPrivateUsage]

    assert base2_deps["service_provider"].save_service_config.call_count == 1


@pytest.mark.asyncio
async def test_save_preserves_model_that_failed_to_load(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    persisted_model = ModelConfig(model_id="m1", options=InstallModelIn(), definition={"hf_id": "m1"})
    base2_svc.instances_info["default"].config.models = [persisted_model]

    with patch.object(base2_svc, "_install_model", new=AsyncMock(side_effect=RuntimeError("boom"))):
        await base2_svc.load_model("default", persisted_model)

    # The service's live registry no longer has the model (it failed to load), matching a real
    # `_generate_instance_config` implementation that only reports live-registered models.
    with patch.object(base2_svc, "_generate_instance_config", return_value=InstanceConfig(options=InstallServiceIn(spec={}), models=[])):
        await base2_svc._save()  # pyright: ignore[reportPrivateUsage]

    saved_models = base2_svc.instances_info["default"].config.models
    assert saved_models is not None
    assert [m.model_id for m in saved_models] == ["m1"]
    assert saved_models[0].definition == {"hf_id": "m1"}


@pytest.mark.asyncio
async def test_load_model_success_clears_failed_marker(base2_svc: _Base2Impl) -> None:
    model = ModelConfig(model_id="m1", options=InstallModelIn())

    with patch.object(base2_svc, "_install_model", new=AsyncMock(side_effect=RuntimeError("boom"))):
        await base2_svc.load_model("default", model)

    assert "m1" in base2_svc._failed_models.get("default", set())  # pyright: ignore[reportPrivateUsage]

    value = InstallModelOut(status="OK", details="done")
    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(value=value)

    with patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=promise)):
        await base2_svc.load_model("default", model)

    assert "m1" not in base2_svc._failed_models.get("default", set())  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_load_model_success_dismisses_matching_warning(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    model = ModelConfig(model_id="m1", options=InstallModelIn())
    value = InstallModelOut(status="OK", details="done")
    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(value=value)

    with patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=promise)):
        await base2_svc.load_model("default", model)

    base2_deps["service_provider"].dismiss_warnings_matching.assert_awaited_once_with(
        base2_svc.get_type(), instance="default", model_id="m1"
    )


@pytest.mark.asyncio
async def test_uninstall_model_stops_preserving_failed_model(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    persisted_model = ModelConfig(model_id="m1", options=InstallModelIn(), definition={"hf_id": "m1"})
    base2_svc.instances_info["default"].config.models = [persisted_model]

    with patch.object(base2_svc, "_install_model", new=AsyncMock(side_effect=RuntimeError("boom"))):
        await base2_svc.load_model("default", persisted_model)

    await base2_svc.uninstall_model("default", "m1", UninstallModelIn(purge=False))

    with patch.object(base2_svc, "_generate_instance_config", return_value=InstanceConfig(options=InstallServiceIn(spec={}), models=[])):
        await base2_svc._save()  # pyright: ignore[reportPrivateUsage]

    assert base2_svc.instances_info["default"].config.models == []


def test_service_config_returns_config(base2_svc: _Base2Impl) -> None:
    instances = {"default": InstanceConfig()}

    cfg = base2_svc.service_config(instances)

    assert cfg.instances == instances
    assert cfg.downloaded == {}
    assert cfg.service_downloaded is False


@pytest.mark.asyncio
async def test_install_instance_raises_if_already_installed(base2_svc: _Base2Impl) -> None:
    base2_svc.instances_info["default"].installed = object()

    with pytest.raises(HTTPException) as exc_info:
        await base2_svc.install_instance("default", InstallServiceIn(spec={}))

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_instance_raises_if_already_installing(base2_svc: _Base2Impl) -> None:
    base2_svc.instances_info["default"].installing = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await base2_svc.install_instance("default", InstallServiceIn(spec={}))

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_instance_happy_path(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    installed_value: dict[str, Any] = {}
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value=installed_value)

    with patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=promise)):
        result = await base2_svc.install_instance("default", InstallServiceIn(spec={}))
        await result.wait()

    assert base2_svc.instances_info["default"].installed is installed_value
    assert base2_svc.instances_info["default"].installing is None


@pytest.mark.asyncio
async def test_uninstall_instance_calls_uninstall_and_save(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    options = UninstallServiceIn(purge=False)

    with patch.object(base2_svc, "_uninstall_instance", new=AsyncMock()) as mock_uninstall:
        await base2_svc.uninstall_instance("default", options)

    assert mock_uninstall.call_count == 1
    assert mock_uninstall.call_args == call("default", options)
    assert base2_deps["service_provider"].save_service_config.call_count == 1
    base2_deps["service_provider"].dismiss_warnings_for_instance.assert_awaited_once_with(base2_svc.get_type(), "default")


@pytest.mark.asyncio
async def test_add_custom_model_base_raises_400(base2_svc: _Base2Impl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await base2_svc.add_custom_model("default", AddCustomModelIn(spec={"name": "test"}))

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_add_custom_model_happy_path(custom_svc: _Base2ImplWithCustom) -> None:
    model_id = await custom_svc.add_custom_model("default", AddCustomModelIn(spec={"name": "test"}))

    assert model_id in custom_svc._custom_store  # pyright: ignore[reportPrivateUsage]
    assert custom_svc.instances_info["default"].config.custom is not None


@pytest.mark.asyncio
async def test_add_custom_model_sets_size_unknown_when_no_size_and_no_resolver(custom_svc: _Base2ImplWithCustom) -> None:
    # When spec has no "size" and _resolve_custom_model_size returns None → "unknown"
    model_id = await custom_svc.add_custom_model("default", AddCustomModelIn(spec={"name": "test"}))

    stored = custom_svc._custom_store[model_id]  # pyright: ignore[reportPrivateUsage]

    assert stored.data["size"] == "unknown"


@pytest.mark.asyncio
async def test_add_custom_model_uses_resolved_size(custom_svc: _Base2ImplWithCustom) -> None:
    # When _resolve_custom_model_size returns a string, it is stored
    with patch.object(custom_svc, "_resolve_custom_model_size", new=AsyncMock(return_value="1.5 GB")):
        model_id = await custom_svc.add_custom_model("default", AddCustomModelIn(spec={"name": "test"}))

    stored = custom_svc._custom_store[model_id]  # pyright: ignore[reportPrivateUsage]

    assert stored.data["size"] == "1.5 GB"


@pytest.mark.asyncio
async def test_add_custom_model_preserves_existing_size(custom_svc: _Base2ImplWithCustom) -> None:
    # When spec already has "size", _resolve_custom_model_size is NOT called
    with patch.object(custom_svc, "_resolve_custom_model_size", new=AsyncMock(return_value="99 GB")) as mock_resolve:
        model_id = await custom_svc.add_custom_model("default", AddCustomModelIn(spec={"name": "test", "size": "2 GB"}))

    stored = custom_svc._custom_store[model_id]  # pyright: ignore[reportPrivateUsage]

    assert stored.data["size"] == "2 GB"
    mock_resolve.assert_not_called()


@pytest.mark.asyncio
async def test_remove_custom_model_base_raises_400(base2_svc: _Base2Impl) -> None:
    base2_svc.instances_info["default"].config.custom = [CustomModel(id="cm1", data={})]

    with pytest.raises(HTTPException) as exc_info:
        await base2_svc.remove_custom_model("default", "cm1")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_remove_custom_model_happy_path(custom_svc: _Base2ImplWithCustom) -> None:
    model_id = await custom_svc.add_custom_model("default", AddCustomModelIn(spec={"x": 1}))
    await custom_svc.remove_custom_model("default", model_id)
    assert model_id not in custom_svc._custom_store  # pyright: ignore[reportPrivateUsage]

    config_custom = custom_svc.instances_info["default"].config.custom or []

    assert all(m.id != model_id for m in config_custom)


@pytest.mark.asyncio
async def test_install_model_happy_path(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    base2_svc._failed_models["default"] = {"m1"}  # pyright: ignore[reportPrivateUsage]
    model_out = InstallModelOut(status="OK", details="done")
    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(value=model_out)

    with patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=promise)):
        result_promise = await base2_svc.install_model("default", "m1", InstallModelIn())
        result = await result_promise.wait()

    assert result.status == "OK"
    assert "m1" not in base2_svc.instances_info["default"].installing_model_progress
    assert "m1" not in base2_svc._failed_models.get("default", set())  # pyright: ignore[reportPrivateUsage]
    base2_deps["service_provider"].dismiss_warnings_matching.assert_awaited_once_with(
        base2_svc.get_type(), instance="default", model_id="m1"
    )


@pytest.mark.asyncio
async def test_install_model_success_with_failing_dismiss_still_completes(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    """A successful install must not fail just because clearing its warnings afterwards errors."""
    base2_deps["service_provider"].save_service_config = AsyncMock()
    base2_deps["service_provider"].dismiss_warnings_matching = AsyncMock(side_effect=RuntimeError("disk full"))
    model_out = InstallModelOut(status="OK", details="done")
    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(value=model_out)

    with patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=promise)):
        result_promise = await base2_svc.install_model("default", "m1", InstallModelIn())
        result = await result_promise.wait()

    assert result.status == "OK"


@pytest.mark.asyncio
async def test_install_model_on_error_cleans_up(base2_svc: _Base2Impl) -> None:
    async def fail_func(stream: Any) -> Any:
        raise RuntimeError("fail")

    fail_promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(func=fail_func)

    with patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=fail_promise)):
        result_promise = await base2_svc.install_model("default", "m1", InstallModelIn())
        with pytest.raises(RuntimeError):
            await result_promise.wait()

    assert "m1" not in base2_svc.instances_info["default"].installing_model_progress


@pytest.mark.asyncio
async def test_install_model_on_error_records_warning(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    async def fail_func(stream: Any) -> Any:
        raise RuntimeError("fail")

    fail_promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(func=fail_func)

    with patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=fail_promise)):
        result_promise = await base2_svc.install_model("default", "m1", InstallModelIn())
        with pytest.raises(RuntimeError):
            await result_promise.wait()

    await asyncio.gather(*base2_svc._warning_tasks)  # pyright: ignore[reportPrivateUsage]

    base2_deps["service_provider"].add_warning.assert_awaited_once()
    call_kwargs = base2_deps["service_provider"].add_warning.await_args.kwargs
    assert call_kwargs["model_id"] == "m1"
    assert call_kwargs["instance"] == "default"


@pytest.mark.asyncio
async def test_install_model_already_installed_fast_path_resolves_immediately(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    """A model that a service's own `_install_model()` recognizes as already installed still resolves right away."""
    base2_deps["service_provider"].save_service_config = AsyncMock()
    value = InstallModelOut(status="OK", details="Already installed or being installed right now.")
    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(value=value)

    with patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=promise)):
        result_promise = await base2_svc.install_model("default", "m1", InstallModelIn())
        result = await result_promise.wait()

    assert result.status == "OK"
    assert "m1" not in base2_svc.instances_info["default"].installing_model_progress


@pytest.mark.asyncio
async def test_install_model_second_call_during_slow_install_reattaches_instead_of_duplicating(
    base2_svc: _Base2Impl, base2_deps: dict[str, Any]
) -> None:
    """Simulates vllm/sglang/mcp's shape: `_install_model()` itself awaits before producing a promise.

    A concurrent second call landing in that window must not start a second real install, and must not
    get an instant fabricated success - it has to wait for and reflect the real, eventual outcome.
    """
    base2_deps["service_provider"].save_service_config = AsyncMock()
    call_count = 0
    resume = asyncio.Event()

    async def slow_install_model(_instance: str, _model_id: str, _options: InstallModelIn) -> Any:
        nonlocal call_count
        call_count += 1
        await resume.wait()
        value = InstallModelOut(status="OK", details="Installed")
        return PromiseWithProgress(value=value)

    with patch.object(base2_svc, "_install_model", new=slow_install_model):
        first_task = asyncio.create_task(base2_svc.install_model("default", "m1", InstallModelIn()))
        await asyncio.sleep(0)  # let the first call reserve the slot and enter the simulated slow work

        second_task = asyncio.create_task(base2_svc.install_model("default", "m1", InstallModelIn()))
        await asyncio.sleep(0)  # let the second call run its check - it must not call _install_model again

        assert call_count == 1

        resume.set()
        first_promise = await first_task
        second_promise = await second_task

    first_result = await first_promise.wait()
    second_result = await second_promise.wait()

    assert first_result.status == "OK"
    assert second_result.status == "OK"
    assert call_count == 1


@pytest.mark.asyncio
async def test_load_model_reservation_prevents_concurrent_install_model_from_duplicating(base2_svc: _Base2Impl) -> None:
    """load_model() (used to restore models at startup) must reserve model_id the same way
    install_model() does. A concurrent install_model() call landing while load_model() is still
    inside its own (possibly slow) `_install_model()` setup must join that same reservation instead
    of racing a duplicate install or getting a fabricated instant "OK".
    """
    call_count = 0
    resume = asyncio.Event()

    async def slow_install_model(_instance: str, _model_id: str, _options: InstallModelIn) -> Any:
        nonlocal call_count
        call_count += 1
        await resume.wait()
        value = InstallModelOut(status="OK", details="Installed")
        return PromiseWithProgress(value=value)

    model = ModelConfig(model_id="m1", options=InstallModelIn())

    with patch.object(base2_svc, "_install_model", new=slow_install_model):
        load_task = asyncio.create_task(base2_svc.load_model("default", model))
        await asyncio.sleep(0)  # let load_model() reserve the slot and enter the simulated slow work

        install_task = asyncio.create_task(base2_svc.install_model("default", "m1", InstallModelIn()))
        await asyncio.sleep(0)  # let install_model() run its check...
        await asyncio.sleep(0)  # ...and, if it wrongly reserved a second time, let that pending_task actually start

        assert call_count == 1
        assert not install_task.done()  # must genuinely wait, not fabricate an instant result

        resume.set()
        await load_task
        install_promise = await install_task

    install_result = await install_promise.wait()
    assert install_result.status == "OK"
    assert call_count == 1


@pytest.mark.asyncio
async def test_install_model_concurrent_callers_run_bookkeeping_only_once(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    """Two callers joining the same install must share one post-install bookkeeping run, not each
    trigger their own copy of it - a second concurrent caller must not cause a second config save or
    a second warnings-dismiss call for what is really one successful install.
    """
    base2_deps["service_provider"].save_service_config = AsyncMock()
    value = InstallModelOut(status="OK", details="Installed")
    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(value=value)

    with patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=promise)):
        first_promise = await base2_svc.install_model("default", "m1", InstallModelIn())
        second_promise = await base2_svc.install_model("default", "m1", InstallModelIn())

    assert first_promise is second_promise

    first_result = await first_promise.wait()
    second_result = await second_promise.wait()

    assert first_result.status == "OK"
    assert second_result.status == "OK"
    assert base2_deps["service_provider"].save_service_config.call_count == 1
    base2_deps["service_provider"].dismiss_warnings_matching.assert_awaited_once()


@pytest.mark.asyncio
async def test_install_model_concurrent_callers_record_only_one_warning_on_failure(
    base2_svc: _Base2Impl, base2_deps: dict[str, Any]
) -> None:
    """Two callers joining an install that ultimately fails must not each record their own duplicate
    warning - exactly one warning should be recorded for the one real failure.
    """
    resume = asyncio.Event()

    async def failing_func(_stream: Stream[StreamChunk]) -> InstallModelOut:
        await resume.wait()
        raise RuntimeError("install blew up")

    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(func=failing_func)

    with patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=promise)):
        first_promise = await base2_svc.install_model("default", "m1", InstallModelIn())
        second_promise = await base2_svc.install_model("default", "m1", InstallModelIn())

    assert first_promise is second_promise

    resume.set()
    with pytest.raises(RuntimeError):
        await first_promise.wait()
    with pytest.raises(RuntimeError):
        await second_promise.wait()

    await asyncio.gather(*base2_svc._warning_tasks)  # pyright: ignore[reportPrivateUsage]

    base2_deps["service_provider"].add_warning.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_model_install_progress_reattach_shows_history_then_live_updates(
    base2_svc: _Base2Impl, base2_deps: dict[str, Any]
) -> None:
    """A caller that joins an in-progress install via the progress endpoint sees prior progress, not a blank slate."""
    base2_deps["service_provider"].save_service_config = AsyncMock()
    reached_midpoint = asyncio.Event()
    proceed = asyncio.Event()

    async def func(stream: Stream[StreamChunk]) -> InstallModelOut:
        stream.emit(StreamChunkProgress(type="progress", stage="download", value=0.3, data={}))
        reached_midpoint.set()
        await proceed.wait()
        stream.emit(StreamChunkProgress(type="progress", stage="download", value=0.9, data={}))
        return InstallModelOut(status="OK", details="done")

    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(func=func)

    with patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=promise)):
        await base2_svc.install_model("default", "m1", InstallModelIn())
        await reached_midpoint.wait()

        rejoined_promise = await base2_svc.get_model_install_progress("default", "m1")
        seen: list[StreamChunk] = []

        async def collect() -> None:
            async for chunk in rejoined_promise.progress.as_generator():
                seen.append(chunk)

        collector = asyncio.create_task(collect())
        await asyncio.sleep(0)

        assert {"type": "progress", "stage": "download", "value": 0.3, "data": {}} in seen

        proceed.set()
        await collector

    assert any(chunk.get("type") == "progress" and chunk.get("value") == 0.9 for chunk in seen)


@pytest.mark.asyncio
async def test_get_model_install_progress_times_out_when_setup_hangs(base2_svc: _Base2Impl) -> None:
    """A genuinely hung pre-promise setup must not block a progress-watcher forever - it should
    time out with a clean error instead.
    """
    reached = asyncio.Event()

    async def hanging_install_model(_instance: str, _model_id: str, _options: InstallModelIn) -> Any:
        reached.set()
        await asyncio.Event().wait()  # never completes

    with (
        patch.object(base2_svc, "_install_model", new=hanging_install_model),
        patch("server.services.base2_service._INSTALL_STATUS_WAIT_TIMEOUT_SECONDS", 0.05),
    ):
        install_task = asyncio.create_task(base2_svc.install_model("default", "m1", InstallModelIn()))
        await reached.wait()

        with pytest.raises(HTTPException) as exc_info:
            await base2_svc.get_model_install_progress("default", "m1")
        assert exc_info.value.status_code == 504

        install_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await install_task


@pytest.mark.asyncio
async def test_install_model_join_branch_times_out_when_setup_hangs(base2_svc: _Base2Impl) -> None:
    """A concurrent caller joining an install stuck in a genuinely hung pre-promise setup must not
    wait forever for it - it should time out with a clean error, and the reservation must be left
    untouched so the real eventual outcome is still observable afterward.
    """
    reached = asyncio.Event()

    async def hanging_install_model(_instance: str, _model_id: str, _options: InstallModelIn) -> Any:
        reached.set()
        await asyncio.Event().wait()  # never completes

    with (
        patch.object(base2_svc, "_install_model", new=hanging_install_model),
        patch("server.services.base2_service._INSTALL_STATUS_WAIT_TIMEOUT_SECONDS", 0.05),
    ):
        first_task = asyncio.create_task(base2_svc.install_model("default", "m1", InstallModelIn()))
        await reached.wait()

        with pytest.raises(HTTPException) as exc_info:
            await base2_svc.install_model("default", "m1", InstallModelIn())
        assert exc_info.value.status_code == 504

        assert "m1" in base2_svc.instances_info["default"].installing_model_progress

        first_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_task


@pytest.mark.asyncio
async def test_cancel_model_install_times_out_and_leaves_reservation_untouched(base2_svc: _Base2Impl) -> None:
    """If the pending setup doesn't respond to cancellation in time, cancel_model_install() must
    time out with a clean error rather than hanging forever - and must leave the reservation as-is,
    since it genuinely doesn't know whether the cancel eventually lands or the install keeps going.
    """
    reached = asyncio.Event()

    async def stubborn_install_model(_instance: str, _model_id: str, _options: InstallModelIn) -> Any:
        reached.set()
        # Swallow exactly one cancellation to simulate setup that doesn't respond within the
        # timeout window - but still respond to a second cancel, so the test can clean up its task.
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(1000)
        await asyncio.sleep(1000)

    with (
        patch.object(base2_svc, "_install_model", new=stubborn_install_model),
        patch("server.services.base2_service._CANCEL_WAIT_TIMEOUT_SECONDS", 0.05),
    ):
        install_task = asyncio.create_task(base2_svc.install_model("default", "m1", InstallModelIn()))
        await reached.wait()

        with pytest.raises(HTTPException) as exc_info:
            await base2_svc.cancel_model_install("default", "m1")
        assert exc_info.value.status_code == 504

        assert "m1" in base2_svc.instances_info["default"].installing_model_progress

        install_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await install_task


@pytest.mark.asyncio
async def test_install_model_reservation_removed_when_install_model_raises_before_promise(
    base2_svc: _Base2Impl, base2_deps: dict[str, Any]
) -> None:
    """If `_install_model()` itself raises before ever producing a promise, the model must not get stuck as installing."""
    base2_deps["service_provider"].save_service_config = AsyncMock()

    async def raising_install_model(_instance: str, _model_id: str, _options: InstallModelIn) -> Any:
        raise HTTPException(400, "bad request")

    with patch.object(base2_svc, "_install_model", new=raising_install_model), pytest.raises(HTTPException):
        await base2_svc.install_model("default", "m1", InstallModelIn())

    assert "m1" not in base2_svc.instances_info["default"].installing_model_progress

    value = InstallModelOut(status="OK", details="done")
    promise: PromiseWithProgress[InstallModelOut, StreamChunk] = PromiseWithProgress(value=value)

    with patch.object(base2_svc, "_install_model", new=AsyncMock(return_value=promise)):
        result_promise = await base2_svc.install_model("default", "m1", InstallModelIn())
        result = await result_promise.wait()

    assert result.status == "OK"


@pytest.mark.asyncio
async def test_install_model_concurrent_waiter_gets_error_instead_of_hanging_when_install_model_raises(
    base2_svc: _Base2Impl,
) -> None:
    """A caller blocked on an in-flight reservation must learn of the real failure, not hang forever."""
    reached_slow_point = asyncio.Event()
    fail_now = asyncio.Event()

    async def slow_failing_install_model(_instance: str, _model_id: str, _options: InstallModelIn) -> Any:
        reached_slow_point.set()
        await fail_now.wait()
        raise RuntimeError("install blew up")

    with patch.object(base2_svc, "_install_model", new=slow_failing_install_model):
        first_task = asyncio.create_task(base2_svc.install_model("default", "m1", InstallModelIn()))
        await reached_slow_point.wait()

        second_task = asyncio.create_task(base2_svc.install_model("default", "m1", InstallModelIn()))
        await asyncio.sleep(0)  # let the second call reserve-check and start waiting on the first's reservation

        fail_now.set()

        with pytest.raises(RuntimeError, match="install blew up"):
            await asyncio.wait_for(first_task, timeout=1)
        with pytest.raises(RuntimeError, match="install blew up"):
            await asyncio.wait_for(second_task, timeout=1)

    assert "m1" not in base2_svc.instances_info["default"].installing_model_progress


@pytest.mark.asyncio
async def test_install_model_concurrent_waiter_gets_error_instead_of_hanging_when_instance_removed_mid_install(
    base2_svc: _Base2Impl,
) -> None:
    """The instance can be removed (uninstall_instance(purge=True)) while one of its models is still
    mid-install - it doesn't wait on in-progress installs. A caller blocked on that reservation must
    still learn of the real failure once it lands, not hang forever because cleanup can no longer
    find the now-missing instance.
    """
    reached_slow_point = asyncio.Event()
    fail_now = asyncio.Event()

    async def slow_failing_install_model(_instance: str, _model_id: str, _options: InstallModelIn) -> Any:
        reached_slow_point.set()
        await fail_now.wait()
        raise RuntimeError("install blew up")

    with patch.object(base2_svc, "_install_model", new=slow_failing_install_model):
        first_task = asyncio.create_task(base2_svc.install_model("default", "m1", InstallModelIn()))
        await reached_slow_point.wait()

        second_task = asyncio.create_task(base2_svc.install_model("default", "m1", InstallModelIn()))
        await asyncio.sleep(0)  # let the second call reserve-check and start waiting on the first's reservation

        del base2_svc.instances_info["default"]  # simulates uninstall_instance(purge=True)
        fail_now.set()

        with pytest.raises(RuntimeError, match="install blew up"):
            await asyncio.wait_for(first_task, timeout=1)
        with pytest.raises(RuntimeError, match="install blew up"):
            await asyncio.wait_for(second_task, timeout=1)


@pytest.mark.asyncio
async def test_install_model_reservation_cleaned_up_when_install_model_cancelled(base2_svc: _Base2Impl) -> None:
    """A cancelled `_install_model()` call (e.g. client disconnect) must clean up its reservation too, not just ordinary exceptions."""
    started = asyncio.Event()

    async def hanging_install_model(_instance: str, _model_id: str, _options: InstallModelIn) -> Any:
        started.set()
        await asyncio.Event().wait()  # never completes on its own - must be cancelled to end

    with patch.object(base2_svc, "_install_model", new=hanging_install_model):
        task = asyncio.create_task(base2_svc.install_model("default", "m1", InstallModelIn()))
        await started.wait()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert "m1" not in base2_svc.instances_info["default"].installing_model_progress


@pytest.mark.asyncio
async def test_uninstall_model_calls_uninstall_and_save(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    options = UninstallModelIn()

    with patch.object(base2_svc, "_uninstall_model", new=AsyncMock()) as mock_uninstall:
        await base2_svc.uninstall_model("default", "m1", options)

    assert mock_uninstall.call_count == 1
    assert mock_uninstall.call_args == call("default", "m1", options)


@pytest.mark.asyncio
async def test_uninstall_model_with_failing_dismiss_still_completes(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    """A successful uninstall must not fail just because clearing its warnings afterwards errors."""
    base2_deps["service_provider"].save_service_config = AsyncMock()
    base2_deps["service_provider"].dismiss_warnings_matching = AsyncMock(side_effect=RuntimeError("disk full"))
    options = UninstallModelIn()

    with patch.object(base2_svc, "_uninstall_model", new=AsyncMock()):
        await base2_svc.uninstall_model("default", "m1", options)


@pytest.mark.asyncio
async def test_get_docker_logs_calls_docker_service(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["docker_service"].get_docker_compose_logs = AsyncMock(return_value="logs")

    with patch.object(base2_svc, "get_docker_compose_file_path", return_value=Path("/tmp/dc.yml")):
        result = await base2_svc.get_docker_logs("default", None)

    assert result == "logs"


def test_get_model_installed_info_returns_false_when_not_in_progress(base2_svc: _Base2Impl) -> None:
    assert base2_svc._get_model_installed_info("default", "m1") is False  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("last_chunk", "expected_stage", "expected_value"),
    [
        ({"type": "progress", "stage": "download", "value": 0.5}, "download", 0.5),
        ({"type": "finish"}, "install", 1),
        (None, "download", 0),
    ],
)
def test_get_model_installed_info_chunk(
    base2_svc: _Base2Impl,
    last_chunk: dict[str, object] | None,
    expected_stage: str,
    expected_value: float,
) -> None:
    mock_installing = MagicMock()
    mock_installing.last_chunk = last_chunk
    base2_svc.instances_info["default"].installing_model_progress["m1"] = mock_installing

    result = base2_svc._get_model_installed_info("default", "m1")  # pyright: ignore[reportPrivateUsage]

    assert result.stage == expected_stage  # type: ignore[union-attr]
    assert result.value == expected_value  # type: ignore[union-attr]


def test_get_service_installed_info_returns_false_when_not_installing(base2_svc: _Base2Impl) -> None:
    assert base2_svc._get_service_installed_info("default") is False  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("last_chunk", "expected_stage", "expected_value"),
    [
        ({"type": "progress", "stage": "download", "value": 0.3}, "download", 0.3),
        ({"type": "finish"}, "install", 1),
        (None, "download", 0),
    ],
)
def test_get_service_installed_info_chunk(
    base2_svc: _Base2Impl,
    last_chunk: dict[str, object] | None,
    expected_stage: str,
    expected_value: float,
) -> None:
    mock_installing = MagicMock()
    mock_installing.last_chunk = last_chunk
    base2_svc.instances_info["default"].installing = mock_installing

    result = base2_svc._get_service_installed_info("default")  # pyright: ignore[reportPrivateUsage]

    assert result.stage == expected_stage  # type: ignore[union-attr]
    assert result.value == expected_value  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_get_docker_compose_file_reads_file(base2_svc: _Base2Impl) -> None:
    with (
        patch.object(base2_svc, "get_docker_compose_file_path", return_value=Path("/tmp/dc.yml")),
        patch("server.services.base2_service.Utils.read_file", new=AsyncMock(return_value="content")),
    ):
        result = await base2_svc.get_docker_compose_file("default", None)

    assert result == "content"


@pytest.mark.asyncio
async def test_restart_docker_calls_docker_service(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["docker_service"].restart_docker_compose = AsyncMock()

    with patch.object(base2_svc, "get_docker_compose_file_path", return_value=Path("/tmp/dc.yml")):
        await base2_svc.restart_docker("default", None)

    assert base2_deps["docker_service"].restart_docker_compose.call_count == 1
    assert base2_deps["docker_service"].restart_docker_compose.call_args == call(Path("/tmp/dc.yml"))


def test_get_docker_compose_file_path_raises_when_not_installed(base2_svc: _Base2Impl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        base2_svc.get_docker_compose_file_path("default", None)

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_raises_no_docker_when_installed(base2_svc: _Base2Impl) -> None:
    base2_svc.instances_info["default"].installed = object()

    with pytest.raises(HTTPException) as exc_info:
        base2_svc.get_docker_compose_file_path("default", None)

    assert exc_info.value.status_code == 400


def test_get_service_dir_creates_dir_if_not_exists(base2_svc: _Base2Impl, base2_deps: dict[str, Any], tmp_path: Path) -> None:
    base2_deps["config"].get_storage_services_dir.return_value = tmp_path

    result = base2_svc._get_service_dir("my-svc")  # pyright: ignore[reportPrivateUsage]

    assert result.is_dir()
    assert result == tmp_path / "my-svc"


@pytest.mark.asyncio
async def test_clear_working_dir_removes_directory(base2_svc: _Base2Impl, tmp_path: Path) -> None:
    working_dir = tmp_path / "test-b2"
    working_dir.mkdir()

    with patch.object(base2_svc, "_get_working_dir", return_value=working_dir):  # pyright: ignore[reportPrivateUsage]
        await base2_svc._clear_working_dir()  # pyright: ignore[reportPrivateUsage]

    assert not working_dir.exists()


def test_get_instance_installed_info_raises_when_not_installed(base2_svc: _Base2Impl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        base2_svc.get_instance_installed_info("default")

    assert exc_info.value.status_code == 400


def test_get_instance_installed_info_returns_value_when_installed(base2_svc: _Base2Impl) -> None:
    sentinel = object()

    base2_svc.instances_info["default"].installed = sentinel

    assert base2_svc.get_instance_installed_info("default") is sentinel


def test_get_hugging_face_token(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["config"].hugging_face_token = SecretStr("hf-abc")

    assert base2_svc.get_hugging_face_token() == "hf-abc"


def test_get_civitai_token(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["config"].civitai_token = SecretStr("civ-xyz")

    assert base2_svc.get_civitai_token() == "civ-xyz"


def test_has_gpu_for_spec_true(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["docker_service"].has_gpu_support = True

    assert base2_svc._has_gpu_for_spec() == "true"  # pyright: ignore[reportPrivateUsage]


def test_has_gpu_for_spec_false(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["docker_service"].has_gpu_support = False

    assert base2_svc._has_gpu_for_spec() == "false"  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_download_image_or_set_progress_forwards_existing_stream(base2_svc: _Base2Impl) -> None:
    existing_stream: Stream[StreamChunk] = Stream()
    chunk = StreamChunkProgress(type="progress", stage="download", value=0.5, data={})
    existing_stream.emit(chunk)
    existing_stream.close()
    image = DockerImage(name="test-image", size="1GB")
    base2_svc.images_download_progress[image.name] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    await base2_svc._download_image_or_set_progress(output_stream, image)  # pyright: ignore[reportPrivateUsage]

    assert output_stream.emit.call_count == 1
    assert output_stream.emit.call_args == call(chunk)


@pytest.mark.asyncio
async def test_download_image_or_set_progress_breaks_on_non_download_chunk(base2_svc: _Base2Impl) -> None:
    existing_stream: Stream[StreamChunk] = Stream()
    finish_chunk: StreamChunk = {"type": "finish", "status": "ok"}  # type: ignore[assignment]
    existing_stream.emit(finish_chunk)
    existing_stream.close()
    image = DockerImage(name="test-image", size="1GB")
    base2_svc.images_download_progress[image.name] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    await base2_svc._download_image_or_set_progress(output_stream, image)  # pyright: ignore[reportPrivateUsage]

    assert output_stream.emit.call_count == 0


@pytest.mark.asyncio
async def test_docker_pull_emits_progress_when_image_not_present(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    image = DockerImage(name="my-image", size="1GB")
    stream = MagicMock()
    base2_deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=False)
    base2_deps["docker_service"].get_docker_image_size = AsyncMock(return_value=1024)

    async def mock_docker_pull(*args: Any):  # type: ignore[misc]
        yield 0.5

    base2_deps["docker_service"].docker_pull = mock_docker_pull

    await base2_svc._docker_pull(image, stream)  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_count >= 3


@pytest.mark.asyncio
async def test_stop_docker_logs_error_on_exception(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["docker_service"].stop_docker = AsyncMock(side_effect=RuntimeError("boom"))
    docker_options = MagicMock()

    await base2_svc._stop_docker(docker_options)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_verify_docker_image_raises_when_warnings_not_ignored(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["docker_service"].get_image_warnings = AsyncMock(return_value=["w1"])

    with pytest.raises(HTTPException) as exc_info:
        await base2_svc._verify_docker_image("img", False)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_verify_docker_image_passes_when_warnings_ignored(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["docker_service"].get_image_warnings = AsyncMock(return_value=["w1"])

    await base2_svc._verify_docker_image("img", True)  # pyright: ignore[reportPrivateUsage]


def test_get_specified_hardware_parts_matches_gpu_by_name(base2_deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 3090", vram="24 GB", id=0)
    hw = MagicMock()
    hw.gpus = [gpu]
    base2_deps["hardware"] = hw
    svc = _Base2Impl(**base2_deps)

    result = svc.get_specified_hardware_parts(f"GPU | {gpu.long_name}")

    assert gpu in result


def test_get_specified_hardware_parts_no_match_returns_empty(base2_deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 3090", vram="24 GB", id=0)
    hw = MagicMock()
    hw.gpus = [gpu]
    base2_deps["hardware"] = hw
    svc = _Base2Impl(**base2_deps)

    result = svc.get_specified_hardware_parts("GPU | Unknown GPU | 0")

    assert list(result) == []


def test_add_hardware_field_to_spec_no_cpu_without_avx512(base2_deps: dict[str, Any]) -> None:
    hw = MagicMock()
    hw.gpus = []
    hw.cpu.avx512 = False
    base2_deps["hardware"] = hw
    svc = _Base2Impl(**base2_deps)

    fields = svc.add_hardware_field_to_spec(add_cpu_option_only_on_avx512_support=True)

    hw_field = next(f for f in fields if f.name == "hardware")
    assert "CPU" not in [getattr(v, "value", v) for v in hw_field.values]  # pyright: ignore[reportOptionalIterable]


def test_add_hardware_field_to_spec_single_gpu(base2_deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 3090", vram="24 GB", id=0)
    hw = MagicMock()
    hw.gpus = [gpu]
    hw.cpu.avx512 = True
    base2_deps["hardware"] = hw
    svc = _Base2Impl(**base2_deps)

    fields = svc.add_hardware_field_to_spec()

    hw_field = next(f for f in fields if f.name == "hardware")
    assert hw_field.default == "GPU"


def test_add_hardware_field_to_spec_multiple_gpus(base2_deps: dict[str, Any]) -> None:
    gpu1 = NvidiaGpuInfo(name="RTX 3090", vram="24 GB", id=0)
    gpu2 = NvidiaGpuInfo(name="RTX 4090", vram="24 GB", id=1)
    hw = MagicMock()
    hw.gpus = [gpu1, gpu2]
    hw.cpu.avx512 = True
    base2_deps["hardware"] = hw
    svc = _Base2Impl(**base2_deps)

    fields = svc.add_hardware_field_to_spec()

    hw_field = next(f for f in fields if f.name == "hardware")
    assert hw_field.default == "GPUs"


def test_add_hardware_field_to_model_spec_no_cpu_without_avx512(base2_deps: dict[str, Any]) -> None:
    hw = MagicMock()
    hw.gpus = []
    hw.cpu.avx512 = False
    base2_deps["hardware"] = hw
    svc = _Base2Impl(**base2_deps)

    fields = svc.add_hardware_field_to_model_spec(add_cpu_option_only_on_avx512_support=True)

    hw_field = next(f for f in fields if f.name == "hardware")
    assert "CPU" not in [getattr(v, "value", v) for v in hw_field.values]  # pyright: ignore[reportOptionalIterable]


def test_add_hardware_field_to_model_spec_multiple_gpus(base2_deps: dict[str, Any]) -> None:
    gpu1 = NvidiaGpuInfo(name="RTX 3090", vram="24 GB", id=0)
    gpu2 = NvidiaGpuInfo(name="RTX 4090", vram="24 GB", id=1)
    hw = MagicMock()
    hw.gpus = [gpu1, gpu2]
    hw.cpu.avx512 = True
    base2_deps["hardware"] = hw
    svc = _Base2Impl(**base2_deps)

    fields = svc.add_hardware_field_to_model_spec()

    hw_field = next(f for f in fields if f.name == "hardware")
    assert hw_field.default == "GPUs"


def test_load_download_info_abstract_body_returns_none(base2_svc: _Base2Impl) -> None:
    result = Base2Service._load_download_info(base2_svc, {})  # type: ignore[arg-type] # pyright: ignore[reportPrivateUsage]
    assert result is None


@pytest.mark.asyncio
async def test_install_instance_new_instance_skips_installed_check(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    installed_value: dict[str, Any] = {}
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value=installed_value)

    with patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=promise)):
        result = await base2_svc.install_instance("brand-new-instance", InstallServiceIn(spec={}))
        await result.wait()

    assert base2_svc.instances_info["brand-new-instance"].installed is installed_value


@pytest.mark.asyncio
async def test_add_custom_model_with_existing_custom_list(custom_svc: _Base2ImplWithCustom) -> None:
    custom_svc.instances_info["default"].config.custom = []

    model_id = await custom_svc.add_custom_model("default", AddCustomModelIn(spec={"name": "test"}))

    assert model_id in custom_svc._custom_store  # pyright: ignore[reportPrivateUsage]
    assert len(custom_svc.instances_info["default"].config.custom) == 1  # pyright: ignore[reportArgumentType]


@pytest.mark.asyncio
async def test_clear_working_dir_no_op_when_not_exists(base2_svc: _Base2Impl, tmp_path: Path) -> None:
    non_existent = tmp_path / "does-not-exist"

    with patch.object(base2_svc, "_get_working_dir", return_value=non_existent):  # pyright: ignore[reportPrivateUsage]
        await base2_svc._clear_working_dir()  # pyright: ignore[reportPrivateUsage]

    assert not non_existent.exists()


@pytest.mark.asyncio
async def test_base_service_get_loaded_model_info_returns_none(base_svc: _BaseImpl) -> None:
    result = await base_svc.get_loaded_model_info("default")

    assert result is None


@pytest.mark.asyncio
async def test_update_instance_raises_when_not_installed(base2_svc: _Base2Impl) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await base2_svc.update_instance("default", InstallServiceIn(spec={}))

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_update_instance_raises_when_already_installing(base2_svc: _Base2Impl) -> None:
    base2_svc.instances_info["default"].installed = object()
    base2_svc.instances_info["default"].installing = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await base2_svc.update_instance("default", InstallServiceIn(spec={}))

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_update_instance_tears_down_then_reinstalls(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    base2_svc.instances_info["default"].installed = object()
    new_installed: dict[str, Any] = {}
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value=new_installed)

    with (
        patch.object(base2_svc, "_uninstall_instance", new=AsyncMock()) as uninstall_mock,
        patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=promise)),
    ):
        result_promise = await base2_svc.update_instance("default", InstallServiceIn(spec={"key": "new"}))
        await result_promise.wait()

    assert uninstall_mock.await_args is not None
    assert uninstall_mock.await_args.args[1].purge is False
    assert base2_svc.instances_info["default"].installed is new_installed
    assert base2_svc.instances_info["default"].installing is None


@pytest.mark.asyncio
async def test_update_instance_validates_options_before_uninstalling(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    base2_svc.instances_info["default"].installed = object()
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value={})
    calls: list[str] = []

    async def record_validate(_options: InstallServiceIn) -> None:
        calls.append("validate")

    async def record_uninstall(_instance: str, _options: Any) -> None:
        calls.append("uninstall")

    with (
        patch.object(base2_svc, "_validate_update_options", new=record_validate),
        patch.object(base2_svc, "_uninstall_instance", new=record_uninstall),
        patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=promise)),
    ):
        result_promise = await base2_svc.update_instance("default", InstallServiceIn(spec={}))
        await result_promise.wait()

    assert calls == ["validate", "uninstall"]


@pytest.mark.asyncio
async def test_update_instance_does_not_uninstall_when_validation_rejects_options(
    base2_svc: _Base2Impl, base2_deps: dict[str, Any]
) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    base2_svc.instances_info["default"].installed = object()

    async def reject(_options: InstallServiceIn) -> None:
        raise HTTPException(400, "Docker image tag 'bogus' is not available for this service")

    with (
        patch.object(base2_svc, "_validate_update_options", new=reject),
        patch.object(base2_svc, "_uninstall_instance", new=AsyncMock()) as uninstall_mock,
        patch.object(base2_svc, "_install_instance", new=AsyncMock()),
        pytest.raises(HTTPException) as exc_info,
    ):
        await base2_svc.update_instance("default", InstallServiceIn(spec={"image_version": "bogus"}))

    assert exc_info.value.status_code == 400
    uninstall_mock.assert_not_awaited()
    assert base2_svc.instances_info["default"].installed is not None


@pytest.mark.asyncio
async def test_update_instance_reloads_preserved_models(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    base2_svc.instances_info["default"].installed = object()
    preserved = ModelConfig(model_id="m1", options=InstallModelIn())
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value={})

    with (
        patch.object(base2_svc, "_uninstall_instance", new=AsyncMock()),
        patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=promise)),
        patch.object(base2_svc, "_generate_instance_config", return_value=InstanceConfig(models=[preserved])),
        patch.object(base2_svc, "load_model", new=AsyncMock()) as load_model_mock,
    ):
        result_promise = await base2_svc.update_instance("default", InstallServiceIn(spec={}))
        await result_promise.wait()

    load_model_mock.assert_awaited_once_with("default", preserved)


@pytest.mark.asyncio
async def test_update_instance_success_with_failing_dismiss_still_completes(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    """A successful update must not fail just because clearing its warnings afterwards errors."""
    base2_deps["service_provider"].save_service_config = AsyncMock()
    base2_deps["service_provider"].dismiss_warnings_matching_any = AsyncMock(side_effect=RuntimeError("disk full"))
    base2_svc.instances_info["default"].installed = object()
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value={})

    with (
        patch.object(base2_svc, "_uninstall_instance", new=AsyncMock()),
        patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=promise)),
        patch.object(base2_svc, "_generate_instance_config", return_value=InstanceConfig(models=[])),
    ):
        result_promise = await base2_svc.update_instance("default", InstallServiceIn(spec={}))
        result = await result_promise.wait()

    assert result.status == "OK"


@pytest.mark.asyncio
async def test_update_instance_retries_previously_failed_model(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_deps["service_provider"].save_service_config = AsyncMock()
    base2_svc.instances_info["default"].installed = object()
    persisted_model = ModelConfig(model_id="m1", options=InstallModelIn(), definition={"hf_id": "m1"})
    base2_svc.instances_info["default"].config.models = [persisted_model]
    base2_svc._failed_models["default"] = {"m1"}  # pyright: ignore[reportPrivateUsage]
    promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(value={})

    with (
        patch.object(base2_svc, "_uninstall_instance", new=AsyncMock()),
        patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=promise)),
        # `m1` failed to load before the update, so it's absent from the live registry `_generate_instance_config` reports.
        patch.object(base2_svc, "_generate_instance_config", return_value=InstanceConfig(models=[])),
        patch.object(base2_svc, "load_model", new=AsyncMock()) as load_model_mock,
    ):
        result_promise = await base2_svc.update_instance("default", InstallServiceIn(spec={}))
        await result_promise.wait()

    load_model_mock.assert_awaited_once_with("default", persisted_model)


@pytest.mark.asyncio
async def test_update_instance_on_error_clears_installing(base2_svc: _Base2Impl) -> None:
    base2_svc.instances_info["default"].installed = object()

    async def fail_func(stream: Any) -> Any:
        raise RuntimeError("update failed")

    fail_promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(func=fail_func)

    with (
        patch.object(base2_svc, "_uninstall_instance", new=AsyncMock()),
        patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=fail_promise)),
    ):
        result_promise = await base2_svc.update_instance("default", InstallServiceIn(spec={}))
        with pytest.raises(RuntimeError):
            await result_promise.wait()

    assert base2_svc.instances_info["default"].installing is None


@pytest.mark.asyncio
async def test_update_instance_on_error_records_warning(base2_svc: _Base2Impl, base2_deps: dict[str, Any]) -> None:
    base2_svc.instances_info["default"].installed = object()

    async def fail_func(stream: Any) -> Any:
        raise RuntimeError("update failed")

    fail_promise: PromiseWithProgress[Any, StreamChunk] = PromiseWithProgress(func=fail_func)

    with (
        patch.object(base2_svc, "_uninstall_instance", new=AsyncMock()),
        patch.object(base2_svc, "_install_instance", new=AsyncMock(return_value=fail_promise)),
    ):
        result_promise = await base2_svc.update_instance("default", InstallServiceIn(spec={}))
        with pytest.raises(RuntimeError):
            await result_promise.wait()
        await base2_svc.drain_warning_tasks()

    base2_deps["service_provider"].add_warning.assert_awaited_once()
    call_kwargs = base2_deps["service_provider"].add_warning.await_args.kwargs
    assert call_kwargs["instance"] == "default"


@pytest.mark.asyncio
async def test_reconcile_hooks_default_to_not_implemented(base2_svc: _Base2Impl) -> None:
    with pytest.raises(NotImplementedError):
        base2_svc._reconcile_container_name("default", None, None)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(NotImplementedError):
        await base2_svc._release_dead_model("default", None, "m1", None)  # pyright: ignore[reportPrivateUsage]
