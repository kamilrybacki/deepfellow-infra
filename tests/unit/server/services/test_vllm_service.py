# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import asyncio
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import HTTPException

from scripts.get_huggingface_models import ModelIncompatibleError
from server.docker import ContainerStatus
from server.models.models import AddCustomModelIn, InstallModelIn, ListModelsFilters, UninstallModelIn
from server.models.services import GpuStats, InstallServiceIn, UninstallServiceIn
from server.services.base2_service import CustomModel, Instance, InstanceConfig, ModelConfig
from server.services.model_catalog_refresh import huggingface_refresh_guard
from server.services.vllm_service import (
    DownloadedInfo,
    InstalledInfo,
    ModelInstalledInfo,
    VllmModel,
    VllmModelOptions,
    VllmOptions,
    VllmService,
    _const,  # pyright: ignore[reportPrivateUsage]
)
from server.utils.core import (
    CommandResult,
    DownloadedPacket,
    PreDownloadPacket,
    Stream,
    StreamChunk,
    StreamChunkProgress,
    SuccessDownloadPacket,
)
from server.utils.exceptions import AppError
from server.utils.hardware import NvidiaGpuInfo


@pytest.fixture
def deps() -> dict[str, Any]:
    hw = MagicMock()
    hw.cpu.avx512 = True
    hw.gpus = []
    hw.nvidia_gpus = []
    hw.intel_gpus = []
    return {
        "config": MagicMock(),
        "endpoint_registry": MagicMock(),
        "service_provider": MagicMock(),
        "model_downloader": MagicMock(),
        "docker_service": MagicMock(),
        "hardware": hw,
    }


@pytest.fixture
def svc(deps: dict[str, Any]) -> VllmService:
    return VllmService(**deps)


def _make_installed_info(hardware: str | bool | None = False) -> InstalledInfo:
    return InstalledInfo(
        models={},
        options=InstallServiceIn(spec={}),
        parsed_options=VllmOptions(hardware=hardware),
    )


def _make_model_installed_info(
    model_id: str = "test-model",
    registration_id: str = "reg-1",
    gpu_memory_utilization: float | None = None,
    model_type: str = "llm",
) -> ModelInstalledInfo:
    docker = MagicMock()
    docker.name = f"df-vllm-{model_id}"
    return ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        options=InstallModelIn(spec={}),
        docker=docker,
        container_host="localhost",
        container_port=8000,
        docker_exposed_port=8000,
        registration_id=registration_id,
        model_path=Path("/tmp/model"),
        base_url="http://localhost:8000",
        gpu_memory_utilization=gpu_memory_utilization,
        model_type=model_type,  # type: ignore[arg-type]
    )


def _setup_install_mocks(svc: VllmService, deps: dict[str, Any], hardware: str | bool | None = False) -> InstalledInfo:
    installed = _make_installed_info(hardware=hardware)
    svc.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8000, True, False))
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8000
    return installed


def test_get_type(svc: VllmService) -> None:
    assert svc.get_type() == "vllm"


def test_get_description_not_empty(svc: VllmService) -> None:
    assert svc.get_description()


def test_default_models_loaded(svc: VllmService) -> None:
    assert "default" in svc.models
    assert len(svc.models["default"]) > 0


def test_get_size_cpu_only(deps: dict[str, Any]) -> None:
    deps["hardware"].cpu.avx512 = True
    deps["hardware"].gpus = []
    svc = VllmService(**deps)

    sizes = svc.get_size()

    assert "cpu" in sizes
    assert "gpu" not in sizes


def test_get_size_with_gpu(deps: dict[str, Any]) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc = VllmService(**deps)

    sizes = svc.get_size()

    assert "gpu" in sizes


def test_get_spec_has_hardware_field(svc: VllmService) -> None:
    spec = svc.get_spec()

    field_names = [f.name for f in spec.fields]
    assert "hardware" in field_names


def test_const_has_cpu_and_gpu_images() -> None:
    assert "cpu" in _const.images
    assert "gpu" in _const.images
    assert _const.images["cpu"].name
    assert _const.images["gpu"].name


def test_get_model_spec_baseline_fields(svc: VllmService) -> None:
    installed = _make_installed_info(hardware=False)
    svc.instances_info["default"].installed = installed

    spec = svc.get_model_spec("default", "llm")

    field_names = [f.name for f in spec.fields]
    for expected in ("alias", "max_model_length", "quantization", "extra_args", "extra_envs"):
        assert expected in field_names


def test_get_model_spec_adds_gpu_memory_utilization_for_gpu_hardware(svc: VllmService) -> None:
    svc.instances_info["default"].config.options = InstallServiceIn(spec={"hardware": True})

    spec = svc.get_model_spec("default", "llm")

    field_names = [f.name for f in spec.fields]
    assert "gpu_memory_utilization" in field_names


def test_get_model_spec_omits_gpu_memory_utilization_for_cpu_hardware(svc: VllmService) -> None:
    spec = svc.get_model_spec("default", "llm")

    field_names = [f.name for f in spec.fields]
    assert "gpu_memory_utilization" not in field_names


def test_get_model_spec_reranker_default_extra_args(svc: VllmService) -> None:
    spec = svc.get_model_spec("default", "reranker")

    extra_args_field = next(f for f in spec.fields if f.name == "extra_args")
    assert extra_args_field.default is not None
    assert "--trust-remote-code" in (extra_args_field.default or "")


def test_get_custom_model_spec_not_none(svc: VllmService) -> None:
    result = svc.get_custom_model_spec()

    assert result is not None
    field_names = [f.name for f in result.fields]
    assert "id" in field_names
    assert "hf_id" in field_names
    assert "revision" in field_names
    assert "size" in field_names


def test_model_installed_info_get_info() -> None:
    model = _make_model_installed_info(registration_id="reg-42")

    info = model.get_info()

    assert info.registration_id == "reg-42"


def test_get_installed_info_returns_spec_when_installed(svc: VllmService) -> None:
    installed = _make_installed_info()
    installed.options.spec["hardware"] = False
    svc.instances_info["default"].installed = installed

    result = svc.get_installed_info("default")

    assert result == installed.options.spec


def test_get_installed_info_delegates_when_not_installed(svc: VllmService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_get_service_installed_info", return_value=False) as mock:  # pyright: ignore[reportPrivateUsage]
        result = svc.get_installed_info("default")

    assert mock.call_count == 1
    assert mock.call_args == call("default")
    assert result is False


def test_generate_instance_config_none_info(svc: VllmService) -> None:
    config = svc._generate_instance_config("default", None, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options is None
    assert config.models == []


def test_generate_instance_config_with_info(svc: VllmService) -> None:
    info = _make_installed_info()
    info.models["m1"] = _make_model_installed_info("m1")

    config = svc._generate_instance_config("default", info, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options == info.options
    assert len(config.models or []) == 1


def test_generate_instance_config_embeds_definition_when_in_registry(svc: VllmService) -> None:
    info = _make_installed_info()
    info.models["m1"] = _make_model_installed_info("m1")
    svc.models["default"]["m1"] = VllmModel(hf_id="m1", size="1 GB")

    config = svc._generate_instance_config("default", info, None)  # pyright: ignore[reportPrivateUsage]

    assert (config.models or [])[0].definition == {
        "hf_id": "m1",
        "revision": None,
        "env_vars": None,
        "quantization": None,
        "dtype": "auto",
        "shm_size": "16gb",
        "ulimits": None,
        "max_model_len": None,
        "gpu_memory_utilization": None,
        "size": "1 GB",
        "custom": None,
        "model_type": "llm",
    }


def test_generate_instance_config_omits_definition_when_missing_from_registry(svc: VllmService) -> None:
    info = _make_installed_info()
    info.models["ghost-model"] = _make_model_installed_info("ghost-model")

    config = svc._generate_instance_config("default", info, None)  # pyright: ignore[reportPrivateUsage]

    assert (config.models or [])[0].definition is None


def test_restore_model_definition_reinstates_model(svc: VllmService) -> None:
    definition = {"hf_id": "ghost-model", "size": "2 GB"}

    svc._restore_model_definition("default", "ghost-model", definition)  # pyright: ignore[reportPrivateUsage]

    assert svc.models["default"]["ghost-model"] == VllmModel(hf_id="ghost-model", size="2 GB")


def test_load_download_info(svc: VllmService) -> None:
    result = svc._load_download_info({"model_path": "/tmp/x"})  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result, DownloadedInfo)
    assert result.model_path == "/tmp/x"


def test_is_given_hardware_support_gpu_raises_400_no_avx512_no_gpu(deps: dict[str, Any]) -> None:
    deps["hardware"].cpu.avx512 = False
    deps["hardware"].gpus = []
    svc = VllmService(**deps)

    with pytest.raises(HTTPException) as exc_info:
        svc.is_given_hardware_support_gpu(None)

    assert exc_info.value.status_code == 400


def test_get_specified_hardware_parts_raises_400_no_avx512_no_gpu(deps: dict[str, Any]) -> None:
    deps["hardware"].cpu.avx512 = False
    deps["hardware"].gpus = []
    svc = VllmService(**deps)

    with pytest.raises(HTTPException) as exc_info:
        svc.get_specified_hardware_parts(None)

    assert exc_info.value.status_code == 400


def test_is_given_hardware_support_gpu_with_gpu_does_not_raise(deps: dict[str, Any]) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc = VllmService(**deps)

    result = svc.is_given_hardware_support_gpu(True)

    assert result is True


def test_get_docker_compose_file_path_raises_400_no_model_id(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", None)

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_raises_400_model_not_installed(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", "some-model")

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_returns_path_for_installed_model(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model")
    svc.instances_info["default"].installed = installed
    expected = Path("/some/path/docker-compose.yml")
    deps["docker_service"].get_docker_compose_file_path.return_value = expected

    result = svc.get_docker_compose_file_path("default", "test-model")

    assert result == expected


def test_get_docker_options_returns_docker_options_for_installed_model(svc: VllmService) -> None:
    installed = _make_installed_info()
    model_installed = _make_model_installed_info("test-model")
    installed.models["test-model"] = model_installed
    svc.instances_info["default"].installed = installed

    result = svc.get_docker_options("default", "test-model")

    assert result is model_installed.docker


@pytest.mark.asyncio
async def test_restart_docker_forwards_installed_model_docker_options(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_installed = _make_model_installed_info("test-model")
    installed.models["test-model"] = model_installed
    svc.instances_info["default"].installed = installed
    deps["docker_service"].restart_docker_compose = AsyncMock()

    await svc.restart_docker("default", "test-model")

    deps["docker_service"].restart_docker_compose.assert_called_once_with(model_installed.docker)


def test_add_custom_model_registers_entry(svc: VllmService) -> None:
    custom = CustomModel(id="c-1", data={"id": "my-model", "hf_id": "google/gemma-3-270m-it", "size": "1GB"})

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert "my-model" in svc.models["default"]
    assert svc.models["default"]["my-model"].custom == "c-1"
    assert svc.models["default"]["my-model"].hf_id == "google/gemma-3-270m-it"
    assert svc.models["default"]["my-model"].revision is None


def test_add_custom_model_registers_revision(svc: VllmService) -> None:
    custom = CustomModel(
        id="c-1b", data={"id": "my-model-rev", "hf_id": "google/gemma-3-270m-it", "revision": "quantized-awq", "size": "1GB"}
    )

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert svc.models["default"]["my-model-rev"].revision == "quantized-awq"


def test_add_custom_model_duplicate_raises_400(svc: VllmService) -> None:
    custom = CustomModel(id="c-2", data={"id": "dup", "hf_id": "google/model", "size": "1GB"})

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(HTTPException) as exc_info:
        svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    assert exc_info.value.status_code == 400


def test_add_custom_model_creates_dict_for_new_instance(svc: VllmService) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    custom = CustomModel(id="c-3", data={"id": "new-model", "hf_id": "google/model", "size": "1GB"})

    svc._add_custom_model("extra", custom)  # pyright: ignore[reportPrivateUsage]

    assert "new-model" in svc.models["extra"]


def test_remove_custom_model_deletes_entry(svc: VllmService) -> None:
    custom = CustomModel(id="c-4", data={"id": "to-remove", "hf_id": "google/model", "size": "1GB"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].installed = _make_installed_info()

    svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert "to-remove" not in svc.models["default"]


def test_remove_custom_model_raises_400_when_in_use(svc: VllmService) -> None:
    custom = CustomModel(id="c-5", data={"id": "in-use", "hf_id": "google/model", "size": "1GB"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    installed = _make_installed_info()
    installed.models["in-use"] = _make_model_installed_info("in-use")
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_list_models_raises_404_for_unknown_instance(svc: VllmService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await svc.list_models("nonexistent", ListModelsFilters())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_list_models_returns_all_models_no_filter(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters())

    assert len(result.list) == len(svc.models["default"])


@pytest.mark.asyncio
async def test_list_models_filters_installed_only(svc: VllmService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters(installed=True))

    assert len(result.list) >= 1
    assert all(bool(m.installed) for m in result.list)


@pytest.mark.asyncio
async def test_get_model_raises_400_for_unknown_model_id(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "nonexistent-model")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_model_returns_correct_model(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = next(iter(svc.models["default"]))

    result = await svc.get_model("default", model_id)

    assert result.id == model_id


@pytest.mark.asyncio
async def test_download_model_emits_initial_and_final_progress(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = VllmModel(hf_id="google/test-model", size="1GB")
    stream = MagicMock()
    local_path = tmp_path / "model-files"
    local_path.mkdir()

    async def mock_download(*args: object, **kwargs: object):  # type: ignore[misc]
        yield SuccessDownloadPacket(local_path=local_path, filename="model-files")

    deps["model_downloader"].download = mock_download

    await svc._download_model(stream, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_count >= 2


@pytest.mark.asyncio
async def test_download_model_updates_progress_on_downloaded_packet(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = VllmModel(hf_id="google/test-model", size="100MB")
    stream = MagicMock()
    local_path = tmp_path / "model-files"
    local_path.mkdir()

    async def mock_download(*args: object, **kwargs: object):  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=1024 * 1024)
        yield SuccessDownloadPacket(local_path=local_path, filename="model-files")

    deps["model_downloader"].download = mock_download

    await svc._download_model(stream, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    progress_calls = [call for call in stream.emit.call_args_list if hasattr(call.args[0], "get") and call.args[0].get("value", 0) > 0]
    assert len(progress_calls) >= 1


@pytest.mark.asyncio
async def test_download_model_or_set_progress_starts_new_download(svc: VllmService, tmp_path: Path) -> None:
    stream = MagicMock()
    model_id = "google/test-model"
    model = VllmModel(hf_id=model_id, size="1GB")
    svc.models["default"][model_id] = model

    with patch.object(  # pyright: ignore[reportPrivateUsage]
        svc, "_download_model", new_callable=AsyncMock, return_value=tmp_path / "model"
    ) as mock_dl:
        await svc._download_model_or_set_progress(stream, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert mock_dl.call_count == 1
    assert model_id not in svc.models_download_progress


@pytest.mark.asyncio
async def test_download_model_or_set_progress_cleans_up_on_failure(svc: VllmService, tmp_path: Path) -> None:
    stream = MagicMock()
    model_id = "google/test-model"
    model = VllmModel(hf_id=model_id, size="1GB")
    svc.models["default"][model_id] = model

    with (
        patch.object(svc, "_download_model", new_callable=AsyncMock, side_effect=HTTPException(400, "boom")),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(HTTPException),
    ):
        await svc._download_model_or_set_progress(stream, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert model_id not in svc.models_download_progress


@pytest.mark.asyncio
async def test_download_model_or_set_progress_retries_download_after_failure(svc: VllmService, tmp_path: Path) -> None:
    stream1 = MagicMock()
    stream2 = MagicMock()
    model_id = "google/test-model"
    model = VllmModel(hf_id=model_id, size="1GB")
    svc.models["default"][model_id] = model

    mock_dl = AsyncMock(side_effect=[HTTPException(400, "boom"), tmp_path / "model"])
    with patch.object(svc, "_download_model", mock_dl):  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(HTTPException):
            await svc._download_model_or_set_progress(stream1, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]
        await svc._download_model_or_set_progress(stream2, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert mock_dl.call_count == 2


@pytest.mark.asyncio
async def test_download_model_or_set_progress_forwards_existing_stream(svc: VllmService, tmp_path: Path) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    chunk = StreamChunkProgress(type="progress", stage="download", value=0.5, data={"local_model_path": str(tmp_path / "model")})
    existing_stream.emit(chunk)
    existing_stream.close()
    model_id = "google/test-model"
    model = VllmModel(hf_id=model_id, size="1GB")
    svc.models_download_progress[model_id] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    await svc._download_model_or_set_progress(output_stream, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert output_stream.emit.call_count == 1
    assert output_stream.emit.call_args == call(chunk)


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_increments_total(svc: VllmService) -> None:
    svc.gpu_memory_utilization = 0.0
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(gpu_memory_utilization=0.5)

    result = await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == 0.5
    assert svc.gpu_memory_utilization == 0.5


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_raises_422_when_sum_exceeds_1(svc: VllmService) -> None:
    svc.gpu_memory_utilization = 0.8
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(gpu_memory_utilization=0.5)

    with pytest.raises(HTTPException) as exc_info:
        await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_get_default_gpu_memory_utilization_uses_free_fraction_with_margin(svc: VllmService, deps: dict[str, Any]) -> None:
    deps["hardware"].get_realtime_stats = AsyncMock(return_value=GpuStats(total_vram_gb=10.0, used_vram_gb=5.0, gpus=None))

    result = await svc._get_default_gpu_memory_utilization()  # pyright: ignore[reportPrivateUsage]

    assert result == 0.45


@pytest.mark.asyncio
async def test_get_default_gpu_memory_utilization_caps_at_default_when_gpu_is_free(svc: VllmService, deps: dict[str, Any]) -> None:
    deps["hardware"].get_realtime_stats = AsyncMock(return_value=GpuStats(total_vram_gb=10.0, used_vram_gb=0.0, gpus=None))

    result = await svc._get_default_gpu_memory_utilization()  # pyright: ignore[reportPrivateUsage]

    assert result == 0.9


@pytest.mark.asyncio
async def test_get_default_gpu_memory_utilization_falls_back_when_stats_unavailable(svc: VllmService, deps: dict[str, Any]) -> None:
    deps["hardware"].get_realtime_stats = AsyncMock(return_value=None)

    result = await svc._get_default_gpu_memory_utilization()  # pyright: ignore[reportPrivateUsage]

    assert result == 0.9


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_uses_dynamic_default_when_unset(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.gpu_memory_utilization = 0.0
    deps["hardware"].get_realtime_stats = AsyncMock(return_value=GpuStats(total_vram_gb=10.0, used_vram_gb=5.0, gpus=None))
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions()

    result = await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == 0.45
    assert svc.gpu_memory_utilization == 0.45


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_reuses_persisted_value_instead_of_live_default(svc: VllmService, deps: dict[str, Any]) -> None:
    """A persisted auto-tuned value must be reused so a reload doesn't drift with currently-free VRAM."""
    svc.gpu_memory_utilization = 0.0
    svc.instances_info["default"].config = InstanceConfig(
        models=[ModelConfig(model_id="google/test", options=InstallModelIn(spec={}), gpu_memory_utilization=0.33)]
    )
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions()

    result = await svc._get_gpu_memory_utilization("default", "google/test", opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == 0.33
    deps["hardware"].get_realtime_stats.assert_not_called()


def test_get_persisted_model_gpu_memory_utilization_returns_none_for_untracked_instance(svc: VllmService) -> None:
    assert svc._get_persisted_model_gpu_memory_utilization("no-such-instance", "model") is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_get_quantization_returns_valid_value(svc: VllmService) -> None:
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(quantization="fp8")

    result = await svc._get_quantization(opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == "fp8"


@pytest.mark.asyncio
async def test_get_quantization_raises_422_on_invalid_characters(svc: VllmService) -> None:
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions(quantization=None)
    opts.quantization = "fp8!@#"  # bypass pydantic

    with pytest.raises(HTTPException) as exc_info:
        await svc._get_quantization(opts, model)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_get_quantization_returns_none_when_not_set(svc: VllmService) -> None:
    model = VllmModel(hf_id="google/test", size="1GB")
    opts = VllmModelOptions()

    result = await svc._get_quantization(opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result is None


def test_register_model_endpoint_llm_calls_chat_completion_proxy(svc: VllmService, deps: dict[str, Any]) -> None:
    model = VllmModel(hf_id="google/test", size="1GB", model_type="llm")
    model_info = _make_model_installed_info("google/test", model_type="llm")

    svc._register_model_endpoint(  # pyright: ignore[reportPrivateUsage]
        model_info=model_info,
        model=model,
        registered_name="google/test",
        model_id="google/test",
        context_window=4096,
        max_context_window=4096,
        capacity=None,
    )

    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 1
    assert deps["endpoint_registry"].register_rerank_as_proxy.call_count == 0


def test_register_model_endpoint_reranker_calls_rerank_proxy(svc: VllmService, deps: dict[str, Any]) -> None:
    model = VllmModel(hf_id="google/reranker", size="1GB", model_type="reranker")
    model_info = _make_model_installed_info("google/reranker", model_type="reranker")

    svc._register_model_endpoint(  # pyright: ignore[reportPrivateUsage]
        model_info=model_info,
        model=model,
        registered_name="google/reranker",
        model_id="google/reranker",
        context_window=None,
        max_context_window=None,
        capacity=None,
    )

    assert deps["endpoint_registry"].register_rerank_as_proxy.call_count == 1
    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 0


def test_register_model_endpoint_embedding_calls_embeddings_proxy(svc: VllmService, deps: dict[str, Any]) -> None:
    model = VllmModel(hf_id="google/embedder", size="1GB", model_type="embedding")
    model_info = _make_model_installed_info("google/embedder", model_type="embedding")

    svc._register_model_endpoint(  # pyright: ignore[reportPrivateUsage]
        model_info=model_info,
        model=model,
        registered_name="google/embedder",
        model_id="google/embedder",
        context_window=None,
        max_context_window=None,
        capacity=None,
    )

    assert deps["endpoint_registry"].register_embeddings_as_proxy.call_count == 1
    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 0
    assert deps["endpoint_registry"].register_rerank_as_proxy.call_count == 0


def test_register_model_endpoint_passes_capacity_through(svc: VllmService, deps: dict[str, Any]) -> None:
    model = VllmModel(hf_id="google/test", size="1GB", model_type="llm")
    model_info = _make_model_installed_info("google/test", model_type="llm")

    svc._register_model_endpoint(  # pyright: ignore[reportPrivateUsage]
        model_info=model_info,
        model=model,
        registered_name="google/test",
        model_id="google/test",
        context_window=4096,
        max_context_window=4096,
        capacity=8,
    )

    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["registration_options"].capacity_state == 8


def test_register_model_endpoint_marks_capacity_unknown_when_none(svc: VllmService, deps: dict[str, Any]) -> None:
    """capacity=None from vLLM always means "couldn't determine it", never "no concurrency concept"."""
    model = VllmModel(hf_id="google/test", size="1GB", model_type="llm")
    model_info = _make_model_installed_info("google/test", model_type="llm")

    svc._register_model_endpoint(  # pyright: ignore[reportPrivateUsage]
        model_info=model_info,
        model=model,
        registered_name="google/test",
        model_id="google/test",
        context_window=4096,
        max_context_window=4096,
        capacity=None,
    )

    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["registration_options"].capacity_state == "unknown"


def test_get_image_true_returns_gpu_image(svc: VllmService) -> None:
    image = svc._get_image(True)  # pyright: ignore[reportPrivateUsage]

    assert image == _const.images["gpu"]


def test_get_image_false_returns_cpu_image(svc: VllmService) -> None:
    image = svc._get_image(False)  # pyright: ignore[reportPrivateUsage]

    assert image == _const.images["cpu"]


@pytest.mark.asyncio
async def test_install_model_returns_already_installed(svc: VllmService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed

    promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    result = await promise.wait()
    assert result.status == "OK"
    assert "Already installed" in result.details


@pytest.mark.asyncio
async def test_install_model_raises_400_for_unknown_model(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model("default", "nonexistent-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_calls_docker_install(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 1


@pytest.mark.asyncio
async def test_install_model_appends_revision_suffix_to_model_dir(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    svc.models["default"]["custom-model"] = VllmModel(hf_id="google/gemma-3-270m-it", revision="quantized/awq", size="1GB")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", "custom-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert "google-gemma-3-270m-it--quantized_awq" in docker_options.volumes[0]


@pytest.mark.asyncio
async def test_install_model_dedups_download_by_revision_aware_key(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    """Two custom models sharing an hf_id but differing only by revision must not share a download dedup key.

    Otherwise the second install would see the first's (bare hf_id) key already "in progress" and be
    handed the first revision's model_dir instead of downloading its own.
    """
    _setup_install_mocks(svc, deps)
    svc.models["default"]["custom-model"] = VllmModel(hf_id="google/gemma-3-270m-it", revision="quantized/awq", size="1GB")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model") as mock_download,  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", "custom-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    download_key = mock_download.call_args.args[1]
    assert download_key == "google-gemma-3-270m-it--quantized_awq"


def test_purge_stale_download_removes_old_directory_when_path_changed(svc: VllmService, tmp_path: Path) -> None:
    old_dir = tmp_path / "old-model"
    old_dir.mkdir()
    new_dir = tmp_path / "new-model"
    svc.models_downloaded["custom-model"] = DownloadedInfo(model_path=str(old_dir))

    svc._purge_stale_download("custom-model", new_dir)  # pyright: ignore[reportPrivateUsage]

    assert not old_dir.exists()


def test_purge_stale_download_keeps_directory_when_path_unchanged(svc: VllmService, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    svc.models_downloaded["custom-model"] = DownloadedInfo(model_path=str(model_dir))

    svc._purge_stale_download("custom-model", model_dir)  # pyright: ignore[reportPrivateUsage]

    assert model_dir.exists()


def test_purge_stale_download_noop_when_nothing_downloaded_yet(svc: VllmService, tmp_path: Path) -> None:
    svc._purge_stale_download("custom-model", tmp_path / "model")  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_model_edit_cleans_up_previous_download_directory(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    """Reinstalling a custom model under a new hf_id/revision must delete its old download directory.

    Otherwise the old directory is orphaned forever - nothing references it once
    `models_downloaded[model_id]` is overwritten with the new path.
    """
    _setup_install_mocks(svc, deps)
    svc.models["default"]["custom-model"] = VllmModel(hf_id="google/gemma-3-270m-it", size="1GB")
    old_dir = tmp_path / "old-model"
    old_dir.mkdir()
    svc.models_downloaded["custom-model"] = DownloadedInfo(model_path=str(old_dir))
    new_dir = tmp_path / "new-model"

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=new_dir),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", "custom-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert not old_dir.exists()
    assert svc.models_downloaded["custom-model"].model_path == str(new_dir)


@pytest.mark.asyncio
async def test_install_model_invalidates_log_cache_before_reading_capacity(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    """A reinstall on the same deterministic container name must not read the previous container's cached logs."""
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))
    svc._log_cache["container"] = (time.monotonic(), "Maximum concurrency for 4096 tokens per request: 2.00x", 8.0)  # pyright: ignore[reportPrivateUsage]

    mock_result = MagicMock()
    mock_result.stdout = "Maximum concurrency for 4096 tokens per request: 16.00x"
    mock_result.stderr = ""

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch("server.services.base2_service.Utils.run_command", new_callable=AsyncMock, return_value=mock_result),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["registration_options"].capacity_state == 16


@pytest.mark.asyncio
async def test_install_model_reuses_persisted_capacity_when_container_not_restarted(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """A no-op reinstall (container already running, config unchanged) must reuse the last known capacity, not re-read logs."""
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8000, False, False))
    svc.instances_info["default"].config = InstanceConfig(
        models=[ModelConfig(model_id=model_id, options=InstallModelIn(spec={}), capacity=16, capacity_known=True)]
    )

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch("server.services.base2_service.Utils.run_command", new_callable=AsyncMock) as mock_run_command,
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    mock_run_command.assert_not_called()
    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["registration_options"].capacity_state == 16


@pytest.mark.asyncio
async def test_install_model_capacity_unknown_when_not_restarted_and_never_persisted(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """A no-op reinstall with no prior known capacity and no recoverable log line must leave capacity unknown."""
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8000, False, False))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch(
            "server.services.base2_service.Utils.run_command",
            new_callable=AsyncMock,
            return_value=CommandResult(exit_code=0, stdout="", stderr=""),
        ) as mock_run_command,
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    mock_run_command.assert_called_once()
    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["registration_options"].capacity_state == "unknown"


@pytest.mark.asyncio
async def test_install_model_capacity_recovered_from_logs_when_not_restarted_and_never_persisted(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """An adopted orphan container with no persisted capacity should still recover it from its startup logs."""
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8000, False, False))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
        patch(
            "server.services.base2_service.Utils.run_command",
            new_callable=AsyncMock,
            return_value=CommandResult(exit_code=0, stdout="Maximum concurrency for 4096 tokens per request: 2.00x", stderr=""),
        ),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    call_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert call_kwargs["registration_options"].capacity_state == 2


def test_get_persisted_model_capacity_returns_unknown_for_untracked_instance(svc: VllmService) -> None:
    assert svc._get_persisted_model_capacity("no-such-instance", "model") == (None, False)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_resolve_model_capacity_not_restarted_reuses_known_none_without_reading_logs(svc: VllmService) -> None:
    """A persisted `capacity=None, capacity_known=True` (a genuinely-checked-but-unbounded reading) must round-trip as-is."""
    svc.instances_info["default"].config = InstanceConfig(
        models=[ModelConfig(model_id="m", options=InstallModelIn(spec={}), capacity=None, capacity_known=True)]
    )

    with patch("server.services.base2_service.Utils.run_command", new_callable=AsyncMock) as mock_run_command:
        capacity, capacity_known = await svc._resolve_model_capacity(  # pyright: ignore[reportPrivateUsage]
            "default",
            "m",
            "vLLM",
            "container",
            False,
            svc._parse_max_concurrency,  # pyright: ignore[reportPrivateUsage]
        )

    mock_run_command.assert_not_called()
    assert capacity is None
    assert capacity_known is True


@pytest.mark.asyncio
async def test_install_model_registers_chat_completion_endpoint(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 1


@pytest.mark.asyncio
async def test_install_model_records_model_as_downloaded(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert model_id in svc.models_downloaded


@pytest.mark.asyncio
async def test_install_model_uses_alias_as_registered_name(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={"alias": "my-alias"}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert installed.models[model_id].registered_name == "my-alias"


@pytest.mark.asyncio
async def test_install_model_registers_reranker_endpoint_for_reranker_model(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, deps)
    reranker_id = "reranker-model"
    svc.models["default"][reranker_id] = VllmModel(hf_id=reranker_id, size="1GB", model_type="reranker")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", reranker_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert deps["endpoint_registry"].register_rerank_as_proxy.call_count == 1
    assert deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 0


@pytest.mark.asyncio
async def test_install_model_docker_failure_decrements_gpu_memory(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=RuntimeError("docker failed"))
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc2.models["default"]))

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert svc2.gpu_memory_utilization == 0.0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("the estimated maximum model length is 110256.", 110256),
        ("estimated maximum model length is 4096", 4096),
        ("docker failed for some other reason", None),
    ],
)
def test_parse_kv_cache_max_len_suggestion(raw: str, expected: int | None) -> None:
    assert VllmService._parse_kv_cache_max_len_suggestion(raw) == expected  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("GPU KV cache size: 100,000 tokens, Maximum concurrency for 4096 tokens per request: 12.34x", 12),
        ("Maximum concurrency for 4096 tokens per request: 1.00x", 1),
        ("docker started successfully with no relevant log line", None),
        ("Maximum concurrency for 4096 tokens per request: 0.73x", 1),
        (
            "Maximum concurrency for 4096 tokens per request: 4.00x\n"
            "restarting with smaller max-model-len\n"
            "Maximum concurrency for 4096 tokens per request: 8.00x",
            8,
        ),
        ("Maximum concurrency for 4096 tokens per request: 1.2.3x", None),
        # A number long enough that float() rounds it to inf must not raise OverflowError out of round().
        (f"Maximum concurrency for 4096 tokens per request: {'9' * 400}.0x", None),
    ],
)
def test_parse_max_concurrency(raw: str, expected: int | None) -> None:
    assert VllmService._parse_max_concurrency(raw) == expected  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_get_max_concurrency_from_logs_returns_parsed_value(svc: VllmService) -> None:
    with patch.object(
        svc, "_get_docker_logs", new_callable=AsyncMock, return_value="Maximum concurrency for 4096 tokens per request: 8.00x"
    ):  # pyright: ignore[reportPrivateUsage]
        result = await svc._get_max_concurrency_from_logs("my-container", "vLLM", svc._parse_max_concurrency)  # pyright: ignore[reportPrivateUsage]

    assert result == 8


@pytest.mark.asyncio
async def test_get_max_concurrency_from_logs_returns_none_on_failure(svc: VllmService) -> None:
    with patch.object(svc, "_get_docker_logs", new_callable=AsyncMock, side_effect=RuntimeError("boom")):  # pyright: ignore[reportPrivateUsage]
        result = await svc._get_max_concurrency_from_logs("my-container", "vLLM", svc._parse_max_concurrency)  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_install_model_retries_with_suggested_max_len_on_kv_cache_error(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    kv_cache_error = AppError(
        "ValueError: To serve at least one request with the models's max seq len (131072), "
        "(4.0 GiB KV cache is needed, which is larger than the available KV cache memory (3.37 GiB). "
        "Based on the available memory, the estimated maximum model length is 110256."
    )
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=[kv_cache_error, (8000, True, False)])
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8000
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc2.models["default"]))

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=131072),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert result.status == "OK"
    assert deps["docker_service"].install_and_run_docker.call_count == 2
    register_kwargs = deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert register_kwargs["props"].context_window == 110256


@pytest.mark.asyncio
async def test_install_model_does_not_retry_when_max_model_length_is_explicit(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    kv_cache_error = AppError("the estimated maximum model length is 110256.")
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=kv_cache_error)
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc2.models["default"]))

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model(  # pyright: ignore[reportPrivateUsage]
            "default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5, "max_model_length": 4096})
        )
        with pytest.raises(AppError):
            await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 1


@pytest.mark.asyncio
async def test_install_model_reraises_when_retry_also_fails(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    kv_cache_error = AppError("the estimated maximum model length is 110256.")
    retry_error = AppError("still failing")
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=[kv_cache_error, retry_error])
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc2.models["default"]))

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(AppError, match="still failing"):
            await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 2
    assert deps["docker_service"].stop_docker.call_count == 2


@pytest.mark.parametrize(
    "raw",
    [
        "ValueError: ... is larger than the available KV cache memory (3.37 GiB) ...",
        "No available memory for the cache blocks.",
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "CUDA error: out of memory",
    ],
)
def test_diagnose_insufficient_vram_matches_known_patterns(raw: str) -> None:
    msg = VllmService._diagnose_insufficient_vram("my-model", raw)  # pyright: ignore[reportPrivateUsage]
    assert msg is not None
    assert "my-model" in msg
    assert "VRAM" in msg


def test_diagnose_insufficient_vram_returns_none_for_unrelated_error() -> None:
    assert VllmService._diagnose_insufficient_vram("my-model", "connection refused") is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    "raw",
    [
        "huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache "
        "and outgoing traffic has been disabled.",
        "OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.",
    ],
)
def test_diagnose_missing_model_files_matches_known_patterns(raw: str) -> None:
    msg = VllmService._diagnose_missing_model_files("my-model", raw)  # pyright: ignore[reportPrivateUsage]
    assert msg is not None
    assert "my-model" in msg
    assert "missing files" in msg


def test_diagnose_missing_model_files_returns_none_for_unrelated_error() -> None:
    assert VllmService._diagnose_missing_model_files("my-model", "connection refused") is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_model_raises_friendly_error_when_retry_fails_on_vram(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    kv_cache_error = RuntimeError("the estimated maximum model length is 4096.")
    vram_error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=[kv_cache_error, vram_error])
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc2.models["default"]))

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(AppError, match="Not enough GPU VRAM"):
            await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 2
    assert deps["docker_service"].stop_docker.call_count == 2


@pytest.mark.asyncio
async def test_install_model_raises_friendly_error_on_vram_when_max_model_length_is_explicit(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    vram_error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=vram_error)
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc2.models["default"]))

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model(  # pyright: ignore[reportPrivateUsage]
            "default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5, "max_model_length": 4096})
        )
        with pytest.raises(AppError, match="Not enough GPU VRAM"):
            await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 1
    assert deps["docker_service"].stop_docker.call_count == 1


@pytest.mark.asyncio
async def test_install_model_raises_friendly_error_when_retry_fails_on_missing_files(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    kv_cache_error = RuntimeError("the estimated maximum model length is 4096.")
    missing_files_error = RuntimeError(
        "huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache "
        "and outgoing traffic has been disabled."
    )
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=[kv_cache_error, missing_files_error])
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc2.models["default"]))

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(AppError, match="missing files"):
            await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 2
    assert deps["docker_service"].stop_docker.call_count == 2


@pytest.mark.asyncio
async def test_install_model_raises_friendly_error_when_model_files_are_missing(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    missing_files_error = RuntimeError(
        "huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache "
        "and outgoing traffic has been disabled."
    )
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=missing_files_error)
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc2.models["default"]))

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(AppError, match="missing files"):
            await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 1
    assert deps["docker_service"].stop_docker.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_model_removes_from_installed_and_unregisters_endpoint(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model", "reg-1")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert "test-model" not in installed.models
    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 1
    assert deps["endpoint_registry"].unregister_chat_completion.call_args == call("test-model", "reg-1")


@pytest.mark.asyncio
async def test_uninstall_model_decrements_gpu_memory_utilization(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.gpu_memory_utilization = 0.5
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model", gpu_memory_utilization=0.5)
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.gpu_memory_utilization == 0.0


@pytest.mark.asyncio
async def test_uninstall_model_floors_gpu_memory_at_zero(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.gpu_memory_utilization = 0.1
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model", gpu_memory_utilization=0.5)
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.gpu_memory_utilization == 0.0


@pytest.mark.asyncio
async def test_uninstall_model_purges_files_and_downloaded_entry(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _make_installed_info()
    model_dir = tmp_path / "model-dir"
    model_dir.mkdir()
    model_info = _make_model_installed_info("test-model")
    model_info.model_path = model_dir
    installed.models["test-model"] = model_info
    svc.instances_info["default"].installed = installed
    svc.models_downloaded["test-model"] = DownloadedInfo(model_path=str(model_dir))
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert not model_dir.exists()
    assert "test-model" not in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_reranker_calls_unregister_rerank(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["reranker-model"] = _make_model_installed_info("reranker-model", "reg-rerank", model_type="reranker")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "reranker-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_rerank.call_count == 1
    assert deps["endpoint_registry"].unregister_rerank.call_args == call("reranker-model", "reg-rerank")
    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_model_embedding_calls_unregister_embeddings(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["embedder-model"] = _make_model_installed_info("embedder-model", "reg-embed", model_type="embedding")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "embedder-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_embeddings.call_count == 1
    assert deps["endpoint_registry"].unregister_embeddings.call_args == call("embedder-model", "reg-embed")
    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0
    assert deps["endpoint_registry"].unregister_rerank.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_model_ignores_unknown_model_id(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "unknown-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["endpoint_registry"].unregister_chat_completion.call_count == 0
    assert deps["docker_service"].uninstall_docker.call_count == 0


@pytest.mark.asyncio
async def test_install_instance_returns_installed_info_with_empty_models(svc: VllmService) -> None:
    options = InstallServiceIn(spec={"hardware": False})

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert isinstance(result, InstalledInfo)
    assert result.models == {}


@pytest.mark.asyncio
async def test_uninstall_instance_calls_uninstall_model_for_each_model(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info("m1")
    installed.models["m2"] = _make_model_installed_info("m2")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall:  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 2


@pytest.mark.asyncio
async def test_uninstall_instance_purge_clears_service_state(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["docker_service"].remove_image = AsyncMock()
    deps["docker_service"].uninstall_docker = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock) as mock_clear,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert svc.service_downloaded is False
    assert mock_clear.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_instance_purge_with_other_instance_installed_does_not_remove_image(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["extra"].installed = _make_installed_info()
    svc.instances_info["default"].installed = _make_installed_info()
    deps["docker_service"].remove_image = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock) as mock_clear,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert deps["docker_service"].remove_image.call_count == 0
    assert mock_clear.call_count == 0
    assert svc.instances_info["default"].installed is None
    assert "extra" in svc.instances_info


@pytest.mark.asyncio
async def test_uninstall_instance_purge_removes_per_model_docker_images(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info("test-model")
    model_info.docker.image = "vllm-model-image:tag"
    installed.models["test-model"] = model_info
    svc.instances_info["default"].installed = installed
    deps["docker_service"].remove_image = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    removed_images = {c.args[0] for c in deps["docker_service"].remove_image.call_args_list}
    assert "vllm-model-image:tag" in removed_images


@pytest.mark.asyncio
async def test_stop_instance_does_nothing_when_not_installed(svc: VllmService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_stop_dockers_parallel", new_callable=AsyncMock) as mock_stop:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock_stop.call_count == 0


@pytest.mark.asyncio
async def test_stop_instance_stops_all_containers(svc: VllmService) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info("m1")
    installed.models["m2"] = _make_model_installed_info("m2")
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_stop_dockers_parallel", new_callable=AsyncMock) as mock_stop:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock_stop.call_count == 1
    called_dockers = mock_stop.call_args[0][0]
    assert len(called_dockers) == 2


def test_get_specified_hardware_parts_delegates_to_super_when_hardware_ok(svc: VllmService) -> None:
    svc.hardware.cpu.avx512 = True

    with patch("server.services.base2_service.Base2Service.get_specified_hardware_parts", return_value=[]) as mock_super:
        result = svc.get_specified_hardware_parts(False)

    assert mock_super.call_count == 1
    assert result == []


def test_get_model_spec_adds_hardware_key_when_missing(svc: VllmService, deps: dict[str, Any]) -> None:
    deps["docker_service"].has_gpu_support = False
    svc.instances_info["default"].config = InstanceConfig(options=InstallServiceIn(spec={}))

    result = svc.get_model_spec("default", "llm")

    assert result is not None


@pytest.mark.asyncio
async def test_install_instance_loads_default_models_for_new_instance(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    options = InstallServiceIn(spec={"hardware": False})
    svc.hardware.cpu.avx512 = True

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "load_default_models") as mock_load,
    ):
        promise = await svc._install_instance("inst2", options)  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert mock_load.call_count == 1
    assert mock_load.call_args == call("inst2")


@pytest.mark.asyncio
async def test_install_instance_adds_hardware_key_when_missing(svc: VllmService, deps: dict[str, Any]) -> None:
    options = InstallServiceIn(spec={})
    deps["docker_service"].has_gpu_support = False
    svc.hardware.cpu.avx512 = True

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]

    assert "hardware" in options.spec


@pytest.mark.asyncio
async def test_uninstall_instance_purge_removes_non_default_instance(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}), parsed_options=VllmOptions())
    svc.instances_info["inst2"].installed = installed
    deps["docker_service"].remove_image = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("inst2", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "inst2" not in svc.instances_info


@pytest.mark.asyncio
async def test_list_models_skips_instances_not_in_filter(svc: VllmService) -> None:
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    svc.models["inst2"] = {"extra-model": VllmModel(hf_id="extra/model", size="1GB")}
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}), parsed_options=VllmOptions())

    result = await svc.list_models("default", ListModelsFilters())

    assert not any(m.id == "extra-model" for m in result.list)


@pytest.mark.asyncio
async def test_download_model_handles_pre_download_packet_with_size(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = VllmModel(hf_id="google/test", size="100MB")
    stream = MagicMock()

    async def mock_download(*args: object, **kwargs: object):  # type: ignore[misc]
        yield PreDownloadPacket(file_bytes_size=50 * 1024 * 1024)
        yield SuccessDownloadPacket(local_path=tmp_path / "model", filename="model")

    deps["model_downloader"].download = mock_download

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        local_path = await svc._download_model(stream, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert local_path is not None


@pytest.mark.asyncio
async def test_download_model_or_set_progress_forwards_chunk_with_data(svc: VllmService, tmp_path: Path) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    chunk = StreamChunkProgress(type="progress", stage="download", value=0.9, data={"local_model_path": str(tmp_path / "model")})
    existing_stream.emit(chunk)
    existing_stream.close()
    model = VllmModel(hf_id="google/test", size="1GB")
    svc.models_download_progress["google/test"] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    result_path = await svc._download_model_or_set_progress(output_stream, "google/test", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert result_path == tmp_path / "model"


@pytest.mark.asyncio
async def test_download_model_or_set_progress_breaks_on_non_download_chunk(svc: VllmService, tmp_path: Path) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    finish_chunk: StreamChunk = {"type": "finish", "status": "ok"}  # type: ignore[assignment]
    progress_chunk = StreamChunkProgress(type="progress", stage="download", value=1.0, data={"local_model_path": str(tmp_path / "model")})
    existing_stream.emit(progress_chunk)
    existing_stream.emit(finish_chunk)
    existing_stream.close()
    model = VllmModel(hf_id="google/test", size="1GB")
    svc.models_download_progress["google/test"] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    result_path = await svc._download_model_or_set_progress(output_stream, "google/test", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert result_path is not None


def test_build_vllm_command_use_gpu_with_model_length(svc: VllmService, tmp_path: Path) -> None:
    opts = VllmModelOptions.model_construct(extra_args={})

    cmd = svc._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        user_model_length=4096,
        use_gpu=True,
    )

    assert "--max-model-len" in cmd
    assert "4096" in cmd


def test_build_vllm_command_no_gpu_no_model_length_adds_disable_sliding_window(svc: VllmService, tmp_path: Path) -> None:
    opts = VllmModelOptions.model_construct(extra_args={})

    cmd = svc._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        user_model_length=None,
        use_gpu=False,
    )

    assert "--disable-sliding-window" in cmd


def test_build_vllm_command_extra_args_with_value(svc: VllmService, tmp_path: Path) -> None:
    opts = VllmModelOptions.model_construct(extra_args={"--dtype": "float16"})

    cmd = svc._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        user_model_length=None,
        use_gpu=True,
    )

    assert "--dtype" in cmd
    assert "float16" in cmd


def test_build_vllm_command_extra_args_flag_only(svc: VllmService, tmp_path: Path) -> None:
    opts = VllmModelOptions.model_construct(extra_args={"--enforce-eager": None})

    cmd = svc._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        user_model_length=None,
        use_gpu=True,
    )

    assert "--enforce-eager" in cmd


def test_build_vllm_command_no_gpu_with_model_length(svc: VllmService, tmp_path: Path) -> None:
    opts = VllmModelOptions.model_construct(extra_args={})

    cmd = svc._build_vllm_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        user_model_length=2048,
        use_gpu=False,
    )

    assert "--max-model-len" in cmd
    assert "2048" in cmd


@pytest.mark.asyncio
async def test_download_model_raises_500_when_no_success_packet(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = VllmModel(hf_id="google/test", size="100MB")
    stream = MagicMock()

    async def mock_download(*args: object, **kwargs: object):  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=1024)

    deps["model_downloader"].download = mock_download

    with (
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._download_model(stream, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_download_model_or_set_progress_raises_500_after_break_with_no_path(svc: VllmService, tmp_path: Path) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    finish_chunk: StreamChunk = {"type": "finish", "status": "ok"}  # type: ignore[assignment]
    existing_stream.emit(finish_chunk)
    existing_stream.close()
    model = VllmModel(hf_id="google/test", size="1GB")
    svc.models_download_progress["google/test"] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await svc._download_model_or_set_progress(output_stream, "google/test", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 500


def test_get_size_no_cpu_without_avx512(deps: dict[str, Any]) -> None:
    deps["hardware"].cpu.avx512 = False
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc = VllmService(**deps)

    sizes = svc.get_size()

    assert "cpu" not in sizes
    assert "gpu" in sizes


@pytest.mark.asyncio
async def test_uninstall_instance_does_nothing_when_not_installed(svc: VllmService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall:  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 0
    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_skips_model_installed_in_other_instance(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info("m1")
    installed.models["m2"] = _make_model_installed_info("m2")
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    def mock_is_in_other(instance: str, model_id: str) -> bool:
        return model_id == "m1"

    with (
        patch.object(svc, "is_model_installed_in_other_instance", side_effect=mock_is_in_other),
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 1
    called_ids = [call.args[1] for call in mock_uninstall.call_args_list]
    assert "m1" not in called_ids
    assert "m2" in called_ids


@pytest.mark.asyncio
async def test_download_model_handles_pre_download_packet_without_file_size(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = VllmModel(hf_id="google/test", size="100MB")
    stream = MagicMock()
    local_path = tmp_path / "model-files"
    local_path.mkdir()

    async def mock_download(*args: object, **kwargs: object):  # type: ignore[misc]
        yield PreDownloadPacket(file_bytes_size=None)  # pyright: ignore[reportArgumentType]
        yield SuccessDownloadPacket(local_path=local_path, filename="model-files")

    deps["model_downloader"].download = mock_download
    result = await svc._download_model(stream, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert result == local_path


@pytest.mark.asyncio
async def test_download_model_ignores_zero_bytes_downloaded_packet(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    model = VllmModel(hf_id="google/test", size="100MB")
    stream = MagicMock()
    local_path = tmp_path / "model-files"
    local_path.mkdir()

    async def mock_download(*args: object, **kwargs: object):  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=0)
        yield SuccessDownloadPacket(local_path=local_path, filename="model-files")

    deps["model_downloader"].download = mock_download
    result = await svc._download_model(stream, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert result == local_path


@pytest.mark.asyncio
async def test_download_model_or_set_progress_emits_chunk_with_empty_data(svc: VllmService, tmp_path: Path) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    chunk_empty_data = StreamChunkProgress(type="progress", stage="download", value=0.3, data={})
    chunk_with_data = StreamChunkProgress(
        type="progress",
        stage="download",
        value=1.0,
        data={"local_model_path": str(tmp_path / "model")},
    )
    existing_stream.emit(chunk_empty_data)
    existing_stream.emit(chunk_with_data)
    existing_stream.close()

    model_id = "google/test"
    model = VllmModel(hf_id=model_id, size="1GB")
    svc.models_download_progress[model_id] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    result_path = await svc._download_model_or_set_progress(output_stream, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert result_path == tmp_path / "model"
    assert output_stream.emit.call_count == 2


@pytest.mark.asyncio
async def test_install_model_docker_failure_no_decrement_when_model_gpu_utilization_zero(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=RuntimeError("docker failed"))
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()

    model_id = "zero-gpu-model"
    svc2.models["default"][model_id] = VllmModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.0)

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert svc2.gpu_memory_utilization >= 0


@pytest.mark.asyncio
async def test_install_model_docker_failure_no_floor_when_utilization_stays_positive(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    svc2.gpu_memory_utilization = 0.5
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=RuntimeError("docker failed"))
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].stop_docker = AsyncMock()

    model_id = "small-gpu-model"
    svc2.models["default"][model_id] = VllmModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.3)

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert svc2.gpu_memory_utilization == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_uninstall_model_purge_with_none_model_path_removes_downloaded_entry(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model")
    svc.instances_info["default"].installed = installed
    svc.models_downloaded["test-model"] = DownloadedInfo(model_path=None)
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "test-model" not in svc.models_downloaded


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_formatted_size(svc: VllmService, deps: dict[str, Any]) -> None:
    with patch("server.services.vllm_service.fetch_huggingface_model_size", new=AsyncMock(return_value="4.0 GB")):
        result = await svc._resolve_custom_model_size({"hf_id": "google/gemma-3-270m-it"})  # pyright: ignore[reportPrivateUsage]

    assert result == "4.0 GB"


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_none_on_exception(svc: VllmService, deps: dict[str, Any]) -> None:
    with patch("server.services.vllm_service.fetch_huggingface_model_size", new=AsyncMock(side_effect=Exception("fail"))):
        result = await svc._resolve_custom_model_size({"hf_id": "google/gemma"})  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_validate_custom_model_skips_when_no_hf_id(svc: VllmService, deps: dict[str, Any]) -> None:
    with patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock()) as mock_validate:
        await svc._validate_custom_model({})  # pyright: ignore[reportPrivateUsage]

    mock_validate.assert_not_called()


@pytest.mark.parametrize("skip_validation", [True, "true", "True", "1", "yes"])
@pytest.mark.asyncio
async def test_validate_custom_model_skips_when_skip_validation_flag_set(
    svc: VllmService, deps: dict[str, Any], skip_validation: object
) -> None:
    with patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock()) as mock_validate:
        await svc._validate_custom_model(  # pyright: ignore[reportPrivateUsage]
            {"hf_id": "org/model", "skip_validation": skip_validation}
        )

    mock_validate.assert_not_called()


@pytest.mark.parametrize("skip_validation", [False, "false", "False", "0", "no", "No", "off", "Off", "", None])
@pytest.mark.asyncio
async def test_validate_custom_model_runs_when_skip_validation_flag_unset(
    svc: VllmService, deps: dict[str, Any], skip_validation: object
) -> None:
    spec: dict[str, Any] = {"hf_id": "org/model"}
    if skip_validation is not None:
        spec["skip_validation"] = skip_validation

    with patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=None)) as mock_validate:
        await svc._validate_custom_model(spec)  # pyright: ignore[reportPrivateUsage]

    mock_validate.assert_called_once()


@pytest.mark.asyncio
async def test_validate_custom_model_passes_hf_id_revision_and_model_type(svc: VllmService, deps: dict[str, Any]) -> None:
    with patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=None)) as mock_validate:
        await svc._validate_custom_model(  # pyright: ignore[reportPrivateUsage]
            {"hf_id": "org/model", "revision": "main", "model_type": "reranker"}
        )

    args = mock_validate.call_args.args
    assert args[1:] == ("org/model", "main", "reranker")


@pytest.mark.asyncio
async def test_validate_custom_model_defaults_model_type_to_llm(svc: VllmService, deps: dict[str, Any]) -> None:
    with patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=None)) as mock_validate:
        await svc._validate_custom_model({"hf_id": "org/model"})  # pyright: ignore[reportPrivateUsage]

    args = mock_validate.call_args.args
    assert args[1:] == ("org/model", None, "llm")


@pytest.mark.asyncio
async def test_validate_custom_model_defaults_empty_model_type_to_llm(svc: VllmService, deps: dict[str, Any]) -> None:
    with patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=None)) as mock_validate:
        await svc._validate_custom_model({"hf_id": "org/model", "model_type": ""})  # pyright: ignore[reportPrivateUsage]

    args = mock_validate.call_args.args
    assert args[1:] == ("org/model", None, "llm")


@pytest.mark.asyncio
async def test_validate_custom_model_raises_400_on_incompatible_model(svc: VllmService, deps: dict[str, Any]) -> None:
    with (
        patch(
            "server.services.vllm_service.validate_model_compatibility",
            new=AsyncMock(side_effect=ModelIncompatibleError("org/model is a GGUF repo")),
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._validate_custom_model({"hf_id": "org/model"})  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400
    assert "GGUF" in exc_info.value.detail


@pytest.mark.asyncio
async def test_validate_custom_model_sends_configured_hf_token_as_bearer_header(svc: VllmService, deps: dict[str, Any]) -> None:
    deps["config"].hugging_face_token.get_secret_value.return_value = "hf_test_token"
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session_cls = MagicMock(return_value=mock_session)

    with (
        patch("server.services.vllm_service.aiohttp.ClientSession", mock_session_cls),
        patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=None)),
    ):
        await svc._validate_custom_model({"hf_id": "org/model"})  # pyright: ignore[reportPrivateUsage]

    assert mock_session_cls.call_args.kwargs["headers"] == {"Authorization": "Bearer hf_test_token"}


@pytest.mark.asyncio
async def test_validate_custom_model_sends_no_auth_header_when_token_unset(svc: VllmService, deps: dict[str, Any]) -> None:
    deps["config"].hugging_face_token.get_secret_value.return_value = ""
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session_cls = MagicMock(return_value=mock_session)

    with (
        patch("server.services.vllm_service.aiohttp.ClientSession", mock_session_cls),
        patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=None)),
    ):
        await svc._validate_custom_model({"hf_id": "org/model"})  # pyright: ignore[reportPrivateUsage]

    assert mock_session_cls.call_args.kwargs["headers"] == {}


@pytest.mark.asyncio
async def test_validate_custom_model_fills_size_from_fetched_data(svc: VllmService, deps: dict[str, Any]) -> None:
    fetched_data = {"siblings": [{"rfilename": "model.safetensors", "size": 4 * 1024**3}]}
    spec: dict[str, Any] = {"hf_id": "org/model"}

    with patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=fetched_data)):
        await svc._validate_custom_model(spec)  # pyright: ignore[reportPrivateUsage]

    assert spec["size"] == "4.0 GB"


@pytest.mark.asyncio
async def test_validate_custom_model_does_not_overwrite_explicit_size(svc: VllmService, deps: dict[str, Any]) -> None:
    fetched_data = {"siblings": [{"rfilename": "model.safetensors", "size": 4 * 1024**3}]}
    spec: dict[str, Any] = {"hf_id": "org/model", "size": "1GB"}

    with patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=fetched_data)):
        await svc._validate_custom_model(spec)  # pyright: ignore[reportPrivateUsage]

    assert spec["size"] == "1GB"


@pytest.mark.asyncio
async def test_validate_custom_model_leaves_size_unset_when_siblings_report_no_size(svc: VllmService, deps: dict[str, Any]) -> None:
    fetched_data = {"siblings": [{"rfilename": "README.md"}]}
    spec: dict[str, Any] = {"hf_id": "org/model"}

    with patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=fetched_data)):
        await svc._validate_custom_model(spec)  # pyright: ignore[reportPrivateUsage]

    assert "size" not in spec


@pytest.mark.asyncio
async def test_validate_custom_model_leaves_size_unset_when_fetch_failed(svc: VllmService, deps: dict[str, Any]) -> None:
    spec: dict[str, Any] = {"hf_id": "org/model"}

    with patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=None)):
        await svc._validate_custom_model(spec)  # pyright: ignore[reportPrivateUsage]

    assert "size" not in spec


@pytest.mark.asyncio
async def test_add_custom_model_size_fill_skips_redundant_size_resolution_call(svc: VllmService, deps: dict[str, Any]) -> None:
    # The whole point of filling `size` inside `_validate_custom_model` is to spare
    # `add_custom_model`'s `if not spec.get("size")` guard a second, near-identical HF request via
    # `_resolve_custom_model_size` - verify that guard is actually skipped end-to-end.
    fetched_data = {"siblings": [{"rfilename": "model.safetensors", "size": 4 * 1024**3}]}
    deps["service_provider"].save_service_config = AsyncMock()
    with (
        patch("server.services.vllm_service.validate_model_compatibility", new=AsyncMock(return_value=fetched_data)),
        patch.object(svc, "_resolve_custom_model_size", new=AsyncMock()) as mock_resolve_size,
    ):
        await svc.add_custom_model("default", AddCustomModelIn(spec={"id": "my-model", "hf_id": "org/model"}))

    mock_resolve_size.assert_not_called()


@pytest.mark.asyncio
async def test_get_docker_logs_cache_hit(svc: VllmService) -> None:
    svc._log_cache["my-container"] = (time.monotonic(), "cached logs", 8.0)  # pyright: ignore[reportPrivateUsage]

    result = await svc._get_docker_logs("my-container")  # pyright: ignore[reportPrivateUsage]

    assert result == "cached logs"


@pytest.mark.asyncio
async def test_get_docker_logs_fetches_and_caches(svc: VllmService) -> None:
    mock_result = MagicMock()
    mock_result.exit_code = 0
    mock_result.stdout = "stdout logs"
    mock_result.stderr = "stderr logs"

    with patch("server.services.base2_service.Utils.run_command", new_callable=AsyncMock, return_value=mock_result):
        result = await svc._get_docker_logs("my-container")  # pyright: ignore[reportPrivateUsage]

    assert result == "stdout logsstderr logs"
    assert "my-container" in svc._log_cache  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_get_docker_logs_caches_failed_command_briefly(svc: VllmService) -> None:
    """A failed read is still cached, but only for the short failed-read TTL - long enough to
    avoid spawning a fresh `docker logs` subprocess on every poll tick of a stuck container,
    short enough to pick up the real startup log soon after it becomes available."""
    mock_result = MagicMock()
    mock_result.exit_code = 1
    mock_result.stdout = ""
    mock_result.stderr = "Error: No such container"

    with patch("server.services.base2_service.Utils.run_command", new_callable=AsyncMock, return_value=mock_result):
        result = await svc._get_docker_logs("my-container")  # pyright: ignore[reportPrivateUsage]

    assert result == "Error: No such container"
    cached = svc._log_cache["my-container"]  # pyright: ignore[reportPrivateUsage]
    assert cached[1] == "Error: No such container"
    assert cached[2] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_get_docker_logs_caches_empty_output_briefly(svc: VllmService) -> None:
    mock_result = MagicMock()
    mock_result.exit_code = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("server.services.base2_service.Utils.run_command", new_callable=AsyncMock, return_value=mock_result):
        result = await svc._get_docker_logs("my-container")  # pyright: ignore[reportPrivateUsage]

    assert result == ""
    cached = svc._log_cache["my-container"]  # pyright: ignore[reportPrivateUsage]
    assert cached[1] == ""
    assert cached[2] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Total (incl. non-KV cache overhead): 8.50 GiB\n", 8.5),
        ("Model weights: 6.00 GiB\nKV Cache: 2.00 GiB\n", 8.0),
        ("model weights take 5.00 GiB\nKV cache memory: 1.50 GiB\n", 6.5),
        ("Model weights: 4.00 GiB\n", 4.0),
        ("KV cache: 3.00 GiB\n", 3.0),
        ("no useful info here", None),
    ],
)
def test_parse_vllm_vram_gb(raw: str, expected: float | None) -> None:
    result = VllmService._parse_vllm_vram_gb(raw)  # pyright: ignore[reportPrivateUsage]

    assert result == expected


@pytest.mark.asyncio
async def test_get_vram_from_logs_returns_parsed_value(svc: VllmService) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info("test-model")
    model_info.docker.container_name = "vllm-container"
    installed.models["test-model"] = model_info
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_get_docker_logs", new_callable=AsyncMock, return_value="Total (incl. non-KV cache overhead): 7.00 GiB"):  # pyright: ignore[reportPrivateUsage]
        result = await svc._get_vram_from_logs("default", "test-model")  # pyright: ignore[reportPrivateUsage]

    assert result == 7.0


@pytest.mark.asyncio
async def test_get_vram_from_logs_returns_none_when_no_container(svc: VllmService) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info("test-model")
    model_info.docker.container_name = ""
    installed.models["test-model"] = model_info
    svc.instances_info["default"].installed = installed

    result = await svc._get_vram_from_logs("default", "test-model")  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_get_vram_from_logs_returns_none_on_docker_logs_error(svc: VllmService) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info("test-model")
    model_info.docker.container_name = "vllm-container"
    installed.models["test-model"] = model_info
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_get_docker_logs", new_callable=AsyncMock, side_effect=Exception("boom")):  # pyright: ignore[reportPrivateUsage]
        result = await svc._get_vram_from_logs("default", "test-model")  # pyright: ignore[reportPrivateUsage]

    assert result is None


def test_get_vram_estimate_returns_value(svc: VllmService) -> None:
    svc.hardware.total_vram_gb = 24.0  # pyright: ignore[reportAttributeAccessIssue]
    model_info = _make_model_installed_info("test-model", gpu_memory_utilization=0.5)

    result = svc._get_vram_estimate(model_info)  # pyright: ignore[reportPrivateUsage]

    assert result == 12.0


def test_get_vram_estimate_returns_none_when_no_utilization(svc: VllmService) -> None:
    model_info = _make_model_installed_info("test-model", gpu_memory_utilization=None)

    result = svc._get_vram_estimate(model_info)  # pyright: ignore[reportPrivateUsage]

    assert result is None


def test_get_vram_estimate_returns_none_when_total_vram_zero(svc: VllmService) -> None:
    svc.hardware.total_vram_gb = 0.0  # pyright: ignore[reportAttributeAccessIssue]
    model_info = _make_model_installed_info("test-model", gpu_memory_utilization=0.5)

    result = svc._get_vram_estimate(model_info)  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_get_cached_vram_estimate_cache_hit(svc: VllmService) -> None:
    svc._vram_cache[("default", "test-model")] = 4.0  # pyright: ignore[reportPrivateUsage]

    result = await svc._get_cached_vram_estimate("default", "test-model", is_loaded=True)  # pyright: ignore[reportPrivateUsage]

    assert result == 4.0


@pytest.mark.asyncio
async def test_get_cached_vram_estimate_from_logs_non_none(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_get_vram_from_logs", new_callable=AsyncMock, return_value=5.5):  # pyright: ignore[reportPrivateUsage]
        result = await svc._get_cached_vram_estimate("default", "test-model", is_loaded=True)  # pyright: ignore[reportPrivateUsage]

    assert result == 5.5
    assert svc._vram_cache[("default", "test-model")] == 5.5  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_get_cached_vram_estimate_model_not_in_installed(svc: VllmService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed  # no models in installed

    with patch.object(svc, "_get_vram_from_logs", new_callable=AsyncMock, return_value=None):  # pyright: ignore[reportPrivateUsage]
        result = await svc._get_cached_vram_estimate("default", "missing-model", is_loaded=True)  # pyright: ignore[reportPrivateUsage]

    assert result is None
    assert ("default", "missing-model") not in svc._vram_cache  # pyright: ignore[reportPrivateUsage]


def test_release_gpu_utilization_with_none_does_not_change_value(svc: VllmService) -> None:
    svc.gpu_memory_utilization = 0.5
    svc._release_gpu_utilization(None)  # pyright: ignore[reportPrivateUsage]
    assert svc.gpu_memory_utilization == pytest.approx(0.5)


def test_release_gpu_utilization_floors_at_zero_when_result_would_be_negative(svc: VllmService) -> None:
    svc.gpu_memory_utilization = 0.1
    svc._release_gpu_utilization(0.5)  # pyright: ignore[reportPrivateUsage]
    assert svc.gpu_memory_utilization == 0


@pytest.mark.asyncio
async def test_install_model_releases_gpu_on_cancelled_error(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=asyncio.CancelledError())
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"

    model_id = "cancel-model"
    svc2.models["default"][model_id] = VllmModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.5)

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(asyncio.CancelledError):
            await promise.wait()

    assert svc2.gpu_memory_utilization == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_install_model_registration_failure_rolls_back_model(svc: VllmService, deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _setup_install_mocks(svc, deps)
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc.models["default"]))
    deps["endpoint_registry"].register_chat_completion_as_proxy.side_effect = RuntimeError("registry down")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert model_id not in installed.models
    deps["docker_service"].stop_docker.assert_called_once()


@pytest.mark.asyncio
async def test_install_model_releases_gpu_when_option_parsing_fails(svc: VllmService, deps: dict[str, Any]) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    svc2.instances_info["default"].installed = _make_installed_info(hardware=True)
    model_id = "parse-fail-model"
    svc2.models["default"][model_id] = VllmModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.5)

    with (
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
        patch.object(svc2, "_get_quantization", new_callable=AsyncMock, side_effect=RuntimeError("bad quantization")),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(RuntimeError),
    ):
        await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]

    assert svc2.gpu_memory_utilization == 0.0
    assert ("default", model_id) not in svc2._installing  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_model_releases_gpu_on_cancelled_error_before_docker_start(svc: VllmService, deps: dict[str, Any]) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    svc2.instances_info["default"].installed = _make_installed_info(hardware=True)
    model_id = "cancel-before-docker-model"
    svc2.models["default"][model_id] = VllmModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.5)

    with (
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
        patch.object(svc2, "_get_quantization", new_callable=AsyncMock, side_effect=asyncio.CancelledError()),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(asyncio.CancelledError),
    ):
        await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]

    assert svc2.gpu_memory_utilization == 0.0
    assert ("default", model_id) not in svc2._installing  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_model_releases_gpu_and_stops_container_when_post_start_fails(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8000, True, False))
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8000
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = "post-start-model"
    svc2.models["default"][model_id] = VllmModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.5)

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch(
            "server.services.vllm_service.get_model_dir_context_window",
            new_callable=AsyncMock,
            side_effect=RuntimeError("context read failed"),
        ),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert svc2.gpu_memory_utilization == 0.0
    assert model_id not in installed.models
    deps["docker_service"].stop_docker.assert_called_once()


@pytest.mark.asyncio
async def test_install_model_does_not_stop_adopted_container_when_post_start_fails(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """An adopted container pre-dates this install attempt, so a later failure must not tear it down (DFINFRA-297)."""
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    installed = _make_installed_info(hardware=True)
    svc2.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8000, False, True))
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 8000
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = "adopted-post-start-model"
    svc2.models["default"][model_id] = VllmModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.5)

    with (
        patch.object(svc2, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch(
            "server.services.vllm_service.get_model_dir_context_window",
            new_callable=AsyncMock,
            side_effect=RuntimeError("context read failed"),
        ),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert svc2.gpu_memory_utilization == 0.0
    assert model_id not in installed.models
    deps["docker_service"].stop_docker.assert_not_called()


@pytest.mark.asyncio
async def test_install_model_releases_gpu_without_stopping_container_when_download_fails(svc: VllmService, deps: dict[str, Any]) -> None:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    svc2 = VllmService(**deps)
    svc2.instances_info["default"].installed = _make_installed_info(hardware=True)
    deps["docker_service"].stop_docker = AsyncMock()
    model_id = "download-fail-model"
    svc2.models["default"][model_id] = VllmModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.5)

    with (
        patch.object(
            svc2,
            "_download_model_or_set_progress",
            new_callable=AsyncMock,
            side_effect=RuntimeError("download failed"),  # pyright: ignore[reportPrivateUsage]
        ),
        patch.object(svc2, "get_specified_hardware_parts", return_value=[nvidia]),
        patch.object(svc2, "is_given_hardware_support_gpu", return_value=True),
    ):
        promise = await svc2._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert svc2.gpu_memory_utilization == 0.0
    deps["docker_service"].stop_docker.assert_not_called()


def test_get_image_with_version_overrides_tag_gpu(svc: VllmService) -> None:
    image = svc._get_image(gpu=True, image_version="v0.9.0")  # pyright: ignore[reportPrivateUsage]
    base = _const.images["gpu"].name.split(":")[0]
    assert image.name == f"{base}:v0.9.0"
    assert image.size == _const.images["gpu"].size


def test_get_image_with_version_overrides_tag_cpu(svc: VllmService) -> None:
    image = svc._get_image(gpu=False, image_version="v0.9.0")  # pyright: ignore[reportPrivateUsage]
    base = _const.images["cpu"].name.split(":")[0]
    assert image.name == f"{base}:v0.9.0"
    assert image.size == _const.images["cpu"].size


@pytest.mark.asyncio
async def test_get_docker_tags_returns_filtered_tags(svc: VllmService) -> None:
    mock_client = AsyncMock()
    mock_client.get_tags = AsyncMock(return_value=["v0.9.0-cu130", "v0.9.0-cpu"])
    with (
        patch("server.services.vllm_service.registry_for", return_value=mock_client),
        patch("server.services.vllm_service.image_without_registry_prefix", return_value="vllm/vllm-openai"),
    ):
        tags = await svc.get_docker_tags("GPU")
    assert "v0.9.0-cu130" in tags
    assert "v0.9.0-cpu" not in tags


@pytest.mark.asyncio
async def test_get_docker_tags_cpu_hardware_returns_tags_unfiltered(svc: VllmService) -> None:
    mock_client = AsyncMock()
    mock_client.get_tags = AsyncMock(return_value=["v0.19.0", "v0.19.1"])
    cpu_base = _const.images["cpu"].name.split(":")[0]
    with (
        patch("server.services.vllm_service.registry_for", return_value=mock_client),
        patch("server.services.vllm_service.image_without_registry_prefix", return_value="cpu/vllm-openai") as mock_img,
    ):
        tags = await svc.get_docker_tags("cpu")
    assert tags == ["v0.19.0", "v0.19.1"]
    mock_img.assert_called_once_with(cpu_base)


def test_get_default_docker_tag_gpu(svc: VllmService) -> None:
    assert svc.get_default_docker_tag("GPU") == _const.images["gpu"].name.split(":")[1]


def test_get_default_docker_tag_cpu(svc: VllmService) -> None:
    assert svc.get_default_docker_tag("cpu") == _const.images["cpu"].name.split(":")[1]


def test_get_default_docker_tag_defaults_to_gpu_when_no_hardware(svc: VllmService) -> None:
    assert svc.get_default_docker_tag(None) == _const.images["gpu"].name.split(":")[1]


def test_get_docker_image_repo_gpu(svc: VllmService) -> None:
    assert svc.get_docker_image_repo("GPU") == _const.images["gpu"].name.split(":")[0]


def test_get_docker_image_repo_cpu(svc: VllmService) -> None:
    assert svc.get_docker_image_repo("cpu") == _const.images["cpu"].name.split(":")[0]


def test_get_docker_image_repo_defaults_to_gpu_when_no_hardware(svc: VllmService) -> None:
    assert svc.get_docker_image_repo(None) == _const.images["gpu"].name.split(":")[0]


@pytest.mark.asyncio
async def test_install_model_registration_failure_skips_rollback_when_already_removed(
    svc: VllmService, deps: dict[str, Any], tmp_path: Path
) -> None:
    installed = _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    def side_effect(*args: object, **kwargs: object) -> None:
        installed.models.pop(model_id, None)
        raise RuntimeError("registry down")

    deps["endpoint_registry"].register_chat_completion_as_proxy.side_effect = side_effect

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.vllm_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.vllm_service.get_base_url", return_value="http://localhost:8000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert model_id not in installed.models


@pytest.mark.asyncio
async def test_install_model_pre_func_exception_discards_installing_key(svc: VllmService, deps: dict[str, Any]) -> None:
    _setup_install_mocks(svc, deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_get_quantization", new_callable=AsyncMock, side_effect=RuntimeError("quantization failed")),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(RuntimeError),
    ):
        await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert ("default", model_id) not in svc._installing  # pyright: ignore[reportPrivateUsage]


def _status(exists: bool = True, state: str = "running", health: str = "healthy", restart_count: int = 0) -> ContainerStatus:
    return ContainerStatus(exists=exists, state=state, health=health, restart_count=restart_count)


@pytest.mark.asyncio
async def test_reconcile_instance_models_running_healthy_does_not_release(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", gpu_memory_utilization=0.5)
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(state="running", health="healthy"))

    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert "m1" in installed.models
    assert svc.gpu_memory_utilization == 0.5
    deps["endpoint_registry"].unregister_chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_instance_models_recovers_within_threshold_resets_counter(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", gpu_memory_utilization=0.5)
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(state="exited", health=""))

    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]
    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert svc._crash_poll_state[("default", "m1")] == 2  # pyright: ignore[reportPrivateUsage]

    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(state="running", health="healthy"))
    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert ("default", "m1") not in svc._crash_poll_state  # pyright: ignore[reportPrivateUsage]
    assert "m1" in installed.models
    assert svc.gpu_memory_utilization == 0.5


@pytest.mark.asyncio
async def test_reconcile_instance_models_releases_after_threshold_bad_polls(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", registration_id="reg-1", gpu_memory_utilization=0.5, model_type="llm")
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_save", new_callable=AsyncMock) as mock_save:
        for _ in range(3):
            await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert "m1" not in installed.models
    assert svc.gpu_memory_utilization == pytest.approx(0.0)
    assert ("default", "m1") not in svc._crash_poll_state  # pyright: ignore[reportPrivateUsage]
    deps["endpoint_registry"].unregister_chat_completion.assert_called_once_with("m1", "reg-1")
    deps["docker_service"].uninstall_docker.assert_called_once_with(model_info.docker)
    mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_instance_models_keeps_bookkeeping_when_teardown_fails_and_retries(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", registration_id="reg-1", gpu_memory_utilization=0.5)
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))
    deps["docker_service"].uninstall_docker = AsyncMock(side_effect=RuntimeError("docker daemon busy"))

    for _ in range(3):
        await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    # Teardown kept failing: bookkeeping stays untouched, nothing released yet.
    assert "m1" in installed.models
    assert svc.gpu_memory_utilization == pytest.approx(0.5)
    deps["endpoint_registry"].unregister_chat_completion.assert_not_called()

    # Teardown finally succeeds on a later tick, without needing 3 more bad polls first.
    deps["docker_service"].uninstall_docker = AsyncMock()
    with patch.object(svc, "_save", new_callable=AsyncMock) as mock_save:
        await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert "m1" not in installed.models
    assert svc.gpu_memory_utilization == pytest.approx(0.0)
    deps["endpoint_registry"].unregister_chat_completion.assert_called_once_with("m1", "reg-1")
    mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_instance_models_recovers_while_teardown_was_failing(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", gpu_memory_utilization=0.5)
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))
    deps["docker_service"].uninstall_docker = AsyncMock(side_effect=RuntimeError("docker daemon busy"))

    for _ in range(3):
        await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    # Container comes back healthy on its own (e.g. manually restarted) before teardown ever succeeded.
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(state="running", health="healthy"))
    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert "m1" in installed.models
    assert svc.gpu_memory_utilization == pytest.approx(0.5)
    assert ("default", "m1") not in svc._crash_poll_state  # pyright: ignore[reportPrivateUsage]
    deps["endpoint_registry"].unregister_chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_instance_models_unhealthy_running_container_counts_as_bad(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", gpu_memory_utilization=0.5)
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(state="running", health="unhealthy"))
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_save", new_callable=AsyncMock):
        for _ in range(3):
            await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert "m1" not in installed.models
    assert svc.gpu_memory_utilization == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_reconcile_instance_models_skips_models_currently_installing(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", gpu_memory_utilization=0.5)
    installed.models["m1"] = model_info
    svc._installing.add(("default", "m1"))  # pyright: ignore[reportPrivateUsage]
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))

    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    deps["docker_service"].get_container_status.assert_not_called()
    assert "m1" in installed.models


@pytest.mark.asyncio
async def test_reconcile_instance_models_skips_model_uninstalled_concurrently(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", gpu_memory_utilization=0.5)
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))

    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]
    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]
    assert svc._crash_poll_state[("default", "m1")] == 2  # pyright: ignore[reportPrivateUsage]

    async def _status_and_uninstall_concurrently(_container_name: str) -> ContainerStatus:
        # Simulate a concurrent manual uninstall completing while this (3rd) poll is in flight.
        del installed.models["m1"]
        return _status(exists=False, state="", health="")

    deps["docker_service"].get_container_status = AsyncMock(side_effect=_status_and_uninstall_concurrently)

    with patch.object(svc, "_save", new_callable=AsyncMock) as mock_save:
        await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    deps["endpoint_registry"].unregister_chat_completion.assert_not_called()
    mock_save.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_instance_models_releases_reranker_model(svc: VllmService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", registration_id="reg-1", gpu_memory_utilization=0.5, model_type="reranker")
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))
    deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_save", new_callable=AsyncMock):
        for _ in range(3):
            await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    deps["endpoint_registry"].unregister_rerank.assert_called_once_with("m1", "reg-1")
    deps["endpoint_registry"].unregister_chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_start_reconciliation_task_cancels_existing_and_stop_cancels_it(svc: VllmService) -> None:
    svc.instances_info["default"].installed = _make_installed_info()

    svc._start_reconciliation_task("default")  # pyright: ignore[reportPrivateUsage]
    first_task = svc._reconciliation_tasks["default"]  # pyright: ignore[reportPrivateUsage]
    svc._start_reconciliation_task("default")  # pyright: ignore[reportPrivateUsage]
    second_task = svc._reconciliation_tasks["default"]  # pyright: ignore[reportPrivateUsage]

    await asyncio.gather(first_task, return_exceptions=True)
    assert first_task is not second_task
    assert first_task.cancelled()

    svc._stop_reconciliation_task("default")  # pyright: ignore[reportPrivateUsage]
    await asyncio.gather(second_task, return_exceptions=True)
    assert second_task.cancelled()
    assert "default" not in svc._reconciliation_tasks  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_reconciliation_loop_ticks_and_stops_once_instance_uninstalled(svc: VllmService) -> None:
    svc.instances_info["default"].installed = _make_installed_info()
    calls = 0

    async def fake_reconcile(_instance: str, _info: InstalledInfo) -> None:
        nonlocal calls
        calls += 1
        svc.instances_info["default"].installed = None

    with (
        patch("server.services.base2_service.asyncio.sleep", new_callable=AsyncMock),
        patch.object(svc, "_reconcile_instance_models", new=AsyncMock(side_effect=fake_reconcile)),
    ):
        svc._start_reconciliation_task("default")  # pyright: ignore[reportPrivateUsage]
        task = svc._reconciliation_tasks["default"]  # pyright: ignore[reportPrivateUsage]
        await asyncio.wait_for(task, timeout=1)

    assert calls == 1


@pytest.mark.asyncio
async def test_reconciliation_loop_logs_and_continues_on_tick_exception(svc: VllmService) -> None:
    svc.instances_info["default"].installed = _make_installed_info()
    calls = 0

    async def fake_reconcile(_instance: str, _info: InstalledInfo) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        svc.instances_info["default"].installed = None

    with (
        patch("server.services.base2_service.asyncio.sleep", new_callable=AsyncMock),
        patch.object(svc, "_reconcile_instance_models", new=AsyncMock(side_effect=fake_reconcile)),
    ):
        svc._start_reconciliation_task("default")  # pyright: ignore[reportPrivateUsage]
        task = svc._reconciliation_tasks["default"]  # pyright: ignore[reportPrivateUsage]
        await asyncio.wait_for(task, timeout=1)

    assert calls == 2


async def _fake_fetch_registry_entries(
    _session: Any,
    _top_by_downloads: int,
    _top_by_likes: int,
    _top_by_trending: int,
    model_type: str,
    log: Any,
    allow_generative_rerankers: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    if model_type == "llm":
        return "llms", [{"name": "new/model", "size": "2GB"}]
    if model_type == "reranker":
        return "rerankers", []
    return "embeddings", []


@pytest.fixture(autouse=True)
def _reset_catalog_refresh_guard() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    huggingface_refresh_guard.release()
    yield
    huggingface_refresh_guard.release()


def test_get_catalog_refresh_unavailable_reason_none_when_supported(svc: VllmService) -> None:
    with patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", True):
        assert svc.get_catalog_refresh_unavailable_reason() is None


def test_get_catalog_refresh_unavailable_reason_set_when_unsupported(svc: VllmService) -> None:
    with patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", False):
        assert svc.get_catalog_refresh_unavailable_reason() is not None


@pytest.mark.asyncio
async def test_refresh_catalog_writes_registry_and_reloads_const(svc: VllmService, tmp_path: Path) -> None:
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "vllm-min.json").write_text(json.dumps({"llms": [{"name": "old/model", "size": "1GB"}]}), encoding="utf-8")

    original_models = dict(_const.models)
    try:
        with (
            patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", True),
            patch("server.services.vllm_service.get_main_dir", return_value=tmp_path),
            patch("server.services.vllm_service.fetch_registry_entries", new=AsyncMock(side_effect=_fake_fetch_registry_entries)),
        ):
            promise = await svc.refresh_catalog()
            result = await promise.wait()

        assert result.added == 1
        assert "new/model" in _const.models
        assert "old/model" not in _const.models
        written = json.loads((tmp_path / "static" / "vllm-min.json").read_text(encoding="utf-8"))
        assert written["llms"] == [{"name": "new/model", "size": "2GB"}]
        assert set(svc.models["default"].keys()) == set(_const.models.keys())
    finally:
        _const.models = original_models


@pytest.mark.asyncio
async def test_refresh_catalog_raises_when_unsupported(svc: VllmService) -> None:
    with (
        patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", False),
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc.refresh_catalog()
    assert exc_info.value.status_code == 405


@pytest.mark.asyncio
async def test_refresh_catalog_rejects_concurrent_refresh(svc: VllmService) -> None:
    huggingface_refresh_guard.acquire("llamacpp")
    try:
        with (
            patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", True),
            pytest.raises(HTTPException) as exc_info,
        ):
            await svc.refresh_catalog()
        assert exc_info.value.status_code == 429
    finally:
        huggingface_refresh_guard.release()


@pytest.mark.asyncio
async def test_refresh_catalog_leaves_catalog_unchanged_on_failure(svc: VllmService, tmp_path: Path) -> None:
    (tmp_path / "static").mkdir()
    original_content = json.dumps({"llms": [{"name": "old/model", "size": "1GB"}]})
    (tmp_path / "static" / "vllm-min.json").write_text(original_content, encoding="utf-8")

    original_models = dict(_const.models)
    try:
        with (
            patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", True),
            patch("server.services.vllm_service.get_main_dir", return_value=tmp_path),
            patch("server.services.vllm_service.fetch_registry_entries", new=AsyncMock(side_effect=RuntimeError("boom"))),
        ):
            promise = await svc.refresh_catalog()
            with pytest.raises(RuntimeError):
                await promise.wait()

        assert (tmp_path / "static" / "vllm-min.json").read_text(encoding="utf-8") == original_content
        assert _const.models == original_models
    finally:
        _const.models = original_models


@pytest.mark.asyncio
async def test_refresh_catalog_raises_502_when_catalog_shrinks_too_much(svc: VllmService, tmp_path: Path) -> None:
    (tmp_path / "static").mkdir()
    original_content = json.dumps({"llms": [{"name": f"old/model-{i}", "size": "1GB"} for i in range(10)]})
    (tmp_path / "static" / "vllm-min.json").write_text(original_content, encoding="utf-8")

    async def _degraded_fetch(
        _session: Any,
        _top_by_downloads: int,
        _top_by_likes: int,
        _top_by_trending: int,
        model_type: str,
        log: Any,
        allow_generative_rerankers: bool = False,
    ) -> tuple[str, list[dict[str, Any]]]:
        if model_type == "llm":
            return "llms", [{"name": "new/model", "size": "2GB"}]
        return ("rerankers" if model_type == "reranker" else "embeddings"), []

    original_models = dict(_const.models)
    try:
        with (
            patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", True),
            patch("server.services.vllm_service.get_main_dir", return_value=tmp_path),
            patch("server.services.vllm_service.fetch_registry_entries", new=AsyncMock(side_effect=_degraded_fetch)),
        ):
            promise = await svc.refresh_catalog()
            with pytest.raises(HTTPException) as exc_info:
                await promise.wait()
            assert exc_info.value.status_code == 502

        assert (tmp_path / "static" / "vllm-min.json").read_text(encoding="utf-8") == original_content
        assert _const.models == original_models
    finally:
        _const.models = original_models


@pytest.mark.asyncio
async def test_refresh_catalog_restores_installed_model_dropped_from_catalog(svc: VllmService, tmp_path: Path) -> None:
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "vllm-min.json").write_text(json.dumps({"llms": [{"name": "old/model", "size": "1GB"}]}), encoding="utf-8")

    installed = _make_installed_info()
    installed.models["old/model"] = _make_model_installed_info("old/model")
    svc.instances_info["default"].installed = installed
    svc.instances_info["default"].config = InstanceConfig(
        models=[ModelConfig(model_id="old/model", options=InstallModelIn(spec={}), definition={"hf_id": "old/model", "size": "1GB"})]
    )

    original_models = dict(_const.models)
    try:
        with (
            patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", True),
            patch("server.services.vllm_service.get_main_dir", return_value=tmp_path),
            patch("server.services.vllm_service.fetch_registry_entries", new=AsyncMock(side_effect=_fake_fetch_registry_entries)),
        ):
            promise = await svc.refresh_catalog()
            await promise.wait()

        assert "old/model" not in _const.models
        assert svc.models["default"]["old/model"] == VllmModel(hf_id="old/model", size="1GB")
    finally:
        _const.models = original_models


@pytest.mark.asyncio
async def test_refresh_catalog_drops_installed_model_with_no_persisted_definition(svc: VllmService, tmp_path: Path) -> None:
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "vllm-min.json").write_text(json.dumps({"llms": [{"name": "old/model", "size": "1GB"}]}), encoding="utf-8")

    installed = _make_installed_info()
    installed.models["old/model"] = _make_model_installed_info("old/model")
    svc.instances_info["default"].installed = installed

    original_models = dict(_const.models)
    try:
        with (
            patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", True),
            patch("server.services.vllm_service.get_main_dir", return_value=tmp_path),
            patch("server.services.vllm_service.fetch_registry_entries", new=AsyncMock(side_effect=_fake_fetch_registry_entries)),
        ):
            promise = await svc.refresh_catalog()
            await promise.wait()

        assert "old/model" not in svc.models["default"]
    finally:
        _const.models = original_models


@pytest.mark.asyncio
async def test_sync_models_triggers_refresh_without_awaiting_completion(svc: VllmService) -> None:
    with patch.object(svc, "refresh_catalog", new=AsyncMock()) as mock_refresh:
        await svc.sync_models("default")
    mock_refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_models_suppresses_unsupported_error(svc: VllmService) -> None:
    with patch("server.services.vllm_service.model_catalog_refresh.catalog_refresh_supported", False):
        await svc.sync_models("default")  # must not raise


@pytest.mark.asyncio
async def test_sync_models_logs_warning_when_background_refresh_fails(svc: VllmService) -> None:
    class _FailingPromise:
        async def wait(self) -> None:
            raise RuntimeError("boom")

    with (
        patch.object(svc, "refresh_catalog", new=AsyncMock(return_value=_FailingPromise())),
        patch("server.services.vllm_service.logger") as mock_logger,
    ):
        await svc.sync_models("default")
        await asyncio.sleep(0)  # let the fire-and-forget task run

    mock_logger.warning.assert_called_once()
