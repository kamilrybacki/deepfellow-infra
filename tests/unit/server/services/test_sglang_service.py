# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi import HTTPException

from server.docker import ContainerStatus
from server.models.models import InstallModelIn, ListModelsFilters, UninstallModelIn
from server.models.services import GpuStats, InstallServiceIn, UninstallServiceIn
from server.services.base2_service import CustomModel, Instance, InstanceConfig
from server.services.sglang_service import (
    DownloadedInfo,
    InstalledInfo,
    ModelInstalledInfo,
    SglangModel,
    SglangModelOptions,
    SglangOptions,
    SglangService,
    _const,  # pyright: ignore[reportPrivateUsage]
)
from server.utils.core import DownloadedPacket, PreDownloadPacket, Stream, StreamChunk, StreamChunkProgress, SuccessDownloadPacket
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
def gpu_deps(deps: dict[str, Any]) -> dict[str, Any]:
    nvidia = MagicMock(spec=NvidiaGpuInfo)
    deps["hardware"].gpus = [nvidia]
    deps["hardware"].get_realtime_stats = AsyncMock(return_value=None)
    return deps


@pytest.fixture
def svc(gpu_deps: dict[str, Any]) -> SglangService:
    return SglangService(**gpu_deps)


def _make_installed_info(hardware: str | bool | None = True) -> InstalledInfo:
    return InstalledInfo(
        models={},
        options=InstallServiceIn(spec={}),
        parsed_options=SglangOptions(hardware=hardware),
    )


def _make_model_installed_info(
    model_id: str = "test-model",
    registration_id: str = "reg-1",
    gpu_memory_utilization: float | None = None,
    model_type: str = "llm",
) -> ModelInstalledInfo:
    docker = MagicMock()
    docker.name = f"df-sglang-{model_id}"
    return ModelInstalledInfo(
        id=model_id,
        registered_name=model_id,
        options=InstallModelIn(spec={}),
        docker=docker,
        container_host="localhost",
        container_port=30000,
        docker_exposed_port=30000,
        registration_id=registration_id,
        model_path=Path("/tmp/model"),
        base_url="http://localhost:30000",
        gpu_memory_utilization=gpu_memory_utilization,
        model_type=model_type,  # type: ignore[arg-type]
    )


def _setup_install_mocks(svc: SglangService, deps: dict[str, Any], hardware: str | bool | None = True) -> InstalledInfo:
    installed = _make_installed_info(hardware=hardware)
    svc.instances_info["default"].installed = installed
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=30000)
    deps["docker_service"].get_docker_subnet.return_value = None
    deps["docker_service"].get_docker_container_name.return_value = "container"
    deps["docker_service"].get_container_host.return_value = "localhost"
    deps["docker_service"].get_container_port.return_value = 30000
    return installed


def test_get_type(svc: SglangService) -> None:
    assert svc.get_type() == "sglang"


def test_get_description_not_empty(svc: SglangService) -> None:
    assert svc.get_description()


def test_default_models_loaded(svc: SglangService) -> None:
    assert "default" in svc.models
    assert len(svc.models["default"]) > 0


def test_get_size_no_gpu_returns_empty(deps: dict[str, Any]) -> None:
    svc = SglangService(**deps)

    sizes = svc.get_size()

    assert "gpu" not in sizes


def test_get_size_with_gpu(svc: SglangService) -> None:
    sizes = svc.get_size()

    assert "gpu" in sizes


def test_get_spec_has_hardware_field_and_no_cpu_option(svc: SglangService) -> None:
    spec = svc.get_spec()

    field_names = [f.name for f in spec.fields]
    assert "hardware" in field_names
    assert "image_version" in field_names

    hardware_field = next(f for f in spec.fields if f.name == "hardware")
    assert "CPU" not in (hardware_field.values or [])


def test_const_has_gpu_image_only() -> None:
    assert _const.image.name
    assert "sglang" in _const.image.name


def test_get_model_spec_baseline_fields(svc: SglangService) -> None:
    installed = _make_installed_info(hardware=True)
    svc.instances_info["default"].installed = installed

    spec = svc.get_model_spec("default", "llm")

    field_names = [f.name for f in spec.fields]
    for expected in ("alias", "max_model_length", "quantization", "extra_args", "extra_envs"):
        assert expected in field_names


def test_get_model_spec_adds_gpu_memory_utilization(svc: SglangService) -> None:
    svc.instances_info["default"].config.options = InstallServiceIn(spec={"hardware": True})

    spec = svc.get_model_spec("default", "llm")

    field_names = [f.name for f in spec.fields]
    assert "gpu_memory_utilization" in field_names


def test_get_model_spec_reranker_default_extra_args(svc: SglangService) -> None:
    spec = svc.get_model_spec("default", "reranker")

    extra_args_field = next(f for f in spec.fields if f.name == "extra_args")
    assert extra_args_field.default is not None
    assert "--trust-remote-code" in (extra_args_field.default or "")


def test_get_custom_model_spec_not_none(svc: SglangService) -> None:
    result = svc.get_custom_model_spec()

    assert result is not None
    field_names = [f.name for f in result.fields]
    assert "id" in field_names
    assert "hf_id" in field_names
    assert "size" in field_names


def test_is_given_hardware_support_gpu_raises_400_without_gpu(deps: dict[str, Any]) -> None:
    svc = SglangService(**deps)

    with pytest.raises(HTTPException) as exc_info:
        svc.is_given_hardware_support_gpu(None)

    assert exc_info.value.status_code == 400


def test_get_specified_hardware_parts_raises_400_without_gpu(deps: dict[str, Any]) -> None:
    svc = SglangService(**deps)

    with pytest.raises(HTTPException) as exc_info:
        svc.get_specified_hardware_parts(None)

    assert exc_info.value.status_code == 400


def test_is_given_hardware_support_gpu_with_gpu_does_not_raise(svc: SglangService) -> None:
    result = svc.is_given_hardware_support_gpu(True)

    assert result is True


@pytest.mark.asyncio
async def test_install_instance_raises_when_no_gpu(deps: dict[str, Any]) -> None:
    svc = SglangService(**deps)
    options = InstallServiceIn(spec={"hardware": True})

    with pytest.raises(HTTPException) as exc_info:
        await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


def test_add_custom_model_registers_entry(svc: SglangService) -> None:
    custom = CustomModel(id="c-1", data={"id": "my-model", "hf_id": "google/gemma-3-270m-it", "size": "1GB"})

    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert "my-model" in svc.models["default"]
    assert svc.models["default"]["my-model"].custom == "c-1"


def test_add_custom_model_duplicate_raises_400(svc: SglangService) -> None:
    custom = CustomModel(id="c-2", data={"id": "dup", "hf_id": "google/model", "size": "1GB"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(HTTPException) as exc_info:
        svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    assert exc_info.value.status_code == 400


def test_remove_custom_model_deletes_entry(svc: SglangService) -> None:
    custom = CustomModel(id="c-4", data={"id": "to-remove", "hf_id": "google/model", "size": "1GB"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].installed = _make_installed_info()

    svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert "to-remove" not in svc.models["default"]


def test_remove_custom_model_raises_400_when_in_use(svc: SglangService) -> None:
    custom = CustomModel(id="c-5", data={"id": "in-use", "hf_id": "google/model", "size": "1GB"})
    svc._add_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]
    installed = _make_installed_info()
    installed.models["in-use"] = _make_model_installed_info("in-use")
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc._remove_custom_model("default", custom)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_list_models_raises_404_for_unknown_instance(svc: SglangService) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await svc.list_models("nonexistent", ListModelsFilters())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_list_models_returns_all_models_no_filter(svc: SglangService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters())

    assert len(result.list) == len(svc.models["default"])


@pytest.mark.asyncio
async def test_get_model_raises_400_for_unknown_model_id(svc: SglangService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc.get_model("default", "nonexistent-model")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_model_returns_correct_model(svc: SglangService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    model_id = next(iter(svc.models["default"]))

    result = await svc.get_model("default", model_id)

    assert result.id == model_id


def test_build_sglang_command_uses_launch_server_prefix(svc: SglangService, tmp_path: Path) -> None:
    model = SglangModel(hf_id="google/test", size="1GB")
    opts = SglangModelOptions.model_construct(extra_args={})

    cmd = svc._build_sglang_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        model=model,
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        max_model_length=None,
    )

    assert cmd[:3] == ["python3", "-m", "sglang.launch_server"]
    assert "--model-path" in cmd
    assert "--host" in cmd
    assert "0.0.0.0" in cmd
    assert "--port" in cmd
    assert "30000" in cmd
    assert "--served-model-name" in cmd
    assert "google/test" in cmd


def test_build_sglang_command_applies_mem_fraction_static(svc: SglangService, tmp_path: Path) -> None:
    model = SglangModel(hf_id="google/test", size="1GB")
    opts = SglangModelOptions.model_construct(extra_args={})

    cmd = svc._build_sglang_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        model=model,
        opts=opts,
        quantization=None,
        gpu_memory_utilization=0.7,
        max_model_length=None,
    )

    assert "--mem-fraction-static" in cmd
    assert "0.7" in cmd


def test_build_sglang_command_applies_context_length(svc: SglangService, tmp_path: Path) -> None:
    model = SglangModel(hf_id="google/test", size="1GB")
    opts = SglangModelOptions.model_construct(extra_args={})

    cmd = svc._build_sglang_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        model=model,
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        max_model_length=4096,
    )

    assert "--context-length" in cmd
    assert "4096" in cmd


def test_build_sglang_command_applies_quantization(svc: SglangService, tmp_path: Path) -> None:
    model = SglangModel(hf_id="google/test", size="1GB")
    opts = SglangModelOptions.model_construct(extra_args={})

    cmd = svc._build_sglang_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        model=model,
        opts=opts,
        quantization="fp8",
        gpu_memory_utilization=None,
        max_model_length=None,
    )

    assert "--quantization" in cmd
    assert "fp8" in cmd


def test_build_sglang_command_extra_args_with_value(svc: SglangService, tmp_path: Path) -> None:
    model = SglangModel(hf_id="google/test", size="1GB")
    opts = SglangModelOptions.model_construct(extra_args={"--dtype": "float16"})

    cmd = svc._build_sglang_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        model=model,
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        max_model_length=None,
    )

    assert "--dtype" in cmd
    assert "float16" in cmd


def test_build_sglang_command_extra_args_flag_only(svc: SglangService, tmp_path: Path) -> None:
    model = SglangModel(hf_id="google/test", size="1GB")
    opts = SglangModelOptions.model_construct(extra_args={"--enable-metrics": None})

    cmd = svc._build_sglang_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        model=model,
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        max_model_length=None,
    )

    assert "--enable-metrics" in cmd


def test_build_sglang_command_embedding_model_gets_is_embedding(svc: SglangService, tmp_path: Path) -> None:
    model = SglangModel(hf_id="google/embedder", size="1GB", model_type="embedding")
    opts = SglangModelOptions.model_construct(extra_args={})

    cmd = svc._build_sglang_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/embedder",
        model=model,
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        max_model_length=None,
    )

    assert "--is-embedding" in cmd


def test_build_sglang_command_cross_encoder_reranker_gets_is_embedding(svc: SglangService, tmp_path: Path) -> None:
    model = SglangModel(hf_id="BAAI/bge-reranker-v2-m3", size="1GB", model_type="reranker")
    opts = SglangModelOptions.model_construct(extra_args={})

    cmd = svc._build_sglang_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="BAAI/bge-reranker-v2-m3",
        model=model,
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        max_model_length=None,
    )

    assert "--is-embedding" in cmd


def test_build_sglang_command_qwen3_reranker_omits_is_embedding(svc: SglangService, tmp_path: Path) -> None:
    model = SglangModel(hf_id="Qwen/Qwen3-Reranker-0.6B", size="1GB", model_type="reranker")
    opts = SglangModelOptions.model_construct(extra_args={})

    cmd = svc._build_sglang_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="Qwen/Qwen3-Reranker-0.6B",
        model=model,
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        max_model_length=None,
    )

    assert "--is-embedding" not in cmd


def test_build_sglang_command_registry_flag_overrides_qwen3_naming_heuristic(svc: SglangService, tmp_path: Path) -> None:
    # A seq-cls fork of Qwen3-Reranker: naming heuristic alone would say "generative, skip the
    # flag", but this is architecturally a cross-encoder, so the registry's `is_generative_reranker
    # =False` verdict must win and the flag must be added.
    model = SglangModel(hf_id="tomaarsen/Qwen3-Reranker-0.6B-seq-cls", size="1GB", model_type="reranker", is_generative_reranker=False)
    opts = SglangModelOptions.model_construct(extra_args={})

    cmd = svc._build_sglang_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
        model=model,
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        max_model_length=None,
    )

    assert "--is-embedding" in cmd


def test_build_sglang_command_llm_model_omits_is_embedding(svc: SglangService, tmp_path: Path) -> None:
    model = SglangModel(hf_id="google/test", size="1GB", model_type="llm")
    opts = SglangModelOptions.model_construct(extra_args={})

    cmd = svc._build_sglang_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        model=model,
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        max_model_length=None,
    )

    assert "--is-embedding" not in cmd


def test_build_sglang_command_reranker_gets_disable_radix_cache(svc: SglangService, tmp_path: Path) -> None:
    model = SglangModel(hf_id="BAAI/bge-reranker-v2-m3", size="1GB", model_type="reranker")
    opts = SglangModelOptions.model_construct(extra_args={})

    cmd = svc._build_sglang_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="BAAI/bge-reranker-v2-m3",
        model=model,
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        max_model_length=None,
    )

    assert "--disable-radix-cache" in cmd


def test_build_sglang_command_embedding_omits_disable_radix_cache(svc: SglangService, tmp_path: Path) -> None:
    model = SglangModel(hf_id="google/embedder", size="1GB", model_type="embedding")
    opts = SglangModelOptions.model_construct(extra_args={})

    cmd = svc._build_sglang_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/embedder",
        model=model,
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        max_model_length=None,
    )

    assert "--disable-radix-cache" not in cmd


def test_build_sglang_command_llm_model_omits_disable_radix_cache(svc: SglangService, tmp_path: Path) -> None:
    model = SglangModel(hf_id="google/test", size="1GB", model_type="llm")
    opts = SglangModelOptions.model_construct(extra_args={})

    cmd = svc._build_sglang_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/test",
        model=model,
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        max_model_length=None,
    )

    assert "--disable-radix-cache" not in cmd


@pytest.mark.parametrize(
    ("hf_id", "expected"),
    [
        ("Qwen/Qwen3-Reranker-0.6B", True),
        ("Qwen/Qwen3-Reranker-4B", True),
        ("qwen/qwen3-reranker-8b", True),
        ("BAAI/bge-reranker-v2-m3", False),
        ("Qwen/Qwen3-0.6B", False),
        ("cross-encoder/ms-marco-reranker", False),
    ],
)
def test_is_qwen3_reranker(hf_id: str, expected: bool) -> None:
    assert SglangService._is_qwen3_reranker(hf_id) == expected  # pyright: ignore[reportPrivateUsage]


def test_is_generative_reranker_trusts_registry_flag_over_naming_heuristic() -> None:
    # Named like a Qwen3-Reranker (naming heuristic would say True), but the registry's
    # architecture-derived flag says it's actually a cross-encoder fork (seq-cls conversion).
    model = SglangModel(hf_id="tomaarsen/Qwen3-Reranker-0.6B-seq-cls", size="1GB", model_type="reranker", is_generative_reranker=False)

    assert SglangService._is_generative_reranker(model) is False  # pyright: ignore[reportPrivateUsage]


def test_is_generative_reranker_trusts_registry_flag_when_true() -> None:
    model = SglangModel(hf_id="BAAI/bge-reranker-v2-gemma", size="1GB", model_type="reranker", is_generative_reranker=True)

    assert SglangService._is_generative_reranker(model) is True  # pyright: ignore[reportPrivateUsage]


def test_is_generative_reranker_falls_back_to_naming_heuristic_when_unset() -> None:
    model = SglangModel(hf_id="Qwen/Qwen3-Reranker-0.6B", size="1GB", model_type="reranker")

    assert SglangService._is_generative_reranker(model) is True  # pyright: ignore[reportPrivateUsage]


def test_describe_install_failure_maps_is_embedding_mismatch(svc: SglangService) -> None:
    exc = RuntimeError("Please relaunch without --is-embedding for this model")

    result = svc._describe_install_failure(exc)  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result, HTTPException)
    assert result.status_code == 400


def test_describe_install_failure_passes_through_other_errors(svc: SglangService) -> None:
    exc = RuntimeError("some other docker failure")

    result = svc._describe_install_failure(exc)  # pyright: ignore[reportPrivateUsage]

    assert result is exc


@pytest.mark.parametrize(
    ("hf_id", "expected"),
    [
        ("deepseek-ai/DeepSeek-R1", "deepseek-r1"),
        ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "deepseek-r1"),
        ("deepseek-ai/DeepSeek-V3", "deepseek-v3"),
        ("Qwen/Qwen3-30B-A3B", "qwen3"),
        ("cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit", None),
        ("google/gemma-3-270m-it", None),
        ("Qwen/Qwen3-Reranker-4B", None),
        ("Qwen/Qwen3-Embedding-4B", None),
    ],
)
def test_detect_reasoning_parser(hf_id: str, expected: str | None) -> None:
    assert SglangService._detect_reasoning_parser(hf_id) == expected  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("hf_id", "expected"),
    [
        ("cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit", "qwen3_coder"),
        ("Qwen/Qwen3-30B-A3B", "qwen"),
        ("deepseek-ai/DeepSeek-V3", "deepseekv3"),
        ("meta-llama/Meta-Llama-3-8B-Instruct", "llama3"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
        ("google/gemma-3-270m-it", None),
        ("Qwen/Qwen3-Reranker-4B", None),
        ("Qwen/Qwen3-Embedding-4B", None),
    ],
)
def test_detect_tool_call_parser(hf_id: str, expected: str | None) -> None:
    assert SglangService._detect_tool_call_parser(hf_id) == expected  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("hf_id", "expected"),
    [
        ("Qwen/Qwen3-Reranker-4B", True),
        ("BAAI/bge-reranker-v2-m3", True),
        ("Qwen/Qwen3-Embedding-4B", True),
        ("google/embeddinggemma-300m", True),
        ("Qwen/Qwen3-30B-A3B", False),
        ("google/gemma-3-270m-it", False),
    ],
)
def test_is_rerank_or_embedding_hf_id(hf_id: str, expected: bool) -> None:
    assert SglangService._is_rerank_or_embedding_hf_id(hf_id) == expected  # pyright: ignore[reportPrivateUsage]


def test_build_sglang_command_llm_reasoning_model_gets_reasoning_parser(svc: SglangService, tmp_path: Path) -> None:
    model = SglangModel(hf_id="deepseek-ai/DeepSeek-R1", size="1GB", model_type="llm")
    opts = SglangModelOptions.model_construct(extra_args={})

    cmd = svc._build_sglang_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="deepseek-ai/DeepSeek-R1",
        model=model,
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        max_model_length=None,
    )

    assert "--reasoning-parser" in cmd
    assert "deepseek-r1" in cmd


def test_build_sglang_command_llm_tool_call_model_gets_tool_call_parser(svc: SglangService, tmp_path: Path) -> None:
    model = SglangModel(hf_id="meta-llama/Meta-Llama-3-8B-Instruct", size="1GB", model_type="llm")
    opts = SglangModelOptions.model_construct(extra_args={})

    cmd = svc._build_sglang_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        model=model,
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        max_model_length=None,
    )

    assert "--tool-call-parser" in cmd
    assert "llama3" in cmd


def test_build_sglang_command_unmatched_llm_omits_parsers(svc: SglangService, tmp_path: Path) -> None:
    model = SglangModel(hf_id="google/gemma-3-270m-it", size="1GB", model_type="llm")
    opts = SglangModelOptions.model_construct(extra_args={})

    cmd = svc._build_sglang_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="google/gemma-3-270m-it",
        model=model,
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        max_model_length=None,
    )

    assert "--reasoning-parser" not in cmd
    assert "--tool-call-parser" not in cmd


def test_build_sglang_command_reranker_omits_parsers(svc: SglangService, tmp_path: Path) -> None:
    model = SglangModel(hf_id="Qwen/Qwen3-Reranker-0.6B", size="1GB", model_type="reranker")
    opts = SglangModelOptions.model_construct(extra_args={})

    cmd = svc._build_sglang_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="Qwen/Qwen3-Reranker-0.6B",
        model=model,
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        max_model_length=None,
    )

    assert "--reasoning-parser" not in cmd
    assert "--tool-call-parser" not in cmd


def test_build_sglang_command_extra_args_override_reasoning_parser(svc: SglangService, tmp_path: Path) -> None:
    model = SglangModel(hf_id="deepseek-ai/DeepSeek-R1", size="1GB", model_type="llm")
    opts = SglangModelOptions.model_construct(extra_args={"--reasoning-parser": "deepseek-v3"})

    cmd = svc._build_sglang_command(  # pyright: ignore[reportPrivateUsage]
        docker_model_path=tmp_path / "model",
        model_id="deepseek-ai/DeepSeek-R1",
        model=model,
        opts=opts,
        quantization=None,
        gpu_memory_utilization=None,
        max_model_length=None,
    )

    assert cmd.count("--reasoning-parser") == 1
    assert "deepseek-v3" in cmd


def test_register_model_endpoint_llm_calls_chat_completion_proxy(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    model = SglangModel(hf_id="google/test", size="1GB", model_type="llm")
    model_info = _make_model_installed_info("google/test", model_type="llm")

    svc._register_model_endpoint(  # pyright: ignore[reportPrivateUsage]
        model_info=model_info,
        model=model,
        registered_name="google/test",
        model_id="google/test",
        context_window=4096,
        max_context_window=4096,
    )

    assert gpu_deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 1
    assert gpu_deps["endpoint_registry"].register_rerank_as_proxy.call_count == 0
    assert gpu_deps["endpoint_registry"].register_embeddings_as_proxy.call_count == 0


def test_register_model_endpoint_reranker_calls_rerank_proxy(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    model = SglangModel(hf_id="google/reranker", size="1GB", model_type="reranker")
    model_info = _make_model_installed_info("google/reranker", model_type="reranker")

    svc._register_model_endpoint(  # pyright: ignore[reportPrivateUsage]
        model_info=model_info,
        model=model,
        registered_name="google/reranker",
        model_id="google/reranker",
        context_window=None,
        max_context_window=None,
    )

    assert gpu_deps["endpoint_registry"].register_rerank_as_proxy.call_count == 1
    assert gpu_deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 0
    assert gpu_deps["endpoint_registry"].register_rerank_as_proxy.call_args.kwargs["normalize_sglang_response"] is True


def test_register_model_endpoint_embedding_calls_embeddings_proxy(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    model = SglangModel(hf_id="google/embedder", size="1GB", model_type="embedding")
    model_info = _make_model_installed_info("google/embedder", model_type="embedding")

    svc._register_model_endpoint(  # pyright: ignore[reportPrivateUsage]
        model_info=model_info,
        model=model,
        registered_name="google/embedder",
        model_id="google/embedder",
        context_window=None,
        max_context_window=None,
    )

    assert gpu_deps["endpoint_registry"].register_embeddings_as_proxy.call_count == 1
    assert gpu_deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 0
    assert gpu_deps["endpoint_registry"].register_rerank_as_proxy.call_count == 0


def test_register_model_endpoint_llm_uses_v1_paths(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    model = SglangModel(hf_id="google/test", size="1GB", model_type="llm")
    model_info = _make_model_installed_info("google/test", model_type="llm")
    model_info.base_url = "http://localhost:30000"

    svc._register_model_endpoint(  # pyright: ignore[reportPrivateUsage]
        model_info=model_info,
        model=model,
        registered_name="google/test",
        model_id="google/test",
        context_window=None,
        max_context_window=None,
    )

    kwargs = gpu_deps["endpoint_registry"].register_chat_completion_as_proxy.call_args.kwargs
    assert kwargs["chat_completions"].url == "http://localhost:30000/v1/chat/completions"
    assert kwargs["completions"].url == "http://localhost:30000/v1/completions"
    assert kwargs["responses"].url == "http://localhost:30000/v1/responses"
    assert kwargs["messages"].url == "http://localhost:30000/v1/messages"


@pytest.mark.asyncio
async def test_install_model_returns_already_installed(svc: SglangService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed

    promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    result = await promise.wait()
    assert result.status == "OK"
    assert "Already installed" in result.details


@pytest.mark.asyncio
async def test_install_model_raises_400_for_unknown_model(svc: SglangService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        await svc._install_model("default", "nonexistent-model", InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_calls_docker_install(svc: SglangService, gpu_deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, gpu_deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.sglang_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.sglang_service.get_base_url", return_value="http://localhost:30000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert gpu_deps["docker_service"].install_and_run_docker.call_count == 1
    docker_options = gpu_deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.image_port == 30000
    assert "python3" in docker_options.command
    assert "sglang.launch_server" in docker_options.command
    assert "health_generate" in docker_options.healthcheck["test"]


@pytest.mark.asyncio
async def test_install_model_registers_chat_completion_endpoint(svc: SglangService, gpu_deps: dict[str, Any], tmp_path: Path) -> None:
    _setup_install_mocks(svc, gpu_deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.sglang_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.sglang_service.get_base_url", return_value="http://localhost:30000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert gpu_deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 1


@pytest.mark.asyncio
async def test_install_model_registers_reranker_endpoint_for_reranker_model(
    svc: SglangService, gpu_deps: dict[str, Any], tmp_path: Path
) -> None:
    _setup_install_mocks(svc, gpu_deps)
    reranker_id = "reranker-model"
    svc.models["default"][reranker_id] = SglangModel(hf_id=reranker_id, size="1GB", model_type="reranker")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.sglang_service.get_base_url", return_value="http://localhost:30000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", reranker_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        await promise.wait()

    assert gpu_deps["endpoint_registry"].register_rerank_as_proxy.call_count == 1
    assert gpu_deps["endpoint_registry"].register_chat_completion_as_proxy.call_count == 0


@pytest.mark.asyncio
async def test_install_model_docker_failure_decrements_gpu_memory(svc: SglangService, gpu_deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _make_installed_info(hardware=True)
    svc.instances_info["default"].installed = installed
    gpu_deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=RuntimeError("docker failed"))
    gpu_deps["docker_service"].get_docker_subnet.return_value = None
    gpu_deps["docker_service"].get_docker_container_name.return_value = "container"
    gpu_deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.sglang_service.get_base_url", return_value="http://localhost:30000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert svc.gpu_memory_utilization == 0.0


@pytest.mark.asyncio
async def test_install_model_is_embedding_mismatch_raises_actionable_400(
    svc: SglangService, gpu_deps: dict[str, Any], tmp_path: Path
) -> None:
    installed = _make_installed_info(hardware=True)
    svc.instances_info["default"].installed = installed
    gpu_deps["docker_service"].install_and_run_docker = AsyncMock(
        side_effect=RuntimeError("Please relaunch without --is-embedding for Qwen3-Reranker models")
    )
    gpu_deps["docker_service"].get_docker_subnet.return_value = None
    gpu_deps["docker_service"].get_docker_container_name.return_value = "container"
    gpu_deps["docker_service"].stop_docker = AsyncMock()
    reranker_id = "Qwen/Qwen3-Reranker-0.6B"
    svc.models["default"][reranker_id] = SglangModel(hf_id=reranker_id, size="1GB", model_type="reranker")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.sglang_service.get_base_url", return_value="http://localhost:30000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", reranker_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(HTTPException) as exc_info:
            await promise.wait()

    assert exc_info.value.status_code == 400
    assert "is-embedding" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_install_model_retries_without_is_embedding_flag_on_mismatch(
    svc: SglangService, gpu_deps: dict[str, Any], tmp_path: Path
) -> None:
    installed = _make_installed_info(hardware=True)
    svc.instances_info["default"].installed = installed
    observed_commands: list[str] = []

    async def fake_install_and_run_docker(docker_options: Any) -> int:
        observed_commands.append(docker_options.command)
        if len(observed_commands) == 1:
            raise RuntimeError("Please relaunch without --is-embedding for this model")
        return 30000

    gpu_deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=fake_install_and_run_docker)
    gpu_deps["docker_service"].get_docker_subnet.return_value = None
    gpu_deps["docker_service"].get_docker_container_name.return_value = "container"
    gpu_deps["docker_service"].get_container_host.return_value = "localhost"
    gpu_deps["docker_service"].get_container_port.return_value = 30000
    gpu_deps["docker_service"].stop_docker = AsyncMock()
    embedding_id = "embedding-model"
    svc.models["default"][embedding_id] = SglangModel(hf_id=embedding_id, size="1GB", model_type="embedding")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.sglang_service.get_base_url", return_value="http://localhost:30000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", embedding_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert result.status == "OK"
    assert len(observed_commands) == 2
    assert "--is-embedding" in observed_commands[0]
    assert "--is-embedding" not in observed_commands[1]


@pytest.mark.asyncio
async def test_install_model_retry_without_is_embedding_flag_still_fails(
    svc: SglangService, gpu_deps: dict[str, Any], tmp_path: Path
) -> None:
    installed = _make_installed_info(hardware=True)
    svc.instances_info["default"].installed = installed
    call_count = 0

    async def fake_install_and_run_docker(_docker_options: Any) -> int:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Please relaunch without --is-embedding for this model")
        raise RuntimeError("still broken")

    gpu_deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=fake_install_and_run_docker)
    gpu_deps["docker_service"].get_docker_subnet.return_value = None
    gpu_deps["docker_service"].get_docker_container_name.return_value = "container"
    gpu_deps["docker_service"].stop_docker = AsyncMock()
    embedding_id = "embedding-model"
    svc.models["default"][embedding_id] = SglangModel(hf_id=embedding_id, size="1GB", model_type="embedding")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.sglang_service.get_base_url", return_value="http://localhost:30000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", embedding_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError, match="still broken"):
            await promise.wait()

    assert call_count == 2
    assert gpu_deps["docker_service"].stop_docker.call_count == 3


@pytest.mark.asyncio
async def test_install_model_releases_gpu_on_cancelled_error(svc: SglangService, gpu_deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _make_installed_info(hardware=True)
    svc.instances_info["default"].installed = installed
    gpu_deps["docker_service"].install_and_run_docker = AsyncMock(side_effect=asyncio.CancelledError())
    gpu_deps["docker_service"].get_docker_subnet.return_value = None
    gpu_deps["docker_service"].get_docker_container_name.return_value = "container"
    gpu_deps["docker_service"].stop_docker = AsyncMock()

    model_id = "cancel-model"
    svc.models["default"][model_id] = SglangModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.5)

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.sglang_service.get_base_url", return_value="http://localhost:30000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(asyncio.CancelledError):
            await promise.wait()

    assert svc.gpu_memory_utilization == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_uninstall_model_removes_from_installed_and_unregisters_endpoint(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model", "reg-1")
    svc.instances_info["default"].installed = installed
    gpu_deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert "test-model" not in installed.models
    assert gpu_deps["endpoint_registry"].unregister_chat_completion.call_count == 1
    assert gpu_deps["endpoint_registry"].unregister_chat_completion.call_args == call("test-model", "reg-1")


@pytest.mark.asyncio
async def test_uninstall_model_decrements_gpu_memory_utilization(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    svc.gpu_memory_utilization = 0.5
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model", gpu_memory_utilization=0.5)
    svc.instances_info["default"].installed = installed
    gpu_deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.gpu_memory_utilization == 0.0


@pytest.mark.asyncio
async def test_uninstall_model_reranker_calls_unregister_rerank(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["reranker-model"] = _make_model_installed_info("reranker-model", "reg-rerank", model_type="reranker")
    svc.instances_info["default"].installed = installed
    gpu_deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "reranker-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert gpu_deps["endpoint_registry"].unregister_rerank.call_count == 1
    assert gpu_deps["endpoint_registry"].unregister_chat_completion.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_model_embedding_calls_unregister_embeddings(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["embedder-model"] = _make_model_installed_info("embedder-model", "reg-embed", model_type="embedding")
    svc.instances_info["default"].installed = installed
    gpu_deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "embedder-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert gpu_deps["endpoint_registry"].unregister_embeddings.call_count == 1
    assert gpu_deps["endpoint_registry"].unregister_chat_completion.call_count == 0
    assert gpu_deps["endpoint_registry"].unregister_rerank.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_instance_calls_uninstall_model_for_each_model(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info("m1")
    installed.models["m2"] = _make_model_installed_info("m2")
    svc.instances_info["default"].installed = installed
    gpu_deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall:  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 2


@pytest.mark.asyncio
async def test_stop_instance_stops_all_containers(svc: SglangService) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info("m1")
    installed.models["m2"] = _make_model_installed_info("m2")
    svc.instances_info["default"].installed = installed

    with patch.object(svc, "_stop_dockers_parallel", new_callable=AsyncMock) as mock_stop:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock_stop.call_count == 1


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_increments_total(svc: SglangService) -> None:
    svc.gpu_memory_utilization = 0.0
    model = SglangModel(hf_id="google/test", size="1GB")
    opts = SglangModelOptions(gpu_memory_utilization=0.5)

    result = await svc._get_gpu_memory_utilization(opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == 0.5
    assert svc.gpu_memory_utilization == 0.5


@pytest.mark.asyncio
async def test_get_gpu_memory_utilization_raises_422_when_sum_exceeds_1(svc: SglangService) -> None:
    svc.gpu_memory_utilization = 0.8
    model = SglangModel(hf_id="google/test", size="1GB")
    opts = SglangModelOptions(gpu_memory_utilization=0.5)

    with pytest.raises(HTTPException) as exc_info:
        await svc._get_gpu_memory_utilization(opts, model)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_get_default_gpu_memory_utilization_uses_free_fraction_with_margin(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    gpu_deps["hardware"].get_realtime_stats = AsyncMock(return_value=GpuStats(total_vram_gb=10.0, used_vram_gb=5.0, gpus=None))

    result = await svc._get_default_gpu_memory_utilization()  # pyright: ignore[reportPrivateUsage]

    assert result == 0.45


def test_release_gpu_utilization_floors_at_zero(svc: SglangService) -> None:
    svc.gpu_memory_utilization = 0.1
    svc._release_gpu_utilization(0.5)  # pyright: ignore[reportPrivateUsage]
    assert svc.gpu_memory_utilization == 0


@pytest.mark.asyncio
async def test_get_quantization_returns_valid_value(svc: SglangService) -> None:
    model = SglangModel(hf_id="google/test", size="1GB")
    opts = SglangModelOptions(quantization="fp8")

    result = await svc._get_quantization(opts, model)  # pyright: ignore[reportPrivateUsage]

    assert result == "fp8"


@pytest.mark.asyncio
async def test_get_quantization_raises_422_on_invalid_characters(svc: SglangService) -> None:
    model = SglangModel(hf_id="google/test", size="1GB")
    opts = SglangModelOptions(quantization=None)
    opts.quantization = "fp8!@#"

    with pytest.raises(HTTPException) as exc_info:
        await svc._get_quantization(opts, model)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 422


def test_get_image_default(svc: SglangService) -> None:
    image = svc._get_image()  # pyright: ignore[reportPrivateUsage]

    assert image == _const.image


def test_get_image_with_version_overrides_tag(svc: SglangService) -> None:
    image = svc._get_image(image_version="v0.9.0-cu130")  # pyright: ignore[reportPrivateUsage]

    base = _const.image.name.split(":")[0]
    assert image.name == f"{base}:v0.9.0-cu130"
    assert image.size == _const.image.size


def test_get_docker_image_repo(svc: SglangService) -> None:
    assert svc.get_docker_image_repo(None) == _const.image.name.split(":")[0]


def test_get_default_docker_tag(svc: SglangService) -> None:
    assert svc.get_default_docker_tag(None) == _const.image.name.split(":")[1]


@pytest.mark.asyncio
async def test_get_docker_tags_filters_to_cuda_tags(svc: SglangService) -> None:
    mock_client = AsyncMock()
    mock_client.get_tags = AsyncMock(
        return_value=["v0.5.16-cu129", "v0.5.16-cu130", "v0.5.16-xeon", "v0.5.16-rocm700-mi35x", "dev", "v0.5.16"]
    )
    with (
        patch("server.services.sglang_service.registry_for", return_value=mock_client),
        patch("server.services.sglang_service.image_without_registry_prefix", return_value="lmsysorg/sglang"),
    ):
        tags = await svc.get_docker_tags("GPU")

    assert "v0.5.16-cu129" in tags
    assert "v0.5.16-cu130" in tags
    assert "v0.5.16-xeon" not in tags
    assert "v0.5.16-rocm700-mi35x" not in tags
    assert "dev" not in tags
    assert "v0.5.16" not in tags


def test_filter_docker_tags_excludes_xeon_and_rocm(svc: SglangService) -> None:
    tags = ["v0.5.16-cu129", "v0.5.16-cu130", "v0.5.16-xeon", "v0.5.16-rocm700-mi35x", "v0.5.16-cann9.0.0-a3"]

    filtered = svc.filter_docker_tags(tags, "GPU")

    assert filtered == ["v0.5.16-cu129", "v0.5.16-cu130"]


def _status(exists: bool = True, state: str = "running", health: str = "healthy", restart_count: int = 0) -> ContainerStatus:
    return ContainerStatus(exists=exists, state=state, health=health, restart_count=restart_count)


@pytest.mark.asyncio
async def test_reconcile_instance_models_releases_after_threshold_bad_polls(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", registration_id="reg-1", gpu_memory_utilization=0.5, model_type="llm")
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    gpu_deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))
    gpu_deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_save", new_callable=AsyncMock):
        for _ in range(3):
            await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert "m1" not in installed.models
    assert svc.gpu_memory_utilization == pytest.approx(0.0)
    gpu_deps["endpoint_registry"].unregister_chat_completion.assert_called_once_with("m1", "reg-1")


@pytest.mark.asyncio
async def test_reconcile_instance_models_releases_embedding_model(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", registration_id="reg-1", gpu_memory_utilization=0.5, model_type="embedding")
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    gpu_deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))
    gpu_deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_save", new_callable=AsyncMock):
        for _ in range(3):
            await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    gpu_deps["endpoint_registry"].unregister_embeddings.assert_called_once_with("m1", "reg-1")
    gpu_deps["endpoint_registry"].unregister_chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_instance_models_recovers_within_threshold_resets_counter(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", gpu_memory_utilization=0.5)
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    gpu_deps["docker_service"].get_container_status = AsyncMock(return_value=_status(state="exited", health=""))

    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]
    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    gpu_deps["docker_service"].get_container_status = AsyncMock(return_value=_status(state="running", health="healthy"))
    await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    assert ("default", "m1") not in svc._crash_poll_state  # pyright: ignore[reportPrivateUsage]
    assert "m1" in installed.models
    assert svc.gpu_memory_utilization == 0.5


@pytest.mark.asyncio
async def test_download_model_emits_initial_and_final_progress(svc: SglangService, gpu_deps: dict[str, Any], tmp_path: Path) -> None:
    model = SglangModel(hf_id="google/test-model", size="1GB")
    stream = MagicMock()
    local_path = tmp_path / "model-files"
    local_path.mkdir()

    async def mock_download(*args: object):  # type: ignore[misc]
        yield SuccessDownloadPacket(local_path=local_path, filename="model-files")

    gpu_deps["model_downloader"].download = mock_download

    await svc._download_model(stream, "google/test-model", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert stream.emit.call_count >= 2


@pytest.mark.asyncio
async def test_download_model_raises_500_when_no_success_packet(svc: SglangService, gpu_deps: dict[str, Any], tmp_path: Path) -> None:
    model = SglangModel(hf_id="google/test", size="100MB")
    stream = MagicMock()

    async def mock_download(*args: object):  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=1024)

    gpu_deps["model_downloader"].download = mock_download

    with (
        patch.object(svc, "_get_working_dir", return_value=tmp_path),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(HTTPException) as exc_info,
    ):
        await svc._download_model(stream, "google/test", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_download_model_handles_pre_download_packet_with_size(svc: SglangService, gpu_deps: dict[str, Any], tmp_path: Path) -> None:
    model = SglangModel(hf_id="google/test", size="100MB")
    stream = MagicMock()

    async def mock_download(*args: object):  # type: ignore[misc]
        yield PreDownloadPacket(file_bytes_size=50 * 1024 * 1024)
        yield SuccessDownloadPacket(local_path=tmp_path / "model", filename="model")

    gpu_deps["model_downloader"].download = mock_download

    with patch.object(svc, "_get_working_dir", return_value=tmp_path):  # pyright: ignore[reportPrivateUsage]
        local_path = await svc._download_model(stream, "google/test", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert local_path is not None


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_formatted_size(svc: SglangService) -> None:
    with patch("server.services.sglang_service.fetch_huggingface_model_size", new=AsyncMock(return_value="4.0 GB")):
        result = await svc._resolve_custom_model_size({"hf_id": "google/gemma-3-270m-it"})  # pyright: ignore[reportPrivateUsage]

    assert result == "4.0 GB"


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_none_on_exception(svc: SglangService) -> None:
    with patch("server.services.sglang_service.fetch_huggingface_model_size", new=AsyncMock(side_effect=Exception("fail"))):
        result = await svc._resolve_custom_model_size({"hf_id": "google/gemma"})  # pyright: ignore[reportPrivateUsage]

    assert result is None


def test_get_vram_estimate_returns_value(svc: SglangService) -> None:
    svc.hardware.total_vram_gb = 24.0  # pyright: ignore[reportAttributeAccessIssue]
    model_info = _make_model_installed_info("test-model", gpu_memory_utilization=0.5)

    result = svc._get_vram_estimate(model_info)  # pyright: ignore[reportPrivateUsage]

    assert result == 12.0


def test_get_vram_estimate_returns_none_when_no_utilization(svc: SglangService) -> None:
    model_info = _make_model_installed_info("test-model", gpu_memory_utilization=None)

    result = svc._get_vram_estimate(model_info)  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_get_cached_vram_estimate_cache_hit(svc: SglangService) -> None:
    svc._vram_cache[("default", "test-model")] = 4.0  # pyright: ignore[reportPrivateUsage]

    result = await svc._get_cached_vram_estimate("default", "test-model", is_loaded=True)  # pyright: ignore[reportPrivateUsage]

    assert result == 4.0


@pytest.mark.asyncio
async def test_get_cached_vram_estimate_not_loaded_evicts_cache(svc: SglangService) -> None:
    svc._vram_cache[("default", "test-model")] = 4.0  # pyright: ignore[reportPrivateUsage]

    result = await svc._get_cached_vram_estimate("default", "test-model", is_loaded=False)  # pyright: ignore[reportPrivateUsage]

    assert result is None
    assert ("default", "test-model") not in svc._vram_cache  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_get_cached_vram_estimate_model_not_in_installed(svc: SglangService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    result = await svc._get_cached_vram_estimate("default", "missing-model", is_loaded=True)  # pyright: ignore[reportPrivateUsage]

    assert result is None
    assert ("default", "missing-model") not in svc._vram_cache  # pyright: ignore[reportPrivateUsage]


def test_model_installed_info_get_info() -> None:
    model = _make_model_installed_info(registration_id="reg-42")

    info = model.get_info()

    assert info.registration_id == "reg-42"


def test_generate_instance_config_none_info(svc: SglangService) -> None:
    config = svc._generate_instance_config(None, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options is None
    assert config.models == []


def test_generate_instance_config_with_info(svc: SglangService) -> None:
    info = _make_installed_info()
    info.models["m1"] = _make_model_installed_info("m1")

    config = svc._generate_instance_config(info, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options == info.options
    assert len(config.models or []) == 1


def test_load_download_info(svc: SglangService) -> None:
    result = svc._load_download_info({"model_path": "/tmp/x"})  # pyright: ignore[reportPrivateUsage]

    assert isinstance(result, DownloadedInfo)
    assert result.model_path == "/tmp/x"


def test_get_installed_info_returns_spec_when_installed(svc: SglangService) -> None:
    installed = _make_installed_info()
    installed.options.spec["hardware"] = True
    svc.instances_info["default"].installed = installed

    result = svc.get_installed_info("default")

    assert result == installed.options.spec


def test_get_installed_info_delegates_when_not_installed(svc: SglangService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_get_service_installed_info", return_value=False) as mock:  # pyright: ignore[reportPrivateUsage]
        result = svc.get_installed_info("default")

    assert mock.call_count == 1
    assert mock.call_args == call("default")
    assert result is False


def test_get_specified_hardware_parts_delegates_to_super_when_gpu_present(svc: SglangService) -> None:
    with patch("server.services.base2_service.Base2Service.get_specified_hardware_parts", return_value=[]) as mock_super:
        result = svc.get_specified_hardware_parts(True)

    assert mock_super.call_count == 1
    assert result == []


def test_get_spec_multiple_gpus_offers_all_gpus_option(deps: dict[str, Any]) -> None:
    deps["hardware"].gpus = [MagicMock(spec=NvidiaGpuInfo), MagicMock(spec=NvidiaGpuInfo)]
    svc = SglangService(**deps)

    spec = svc.get_spec()

    hardware_field = next(f for f in spec.fields if f.name == "hardware")
    assert hardware_field.default == "GPUs"


def test_get_model_spec_adds_hardware_key_when_missing(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    gpu_deps["docker_service"].has_gpu_support = True
    svc.instances_info["default"].config = InstanceConfig(options=InstallServiceIn(spec={}))

    result = svc.get_model_spec("default", "llm")

    assert result is not None


def test_get_model_spec_omits_gpu_memory_utilization_for_non_gpu_hardware(svc: SglangService) -> None:
    svc.instances_info["default"].config.options = InstallServiceIn(spec={"hardware": False})

    spec = svc.get_model_spec("default", "llm")

    field_names = [f.name for f in spec.fields]
    assert "gpu_memory_utilization" not in field_names


@pytest.mark.asyncio
async def test_install_instance_loads_default_models_for_new_instance(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    options = InstallServiceIn(spec={"hardware": True})

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
async def test_install_instance_returns_installed_info_with_empty_models(svc: SglangService) -> None:
    options = InstallServiceIn(spec={"hardware": True})

    with (
        patch.object(svc, "_download_image_or_set_progress", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_verify_docker_image", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
        result = await promise.wait()

    assert isinstance(result, InstalledInfo)
    assert result.models == {}


@pytest.mark.asyncio
async def test_uninstall_instance_does_nothing_when_not_installed(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall:  # pyright: ignore[reportPrivateUsage]
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 0
    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_purge_clears_service_state(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    gpu_deps["docker_service"].remove_image = AsyncMock()
    gpu_deps["docker_service"].uninstall_docker = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock) as mock_clear,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert svc.service_downloaded is False
    assert mock_clear.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_instance_purge_removes_non_default_instance(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}), parsed_options=SglangOptions())
    svc.instances_info["inst2"].installed = installed
    gpu_deps["docker_service"].remove_image = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("inst2", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "inst2" not in svc.instances_info


@pytest.mark.asyncio
async def test_uninstall_instance_purge_with_other_instance_installed_does_not_remove_image(
    svc: SglangService, gpu_deps: dict[str, Any]
) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["extra"].installed = _make_installed_info()
    svc.instances_info["default"].installed = _make_installed_info()
    gpu_deps["docker_service"].remove_image = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock) as mock_clear,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert gpu_deps["docker_service"].remove_image.call_count == 0
    assert mock_clear.call_count == 0
    assert svc.instances_info["default"].installed is None
    assert "extra" in svc.instances_info


@pytest.mark.asyncio
async def test_uninstall_instance_skips_model_installed_in_other_instance(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["m1"] = _make_model_installed_info("m1")
    installed.models["m2"] = _make_model_installed_info("m2")
    svc.instances_info["default"].installed = installed
    gpu_deps["docker_service"].uninstall_docker = AsyncMock()

    def mock_is_in_other(instance: str, model_id: str) -> bool:
        return model_id == "m1"

    with (
        patch.object(svc, "is_model_installed_in_other_instance", side_effect=mock_is_in_other),
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock) as mock_uninstall,  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert mock_uninstall.call_count == 1
    called_ids = [c.args[1] for c in mock_uninstall.call_args_list]
    assert "m1" not in called_ids
    assert "m2" in called_ids


def test_get_docker_compose_file_path_raises_400_no_model_id(svc: SglangService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", None)

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_raises_400_model_not_installed(svc: SglangService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    with pytest.raises(HTTPException) as exc_info:
        svc.get_docker_compose_file_path("default", "some-model")

    assert exc_info.value.status_code == 400


def test_get_docker_compose_file_path_returns_path_for_installed_model(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model")
    svc.instances_info["default"].installed = installed
    expected = Path("/some/path/docker-compose.yml")
    gpu_deps["docker_service"].get_docker_compose_file_path.return_value = expected

    result = svc.get_docker_compose_file_path("default", "test-model")

    assert result == expected


def test_add_custom_model_creates_dict_for_new_instance(svc: SglangService) -> None:
    svc.instances_info["extra"] = Instance(None, None, {}, InstanceConfig())
    custom = CustomModel(id="c-3", data={"id": "new-model", "hf_id": "google/model", "size": "1GB"})

    svc._add_custom_model("extra", custom)  # pyright: ignore[reportPrivateUsage]

    assert "new-model" in svc.models["extra"]


@pytest.mark.asyncio
async def test_list_models_skips_instances_not_in_filter(svc: SglangService) -> None:
    svc.instances_info["inst2"] = Instance(None, None, {}, InstanceConfig())
    svc.models["inst2"] = {"extra-model": SglangModel(hf_id="extra/model", size="1GB")}
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}), parsed_options=SglangOptions())

    result = await svc.list_models("default", ListModelsFilters())

    assert not any(m.id == "extra-model" for m in result.list)


@pytest.mark.asyncio
async def test_list_models_filters_installed_only(svc: SglangService) -> None:
    installed = _make_installed_info()
    model_id = next(iter(svc.models["default"]))
    installed.models[model_id] = _make_model_installed_info(model_id)
    svc.instances_info["default"].installed = installed

    result = await svc.list_models("default", ListModelsFilters(installed=True))

    assert len(result.list) >= 1
    assert all(bool(m.installed) for m in result.list)


@pytest.mark.asyncio
async def test_download_model_handles_pre_download_packet_without_file_size(
    svc: SglangService, gpu_deps: dict[str, Any], tmp_path: Path
) -> None:
    model = SglangModel(hf_id="google/test", size="100MB")
    stream = MagicMock()
    local_path = tmp_path / "model-files"
    local_path.mkdir()

    async def mock_download(*args: object):  # type: ignore[misc]
        yield PreDownloadPacket(file_bytes_size=None)  # pyright: ignore[reportArgumentType]
        yield SuccessDownloadPacket(local_path=local_path, filename="model-files")

    gpu_deps["model_downloader"].download = mock_download
    result = await svc._download_model(stream, "google/test", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert result == local_path


@pytest.mark.asyncio
async def test_download_model_ignores_zero_bytes_downloaded_packet(svc: SglangService, gpu_deps: dict[str, Any], tmp_path: Path) -> None:
    model = SglangModel(hf_id="google/test", size="100MB")
    stream = MagicMock()
    local_path = tmp_path / "model-files"
    local_path.mkdir()

    async def mock_download(*args: object):  # type: ignore[misc]
        yield DownloadedPacket(downloaded_bytes_size=0)
        yield SuccessDownloadPacket(local_path=local_path, filename="model-files")

    gpu_deps["model_downloader"].download = mock_download
    result = await svc._download_model(stream, "google/test", model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert result == local_path


@pytest.mark.asyncio
async def test_download_model_or_set_progress_starts_new_download(svc: SglangService, tmp_path: Path) -> None:
    stream = MagicMock()
    model_id = "google/test-model"
    model = SglangModel(hf_id=model_id, size="1GB")
    svc.models["default"][model_id] = model

    with patch.object(  # pyright: ignore[reportPrivateUsage]
        svc, "_download_model", new_callable=AsyncMock, return_value=tmp_path / "model"
    ) as mock_dl:
        await svc._download_model_or_set_progress(stream, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert mock_dl.call_count == 1
    assert model_id not in svc.models_download_progress


@pytest.mark.asyncio
async def test_download_model_or_set_progress_cleans_up_on_failure(svc: SglangService, tmp_path: Path) -> None:
    stream = MagicMock()
    model_id = "google/test-model"
    model = SglangModel(hf_id=model_id, size="1GB")
    svc.models["default"][model_id] = model

    with (
        patch.object(svc, "_download_model", new_callable=AsyncMock, side_effect=HTTPException(400, "boom")),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(HTTPException),
    ):
        await svc._download_model_or_set_progress(stream, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert model_id not in svc.models_download_progress


@pytest.mark.asyncio
async def test_download_model_or_set_progress_forwards_existing_stream(svc: SglangService, tmp_path: Path) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    chunk = StreamChunkProgress(type="progress", stage="download", value=0.5, data={"local_model_path": str(tmp_path / "model")})
    existing_stream.emit(chunk)
    existing_stream.close()
    model_id = "google/test-model"
    model = SglangModel(hf_id=model_id, size="1GB")
    svc.models_download_progress[model_id] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    await svc._download_model_or_set_progress(output_stream, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert output_stream.emit.call_count == 1
    assert output_stream.emit.call_args == call(chunk)


@pytest.mark.asyncio
async def test_download_model_or_set_progress_breaks_on_non_download_chunk(svc: SglangService, tmp_path: Path) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    finish_chunk: StreamChunk = {"type": "finish", "status": "ok"}  # type: ignore[assignment]
    progress_chunk = StreamChunkProgress(type="progress", stage="download", value=1.0, data={"local_model_path": str(tmp_path / "model")})
    existing_stream.emit(progress_chunk)
    existing_stream.emit(finish_chunk)
    existing_stream.close()
    model_id = "google/test-model"
    model = SglangModel(hf_id=model_id, size="1GB")
    svc.models_download_progress[model_id] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    result_path = await svc._download_model_or_set_progress(output_stream, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert result_path is not None


@pytest.mark.asyncio
async def test_download_model_or_set_progress_raises_500_after_break_with_no_path(svc: SglangService, tmp_path: Path) -> None:
    existing_stream: Stream[StreamChunk] = Stream()  # type: ignore[type-arg]
    finish_chunk: StreamChunk = {"type": "finish", "status": "ok"}  # type: ignore[assignment]
    existing_stream.emit(finish_chunk)
    existing_stream.close()
    model_id = "google/test-model"
    model = SglangModel(hf_id=model_id, size="1GB")
    svc.models_download_progress[model_id] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await svc._download_model_or_set_progress(output_stream, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert exc_info.value.status_code == 500


def test_release_gpu_utilization_with_none_does_not_change_value(svc: SglangService) -> None:
    svc.gpu_memory_utilization = 0.5
    svc._release_gpu_utilization(None)  # pyright: ignore[reportPrivateUsage]
    assert svc.gpu_memory_utilization == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_install_model_releases_gpu_when_option_parsing_fails(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = _make_installed_info(hardware=True)
    model_id = "parse-fail-model"
    svc.models["default"][model_id] = SglangModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.5)

    with (
        patch.object(svc, "_get_quantization", new_callable=AsyncMock, side_effect=RuntimeError("bad quantization")),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(RuntimeError),
    ):
        await svc._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]

    assert svc.gpu_memory_utilization == 0.0
    assert ("default", model_id) not in svc._installing  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_install_model_registration_failure_rolls_back_model(svc: SglangService, gpu_deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _setup_install_mocks(svc, gpu_deps)
    gpu_deps["docker_service"].stop_docker = AsyncMock()
    model_id = next(iter(svc.models["default"]))
    gpu_deps["endpoint_registry"].register_chat_completion_as_proxy.side_effect = RuntimeError("registry down")

    with (
        patch.object(svc, "_download_model_or_set_progress", new_callable=AsyncMock, return_value=tmp_path / "model"),  # pyright: ignore[reportPrivateUsage]
        patch("server.services.sglang_service.get_model_dir_context_window", new_callable=AsyncMock, return_value=4096),
        patch("server.services.sglang_service.get_base_url", return_value="http://localhost:30000"),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert model_id not in installed.models
    gpu_deps["docker_service"].stop_docker.assert_called_once()


@pytest.mark.asyncio
async def test_install_model_pre_func_exception_discards_installing_key(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    _setup_install_mocks(svc, gpu_deps)
    model_id = next(iter(svc.models["default"]))

    with (
        patch.object(svc, "_get_quantization", new_callable=AsyncMock, side_effect=RuntimeError("quantization failed")),  # pyright: ignore[reportPrivateUsage]
        pytest.raises(RuntimeError),
    ):
        await svc._install_model("default", model_id, InstallModelIn(spec={}))  # pyright: ignore[reportPrivateUsage]

    assert ("default", model_id) not in svc._installing  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_uninstall_model_floors_gpu_memory_at_zero(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    svc.gpu_memory_utilization = 0.1
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model", gpu_memory_utilization=0.5)
    svc.instances_info["default"].installed = installed
    gpu_deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.gpu_memory_utilization == 0.0


@pytest.mark.asyncio
async def test_uninstall_model_purges_files_and_downloaded_entry(svc: SglangService, gpu_deps: dict[str, Any], tmp_path: Path) -> None:
    installed = _make_installed_info()
    model_dir = tmp_path / "model-dir"
    model_dir.mkdir()
    model_info = _make_model_installed_info("test-model")
    model_info.model_path = model_dir
    installed.models["test-model"] = model_info
    svc.instances_info["default"].installed = installed
    svc.models_downloaded["test-model"] = DownloadedInfo(model_path=str(model_dir))
    gpu_deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert not model_dir.exists()
    assert "test-model" not in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_purge_with_none_model_path_removes_downloaded_entry(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model")
    svc.instances_info["default"].installed = installed
    svc.models_downloaded["test-model"] = DownloadedInfo(model_path=None)
    gpu_deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "test-model", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "test-model" not in svc.models_downloaded


@pytest.mark.asyncio
async def test_uninstall_model_ignores_unknown_model_id(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    gpu_deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_model("default", "unknown-model", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert gpu_deps["endpoint_registry"].unregister_chat_completion.call_count == 0
    assert gpu_deps["docker_service"].uninstall_docker.call_count == 0


@pytest.mark.asyncio
async def test_stop_instance_does_nothing_when_not_installed(svc: SglangService) -> None:
    svc.instances_info["default"].installed = None

    with patch.object(svc, "_stop_dockers_parallel", new_callable=AsyncMock) as mock_stop:  # pyright: ignore[reportPrivateUsage]
        await svc.stop_instance("default")

    assert mock_stop.call_count == 0


@pytest.mark.asyncio
async def test_reconcile_instance_models_releases_reranker_model(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info(model_id="m1", registration_id="reg-1", gpu_memory_utilization=0.5, model_type="reranker")
    installed.models["m1"] = model_info
    svc.gpu_memory_utilization = 0.5
    gpu_deps["docker_service"].get_container_status = AsyncMock(return_value=_status(exists=False, state="", health=""))
    gpu_deps["docker_service"].uninstall_docker = AsyncMock()

    with patch.object(svc, "_save", new_callable=AsyncMock):
        for _ in range(3):
            await svc._reconcile_instance_models("default", installed)  # pyright: ignore[reportPrivateUsage]

    gpu_deps["endpoint_registry"].unregister_rerank.assert_called_once_with("m1", "reg-1")
    gpu_deps["endpoint_registry"].unregister_chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_validate_update_options_adds_hardware_key_when_missing(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    gpu_deps["docker_service"].has_gpu_support = True
    options = InstallServiceIn(spec={})

    with patch.object(svc, "validate_docker_image_version", new_callable=AsyncMock):
        await svc._validate_update_options(options)  # pyright: ignore[reportPrivateUsage]

    assert "hardware" in options.spec


@pytest.mark.asyncio
async def test_uninstall_instance_purge_removes_per_model_docker_images(svc: SglangService, gpu_deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    model_info = _make_model_installed_info("test-model")
    model_info.docker.image = "sglang-model-image:tag"
    installed.models["test-model"] = model_info
    svc.instances_info["default"].installed = installed
    gpu_deps["docker_service"].remove_image = AsyncMock()

    with (
        patch.object(svc, "_uninstall_model", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
        patch.object(svc, "_clear_working_dir", new_callable=AsyncMock),  # pyright: ignore[reportPrivateUsage]
    ):
        await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    removed_images = {c.args[0] for c in gpu_deps["docker_service"].remove_image.call_args_list}
    assert "sglang-model-image:tag" in removed_images


@pytest.mark.asyncio
async def test_get_cached_vram_estimate_computes_from_installed_model(svc: SglangService) -> None:
    svc.hardware.total_vram_gb = 24.0  # pyright: ignore[reportAttributeAccessIssue]
    installed = _make_installed_info()
    installed.models["test-model"] = _make_model_installed_info("test-model", gpu_memory_utilization=0.5)
    svc.instances_info["default"].installed = installed

    result = await svc._get_cached_vram_estimate("default", "test-model", is_loaded=True)  # pyright: ignore[reportPrivateUsage]

    assert result == 12.0
    assert svc._vram_cache[("default", "test-model")] == 12.0  # pyright: ignore[reportPrivateUsage]


def test_get_spec_no_gpu_offers_no_default_hardware_option(deps: dict[str, Any]) -> None:
    svc = SglangService(**deps)

    spec = svc.get_spec()

    hardware_field = next(f for f in spec.fields if f.name == "hardware")
    assert hardware_field.values == []
    assert hardware_field.default is None


@pytest.mark.asyncio
async def test_download_model_or_set_progress_emits_chunk_with_empty_data(svc: SglangService, tmp_path: Path) -> None:
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
    model = SglangModel(hf_id=model_id, size="1GB")
    svc.models_download_progress[model_id] = existing_stream  # type: ignore[assignment]
    output_stream = MagicMock()

    result_path = await svc._download_model_or_set_progress(output_stream, model_id, model, tmp_path)  # pyright: ignore[reportPrivateUsage]

    assert result_path == tmp_path / "model"
    assert output_stream.emit.call_count == 2


@pytest.mark.asyncio
async def test_install_model_releases_gpu_without_stopping_container_when_download_fails(
    svc: SglangService, gpu_deps: dict[str, Any]
) -> None:
    svc.instances_info["default"].installed = _make_installed_info(hardware=True)
    gpu_deps["docker_service"].stop_docker = AsyncMock()
    model_id = "download-fail-model"
    svc.models["default"][model_id] = SglangModel(hf_id=model_id, size="1GB", gpu_memory_utilization=0.5)

    with (
        patch.object(
            svc,
            "_download_model_or_set_progress",  # pyright: ignore[reportPrivateUsage]
            new_callable=AsyncMock,
            side_effect=RuntimeError("download failed"),
        ),
        patch.object(svc, "get_specified_hardware_parts", return_value=[]),
    ):
        promise = await svc._install_model("default", model_id, InstallModelIn(spec={"gpu_memory_utilization": 0.5}))  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError):
            await promise.wait()

    assert svc.gpu_memory_utilization == 0.0
    gpu_deps["docker_service"].stop_docker.assert_not_called()
