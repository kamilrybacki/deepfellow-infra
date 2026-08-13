# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

from unittest.mock import MagicMock

import pytest

from server.models.models import InstallModelIn
from server.services.base2_service import Instance, InstanceConfig, ModelConfig
from server.services.llamacpp_service import InstalledInfo, LLamacppService
from server.utils.hardware import CpuInfo, NvidiaGpuInfo

_CPU = CpuInfo(model="Test CPU", avx512=True)
_GPU_0 = NvidiaGpuInfo(name="NVIDIA RTX 4090", vram="24 GB", id=0)
_GPU_1 = NvidiaGpuInfo(name="NVIDIA RTX 4090", vram="24 GB", id=1)


def _make_service(gpus: list[NvidiaGpuInfo]) -> LLamacppService:
    svc = object.__new__(LLamacppService)
    hardware = MagicMock()
    hardware.gpus = gpus
    hardware.cpu = _CPU
    svc.hardware = hardware
    return svc


@pytest.mark.parametrize(
    ("gpus", "spec", "expected"),
    [
        ([_GPU_0], False, "CPU"),
        ([], False, "CPU"),
        ([], True, "CPU"),
        ([], None, "CPU"),
        ([_GPU_0], True, f"GPU | {_GPU_0.long_name}"),
        ([_GPU_0], None, f"GPU | {_GPU_0.long_name}"),
        ([_GPU_0, _GPU_1], True, "GPUs"),
        ([_GPU_0, _GPU_1], None, "GPUs"),
    ],
)
def test_canonicalize_hardware_spec_from_bool(gpus: list[NvidiaGpuInfo], spec: bool | None, expected: str):
    svc = _make_service(gpus)
    assert svc.canonicalize_hardware_spec(spec) == expected


@pytest.mark.parametrize("spec", ["CPU", "GPU", "GPUs", f"GPU | {_GPU_0.long_name}"])
def test_canonicalize_hardware_spec_passes_through_strings(spec: str):
    svc = _make_service([_GPU_0, _GPU_1])
    assert svc.canonicalize_hardware_spec(spec) == spec


def _make_base2_service(instances_info: dict[str, Instance[InstalledInfo]]) -> LLamacppService:
    svc = object.__new__(LLamacppService)
    svc.instances_info = instances_info
    return svc


def test_get_persisted_model_definition_returns_none_for_unknown_instance():
    svc = _make_base2_service({})
    assert svc._get_persisted_model_definition("default", "some-model") is None  # pyright: ignore[reportPrivateUsage]


def test_get_persisted_model_definition_returns_none_when_model_not_persisted():
    config = InstanceConfig(models=[ModelConfig(model_id="other-model", options=InstallModelIn(), definition={"a": 1})])
    svc = _make_base2_service({"default": Instance(None, None, {}, config)})
    assert svc._get_persisted_model_definition("default", "some-model") is None  # pyright: ignore[reportPrivateUsage]


def test_get_persisted_model_definition_returns_definition_when_persisted():
    definition = {"a": 1}
    config = InstanceConfig(models=[ModelConfig(model_id="some-model", options=InstallModelIn(), definition=definition)])
    svc = _make_base2_service({"default": Instance(None, None, {}, config)})
    assert svc._get_persisted_model_definition("default", "some-model") == definition  # pyright: ignore[reportPrivateUsage]
