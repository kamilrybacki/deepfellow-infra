# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Unit tests for reading the driver's CUDA version in server/utils/hardware.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.utils.hardware import get_nvidia_max_cuda_version, parse_nvidia_smi_cuda_version


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("| NVIDIA-SMI 550.54.14   Driver Version: 550.54.14   CUDA Version: 12.4     |", (12, 4)),
        ("| NVIDIA-SMI 615.71.09   KMD Version: 615.71.09   CUDA UMD Version: 13.4     |", (13, 4)),
        ("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.", None),
        ("", None),
    ],
)
def test_parse_nvidia_smi_cuda_version(output: str, expected: tuple[int, int] | None) -> None:
    assert parse_nvidia_smi_cuda_version(output) == expected


@pytest.fixture(autouse=True)
def reset_cuda_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("server.utils.hardware._cuda_version_cache", None)


@pytest.mark.asyncio
async def test_get_nvidia_max_cuda_version_reads_and_caches_the_driver_version() -> None:
    result = MagicMock(stdout="| NVIDIA-SMI 550.54.14   Driver Version: 550.54.14   CUDA Version: 12.4     |")
    with patch("server.utils.hardware.Utils.run_command_for_success", new_callable=AsyncMock, return_value=result) as run:
        assert await get_nvidia_max_cuda_version() == (12, 4)
        assert await get_nvidia_max_cuda_version() == (12, 4)
    run.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("no gpu"), OSError("no docker")])
async def test_get_nvidia_max_cuda_version_is_none_when_nvidia_smi_fails(error: Exception) -> None:
    with patch("server.utils.hardware.Utils.run_command_for_success", new_callable=AsyncMock, side_effect=error):
        assert await get_nvidia_max_cuda_version() is None
