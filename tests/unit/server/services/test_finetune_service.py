# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import aiohttp
import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from server.models.models import InstallModelIn, ListModelsFilters, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import Instance, InstanceConfig
from server.services.finetune_service import FinetuneOptions, FinetuneService, InstalledInfo


@pytest.fixture
def deps() -> dict[str, Any]:
    docker_svc = MagicMock()
    docker_svc.get_docker_subnet.return_value = "172.20.0.0/16"
    docker_svc.get_docker_container_name.side_effect = lambda name: f"df-{name}"  # pyright: ignore[reportUnknownLambdaType]
    docker_svc.get_user_for_docker = AsyncMock(return_value=None)
    config = MagicMock()
    config.standard_proxy_timeout_seconds = 300
    config.adapter_registry_secret = SecretStr("test-secret")
    return {
        "config": config,
        "endpoint_registry": MagicMock(),
        "service_provider": MagicMock(),
        "model_downloader": MagicMock(),
        "docker_service": docker_svc,
        "hardware": MagicMock(gpus=[]),
        "services_manager": MagicMock(),
    }


@pytest.fixture
def svc(deps: dict[str, Any]) -> FinetuneService:
    return FinetuneService(**deps)


def _make_installed_info(base_url: str = "http://172.20.0.1:8333") -> InstalledInfo:
    docker = MagicMock()
    docker.container_name = "df-finetune"
    return InstalledInfo(
        docker=docker,
        options=InstallServiceIn(spec={}),
        parsed_options=FinetuneOptions(),
        container_host="172.20.0.1",
        container_port=8333,
        docker_exposed_port=8333,
        registration_id="reg-1",
        base_url=base_url,
    )


def _mock_adapter_registry_session_cls(status: int, json_data: dict[str, Any] | None = None) -> MagicMock:
    """Build a mock `aiohttp.ClientSession` class whose `session.get(url)` yields a mock response."""
    mock_response = AsyncMock()
    mock_response.status = status
    mock_response.raise_for_status = MagicMock()
    mock_response.json = AsyncMock(return_value=json_data or {})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(return_value=mock_response)

    return MagicMock(return_value=mock_session)


@pytest.mark.asyncio
async def test_load_service_strips_any_persisted_model_entry(svc: FinetuneService) -> None:
    """A `services.json` written before the single-step install refactor may still list a
    per-model entry for deepfellow-finetune; it must not resurface as a "failed to load" warning."""
    raw_config = {
        "instances": {
            "default": {
                "options": {"spec": {"hardware": "GPU"}},
                "models": [{"model_id": "deepfellow-finetune", "options": {"spec": {}}}],
            }
        }
    }

    with patch("server.services.base2_service.Base2Service.load_service", new=AsyncMock()) as mock_super_load:
        await svc.load_service(raw_config)

    loaded_config = mock_super_load.call_args.args[0]
    assert loaded_config["instances"]["default"]["models"] == []


@pytest.mark.asyncio
async def test_load_service_without_instances_key_still_calls_super(svc: FinetuneService) -> None:
    raw_config: dict[str, Any] = {}

    with patch("server.services.base2_service.Base2Service.load_service", new=AsyncMock()) as mock_super_load:
        await svc.load_service(raw_config)

    assert mock_super_load.call_count == 1
    assert mock_super_load.call_args == call(raw_config)


@pytest.mark.asyncio
async def test_load_service_skips_non_dict_instance_entries(svc: FinetuneService) -> None:
    raw_config = {"instances": {"default": "not-a-dict"}}

    with patch("server.services.base2_service.Base2Service.load_service", new=AsyncMock()) as mock_super_load:
        await svc.load_service(raw_config)

    loaded_config = mock_super_load.call_args.args[0]
    assert loaded_config["instances"]["default"] == "not-a-dict"


def test_get_type(svc: FinetuneService) -> None:
    assert svc.get_type() == "finetune"


def test_get_description_not_empty(svc: FinetuneService) -> None:
    assert svc.get_description()


def test_get_size(svc: FinetuneService) -> None:
    assert svc.get_size() == "8.56GB"


def test_get_spec_contains_hardware_and_prefix_fields(svc: FinetuneService) -> None:
    field_names = [f.name for f in svc.get_spec().fields]
    assert "hardware" in field_names
    assert "prefix" in field_names
    assert "proxy_timeout_seconds" in field_names


def test_get_custom_model_spec_is_none(svc: FinetuneService) -> None:
    assert svc.get_custom_model_spec() is None


def test_get_installed_base_url_when_not_installed_returns_none(svc: FinetuneService) -> None:
    assert svc.get_installed_base_url() is None


def test_get_installed_base_url_when_installed_returns_url(svc: FinetuneService) -> None:
    svc.instances_info["default"].installed = _make_installed_info("http://172.20.0.1:8333")

    assert svc.get_installed_base_url() == "http://172.20.0.1:8333"


def test_get_installed_base_url_prefers_default_instance(svc: FinetuneService) -> None:
    svc.instances_info["second"] = Instance(_make_installed_info("http://172.20.0.2:8333"), None, {}, InstanceConfig())
    svc.instances_info["default"].installed = _make_installed_info("http://172.20.0.1:8333")

    assert svc.get_installed_base_url() == "http://172.20.0.1:8333"


def test_get_installed_base_url_falls_back_to_non_default_instance(svc: FinetuneService) -> None:
    """The adapters directory is shared, so a vLLM adapter install must work with only a non-default instance."""
    svc.instances_info["second"] = Instance(_make_installed_info("http://172.20.0.2:8333"), None, {}, InstanceConfig())

    assert svc.get_installed_base_url() == "http://172.20.0.2:8333"


def test_get_installed_base_url_for_explicit_instance_does_not_fall_back(svc: FinetuneService) -> None:
    svc.instances_info["second"] = Instance(_make_installed_info("http://172.20.0.2:8333"), None, {}, InstanceConfig())

    assert svc.get_installed_base_url("default") is None


@pytest.mark.asyncio
async def test_list_adapters_raises_409_when_not_installed(svc: FinetuneService) -> None:
    with pytest.raises(HTTPException) as exc:
        await svc.list_adapters()

    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_list_adapters_returns_parsed_response(svc: FinetuneService) -> None:
    svc.instances_info["default"].installed = _make_installed_info("http://172.20.0.1:8333")
    payload = {
        "object": "list",
        "data": [
            {
                "id": "pirate-0c571d9a",
                "object": "model",
                "created": 1777548607,
                "owned_by": "deepfellow",
                "base_model": "meta-llama/llama-3.1-8b-instruct",
                "formats": ["safetensors", "gguf"],
                "lora": {"rank": 16, "alpha": 32},
                "gguf": {"format": "f16", "file": "adapter.gguf"},
            }
        ],
    }
    mock_session_cls = _mock_adapter_registry_session_cls(200, payload)

    with patch("server.services.finetune_service.aiohttp.ClientSession", mock_session_cls):
        result = await svc.list_adapters()

    assert len(result.data) == 1
    assert result.data[0].id == "pirate-0c571d9a"
    assert result.data[0].base_model == "meta-llama/llama-3.1-8b-instruct"
    mock_session_cls.return_value.get.assert_called_once_with("http://172.20.0.1:8333/v1/adapters")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [aiohttp.ClientConnectionError("connection refused"), TimeoutError()],
    ids=["unreachable", "timeout"],
)
async def test_list_adapters_raises_503_when_container_unavailable(svc: FinetuneService, error: Exception) -> None:
    svc.instances_info["default"].installed = _make_installed_info("http://172.20.0.1:8333")
    mock_session_cls = MagicMock()
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.get = MagicMock(side_effect=error)
    mock_session_cls.return_value = mock_session

    with (
        patch("server.services.finetune_service.aiohttp.ClientSession", mock_session_cls),
        pytest.raises(HTTPException) as exc,
    ):
        await svc.list_adapters()

    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_list_adapters_raises_502_when_container_returns_error_status(svc: FinetuneService) -> None:
    svc.instances_info["default"].installed = _make_installed_info("http://172.20.0.1:8333")
    mock_session_cls = _mock_adapter_registry_session_cls(401)
    mock_response = mock_session_cls.return_value.get.return_value
    mock_response.raise_for_status = MagicMock(
        side_effect=aiohttp.ClientResponseError(request_info=MagicMock(), history=(), status=401, message="Unauthorized")
    )

    with (
        patch("server.services.finetune_service.aiohttp.ClientSession", mock_session_cls),
        pytest.raises(HTTPException) as exc,
    ):
        await svc.list_adapters()

    assert exc.value.status_code == 502
    assert "HTTP 401" in exc.value.detail


def test_get_docker_options_returns_installed_container(svc: FinetuneService) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed

    assert svc.get_docker_options("default", None) is installed.docker


def test_get_docker_options_rejects_model_id(svc: FinetuneService) -> None:
    svc.instances_info["default"].installed = _make_installed_info()

    with pytest.raises(HTTPException) as exc:
        svc.get_docker_options("default", "some-model")

    assert exc.value.status_code == 400


def test_service_has_docker_returns_true(svc: FinetuneService) -> None:
    assert svc.service_has_docker() is True


@pytest.mark.asyncio
async def test_stop_instance_no_installed_is_noop(svc: FinetuneService, deps: dict[str, Any]) -> None:
    deps["docker_service"].stop_docker = AsyncMock()

    await svc.stop_instance("default")

    assert deps["docker_service"].stop_docker.call_count == 0


@pytest.mark.asyncio
async def test_stop_instance_with_installed_stops_container(svc: FinetuneService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["docker_service"].stop_docker = AsyncMock()

    await svc.stop_instance("default")

    assert deps["docker_service"].stop_docker.call_count == 1
    assert deps["docker_service"].stop_docker.call_args == call(installed.docker)


def test_get_installed_info_when_not_installed_returns_false(svc: FinetuneService) -> None:
    assert svc.get_installed_info("default") is False


def test_generate_instance_config_without_info(svc: FinetuneService) -> None:
    config = svc._generate_instance_config("default", None, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options is None
    assert config.models == []


def test_generate_instance_config_with_info_has_no_models(svc: FinetuneService) -> None:
    installed = _make_installed_info()

    config = svc._generate_instance_config("default", installed, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options == installed.options
    assert config.models == []
    assert config.container_name == "df-finetune"


def test_load_download_info(svc: FinetuneService) -> None:
    info = svc._load_download_info({"image": "test/image:latest"})  # pyright: ignore[reportPrivateUsage]

    assert info.image == "test/image:latest"


def _mock_install_deps(deps: dict[str, Any]) -> None:
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(20000, True, False))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8333
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"


@pytest.mark.asyncio
@patch.object(FinetuneService, "_get_working_dir")
async def test_install_instance_creates_container_in_one_step(
    mock_get_working_dir: MagicMock, svc: FinetuneService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """Installing the service must create and start the container directly - no separate per-model install step."""
    mock_get_working_dir.return_value = tmp_path
    _mock_install_deps(deps)

    promise = await svc._install_instance("default", InstallServiceIn(spec={"prefix": "deepfellow-finetune"}))  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()

    assert isinstance(result, InstalledInfo)
    assert svc.service_downloaded is True
    assert deps["docker_service"].install_and_run_docker.call_count == 1
    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.image_port == 8333
    assert docker_options.name == "finetune"
    assert docker_options.container_name == "df-finetune"
    assert docker_options.image == "hub.simplito.com/deepfellow/deepfellow-finetune:0.0.18"
    assert docker_options.volumes == [
        f"{tmp_path}/models:/workspace/models",
        f"{tmp_path}/adapters:/workspace/adapters",
        f"{tmp_path}/finetune/configs:/workspace/configs",
        f"{tmp_path}/finetune/data:/workspace/data",
        f"{tmp_path}/finetune/db:/data",
    ]
    assert docker_options.env_vars["DFFT_DB_PATH"] == "/data/worker.db"
    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "adapters").is_dir()
    registered_options = deps["endpoint_registry"].register_custom_endpoint_as_proxy.call_args.kwargs["options"]
    assert registered_options.read_timeout_seconds == 300


@pytest.mark.asyncio
@patch.object(FinetuneService, "_get_working_dir")
async def test_install_instance_uses_container_name_override_when_given(
    mock_get_working_dir: MagicMock, svc: FinetuneService, deps: dict[str, Any], tmp_path: Path
) -> None:
    mock_get_working_dir.return_value = tmp_path
    _mock_install_deps(deps)

    promise = await svc._install_instance("default", InstallServiceIn(spec={}), "pinned-name")  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.container_name == "pinned-name"


@pytest.mark.asyncio
@patch.object(FinetuneService, "_get_working_dir")
async def test_install_second_instance_gets_own_container_and_shares_models_and_adapters(
    mock_get_working_dir: MagicMock, svc: FinetuneService, deps: dict[str, Any], tmp_path: Path
) -> None:
    """A non-default instance must not collide with the default one on container name or per-instance state."""
    mock_get_working_dir.return_value = tmp_path
    _mock_install_deps(deps)

    promise = await svc._install_instance("second", InstallServiceIn(spec={"prefix": "finetune-second"}))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    docker_options = deps["docker_service"].install_and_run_docker.call_args.args[0]
    assert docker_options.name == "finetune-second"
    assert docker_options.container_name == "df-finetune-second"
    assert docker_options.volumes == [
        f"{tmp_path}/models:/workspace/models",
        f"{tmp_path}/adapters:/workspace/adapters",
        f"{tmp_path}/finetune-second/configs:/workspace/configs",
        f"{tmp_path}/finetune-second/data:/workspace/data",
        f"{tmp_path}/finetune-second/db:/data",
    ]
    deps["docker_service"].get_container_host.assert_called_once_with("172.20.0.0/16", docker_options)


@pytest.mark.asyncio
async def test_install_instance_raises_400_when_prefix_used_by_another_instance(svc: FinetuneService, deps: dict[str, Any]) -> None:
    """Each instance has its own job database, so two instances must not share one proxy prefix."""
    svc.instances_info["default"].installed = _make_installed_info()
    _mock_install_deps(deps)

    with pytest.raises(HTTPException) as exc:
        await svc._install_instance("second", InstallServiceIn(spec={"prefix": "deepfellow-finetune"}))  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400
    assert deps["docker_service"].install_and_run_docker.call_count == 0


@pytest.mark.asyncio
@patch.object(FinetuneService, "_get_working_dir")
async def test_reinstall_instance_keeps_its_own_prefix(
    mock_get_working_dir: MagicMock, svc: FinetuneService, deps: dict[str, Any], tmp_path: Path
) -> None:
    mock_get_working_dir.return_value = tmp_path
    svc.instances_info["default"].installed = _make_installed_info()
    _mock_install_deps(deps)

    promise = await svc._install_instance("default", InstallServiceIn(spec={"prefix": "deepfellow-finetune"}))  # pyright: ignore[reportPrivateUsage]
    await promise.wait()

    assert deps["docker_service"].install_and_run_docker.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_instance_tears_down_container(svc: FinetuneService, deps: dict[str, Any]) -> None:
    installed = _make_installed_info()
    svc.instances_info["default"].installed = installed
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None
    assert deps["docker_service"].uninstall_docker.call_count == 1
    assert deps["docker_service"].uninstall_docker.call_args == call(installed.docker)
    assert deps["endpoint_registry"].unregister_custom_endpoint.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_instance_with_purge_removes_image(svc: FinetuneService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = _make_installed_info()
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["docker_service"].remove_image = AsyncMock()
    svc._clear_working_dir = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert svc.service_downloaded is False
    deps["docker_service"].remove_image.assert_called_once_with("hub.simplito.com/deepfellow/deepfellow-finetune:0.0.18")


@pytest.mark.asyncio
async def test_uninstall_instance_when_not_installed_skips_teardown(svc: FinetuneService, deps: dict[str, Any]) -> None:
    deps["docker_service"].uninstall_docker = AsyncMock()

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None
    assert deps["docker_service"].uninstall_docker.call_count == 0
    assert deps["endpoint_registry"].unregister_custom_endpoint.call_count == 0


@pytest.mark.asyncio
async def test_uninstall_instance_with_purge_keeps_image_when_other_instance_still_installed(
    svc: FinetuneService, deps: dict[str, Any]
) -> None:
    svc.instances_info["default"].installed = _make_installed_info()
    svc.instances_info["other"] = Instance(_make_installed_info(), None, {}, InstanceConfig())
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["docker_service"].remove_image = AsyncMock()
    svc._clear_working_dir = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert deps["docker_service"].remove_image.call_count == 0
    assert svc._clear_working_dir.call_count == 0  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_with_purge_deletes_non_default_instance(svc: FinetuneService, deps: dict[str, Any]) -> None:
    svc.instances_info["other"] = Instance(_make_installed_info(), None, {}, InstanceConfig())
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["docker_service"].remove_image = AsyncMock()
    svc._clear_working_dir = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await svc._uninstall_instance("other", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "other" not in svc.instances_info


@pytest.mark.asyncio
async def test_list_models_always_empty(svc: FinetuneService) -> None:
    result = await svc.list_models("default", ListModelsFilters())

    assert result.list == []


@pytest.mark.asyncio
async def test_get_model_raises(svc: FinetuneService) -> None:
    with pytest.raises(HTTPException) as exc:
        await svc.get_model("default", "deepfellow-finetune")

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_raises(svc: FinetuneService) -> None:
    with pytest.raises(HTTPException) as exc:
        await svc._install_model("default", "deepfellow-finetune", InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_uninstall_model_raises(svc: FinetuneService) -> None:
    with pytest.raises(HTTPException) as exc:
        await svc._uninstall_model("default", "deepfellow-finetune", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400
