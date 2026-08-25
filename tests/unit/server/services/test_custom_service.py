# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from fastapi import HTTPException

from server.docker import DockerOptions
from server.models.models import AddCustomModelIn, InstallModelIn, ListModelsFilters, ModelInfo, UninstallModelIn
from server.models.services import InstallServiceIn, UninstallServiceIn
from server.services.base2_service import CustomModel, Instance, InstanceConfig
from server.services.custom_service import (
    CustomService,
    DownloadedInfo,
    InstalledInfo,
    ModelInstalledInfo,
    _const,  # pyright: ignore[reportPrivateUsage]
    create_bge_m3_model,
    create_doc_chunker_model,
    create_finetune_model,
    create_gliner_model,
    create_lemmatizer_model,
)
from server.utils.hardware import NvidiaGpuInfo

_CUSTOM_MODEL_DATA: dict[str, Any] = {
    "id": "my-custom",
    "default_prefix": "my-custom",
    "size": "1GB",
    "image": "test/image:latest",
    "image_port": 8000,
}


@pytest.fixture
def deps() -> dict[str, Any]:
    docker_svc = MagicMock()
    docker_svc.get_docker_subnet.return_value = "172.20.0.0/16"
    docker_svc.get_docker_container_name.side_effect = lambda name: f"df-{name}"  # pyright: ignore[reportUnknownLambdaType]
    config = MagicMock()
    config.custom_endpoint_read_timeout_seconds = 900
    return {
        "config": config,
        "endpoint_registry": MagicMock(),
        "service_provider": MagicMock(),
        "model_downloader": MagicMock(),
        "docker_service": docker_svc,
        "hardware": MagicMock(gpus=[]),
    }


@pytest.fixture
def svc(deps: dict[str, Any]) -> CustomService:
    return CustomService(**deps)


def test_get_type(svc: CustomService) -> None:
    assert svc.get_type() == "custom"


def test_get_description_not_empty(svc: CustomService) -> None:
    assert svc.get_description()


def test_get_size_empty(svc: CustomService) -> None:
    assert svc.get_size() == ""


def test_get_spec_returns_empty_fields(svc: CustomService) -> None:
    spec = svc.get_spec()
    assert spec.fields == []


def test_default_models_loaded_on_init(svc: CustomService) -> None:
    assert "default" in svc.models
    assert len(svc.models["default"]) > 0


def test_const_models_not_empty() -> None:
    assert len(_const.models) > 0


def test_service_has_docker(svc: CustomService) -> None:
    assert svc.service_has_docker() is False


def test_get_custom_model_spec_not_none(svc: CustomService) -> None:
    assert svc.get_custom_model_spec() is not None


def test_get_default_model_spec_contains_prefix_field(svc: CustomService) -> None:
    spec = svc.get_default_model_spec("my-prefix")
    field_names = [f.name for f in spec.fields]
    assert "prefix" in field_names


def test_model_installed_info_get_info() -> None:
    options = InstallModelIn(spec={"prefix": "test"})
    model_info = ModelInstalledInfo(
        id="test-model",
        options=options,
        docker_options=MagicMock(),
        container_host="127.0.0.1",
        container_port=8080,
        docker_exposed_port=8080,
        registration_id="reg-123",
        prefix="test",
        base_url="http://127.0.0.1:8080",
    )

    info = model_info.get_info()

    assert info.spec == options.spec
    assert info.registration_id == "reg-123"


@pytest.mark.asyncio
async def test_stop_instance_no_installed_is_noop(svc: CustomService, deps: dict[str, Any]) -> None:
    deps["docker_service"].stop_docker = AsyncMock()

    await svc.stop_instance("default")

    assert deps["docker_service"].stop_docker.call_count == 0


@pytest.mark.asyncio
async def test_stop_instance_with_installed_stops_containers(svc: CustomService, deps: dict[str, Any]) -> None:
    docker_opts = MagicMock()
    mock_model = MagicMock()
    mock_model.docker_options = docker_opts
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": mock_model},
        options=InstallServiceIn(spec={}),
    )
    deps["docker_service"].stop_docker = AsyncMock()

    await svc.stop_instance("default")

    assert deps["docker_service"].stop_docker.call_count == 1
    assert deps["docker_service"].stop_docker.call_args == call(docker_opts)


def test_get_installed_info_when_not_installed_returns_false(svc: CustomService) -> None:
    result = svc.get_installed_info("default")
    assert result is False


def test_get_installed_info_when_installed_returns_spec(svc: CustomService) -> None:
    options = InstallServiceIn(spec={"key": "val"})
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=options)

    result = svc.get_installed_info("default")

    assert result == {"key": "val"}


def test_generate_instance_config_without_info(svc: CustomService) -> None:
    config = svc._generate_instance_config("default", None, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options is None
    assert config.models == []


def test_generate_instance_config_with_info(svc: CustomService) -> None:
    mock_model_info = MagicMock()
    mock_model_info.id = "lemmatizer"
    mock_model_info.options = InstallModelIn(spec={})
    installed = InstalledInfo(models={"lemmatizer": mock_model_info}, options=InstallServiceIn(spec={}))

    config = svc._generate_instance_config("default", installed, None)  # pyright: ignore[reportPrivateUsage]

    assert config.options == installed.options
    assert config.models is not None
    assert len(config.models) == 1
    assert config.models[0].model_id == "lemmatizer"


def test_load_download_info(svc: CustomService) -> None:
    info = svc._load_download_info({"image": "test/image:latest"})  # pyright: ignore[reportPrivateUsage]

    assert info.image == "test/image:latest"


@pytest.mark.asyncio
async def test_install_instance_returns_installed_info(svc: CustomService) -> None:
    options = InstallServiceIn(spec={})

    promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]

    result = await promise.wait()
    assert isinstance(result, InstalledInfo)
    assert svc.service_downloaded is True


@pytest.mark.asyncio
async def test_uninstall_instance_without_purge_clears_installed(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_with_purge_resets_service(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc._clear_working_dir = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert svc.service_downloaded is False
    assert svc._clear_working_dir.call_count == 1  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]


@pytest.mark.asyncio
async def test_uninstall_instance_with_models_uninstalls_each(svc: CustomService, deps: dict[str, Any]) -> None:
    mock_model = MagicMock()
    mock_model.id = "lemmatizer"
    mock_model.prefix = "lemmatizer"
    mock_model.registration_id = "reg-1"
    mock_model.docker_options = MagicMock()
    deps["docker_service"].uninstall_docker = AsyncMock()
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": mock_model},
        options=InstallServiceIn(spec={}),
    )

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None
    assert deps["docker_service"].uninstall_docker.call_count == 1


def test_get_docker_compose_file_path_no_model_id_raises(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc:
        svc.get_docker_compose_file_path("default", None)

    assert exc.value.status_code == 400


def test_get_docker_compose_file_path_model_not_found_raises(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc:
        svc.get_docker_compose_file_path("default", "nonexistent")

    assert exc.value.status_code == 400


def test_get_docker_compose_file_path_success(svc: CustomService, deps: dict[str, Any]) -> None:
    expected = MagicMock()
    deps["docker_service"].get_docker_compose_file_path.return_value = expected
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": MagicMock()},
        options=InstallServiceIn(spec={}),
    )

    result = svc.get_docker_compose_file_path("default", "lemmatizer")

    assert result == expected


def test_add_custom_model_adds_to_models(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA)

    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]

    assert "my-custom" in svc.models["default"]


def test_get_custom_model_definition_returns_stored_spec(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA)
    svc.instances_info["default"].config.custom = [model]

    assert svc.get_custom_model_definition("cm-1") == _CUSTOM_MODEL_DATA


def test_get_custom_model_definition_unknown_id_returns_none(svc: CustomService) -> None:
    svc.instances_info["default"].config.custom = [CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA)]

    assert svc.get_custom_model_definition("nonexistent-id") is None


def test_add_custom_model_duplicate_raises(svc: CustomService) -> None:
    svc._add_custom_model("default", CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA))  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(HTTPException) as exc:
        svc._add_custom_model("default", CustomModel(id="cm-2", data=_CUSTOM_MODEL_DATA))  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400


def test_add_custom_model_prefix_collision_raises_400(svc: CustomService) -> None:
    """Duplicating a custom model without changing `default_prefix` must be rejected on add - a
    different `id` alone previously wasn't enough to avoid two models sharing the same endpoint."""
    svc._add_custom_model("default", CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA))  # pyright: ignore[reportPrivateUsage]
    colliding_data = {**_CUSTOM_MODEL_DATA, "id": "other-custom"}

    with pytest.raises(HTTPException) as exc:
        svc._add_custom_model("default", CustomModel(id="cm-2", data=colliding_data))  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400


def test_remove_custom_model_when_in_use_raises(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA)
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].installed = InstalledInfo(
        models={"my-custom": MagicMock()},
        options=InstallServiceIn(spec={}),
    )

    with pytest.raises(HTTPException) as exc:
        svc._remove_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_update_custom_model_not_found_raises_404(svc: CustomService) -> None:
    with pytest.raises(HTTPException) as exc:
        await svc.update_custom_model("default", "nonexistent-id", AddCustomModelIn(spec={"name": "x"}))

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_update_custom_model_persists_and_rebuilds_model(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=dict(_CUSTOM_MODEL_DATA))
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [model]
    svc.service_provider.save_service_config = AsyncMock()

    new_data = {**_CUSTOM_MODEL_DATA, "image": "test/updated:latest"}
    await svc.update_custom_model("default", "cm-1", AddCustomModelIn(spec=new_data))

    assert svc.instances_info["default"].config.custom[0].data["image"] == "test/updated:latest"
    options = svc.models["default"]["my-custom"].options
    assert isinstance(options, DockerOptions)
    assert options.image == "test/updated:latest"


@pytest.mark.asyncio
async def test_update_custom_model_preserves_size_when_form_omits_it(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=dict(_CUSTOM_MODEL_DATA))
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [model]
    svc.service_provider.save_service_config = AsyncMock()

    spec_without_size = {k: v for k, v in _CUSTOM_MODEL_DATA.items() if k != "size"}
    spec_without_size["image"] = "test/updated:latest"
    await svc.update_custom_model("default", "cm-1", AddCustomModelIn(spec=spec_without_size))

    assert svc.instances_info["default"].config.custom[0].data["size"] == "1GB"


@pytest.mark.asyncio
async def test_update_custom_model_rolls_back_when_add_fails(svc: CustomService) -> None:
    model_a = CustomModel(id="cm-1", data={**_CUSTOM_MODEL_DATA, "id": "model-a", "default_prefix": "model-a"})
    model_b = CustomModel(id="cm-2", data={**_CUSTOM_MODEL_DATA, "id": "model-b", "default_prefix": "model-b"})
    svc._add_custom_model("default", model_a)  # pyright: ignore[reportPrivateUsage]
    svc._add_custom_model("default", model_b)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [model_a, model_b]
    svc.service_provider.save_service_config = AsyncMock()

    with pytest.raises(HTTPException) as exc:
        await svc.update_custom_model(
            "default", "cm-1", AddCustomModelIn(spec={**_CUSTOM_MODEL_DATA, "id": "model-b", "default_prefix": "model-b"})
        )

    assert exc.value.status_code == 400
    assert "model-a" in svc.models["default"]
    assert "model-b" in svc.models["default"]
    assert svc.instances_info["default"].config.custom[0].data["id"] == "model-a"


@pytest.mark.asyncio
async def test_update_custom_model_when_in_use_raises_400(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=dict(_CUSTOM_MODEL_DATA))
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [model]
    svc.instances_info["default"].installed = InstalledInfo(
        models={"my-custom": MagicMock()},
        options=InstallServiceIn(spec={}),
    )

    with pytest.raises(HTTPException) as exc:
        await svc.update_custom_model("default", "cm-1", AddCustomModelIn(spec=_CUSTOM_MODEL_DATA))

    assert exc.value.status_code == 400


def test_remove_custom_model_success(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA)
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]

    svc._remove_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]

    assert "my-custom" not in svc.models["default"]


@pytest.mark.asyncio
async def test_remove_custom_model_that_never_registered_still_prunes_config(svc: CustomService) -> None:
    """A model that failed to register at load time (e.g. a prefix collision caught by load_instance)
    is persisted in config.custom but never entered self.models - removing it must still succeed and
    prune the persisted entry, not KeyError on an unconditional del."""
    model = CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA)
    svc.instances_info["default"].config.custom = [model]
    svc.service_provider.save_service_config = AsyncMock()
    assert "my-custom" not in svc.models.get("default", {})

    await svc.remove_custom_model("default", "cm-1")

    assert svc.instances_info["default"].config.custom == []


@pytest.mark.asyncio
async def test_get_duplicate_spec_catalog_model_synthesizes_from_defaults(svc: CustomService) -> None:
    """DFINFRA-271: lemmatizer's `options` is a callable resolving docker options from install fields -
    with nothing installed, get_duplicate_spec must resolve it using the fields' own defaults."""
    spec = await svc.get_duplicate_spec("default", "lemmatizer")

    assert spec["id"] == "lemmatizer"
    assert spec["default_prefix"] == "lemmatizer"
    assert spec["size"] == "1.89GB"
    assert spec["image"] == "hub.simplito.com/deepfellow/deepfellow-lemmatizer:1.0.0-cpu"
    assert spec["image_port"] == 8090
    # generate_docker_options defaults these to int/bool literals, not strings - the duplicate spec
    # must stringify them, since SrvCustomCustomModel.envs is dict[str, str] (add_custom_model would
    # otherwise reject them with a validation error).
    assert spec["envs"] == {
        "DF_LEMMATIZER_NUM_WORKERS": "4",
        "DF_LEMMATIZER_QUEUE_MAX": "100000",
        "DF_LEMMATIZER_SHUTDOWN_TIMEOUT": "30",
    }
    assert all(isinstance(v, str) for v in spec["envs"].values())
    assert spec["healthcheck_cmd"] == "wget -q --spider http://localhost:8090/health"
    assert spec["healthcheck_start_period"] == "30s"
    assert spec["description"] == "Multilingual text lemmatization API."


@pytest.mark.asyncio
async def test_get_duplicate_spec_catalog_model_uses_installed_options(svc: CustomService) -> None:
    """When the source model is installed, duplicate resolves docker options from its actual install
    spec (e.g. a non-default num_workers), not the field defaults."""
    installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    installed.models["lemmatizer"] = ModelInstalledInfo(
        id="lemmatizer",
        options=InstallModelIn(spec={"num_workers": 8}),
        docker_options=MagicMock(),
        container_host="",
        container_port=0,
        docker_exposed_port=0,
        registration_id="reg-1",
        prefix="lemmatizer",
        base_url="",
    )
    svc.instances_info["default"].installed = installed

    spec = await svc.get_duplicate_spec("default", "lemmatizer")

    assert spec["envs"]["DF_LEMMATIZER_NUM_WORKERS"] == "8"


@pytest.mark.asyncio
async def test_get_duplicate_spec_catalog_model_spec_is_addable(svc: CustomService) -> None:
    """End-to-end regression: the synthesized spec must actually pass `_add_custom_model`'s validation
    (SrvCustomCustomModel), not just look plausible - catches type mismatches like int-valued envs."""
    spec = await svc.get_duplicate_spec("default", "lemmatizer")
    spec["id"] = "lemmatizer-copy"
    spec["default_prefix"] = "lemmatizer-copy"

    svc._add_custom_model("default", CustomModel(id="new-uuid", data=spec))  # pyright: ignore[reportPrivateUsage]

    assert "lemmatizer-copy" in svc.models["default"]
    options = svc.models["default"]["lemmatizer-copy"].options
    assert isinstance(options, DockerOptions)
    # A single-colon, two-segment bind mount under the duplicate's own working directory - not the
    # three-segment, two-colon string produced by re-prefixing an already-expanded host:container path.
    assert options.volumes == [f"{svc.get_working_dir()}/lemmatizer_copy_default/volume_0:/root/.cache/stanza"]


@pytest.mark.asyncio
async def test_get_duplicate_spec_strips_host_side_off_an_expanded_volume(svc: CustomService) -> None:
    """`generate_docker_options` callables (e.g. lemmatizer) return an already-expanded host:container
    volume string - get_duplicate_spec must return just the container side, the raw form
    `_add_custom_model` expects, not hand it straight back to be re-prefixed a second time."""
    spec = await svc.get_duplicate_spec("default", "lemmatizer")

    assert spec["volumes"] == ["/root/.cache/stanza"]


@pytest.mark.asyncio
async def test_get_duplicate_spec_custom_backed_model_returns_stored_definition(svc: CustomService) -> None:
    model = CustomModel(id="cm-1", data=dict(_CUSTOM_MODEL_DATA))
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    svc.instances_info["default"].config.custom = [model]

    spec = await svc.get_duplicate_spec("default", "my-custom")

    assert spec == _CUSTOM_MODEL_DATA


@pytest.mark.asyncio
async def test_get_duplicate_spec_custom_backed_missing_definition_raises_404(svc: CustomService) -> None:
    """Registry has `custom=<id>` but no matching CustomModel definition (data inconsistency)."""
    model = CustomModel(id="cm-1", data=dict(_CUSTOM_MODEL_DATA))
    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]
    # Deliberately not setting `svc.instances_info["default"].config.custom` here.

    with pytest.raises(HTTPException) as exc:
        await svc.get_duplicate_spec("default", "my-custom")

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_get_duplicate_spec_unknown_model_raises_400(svc: CustomService) -> None:
    with pytest.raises(HTTPException) as exc:
        await svc.get_duplicate_spec("default", "ghost")

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_list_models_unknown_instance_raises(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc:
        await svc.list_models("ghost", ListModelsFilters())

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_list_models_returns_all_when_no_filter(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result = await svc.list_models("default", ListModelsFilters(installed=None))

    assert len(result.list) > 0


@pytest.mark.asyncio
async def test_list_models_filter_not_installed(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result = await svc.list_models("default", ListModelsFilters(installed=False))

    assert len(result.list) > 0
    assert all(item.installed is False for item in result.list)


@pytest.mark.asyncio
async def test_list_models_filter_installed_returns_only_installed(svc: CustomService) -> None:
    mock_installed = MagicMock()
    mock_installed.get_info.return_value = ModelInfo(spec={"prefix": "lemmatizer"}, registration_id="r1")
    mock_installed.prefix = "lemmatizer"
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": mock_installed},
        options=InstallServiceIn(spec={}),
    )

    result = await svc.list_models("default", ListModelsFilters(installed=True))

    assert len(result.list) == 1
    assert result.list[0].id == "lemmatizer"


@pytest.mark.asyncio
async def test_list_models_none_instance_iterates_all(svc: CustomService) -> None:
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    svc.load_default_models("gpu-1")
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.instances_info["gpu-1"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result = await svc.list_models(None, ListModelsFilters(installed=None))

    assert len(result.list) > 0


@pytest.mark.asyncio
async def test_get_model_not_found_raises(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc:
        await svc.get_model("default", "ghost")

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_get_model_not_installed_returns_false(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result = await svc.get_model("default", "lemmatizer")

    assert result.id == "lemmatizer"
    assert result.installed is False


@pytest.mark.asyncio
async def test_get_model_installed_returns_model_info(svc: CustomService) -> None:
    model_info = ModelInfo(spec={"prefix": "lemmatizer"}, registration_id="r1")
    mock_installed = MagicMock()
    mock_installed.get_info.return_value = model_info
    mock_installed.prefix = "lemmatizer"
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": mock_installed},
        options=InstallServiceIn(spec={}),
    )

    result = await svc.get_model("default", "lemmatizer")

    assert result.installed == model_info


@pytest.mark.asyncio
async def test_install_model_already_installed_returns_ok(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": MagicMock()},
        options=InstallServiceIn(spec={}),
    )

    promise = await svc._install_model("default", "lemmatizer", InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    result = await promise.wait()
    assert result.status == "OK"
    assert "Already installed" in result.details


@pytest.mark.asyncio
async def test_install_model_not_found_raises(svc: CustomService) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException) as exc:
        await svc._install_model("default", "ghost", InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_install_model_success(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8090, True))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8090
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"

    promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
        "default", "lemmatizer", InstallModelIn(spec={"prefix": "lemmatizer"})
    )
    result = await promise.wait()

    assert result.status == "OK"
    assert "lemmatizer" in svc.instances_info["default"].installed.models  # pyright: ignore[reportOptionalMemberAccess]
    registered_options = deps["endpoint_registry"].register_custom_endpoint_as_proxy.call_args.kwargs["options"]
    assert registered_options.read_timeout_seconds == 900


@pytest.mark.asyncio
async def test_install_model_uses_configured_custom_endpoint_read_timeout(svc: CustomService, deps: dict[str, Any]) -> None:
    deps["config"].custom_endpoint_read_timeout_seconds = 1800
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8090, True))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8090
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"

    promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
        "default", "lemmatizer", InstallModelIn(spec={"prefix": "lemmatizer"})
    )
    await promise.wait()

    registered_options = deps["endpoint_registry"].register_custom_endpoint_as_proxy.call_args.kwargs["options"]
    assert registered_options.read_timeout_seconds == 1800


@pytest.mark.asyncio
async def test_uninstall_model_removes_from_installed(svc: CustomService, deps: dict[str, Any]) -> None:
    mock_model = MagicMock()
    mock_model.prefix = "lemmatizer"
    mock_model.registration_id = "r1"
    deps["docker_service"].uninstall_docker = AsyncMock()
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": mock_model},
        options=InstallServiceIn(spec={}),
    )

    await svc._uninstall_model("default", "lemmatizer", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert "lemmatizer" not in svc.instances_info["default"].installed.models  # pyright: ignore[reportOptionalMemberAccess]
    assert deps["endpoint_registry"].unregister_custom_endpoint.call_count == 1


@pytest.mark.asyncio
async def test_uninstall_model_with_purge_removes_downloaded(svc: CustomService, deps: dict[str, Any]) -> None:
    mock_model = MagicMock()
    mock_model.prefix = "lemmatizer"
    mock_model.registration_id = "r1"
    deps["docker_service"].uninstall_docker = AsyncMock()
    deps["docker_service"].remove_image = AsyncMock()
    svc.models_downloaded["lemmatizer"] = DownloadedInfo(image="lemmatizer:latest")
    svc.instances_info["default"].installed = InstalledInfo(
        models={"lemmatizer": mock_model},
        options=InstallServiceIn(spec={}),
    )

    await svc._uninstall_model("default", "lemmatizer", UninstallModelIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "lemmatizer" not in svc.models_downloaded
    assert deps["docker_service"].remove_image.call_count == 1
    assert deps["docker_service"].remove_image.call_args == call("lemmatizer:latest")


def test_create_lemmatizer_model_cpu_image(svc: CustomService) -> None:
    model = create_lemmatizer_model(svc, "172.20.0.0/16")

    assert callable(model.options)
    docker_opts = model.options({"hardware": "CPU"})  # pyright: ignore[reportCallIssue]
    assert "cpu" in docker_opts.image


def test_create_lemmatizer_model_gpu_image(deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24GB", id=0)
    deps["hardware"] = MagicMock(gpus=[gpu], cpu=MagicMock())
    svc_gpu = CustomService(**deps)

    model = create_lemmatizer_model(svc_gpu, "172.20.0.0/16")

    docker_opts = model.options({"hardware": "GPU"})  # pyright: ignore[reportCallIssue]
    assert "cuda" in docker_opts.image


def test_create_gliner_model_cpu_image(svc: CustomService) -> None:
    model = create_gliner_model(svc, "172.20.0.0/16")

    assert callable(model.options)
    docker_opts = model.options({"hardware": "CPU"})  # pyright: ignore[reportCallIssue]
    assert "cpu" in docker_opts.image


def test_create_gliner_model_gpu_image(deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24GB", id=0)
    deps["hardware"] = MagicMock(gpus=[gpu], cpu=MagicMock())
    svc_gpu = CustomService(**deps)

    model = create_gliner_model(svc_gpu, "172.20.0.0/16")

    docker_opts = model.options({"hardware": "GPU"})  # pyright: ignore[reportCallIssue]
    assert "gpu" in docker_opts.image


def test_create_bge_m3_model_cpu_image(svc: CustomService) -> None:
    model = create_bge_m3_model(svc, "172.20.0.0/16")

    assert callable(model.options)
    docker_opts = model.options({"hardware": "CPU"})  # pyright: ignore[reportCallIssue]
    assert "cpu" in docker_opts.image


def test_create_bge_m3_model_declares_context_window(svc: CustomService) -> None:
    """Consumers chunk against this ceiling; leaving it unset made them fall back to a smaller guess."""
    model = create_bge_m3_model(svc, "172.20.0.0/16")

    assert model.model_props.max_context_window == 8192
    assert model.model_props.context_window == 8192


def test_create_bge_m3_model_gpu_image(deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24GB", id=0)
    deps["hardware"] = MagicMock(gpus=[gpu], cpu=MagicMock())
    svc_gpu = CustomService(**deps)

    model = create_bge_m3_model(svc_gpu, "172.20.0.0/16")

    docker_opts = model.options({"hardware": "GPU"})  # pyright: ignore[reportCallIssue]
    assert "cuda" in docker_opts.image


def test_create_doc_chunker_model_cpu_image(svc: CustomService) -> None:
    model = create_doc_chunker_model(svc, "172.20.0.0/16")

    assert callable(model.options)
    docker_opts = model.options({"hardware": "CPU"})  # pyright: ignore[reportCallIssue]
    assert "cpu" in docker_opts.image


def test_create_doc_chunker_model_gpu_image(deps: dict[str, Any]) -> None:
    gpu = NvidiaGpuInfo(name="RTX 4090", vram="24GB", id=0)
    deps["hardware"] = MagicMock(gpus=[gpu], cpu=MagicMock())
    svc_gpu = CustomService(**deps)

    model = create_doc_chunker_model(svc_gpu, "172.20.0.0/16")

    docker_opts = model.options({"hardware": "GPU"})  # pyright: ignore[reportCallIssue]
    assert "gpu" in docker_opts.image


def test_create_finetune_model_returns_docker_options(svc: CustomService) -> None:
    model = create_finetune_model(svc, "172.20.0.0/16")

    assert callable(model.options)
    docker_opts = model.options({})  # pyright: ignore[reportCallIssue]
    assert docker_opts.image_port == 8333


@pytest.mark.asyncio
async def test_install_instance_loads_models_for_new_instance(svc: CustomService) -> None:
    # Remove models for "default" to trigger the load branch
    del svc.models["default"]
    options = InstallServiceIn(spec={})
    promise = await svc._install_instance("default", options)  # pyright: ignore[reportPrivateUsage]
    result = await promise.wait()
    assert isinstance(result, InstalledInfo)
    assert "default" in svc.models


@pytest.mark.asyncio
async def test_uninstall_instance_with_purge_deletes_non_default_instance(svc: CustomService) -> None:
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["gpu-1"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc._clear_working_dir = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await svc._uninstall_instance("gpu-1", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert "gpu-1" not in svc.instances_info


@pytest.mark.asyncio
async def test_uninstall_instance_purge_with_other_instance_installed_does_not_clear_working_dir(
    svc: CustomService,
) -> None:
    # Branch 267->272: another instance is still installed, so the shared working dir
    # cleanup must be skipped even though this instance is being purged.
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["gpu-1"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.service_downloaded = True
    svc._clear_working_dir = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await svc._uninstall_instance("default", UninstallServiceIn(purge=True))  # pyright: ignore[reportPrivateUsage]

    assert svc._clear_working_dir.call_count == 0  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
    assert svc.service_downloaded is True
    assert svc.instances_info["default"].installed is None
    assert "gpu-1" in svc.instances_info


def test_add_custom_model_creates_instance_dict_when_missing(svc: CustomService) -> None:
    del svc.models["default"]
    model = CustomModel(id="cm-1", data=_CUSTOM_MODEL_DATA)

    svc._add_custom_model("default", model)  # pyright: ignore[reportPrivateUsage]

    assert "my-custom" in svc.models["default"]


@pytest.mark.asyncio
async def test_get_model_initialises_empty_models_for_instance(svc: CustomService) -> None:
    del svc.models["default"]
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    with pytest.raises(HTTPException):
        await svc.get_model("default", "lemmatizer")

    assert "default" in svc.models


@pytest.mark.asyncio
async def test_list_models_skips_instance_not_in_requested(svc: CustomService) -> None:
    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    svc.load_default_models("gpu-1")
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    svc.instances_info["gpu-1"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    result_default = await svc.list_models("default", ListModelsFilters(installed=None))
    result_gpu = await svc.list_models("gpu-1", ListModelsFilters(installed=None))

    # Each instance returns only its own models
    assert len(result_default.list) > 0
    assert len(result_gpu.list) > 0
    assert len(result_default.list) == len(result_gpu.list)


@pytest.mark.asyncio
async def test_install_model_uses_default_prefix_when_no_spec(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8090, True))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8090
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"

    # No spec at all → should use default_prefix "lemmatizer"
    promise = await svc._install_model("default", "lemmatizer", InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    result = await promise.wait()
    assert result.status == "OK"


@pytest.mark.asyncio
async def test_install_model_uses_default_prefix_when_prefix_missing(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8090, True))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8090
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.return_value = "reg-1"

    # spec without "prefix" → should set default_prefix
    promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
        "default", "lemmatizer", InstallModelIn(spec={"hardware": "CPU"})
    )

    result = await promise.wait()
    assert result.status == "OK"


@pytest.mark.asyncio
async def test_install_model_initialises_models_for_instance(svc: CustomService, deps: dict[str, Any]) -> None:
    del svc.models["default"]
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])

    with pytest.raises(HTTPException):
        await svc._install_model("default", "ghost", InstallModelIn())  # pyright: ignore[reportPrivateUsage]

    assert "default" in svc.models


@pytest.mark.asyncio
async def test_uninstall_instance_when_not_installed_is_noop(svc: CustomService) -> None:
    # Branch 239->244: installed is None, skip the if block entirely
    assert svc.instances_info["default"].installed is None

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc.instances_info["default"].installed is None


@pytest.mark.asyncio
async def test_uninstall_instance_skips_model_shared_with_other_instance(svc: CustomService) -> None:
    # Branch 241->240: is_model_installed_in_other_instance returns True,
    # so the inner if condition is False and the for loop continues without calling _uninstall_model
    mock_model = MagicMock()
    mock_model.id = "lemmatizer"
    mock_model.prefix = "lemmatizer"
    mock_model.registration_id = "reg-1"
    mock_model.docker_options = MagicMock()

    shared_installed = InstalledInfo(models={"lemmatizer": mock_model}, options=InstallServiceIn(spec={}))
    svc.instances_info["default"].installed = shared_installed

    svc.instances_info["gpu-1"] = Instance(None, None, {}, InstanceConfig())
    svc.instances_info["gpu-1"].installed = InstalledInfo(
        models={"lemmatizer": mock_model},
        options=InstallServiceIn(spec={}),
    )

    svc._uninstall_model = AsyncMock()  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]

    await svc._uninstall_instance("default", UninstallServiceIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert svc._uninstall_model.call_count == 0  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]


@pytest.mark.asyncio
async def test_uninstall_model_when_not_in_installed_is_noop(svc: CustomService, deps: dict[str, Any]) -> None:
    # Branch 430->436: model_id not in info.models, skip the if block and go to purge check
    deps["docker_service"].uninstall_docker = AsyncMock()
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))

    await svc._uninstall_model("default", "lemmatizer", UninstallModelIn(purge=False))  # pyright: ignore[reportPrivateUsage]

    assert deps["docker_service"].uninstall_docker.call_count == 0
    assert deps["endpoint_registry"].unregister_custom_endpoint.call_count == 0


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_formatted_size(svc: CustomService, deps: dict[str, Any]) -> None:
    deps["docker_service"].get_docker_image_size = AsyncMock(return_value=1024**2)

    result = await svc._resolve_custom_model_size({"image": "myimage:latest"})  # pyright: ignore[reportPrivateUsage]

    assert result == "1.0 MB"


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_none_when_image_size_is_none(svc: CustomService, deps: dict[str, Any]) -> None:
    deps["docker_service"].get_docker_image_size = AsyncMock(return_value=None)

    result = await svc._resolve_custom_model_size({"image": "myimage:latest"})  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_resolve_custom_model_size_returns_none_on_exception(svc: CustomService, deps: dict[str, Any]) -> None:
    deps["docker_service"].get_docker_image_size = AsyncMock(side_effect=Exception("fail"))

    result = await svc._resolve_custom_model_size({"image": "myimage:latest"})  # pyright: ignore[reportPrivateUsage]

    assert result is None


@pytest.mark.asyncio
async def test_install_model_registration_failure_rolls_back_model(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8090, True))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8090
    deps["endpoint_registry"].register_custom_endpoint_as_proxy.side_effect = RuntimeError("registry down")

    promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
        "default", "lemmatizer", InstallModelIn(spec={"prefix": "lemmatizer"})
    )
    with pytest.raises(RuntimeError):
        await promise.wait()

    installed = svc.instances_info["default"].installed
    assert installed is not None
    assert "lemmatizer" not in installed.models


@pytest.mark.asyncio
async def test_install_model_registration_failure_skips_rollback_when_already_removed(svc: CustomService, deps: dict[str, Any]) -> None:
    svc.instances_info["default"].installed = InstalledInfo(models={}, options=InstallServiceIn(spec={}))
    deps["docker_service"].get_image_warnings = AsyncMock(return_value=[])
    deps["docker_service"].is_docker_image_pulled = AsyncMock(return_value=True)
    deps["docker_service"].install_and_run_docker = AsyncMock(return_value=(8090, True))
    deps["docker_service"].get_container_host.return_value = "172.20.0.1"
    deps["docker_service"].get_container_port.return_value = 8090

    def side_effect(*args: object, **kwargs: object) -> None:
        installed = svc.instances_info["default"].installed
        assert installed is not None
        installed.models.pop("lemmatizer", None)
        raise RuntimeError("registry down")

    deps["endpoint_registry"].register_custom_endpoint_as_proxy.side_effect = side_effect

    promise = await svc._install_model(  # pyright: ignore[reportPrivateUsage]
        "default", "lemmatizer", InstallModelIn(spec={"prefix": "lemmatizer"})
    )
    with pytest.raises(RuntimeError):
        await promise.wait()

    installed = svc.instances_info["default"].installed
    assert installed is not None
    assert "lemmatizer" not in installed.models
