# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for server/dynamic_config.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from server.config import AppSettings, ConfigError
from server.dynamic_config import (
    CURRENT_SCHEMA_VERSION,
    ConfigEnvelope,
    DynamicSettings,
    _build_legacy_defaults,  # pyright: ignore[reportPrivateUsage]
    _migrate,  # pyright: ignore[reportPrivateUsage]
    apply_dynamic_settings,
    get_config_json_path,
    get_infra_version,
    load_or_init,
    persist,
    persist_settings,
    read_envelope,
    snapshot_dynamic_settings,
    write_envelope_atomic,
)


def _make_config(storage_dir: Path | None = None, **overrides: object) -> MagicMock:
    mock = MagicMock(spec=AppSettings)
    if storage_dir is not None:
        mock.get_storage_dir.return_value = storage_dir
    mock.name = ""
    mock.infra_url = ""
    mock.mesh_key = SecretStr("")
    mock.infra_api_key = SecretStr("")
    mock.connect_to_mesh_url = ""
    mock.connect_to_mesh_key = SecretStr("")
    mock.hugging_face_token = SecretStr("")
    mock.civitai_token = SecretStr("")
    mock.adapter_registry_url = ""
    mock.adapter_registry_secret = SecretStr("")
    mock.log_payloads = ""
    mock.stop_containers_on_shutdown = ""
    mock.mcp_sse_session_ttl_seconds = 300
    mock.mcp_sse_max_sessions = 128
    mock.metrics_username = ""
    mock.metrics_password = SecretStr("")
    mock.otel_exporter_otlp_endpoint = "http://localhost:4317"
    mock.otel_tracing_enabled = False
    mock.otel_logging_enabled = False
    for key, value in overrides.items():
        setattr(mock, key, value)
    return mock


# --- get_config_json_path ---


def test_get_config_json_path_is_inside_storage_dir(tmp_path: Path) -> None:
    config = _make_config(storage_dir=tmp_path)

    assert get_config_json_path(config) == tmp_path / "config.json"


# --- _migrate ---


def test_migrate_is_identity() -> None:
    raw = {"schema_version": 1, "foo": "bar"}

    assert _migrate(raw) is raw


# --- read_envelope ---


def test_read_envelope_returns_none_when_file_missing(tmp_path: Path) -> None:
    config = _make_config(storage_dir=tmp_path)

    assert read_envelope(config) is None


def test_read_envelope_parses_existing_file(tmp_path: Path) -> None:
    data = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "infra_version": get_infra_version(),
        "settings": {"name": "node-a", "mesh_key": "secret-value"},
    }
    (tmp_path / "config.json").write_text(json.dumps(data))
    config = _make_config(storage_dir=tmp_path)

    envelope = read_envelope(config)

    assert envelope is not None
    assert envelope.settings.name == "node-a"
    assert envelope.settings.mesh_key.get_secret_value() == "secret-value"


def test_read_envelope_raises_config_error_on_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{not valid json")
    config = _make_config(storage_dir=tmp_path)

    with pytest.raises(ConfigError, match="not valid JSON"):
        read_envelope(config)


def test_read_envelope_raises_config_error_on_validation_failure(tmp_path: Path) -> None:
    data = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "infra_version": get_infra_version(),
        "settings": {"mcp_sse_max_sessions": "not-an-int"},
    }
    (tmp_path / "config.json").write_text(json.dumps(data))
    config = _make_config(storage_dir=tmp_path)

    with pytest.raises(ConfigError, match="failed validation"):
        read_envelope(config)


def test_read_envelope_raises_config_error_on_unknown_field(tmp_path: Path) -> None:
    data = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "infra_version": get_infra_version(),
        "settings": {"name": "node-a", "typo_field": "x"},
    }
    (tmp_path / "config.json").write_text(json.dumps(data))
    config = _make_config(storage_dir=tmp_path)

    with pytest.raises(ConfigError, match="failed validation"):
        read_envelope(config)


# --- write_envelope_atomic ---


def test_write_envelope_atomic_writes_unmasked_secrets(tmp_path: Path) -> None:
    config = _make_config(storage_dir=tmp_path)
    envelope = ConfigEnvelope(settings=DynamicSettings(mesh_key=SecretStr("real-secret")))

    write_envelope_atomic(config, envelope)

    written = json.loads((tmp_path / "config.json").read_text())
    assert written["settings"]["mesh_key"] == "real-secret"
    assert not (tmp_path / "config.json.tmp").exists()


def test_write_envelope_atomic_creates_storage_dir_if_missing(tmp_path: Path) -> None:
    storage_dir = tmp_path / "storage"
    config = _make_config(storage_dir=storage_dir)
    envelope = ConfigEnvelope(settings=DynamicSettings())

    write_envelope_atomic(config, envelope)

    assert (storage_dir / "config.json").exists()


# --- apply_dynamic_settings ---


def test_apply_dynamic_settings_copies_all_fields_onto_config() -> None:
    config = _make_config()
    settings = DynamicSettings(name="node-b", mcp_sse_max_sessions=999, otel_logging_enabled=True)

    apply_dynamic_settings(config, settings)

    assert config.name == "node-b"
    assert config.mcp_sse_max_sessions == 999
    assert config.otel_logging_enabled is True


# --- snapshot_dynamic_settings ---


def test_snapshot_dynamic_settings_builds_from_config() -> None:
    config = _make_config(name="node-c", metrics_username="admin")

    snapshot = snapshot_dynamic_settings(config)

    assert snapshot.name == "node-c"
    assert snapshot.metrics_username == "admin"


# --- _build_legacy_defaults ---


def test_build_legacy_defaults_returns_snapshot_of_config() -> None:
    config = _make_config(name="node-d")

    settings = _build_legacy_defaults(config)

    assert settings.name == "node-d"


def test_build_legacy_defaults_warns_when_non_default(caplog: pytest.LogCaptureFixture) -> None:
    config = _make_config(name="node-e")

    with caplog.at_level("WARNING", logger="uvicorn.error"):
        _build_legacy_defaults(config)

    assert "Seeding config.json from legacy .env values" in caplog.text


def test_build_legacy_defaults_silent_when_all_default(caplog: pytest.LogCaptureFixture) -> None:
    config = _make_config()

    with caplog.at_level("WARNING", logger="uvicorn.error"):
        _build_legacy_defaults(config)

    assert "Seeding config.json" not in caplog.text


# --- load_or_init ---


def test_load_or_init_seeds_config_json_from_legacy_config_when_missing(tmp_path: Path) -> None:
    config = _make_config(storage_dir=tmp_path, name="node-f")

    load_or_init(config)

    written = json.loads((tmp_path / "config.json").read_text())
    assert written["settings"]["name"] == "node-f"
    assert config.name == "node-f"


def test_load_or_init_loads_existing_config_json_without_rewriting(tmp_path: Path) -> None:
    data = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "infra_version": get_infra_version(),
        "settings": {"name": "on-disk-name"},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(data))
    config = _make_config(storage_dir=tmp_path, name="ignored-legacy-name")

    load_or_init(config)

    assert config.name == "on-disk-name"
    assert json.loads(config_path.read_text())["settings"]["name"] == "on-disk-name"


# --- persist_settings / persist ---


def test_persist_settings_writes_given_settings(tmp_path: Path) -> None:
    config = _make_config(storage_dir=tmp_path)
    settings = DynamicSettings(name="persisted-name")

    persist_settings(config, settings)

    written = json.loads((tmp_path / "config.json").read_text())
    assert written["settings"]["name"] == "persisted-name"


def test_persist_writes_current_config_snapshot(tmp_path: Path) -> None:
    config = _make_config(storage_dir=tmp_path, name="live-name")

    persist(config)

    written = json.loads((tmp_path / "config.json").read_text())
    assert written["settings"]["name"] == "live-name"
