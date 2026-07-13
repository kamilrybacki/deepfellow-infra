# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dynamic, hot-reloadable settings persisted to `config.json`.

`config.json` has exactly one writer: the `/admin/config` handler. There is no
file-watcher; the handler updates the shared `AppSettings` instance in-process
immediately after a successful write, so nothing else needs to poll or watch the
file to stay in sync with its own writes.
"""

import json
import logging
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from server.config import AppSettings, ConfigError, get_main_dir

logger = logging.getLogger("uvicorn.error")

CURRENT_SCHEMA_VERSION = 1


@lru_cache(maxsize=1)
def get_infra_version() -> str:
    """App version reported in config.json, for future migrations/diagnostics.

    Read from `pyproject.toml` lazily (rather than at import time) and cached, since it never
    changes for the lifetime of the process.
    """
    pyproject = get_main_dir() / "pyproject.toml"
    with pyproject.open("rb") as f:
        return tomllib.load(f)["project"]["version"]


class DynamicSettings(BaseModel):
    """The subset of `AppSettings` fields that live in `config.json`."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    infra_url: str = ""

    mesh_key: SecretStr = SecretStr("")
    infra_api_key: SecretStr = SecretStr("")

    connect_to_mesh_url: str = ""
    connect_to_mesh_key: SecretStr = SecretStr("")

    hugging_face_token: SecretStr = SecretStr("")
    civitai_token: SecretStr = SecretStr("")
    adapter_registry_url: str = ""
    adapter_registry_secret: SecretStr = SecretStr("")
    log_payloads: str = ""
    stop_containers_on_shutdown: str = ""

    mcp_sse_session_ttl_seconds: int = 300
    mcp_sse_max_sessions: int = 128

    metrics_username: str = ""
    metrics_password: SecretStr = SecretStr("")

    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_tracing_enabled: bool = False
    otel_logging_enabled: bool = False

    def to_persistable_dict(self) -> dict[str, Any]:
        """Serialize with real secret values — for config.json and internal PUT-merge use only.

        Pydantic's `SecretStr` masks itself as `"**********"` under normal JSON serialization
        (`model_dump(mode="json")`/`model_dump_json()`) to prevent accidental leakage into logs
        or API responses. That's exactly wrong for our own on-disk persistence, which needs the
        real values back on the next load. Never use this for anything sent to a client.
        """
        data = self.model_dump(mode="json")
        for field_name in DynamicSettings.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, SecretStr):
                data[field_name] = value.get_secret_value()
        return data


class ConfigEnvelope(BaseModel):
    """On-disk shape of `config.json`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = CURRENT_SCHEMA_VERSION
    infra_version: str = Field(default_factory=get_infra_version)
    settings: DynamicSettings = DynamicSettings()


def get_config_json_path(config: AppSettings) -> Path:
    """Path to config.json, inside the storage directory so it survives container recreation.

    The storage directory is a mounted volume in Docker deployments; the app's own directory
    (where `.env` lives) is part of the container image layer and does not persist.
    """
    return config.get_storage_dir() / "config.json"


def _migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply schema migrations in order, from `raw["schema_version"]` up to CURRENT_SCHEMA_VERSION.

    No migrations exist yet (this is schema version 1); this is the extension point for
    future versions.
    """
    return raw


def read_envelope(config: AppSettings) -> ConfigEnvelope | None:
    """Read and validate config.json, applying migrations. None if the file doesn't exist.

    Parses the raw JSON directly (not via `ConfigEnvelope.model_validate_json(...).model_dump(...)`)
    so secret fields never round-trip through Pydantic's masking serialization before being
    re-validated — doing so would silently replace real values with the literal `"**********"`.
    """
    path = get_config_json_path(config)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        message = f"config.json is not valid JSON: {e}"
        raise ConfigError(message) from e
    migrated = _migrate(raw)
    try:
        return ConfigEnvelope.model_validate(migrated)
    except ValidationError as e:
        message = f"config.json failed validation: {e}"
        raise ConfigError(message) from e


def write_envelope_atomic(config: AppSettings, envelope: ConfigEnvelope) -> None:
    """Write config.json atomically (temp file + rename), with real (unmasked) secret values."""
    path = get_config_json_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    data = envelope.model_dump(mode="json")
    data["settings"] = envelope.settings.to_persistable_dict()
    tmp_path.write_text(json.dumps(data, indent=2))
    tmp_path.replace(path)


def apply_dynamic_settings(config: AppSettings, settings: DynamicSettings) -> None:
    """Copy every DynamicSettings field onto the shared AppSettings instance in place."""
    for field_name in DynamicSettings.model_fields:
        setattr(config, field_name, getattr(settings, field_name))


def snapshot_dynamic_settings(config: AppSettings) -> DynamicSettings:
    """Build a DynamicSettings snapshot from the current values on the shared AppSettings instance."""
    return DynamicSettings(**{field_name: getattr(config, field_name) for field_name in DynamicSettings.model_fields})


def _build_legacy_defaults(config: AppSettings) -> DynamicSettings:
    """Build initial DynamicSettings from `config`, for the release that introduces config.json.

    `config` was already loaded from `.env` via pydantic-settings using the *current* field
    declarations, which still include every dynamic field (with a default). So any legacy `.env`
    value for a now-dynamic field (e.g. `DF_HUGGING_FACE_TOKEN`) is already present on `config` —
    no separate manual env parsing is needed, just snapshot it before config.json exists.
    """
    settings = snapshot_dynamic_settings(config)
    if settings != DynamicSettings():
        logger.warning(
            "Seeding config.json from legacy .env values. These environment variables are no longer "
            "read once config.json exists — use the /admin/config API going forward."
        )
    return settings


def load_or_init(config: AppSettings) -> None:
    """Load config.json into `config`, creating it (seeded from legacy env if present) if absent."""
    envelope = read_envelope(config)
    if envelope is None:
        settings = _build_legacy_defaults(config)
        envelope = ConfigEnvelope(
            schema_version=CURRENT_SCHEMA_VERSION,
            infra_version=get_infra_version(),
            settings=settings,
        )
        write_envelope_atomic(config, envelope)
    apply_dynamic_settings(config, envelope.settings)


def persist_settings(config: AppSettings, settings: DynamicSettings) -> None:
    """Write the given dynamic settings to config.json atomically.

    Callers should write to disk (this function) before mutating the in-memory `AppSettings`
    instance, so disk remains the source of truth if the process dies mid-update.
    """
    envelope = ConfigEnvelope(
        schema_version=CURRENT_SCHEMA_VERSION,
        infra_version=get_infra_version(),
        settings=settings,
    )
    write_envelope_atomic(config, envelope)


def persist(config: AppSettings) -> None:
    """Write the current in-memory dynamic settings back to config.json."""
    persist_settings(config, snapshot_dynamic_settings(config))
