# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Shared plumbing for the vLLM/llama.cpp/SGLang runtime HuggingFace catalog refresh.

Per-service TTL caching of a completed refresh's result is handled separately, at the route layer
(`server/api/services.py`'s existing `catalog_refresh_cache`), which is already generic across
every service type.
"""

import json
import logging
from pathlib import Path
from typing import Any

from server.services.huggingface_refresh_guard import HuggingFaceRefreshGuard

logger = logging.getLogger("uvicorn.error")

huggingface_refresh_guard = HuggingFaceRefreshGuard()

# SET ONCE AT STARTUP BY `PROBE_STATIC_DIR_WRITABLE`; READ BY VLLMSERVICE/LLAMACPPSERVICE BEFORE
# STARTING A REFRESH SO A DOOMED WRITE IS REJECTED IMMEDIATELY INSTEAD OF AFTER A SLOW HF FETCH.
catalog_refresh_supported = True

# BELOW THIS FRACTION OF THE EXISTING CATALOG'S SIZE, A REFRESH RESULT IS TREATED AS DEGRADED
# (E.G. HUGGINGFACE API CHANGES, RATE-LIMITING) RATHER THAN A GENUINE DROP IN TRENDING MODELS.
MIN_CATALOG_RETENTION_RATE = 0.5


class CatalogTooSmallError(Exception):
    """Raised when a refreshed catalog is suspiciously smaller than the one it would replace."""


def ensure_catalog_not_degraded(registry_key: str, new_count: int, existing_count: int) -> None:
    """Raise `CatalogTooSmallError` if a refresh would shrink `registry_key` below `MIN_CATALOG_RETENTION_RATE`."""
    if existing_count > 0 and new_count < existing_count * MIN_CATALOG_RETENTION_RATE:
        message = (
            f"Refreshed {registry_key!r} catalog has only {new_count} entries, far fewer than the "
            f"existing {existing_count} — refusing to write a possibly-degraded registry."
        )
        raise CatalogTooSmallError(message)


def atomic_write_json(path: Path, data: Any) -> None:  # noqa: ANN401
    """Write `data` as indented JSON to `path` atomically (temp file + rename), never leaving a partial file."""
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(data, indent=4), encoding="utf-8")
    tmp_path.replace(path)


def probe_static_dir_writable(static_dir: Path) -> bool:
    """Attempt to write and delete a marker file in static_dir; return whether it succeeded.

    This only confirms the directory is writable right now — it says nothing about whether that
    directory survives a container restart/redeploy (see the change's open question on `static/`
    durability). Treat it as a floor, not a full durability guarantee.
    """
    marker = static_dir / ".catalog_refresh_write_probe"
    try:
        marker.write_text("", encoding="utf-8")
        marker.unlink()
    except OSError:
        logger.warning("static/ directory %s is not writable; disabling runtime catalog refresh.", static_dir)
        return False
    logger.info("static/ directory %s is writable; runtime catalog refresh enabled.", static_dir)
    return True


def set_catalog_refresh_supported(supported: bool) -> None:
    """Set the module-level capability flag (called once from app startup)."""
    global catalog_refresh_supported
    catalog_refresh_supported = supported
