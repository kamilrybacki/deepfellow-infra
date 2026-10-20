# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Version and build identity of the running Infra."""

import os
import tomllib
from functools import lru_cache

from server.config import get_main_dir


@lru_cache(maxsize=1)
def get_infra_version() -> str:
    """App version reported in config.json, for future migrations/diagnostics.

    Read from `pyproject.toml` lazily (rather than at import time) and cached, since it never
    changes for the lifetime of the process.
    """
    pyproject = get_main_dir() / "pyproject.toml"
    with pyproject.open("rb") as f:
        return tomllib.load(f)["project"]["version"]


@lru_cache(maxsize=1)
def get_build_commit() -> str | None:
    """Return the commit the running image was built from, or None when it is not baked in.

    Baked into the image by the `DF_BUILD_COMMIT` build argument (see the Dockerfile): `.dockerignore`
    keeps `.git` out of the image, so there is nothing to read at runtime and the value has to arrive
    at build time.

    Returns:
        The commit SHA, or `None` for a local run and for any image built before this was added.
    """
    return os.environ.get("DF_BUILD_COMMIT") or None
