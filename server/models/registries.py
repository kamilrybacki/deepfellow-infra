# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Shapes for the static model-registry JSON files (e.g. `static/llamacpp-min.json`).

Shared between the generator scripts (`scripts/get_llamacpp_models.py`) that write these files
and the services (`server/services/llamacpp_service.py`) that read them, so a field rename on one
side is caught by pyright/pydantic on the other instead of silently drifting.
"""

from pydantic import BaseModel


class LlamacppRegistryEntry(BaseModel):
    name: str
    url: str
    size: str
    jinja: bool = False


class LlamacppRegistry(BaseModel):
    llms: list[LlamacppRegistryEntry] = []
