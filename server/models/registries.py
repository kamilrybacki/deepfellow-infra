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
    # Confirmed capability for the *installed* model, as read by `llamacpp_service.py`. Always
    # None from `scripts/get_llamacpp_models.py`: mmproj presence in the catalog repo listing is
    # not proof the installed model is actually served with vision support (llama.cpp never
    # downloads or wires up the mmproj file today - see `has_mmproj` below), so this must never be
    # set from that signal alone. llama.cpp has no audio-projector convention, so `supports_audio`
    # isn't tracked here.
    supports_vision: bool | None = None
    # Presence-only catalog hint from `scripts/get_llamacpp_models.py`: True when a companion
    # mmproj GGUF was found alongside this base model's file in the same HF repo listing (no
    # mmproj download, no proof it's actually usable). None means unresolved (e.g. an entry
    # predating this field, or no mmproj found). Not surfaced as `supports_vision` - see above.
    has_mmproj: bool | None = None


class LlamacppRegistry(BaseModel):
    llms: list[LlamacppRegistryEntry] = []
