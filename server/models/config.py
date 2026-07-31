# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Models for config api."""

from pydantic import BaseModel


class ConfigEntry(BaseModel):
    key: str
    value: str
    is_secret: bool
    field_name: str
    is_editable: bool


class ConfigOut(BaseModel):
    entries: list[ConfigEntry]


class ConfigRevealOut(BaseModel):
    key: str
    value: str
