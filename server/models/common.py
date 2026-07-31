# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Common types."""

from typing import Any

type FormFields = dict[str, Any]
type JsonSerializable = dict[str, Any]
type StarletteResponse = Any  # JsonSerializable | JSONResponse | BaseModel | StreamingResponse
