# SPDX-License-Identifier: MIT

"""Common types."""

from typing import Any

type FormFields = dict[str, Any]
type JsonSerializable = dict[str, Any]
type StarletteResponse = Any  # JsonSerializable | JSONResponse | BaseModel | StreamingResponse
