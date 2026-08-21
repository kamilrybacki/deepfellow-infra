# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Methods for data validation."""

import logging
from typing import Annotated

from pydantic import AfterValidator


def validate_no_traversal(v: str) -> str:
    """Raise if path contains '..'."""
    if v.startswith("../") or "/../" in v:
        raise ValueError("Security risk: Path traversal sequence '/../' is not allowed")
    return v


def clamp_non_positive_to_one(value: int, logger: logging.Logger, message: str) -> int:
    """Clamp a non-positive value to 1 instead of rejecting it, logging `message` when clamped."""
    if value <= 0:
        logger.warning(message, value)
        return 1
    return value


type SafePath = Annotated[str, AfterValidator(validate_no_traversal)]
