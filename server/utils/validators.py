# SPDX-License-Identifier: MIT

"""Methods for data validation."""

from typing import Annotated

from pydantic import AfterValidator


def validate_no_traversal(v: str) -> str:
    """Raise if path contains '..'."""
    if v.startswith("../") or "/../" in v:
        raise ValueError("Security risk: Path traversal sequence '/../' is not allowed")
    return v


type SafePath = Annotated[str, AfterValidator(validate_no_traversal)]
