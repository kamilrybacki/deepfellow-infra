# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""In-flight guard for the shared HuggingFace Hub API rate-limit budget.

vLLM, llama.cpp, and SGLang all fetch from the HuggingFace Hub API, which enforces rate limits on a
fixed 5-minute window shared across all callers from this process. A refresh already in flight for
one of them must block a concurrent trigger for the others, so the guard here is keyed on the
shared resource ("huggingface_api"), not per service_id.
"""

from fastapi import HTTPException


class HuggingFaceRefreshGuard:
    """In-flight guard shared across vLLM, llama.cpp, and SGLang catalog refreshes."""

    def __init__(self) -> None:
        self._holder: str | None = None

    def acquire(self, service_id: str) -> None:
        """Claim the shared HuggingFace API budget for service_id, or raise 429 if any refresh already holds it."""
        if self._holder is not None:
            raise HTTPException(
                429,
                f"A HuggingFace catalog refresh triggered by {self._holder!r} is already in progress; "
                "try again once it completes (vLLM, llama.cpp, and SGLang share the same rate-limit budget).",
            )
        self._holder = service_id

    def release(self) -> None:
        """Release the shared HuggingFace API budget."""
        self._holder = None
