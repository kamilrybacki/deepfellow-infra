# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Shared HuggingFace Hub API helpers for the model-list generator scripts."""

import asyncio
import re
from typing import Any

import aiohttp


async def get_json_with_retry(session: aiohttp.ClientSession, url: str, params: dict[str, str], max_retries: int = 5) -> Any:  # noqa: ANN401
    """GET url as JSON, retrying with backoff when HuggingFace rate-limits (429) or errors transiently (5xx)."""
    for attempt in range(max_retries):
        async with session.get(url, params=params) as resp:
            try:
                resp.raise_for_status()
            except aiohttp.ClientResponseError as exc:
                if (exc.status != 429 and exc.status < 500) or attempt == max_retries - 1:
                    raise
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(2.0**attempt, 30.0)
                await asyncio.sleep(delay)
                continue
            return await resp.json(content_type=None)
    raise AssertionError("unreachable")  # pragma: no cover


def is_gguf(model: dict[str, Any]) -> bool:
    """Return True if the model repo looks like a GGUF (quantized) repo."""
    name = model.get("id", "").lower()
    tags = [t.lower() for t in model.get("tags", [])]
    return "gguf" in name or "gguf" in tags


EMBEDDING_RE = re.compile(
    r"embed"  # generic
    r"|bge-"  # BAAI
    r"|gte-"  # thenlper/Alibaba-NLP
    r"|(?:^|[-/])e5-(?:large|base|small|mini)",  # intfloat; anchored + size suffix, "e5-" alone is too
    # common a substring (e.g. "...-Fable5-Thinking-...") to match unanchored.
    re.IGNORECASE,
)


def is_embedding(model: dict[str, Any]) -> bool:
    """Filter embedding models by name heuristic.

    Covers the generic "embed" substring plus the common family prefixes (BAAI's bge-,
    thenlper/Alibaba's gte-, intfloat's e5-) that ship without "embed" anywhere in the repo name.
    """
    return bool(EMBEDDING_RE.search(model.get("id", "")))
