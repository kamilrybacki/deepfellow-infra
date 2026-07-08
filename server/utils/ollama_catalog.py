# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ollama library catalog client."""

import logging

import aiohttp

logger = logging.getLogger("uvicorn.error")

_CATALOG_URL = "https://ollama.com/api/tags"
_CATALOG_TIMEOUT = 15


def bytes_to_human(size_bytes: int) -> str:
    """Convert byte count to a human-readable string like '8.6 GB'."""
    gb = size_bytes / 1_000_000_000
    return f"{gb:.1f} GB"


class OllamaCatalogClient:
    """Fetches trending models from the Ollama library API."""

    async def fetch_trending(self) -> list[dict[str, str | int]]:
        """Fetch trending models from https://ollama.com/api/tags.

        Returns a list of dicts with keys: id, size, hash.
        Models with size=0 (cloud-hosted) are skipped.
        Returns empty list on any error.
        """
        try:
            timeout = aiohttp.ClientTimeout(total=_CATALOG_TIMEOUT)
            async with aiohttp.ClientSession() as session, session.get(_CATALOG_URL, timeout=timeout) as resp:
                if resp.status != 200:
                    logger.warning("Ollama catalog fetch failed: HTTP %s", resp.status)
                    return []
                data = await resp.json()
        except Exception:
            logger.exception("Ollama catalog fetch error")
            return []

        models: list[dict[str, str | int]] = []
        for entry in data.get("models", []):
            size_bytes: int = entry.get("size", 0)
            if size_bytes == 0:
                continue
            name: str = entry.get("name", "")
            if not name:
                continue
            digest: str = entry.get("digest", "")
            models.append({"id": name, "size": bytes_to_human(size_bytes), "hash": digest})
        return models
