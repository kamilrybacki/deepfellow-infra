# SPDX-License-Identifier: MIT

"""Docker registry tag fetching clients for Docker Hub, GHCR, and ECR Public."""

import logging
import re
from abc import ABC, abstractmethod
from urllib.parse import urljoin

import aiohttp

from server.docker import DockerImageNameInfo

logger = logging.getLogger("uvicorn.error")

_HUB_PAGE_SIZE = 100
_HUB_MAX_TAGS = 200
_OCI_PAGE_SIZE = 100
_OCI_MAX_PAGES = 20
_OCI_MAX_TAGS = 25
_LINK_NEXT_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')


class RegistryUnavailableError(Exception):
    """Raised when a registry's tags can't be fetched (network error, non-2xx response, etc.)."""


class RegistryClient(ABC):
    """Base class for registry tag clients."""

    @abstractmethod
    async def get_tags(self, image: str) -> list[str]:
        """Return available tags for the given image (without registry prefix).

        Raises RegistryUnavailableError if the registry can't be reached or returns an error —
        callers must not confuse that with a legitimate, successfully-fetched empty tag list.
        """

    @abstractmethod
    async def tag_exists(self, image: str, tag: str) -> bool:
        """Return whether tag exists for image, independent of get_tags()'s pagination cap.

        Raises RegistryUnavailableError if the registry can't be reached or returns an error.
        """


class DockerHubClient(RegistryClient):
    """Fetches tags from Docker Hub using its proprietary REST API."""

    def __init__(self, token: str | None = None) -> None:
        self._token = token

    async def _fetch_tags(self, image: str, headers: dict[str, str]) -> list[str]:
        tags: list[str] = []
        url: str | None = f"https://hub.docker.com/v2/repositories/{image}/tags/?page_size={_HUB_PAGE_SIZE}&ordering=last_updated"
        async with aiohttp.ClientSession() as session:
            while url and len(tags) < _HUB_MAX_TAGS:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning("Docker Hub tags fetch failed for %s: HTTP %s", image, resp.status)
                        if tags:
                            # A later page failed after earlier pages already succeeded — return what
                            # was fetched instead of discarding it; only a first-page failure means the
                            # registry is genuinely unavailable.
                            break
                        msg = f"Docker Hub tags fetch failed for {image}: HTTP {resp.status}"
                        raise RegistryUnavailableError(msg)
                    data = await resp.json()
                for result in data.get("results", []):
                    name = result.get("name")
                    if name and name != "latest":
                        tags.append(name)
                url = data.get("next")
        return tags[:_HUB_MAX_TAGS]

    async def get_tags(self, image: str) -> list[str]:
        """Return available tags for the given Docker Hub image, paginated up to 200 results."""
        headers: dict[str, str] = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        try:
            return await self._fetch_tags(image, headers)
        except RegistryUnavailableError:
            raise
        except Exception as exc:
            logger.exception("Docker Hub tags fetch error for %s", image)
            msg = f"Docker Hub tags fetch error for {image}"
            raise RegistryUnavailableError(msg) from exc

    async def _check_tag_exists(self, image: str, tag: str, headers: dict[str, str]) -> bool:
        url = f"https://hub.docker.com/v2/repositories/{image}/tags/{tag}"
        async with aiohttp.ClientSession() as session, session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                return True
            if resp.status == 404:
                return False
            logger.warning("Docker Hub tag existence check failed for %s:%s: HTTP %s", image, tag, resp.status)
            msg = f"Docker Hub tag existence check failed for {image}:{tag}: HTTP {resp.status}"
            raise RegistryUnavailableError(msg)

    async def tag_exists(self, image: str, tag: str) -> bool:
        """Return whether tag exists for image, checked directly instead of via the capped tag list."""
        headers: dict[str, str] = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            return await self._check_tag_exists(image, tag, headers)
        except RegistryUnavailableError:
            raise
        except Exception as exc:
            logger.exception("Docker Hub tag existence check error for %s:%s", image, tag)
            msg = f"Docker Hub tag existence check error for {image}:{tag}"
            raise RegistryUnavailableError(msg) from exc


def _tag_sort_key(tag: str) -> tuple[tuple[int, ...], str]:
    """Sort key that orders tags by their embedded numeric components (e.g. build/version numbers).

    Tags like "server-cuda-b7836" or "v0.19.0" aren't guaranteed to come back from the registry
    in any particular order, so we extract every digit run and compare them numerically, falling
    back to the raw string to keep the sort stable for tags without comparable numbers.
    """
    numbers = tuple(int(n) for n in re.findall(r"\d+", tag))
    return numbers, tag


class OciRegistryClient(RegistryClient):
    """Fetches tags from OCI-compliant registries (GHCR, ECR Public) using anonymous bearer tokens."""

    def __init__(self, registry_base: str, token_url: str, token_scope_template: str | None = None) -> None:
        self._registry_base = registry_base  # e.g. "https://ghcr.io"
        self._token_url = token_url  # e.g. "https://ghcr.io/token"
        self._token_scope_template = token_scope_template  # e.g. "repository:{image}:pull" or None

    async def _get_token(self, image: str, session: aiohttp.ClientSession) -> str:
        params: dict[str, str] = {}
        if self._token_scope_template:
            params["scope"] = self._token_scope_template.format(image=image)
            params["service"] = self._registry_base.removeprefix("https://")
        async with session.get(self._token_url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            data = await resp.json()
            token = str(data.get("token") or data.get("access_token") or "")
            if not token:
                msg = f"Token response for {image} contained no token or access_token field"
                raise ValueError(msg)
            return token

    async def _fetch_tags(self, image: str) -> list[str]:
        async with aiohttp.ClientSession() as session:
            token = await self._get_token(image, session)
            headers = {"Authorization": f"Bearer {token}"}
            all_tags: list[str] = []
            url: str | None = f"{self._registry_base}/v2/{image}/tags/list?n={_OCI_PAGE_SIZE}"
            pages_fetched = 0
            while url and pages_fetched < _OCI_MAX_PAGES:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning("OCI tags fetch failed for %s/%s: HTTP %s", self._registry_base, image, resp.status)
                        if all_tags:
                            # A later page failed after earlier pages already succeeded — return what
                            # was fetched instead of discarding it; only a first-page failure means the
                            # registry is genuinely unavailable.
                            break
                        msg = f"OCI tags fetch failed for {self._registry_base}/{image}: HTTP {resp.status}"
                        raise RegistryUnavailableError(msg)
                    data = await resp.json()
                    all_tags.extend(t for t in data.get("tags", []) if t != "latest")
                    url = self._next_page_url(resp.headers.get("Link"))
                pages_fetched += 1
            if url:
                logger.warning(
                    "OCI tags fetch for %s/%s stopped after %d pages (%d tags); more pages were available",
                    self._registry_base,
                    image,
                    pages_fetched,
                    len(all_tags),
                )
            return sorted(all_tags, key=_tag_sort_key, reverse=True)[:_OCI_MAX_TAGS]

    async def get_tags(self, image: str) -> list[str]:
        """Return the most recent tags for the given image via OCI Distribution Spec bearer token flow.

        Follows the `Link: rel="next"` header across pages (up to `_OCI_MAX_PAGES`) since the
        Distribution Spec makes no guarantee that `tags/list` is chronologically or numerically
        ordered. Tags are sorted by their embedded numeric version/build components (descending)
        only after all fetched pages are collected, so the most recent tags aren't silently
        dropped by a naive "last N" slice of an unordered first page.
        """
        try:
            return await self._fetch_tags(image)
        except RegistryUnavailableError:
            raise
        except Exception as exc:
            logger.exception("OCI tags fetch error for %s/%s", self._registry_base, image)
            msg = f"OCI tags fetch error for {self._registry_base}/{image}"
            raise RegistryUnavailableError(msg) from exc

    async def _check_tag_exists(self, image: str, tag: str, session: aiohttp.ClientSession) -> bool:
        token = await self._get_token(image, session)
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": (
                "application/vnd.oci.image.manifest.v1+json,"
                "application/vnd.oci.image.index.v1+json,"
                "application/vnd.docker.distribution.manifest.v2+json,"
                "application/vnd.docker.distribution.manifest.list.v2+json"
            ),
        }
        url = f"{self._registry_base}/v2/{image}/manifests/{tag}"
        async with session.head(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                return True
            if resp.status == 404:
                return False
            logger.warning("OCI tag existence check failed for %s/%s:%s: HTTP %s", self._registry_base, image, tag, resp.status)
            msg = f"OCI tag existence check failed for {self._registry_base}/{image}:{tag}: HTTP {resp.status}"
            raise RegistryUnavailableError(msg)

    async def tag_exists(self, image: str, tag: str) -> bool:
        """Return whether tag exists for image, checked via a manifest HEAD instead of the capped tag list."""
        try:
            async with aiohttp.ClientSession() as session:
                return await self._check_tag_exists(image, tag, session)
        except RegistryUnavailableError:
            raise
        except Exception as exc:
            logger.exception("OCI tag existence check error for %s/%s:%s", self._registry_base, image, tag)
            msg = f"OCI tag existence check error for {self._registry_base}/{image}:{tag}"
            raise RegistryUnavailableError(msg) from exc

    def _next_page_url(self, link_header: str | None) -> str | None:
        """Extract and resolve the `rel="next"` URL from an RFC 8288 `Link` header, if present."""
        if not link_header:
            return None
        match = _LINK_NEXT_RE.search(link_header)
        if not match:
            return None
        return urljoin(f"{self._registry_base}/", match.group(1))


_GHCR_CLIENT = OciRegistryClient(
    registry_base="https://ghcr.io",
    token_url="https://ghcr.io/token",
    token_scope_template="repository:{image}:pull",
)

_ECR_PUBLIC_CLIENT = OciRegistryClient(
    registry_base="https://public.ecr.aws",
    token_url="https://public.ecr.aws/token/",
    token_scope_template=None,
)


def registry_for(image_name: str, docker_hub_token: str | None = None) -> RegistryClient:
    """Return the appropriate registry client based on the image name's registry host."""
    registry = DockerImageNameInfo.parse(image_name).registry
    if registry == "ghcr.io":
        return _GHCR_CLIENT
    if registry == "public.ecr.aws":
        return _ECR_PUBLIC_CLIENT
    return DockerHubClient(token=docker_hub_token)


def image_without_registry_prefix(image_name: str) -> str:
    """Strip the registry host from an image name, leaving only namespace/repo."""
    info = DockerImageNameInfo.parse(image_name)
    if info.registry in ("ghcr.io", "public.ecr.aws"):
        return f"{info.namespace}/{info.image_name}"
    return image_name
