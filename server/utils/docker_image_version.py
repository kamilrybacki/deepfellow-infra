# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Shared helpers for offering a Docker image-version selector on built-in service models.

Used by services (e.g. CustomService, McpService) that install several built-in models, each
backed by its own Docker image, and want a "docker-tags" install-form field without maintaining a
separate per-model image registry — the repo/tag is derived from each model's own default image.
"""

import copy
import re
from typing import NamedTuple

from server.docker import DockerOptions
from server.models.models import ModelField


def split_image_repo_tag(image: str) -> tuple[str, str]:
    """Split "repo:tag" into (repo, tag); (image, "") if there's no tag or the image is digest-pinned.

    A colon before the last "/" segment is a registry host:port, not a tag separator (e.g.
    "registry.local:5000/myimage" has no tag), so only a colon in the final path segment counts.
    """
    if "@" in image:
        return image, ""
    repo, sep, tag = image.rpartition(":")
    if not sep or "/" in tag:
        return image, ""
    return repo, tag


def docker_tags_model_field(repo: str, depends_on: str | None = None) -> ModelField:
    """Build the "docker-tags" install-form field for a model whose default image resolves to repo."""
    return ModelField(
        type="docker-tags",
        name="image_version",
        description="Docker image version",
        docker_image=repo,
        depends_on=depends_on,
        required=False,
    )


def apply_image_version_override(docker_options: DockerOptions, image_version: str | None) -> DockerOptions:
    """Swap the tag on docker_options.image for a user-selected image_version, if one was given.

    No-op for digest-pinned images (no tag to swap). Copies docker_options before mutating, since
    it may be a single instance shared across installs (a static, non-callable model.options).
    """
    if not image_version:
        return docker_options
    repo, tag = split_image_repo_tag(docker_options.image)
    if not tag:
        return docker_options
    docker_options = copy.copy(docker_options)
    docker_options.image = f"{repo}:{image_version}"
    return docker_options


# "v0.29.0-cu129-ubuntu2404" -> prefix "v", version 0.29.0, suffix "-cu129-ubuntu2404"
_SEMVER_TAG = re.compile(r"^(?P<prefix>v?)(?P<version>\d+\.\d+\.\d+)(?P<suffix>.*)$")
# "server-vulkan-b10902" -> prefix "server-vulkan-b", build 10902
_BUILD_TAG = re.compile(r"^(?P<prefix>.*-b)(?P<version>\d+)$")


class _ParsedTag(NamedTuple):
    variant: tuple[str, str]
    version: tuple[int, ...]
    suffix: str
    cuda: tuple[int, int] | None


def _parse_tag(tag: str, *, keep_cuda_major: bool) -> _ParsedTag | None:
    """Split a tag into its variant (what must stay the same on update), version, suffix and CUDA version, or None.

    The variant is the prefix plus the suffix with its digits dropped, so `-rocm`, `-cu129` and `-cu129-ubuntu2404`
    are different variants. With `keep_cuda_major` a CUDA build also keeps its major version (`cu129` and `cu128`
    match, `cu130` does not), because a newer CUDA major may need a newer host driver; without it the CUDA
    version is not part of the variant and the caller limits it by the driver's CUDA version instead.
    """
    if match := _SEMVER_TAG.match(tag):
        suffix = match["suffix"]
        version = tuple(int(part) for part in match["version"].split("."))
        cuda_match = re.search(r"cu(\d\d)(\d?)", suffix)
        cuda = (int(cuda_match[1]), int(cuda_match[2] or 0)) if cuda_match else None
        variant_suffix = re.sub(r"\d+", "", suffix) + (cuda_match[1] if cuda_match and keep_cuda_major else "")
        return _ParsedTag((match["prefix"], variant_suffix), version, suffix, cuda)
    if match := _BUILD_TAG.match(tag):
        return _ParsedTag((match["prefix"], ""), (int(match["version"]),), "", None)
    return None


def has_cuda_variant(tag: str | None) -> bool:
    """Whether the tag is a CUDA build (`...-cu129...`), i.e. whether the host's CUDA version matters for it."""
    return bool(tag and re.search(r"cu\d\d", tag))


def _select_newest(
    tags: list[str], reference: str | None, *, keep_cuda_major: bool, max_cuda: tuple[int, int] | None
) -> tuple[str, _ParsedTag] | None:
    """Pick the best tag of the reference's variant, `reference` itself when nothing is newer, None if it is unrecognized."""
    parsed_reference = _parse_tag(reference, keep_cuda_major=keep_cuda_major) if reference else None
    if not reference or not parsed_reference:
        return None

    def sort_key(parsed: _ParsedTag) -> tuple[tuple[int, ...], bool, str]:
        return parsed.version, parsed.suffix == parsed_reference.suffix, parsed.suffix

    best_tag, best = reference, parsed_reference
    for tag in tags:
        parsed = _parse_tag(tag, keep_cuda_major=keep_cuda_major)
        if not parsed or parsed.variant != parsed_reference.variant:
            continue
        if max_cuda is not None and parsed.cuda is not None and parsed.cuda > max_cuda:
            continue
        if sort_key(parsed) > sort_key(best):
            best_tag, best = tag, parsed
    return best_tag, best


def newest_docker_tag(tags: list[str], reference: str | None, max_cuda: tuple[int, int] | None = None) -> str | None:
    """Return the newest tag of the same variant as `reference` (the installed or default tag), or None.

    Variants (hardware builds such as `-rocm` or `-cu129`, `server-vulkan-b<build>`) are never mixed, so an update
    does not change the hardware the image was built for. Among tags with the same version the one whose suffix
    equals the reference's wins, then the higher suffix. If the reference is already the newest, it is returned.

    `max_cuda` is the highest CUDA version the host driver supports. When known, CUDA builds up to it are
    candidates, including a newer CUDA major; when None, the CUDA major of the reference is kept.
    """
    selected = _select_newest(tags, reference, keep_cuda_major=max_cuda is None, max_cuda=max_cuda)
    return selected[0] if selected else None


def newer_tag_needing_cuda(tags: list[str], reference: str | None, max_cuda: tuple[int, int] | None) -> tuple[str, tuple[int, int]] | None:
    """Return a tag newer than `newest_docker_tag` that only needs a newer CUDA than `max_cuda`, with that CUDA version.

    None when the host's CUDA version is unknown, or when no newer tag of the reference's variant is held back by it.
    """
    if max_cuda is None:
        return None
    limited = newest_docker_tag(tags, reference, max_cuda)
    unlimited = _select_newest(tags, reference, keep_cuda_major=False, max_cuda=None)
    if not unlimited or unlimited[0] == limited or unlimited[1].cuda is None or unlimited[1].cuda <= max_cuda:
        return None
    return unlimited[0], unlimited[1].cuda
