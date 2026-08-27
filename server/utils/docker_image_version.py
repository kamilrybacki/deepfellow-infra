# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Shared helpers for offering a Docker image-version selector on built-in service models.

Used by services (e.g. CustomService, McpService) that install several built-in models, each
backed by its own Docker image, and want a "docker-tags" install-form field without maintaining a
separate per-model image registry — the repo/tag is derived from each model's own default image.
"""

import copy

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
