# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Unit tests for server/utils/docker_image_version.py."""

from server.docker import DockerOptions
from server.utils.docker_image_version import apply_image_version_override, docker_tags_model_field, split_image_repo_tag


class TestSplitImageRepoTag:
    def test_splits_repo_and_tag(self) -> None:
        assert split_image_repo_tag("hub.simplito.com/deepfellow/doc-chunker-cpu:v1.0.3") == (
            "hub.simplito.com/deepfellow/doc-chunker-cpu",
            "v1.0.3",
        )

    def test_no_colon_returns_whole_string_as_repo(self) -> None:
        assert split_image_repo_tag("hub.simplito.com/deepfellow/doc-chunker-cpu") == (
            "hub.simplito.com/deepfellow/doc-chunker-cpu",
            "",
        )

    def test_digest_pinned_image_returns_no_tag(self) -> None:
        image = "ghcr.io/d4vinci/scrapling@sha256:77af4d59a6d00e40b918358943503ee6cafc44ad21fb60d5a545e17d0d40cd7a"
        assert split_image_repo_tag(image) == (image, "")

    def test_registry_host_with_port_and_no_tag_is_not_mistaken_for_a_tag(self) -> None:
        image = "registry.local:5000/deepfellow/doc-chunker"
        assert split_image_repo_tag(image) == (image, "")

    def test_registry_host_with_port_and_tag(self) -> None:
        assert split_image_repo_tag("registry.local:5000/deepfellow/doc-chunker:1.0.3") == (
            "registry.local:5000/deepfellow/doc-chunker",
            "1.0.3",
        )


class TestDockerTagsModelField:
    def test_builds_field_with_repo_and_depends_on(self) -> None:
        field = docker_tags_model_field("hub.simplito.com/deepfellow/doc-chunker-cpu", depends_on="hardware")

        assert field.type == "docker-tags"
        assert field.name == "image_version"
        assert field.docker_image == "hub.simplito.com/deepfellow/doc-chunker-cpu"
        assert field.depends_on == "hardware"
        assert field.required is False

    def test_depends_on_defaults_to_none(self) -> None:
        field = docker_tags_model_field("hub.simplito.com/deepfellow/open-websearch")

        assert field.depends_on is None


class TestApplyImageVersionOverride:
    def _docker_options(self, image: str) -> DockerOptions:
        return DockerOptions(image_port=8000, name="x", container_name="x", image=image, subnet=None)

    def test_no_version_returns_same_object(self) -> None:
        docker_options = self._docker_options("hub.simplito.com/deepfellow/doc-chunker-cpu:v1.0.3")

        assert apply_image_version_override(docker_options, None) is docker_options
        assert apply_image_version_override(docker_options, "") is docker_options

    def test_swaps_tag_on_a_copy(self) -> None:
        docker_options = self._docker_options("hub.simplito.com/deepfellow/doc-chunker-cpu:v1.0.3")

        overridden = apply_image_version_override(docker_options, "v1.0.5")

        assert overridden.image == "hub.simplito.com/deepfellow/doc-chunker-cpu:v1.0.5"
        # The original (which may be a shared instance for static, non-callable models) is untouched.
        assert docker_options.image == "hub.simplito.com/deepfellow/doc-chunker-cpu:v1.0.3"
        assert overridden is not docker_options

    def test_digest_pinned_image_is_left_unchanged(self) -> None:
        image = "ghcr.io/d4vinci/scrapling@sha256:77af4d59a6d00e40b918358943503ee6cafc44ad21fb60d5a545e17d0d40cd7a"
        docker_options = self._docker_options(image)

        overridden = apply_image_version_override(docker_options, "v9.9.9")

        assert overridden is docker_options
        assert overridden.image == image
