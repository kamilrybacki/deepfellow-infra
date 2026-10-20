# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Unit tests for server/utils/docker_image_version.py."""

import pytest

from server.docker import DockerOptions
from server.utils.docker_image_version import (
    apply_image_version_override,
    docker_tags_model_field,
    has_cuda_variant,
    newer_tag_needing_cuda,
    newest_docker_tag,
    split_image_repo_tag,
)


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


class TestNewestDockerTag:
    @pytest.mark.parametrize(
        ("reference", "tags", "expected"),
        [
            # plain semver ignores variants and non-releases
            ("v2.2.0", ["latest", "v2.4.0-volta", "v2.3.0", "v2.10.0", "v2.9.0"], "v2.10.0"),
            # llama.cpp: build number, variant kept
            (
                "server-intel-b5966",
                ["server-intel-b6100", "server-vulkan-b9000", "server-intel-b999", "server-cuda-b7000"],
                "server-intel-b6100",
            ),
            ("server-vulkan-b10902", ["server-vulkan-b9999", "server-vulkan-b10903"], "server-vulkan-b10903"),
            # vLLM / SGLang: cuda variants
            ("v0.29.0-cu129-ubuntu2404", ["v0.30.0-cu129-ubuntu2404", "v0.31.0-cu129"], "v0.30.0-cu129-ubuntu2404"),
            ("v0.5.19-cu129", ["v0.5.20-cu129", "v0.5.20-cu130", "v0.5.20-cu129-xeon"], "v0.5.20-cu129"),
            # a newer CUDA major is never picked
            ("v0.5.19-cu129", ["v0.5.21-cu130", "v0.5.20-cu130"], "v0.5.19-cu129"),
            ("v0.5.19-cu128", ["v0.5.20-cu129", "v0.5.21-cu130"], "v0.5.20-cu129"),
            # Ollama: ROCm stays ROCm, plain stays plain
            ("0.12.3-rocm", ["0.13.0", "0.13.0-rocm", "0.12.4-rocm"], "0.13.0-rocm"),
            ("0.12.3", ["0.13.0", "0.13.0-rocm", "0.30.10-rc0"], "0.13.0"),
            # already newest -> reference itself
            ("v2.4.0", ["v2.3.0", "v2.4.0"], "v2.4.0"),
            ("v2.5.0", ["v2.4.0"], "v2.5.0"),
        ],
    )
    def test_picks_newest_tag_of_the_same_variant(self, reference: str, tags: list[str], expected: str) -> None:
        assert newest_docker_tag(tags, reference) == expected

    @pytest.mark.parametrize("reference", [None, "", "latest", "dev"])
    def test_unrecognized_reference_gives_none(self, reference: str | None) -> None:
        assert newest_docker_tag(["v1.0.0"], reference) is None


class TestNewestDockerTagWithHostCuda:
    TAGS = ("v0.5.21-cu130", "v0.5.20-cu130", "v0.5.20-cu129", "v0.5.19-cu129", "v0.5.21-cu134")

    @pytest.mark.parametrize(
        ("max_cuda", "expected"),
        [
            ((13, 4), "v0.5.21-cu134"),
            ((13, 0), "v0.5.21-cu130"),
            ((12, 9), "v0.5.20-cu129"),
            ((12, 8), "v0.5.19-cu129"),  # driver too old for the reference itself: stay on it
            (None, "v0.5.20-cu129"),
        ],
    )
    def test_cuda_builds_are_limited_by_the_driver(self, max_cuda: tuple[int, int] | None, expected: str) -> None:
        assert newest_docker_tag(list(self.TAGS), "v0.5.19-cu129", max_cuda) == expected

    def test_keeps_the_reference_cuda_build_at_the_same_version(self) -> None:
        assert newest_docker_tag(["v0.5.19-cu130", "v0.5.19-cu129"], "v0.5.19-cu129", (13, 0)) == "v0.5.19-cu129"

    def test_does_not_mix_other_variants(self) -> None:
        assert newest_docker_tag(["v0.5.21-cu130-runtime", "v0.5.21-cu130-xeon"], "v0.5.19-cu129", (13, 0)) == "v0.5.19-cu129"


class TestHasCudaVariant:
    @pytest.mark.parametrize(("tag", "expected"), [("v0.29.0-cu129-ubuntu2404", True), ("0.12.3-rocm", False), (None, False)])
    def test_detects_cuda_builds(self, tag: str | None, expected: bool) -> None:
        assert has_cuda_variant(tag) is expected


class TestNewerTagNeedingCuda:
    TAGS = ["v0.5.21-cu130", "v0.5.20-cu130", "v0.5.19-cu129", "v0.5.21-cu130-runtime"]

    @pytest.mark.parametrize(
        ("max_cuda", "expected"),
        [
            ((12, 9), ("v0.5.21-cu130", (13, 0))),  # driver too old for CUDA 13
            ((13, 0), None),  # the newest one is selectable
            (None, None),  # unknown driver: nothing to say
        ],
    )
    def test_reports_the_newer_tag_held_back_by_the_driver(
        self, max_cuda: tuple[int, int] | None, expected: tuple[str, tuple[int, int]] | None
    ) -> None:
        assert newer_tag_needing_cuda(self.TAGS, "v0.5.19-cu129", max_cuda) == expected

    def test_is_none_for_tags_without_cuda(self) -> None:
        assert newer_tag_needing_cuda(["0.13.0"], "0.12.3", (12, 9)) is None

    def test_is_none_for_an_unrecognized_reference(self) -> None:
        assert newer_tag_needing_cuda(self.TAGS, "latest", (12, 9)) is None

    def test_reports_the_cuda_needed_when_a_lower_update_is_still_possible(self) -> None:
        tags = ["v0.5.21-cu130", "v0.5.20-cu129", "v0.5.19-cu129"]
        assert newest_docker_tag(tags, "v0.5.19-cu129", (12, 9)) == "v0.5.20-cu129"
        assert newer_tag_needing_cuda(tags, "v0.5.19-cu129", (12, 9)) == ("v0.5.21-cu130", (13, 0))
