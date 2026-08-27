# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Unit tests for server/utils/registry_client.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.utils.registry_client import (
    DockerHubClient,
    OciRegistryClient,
    RegistryUnavailableError,
    image_without_registry_prefix,
    registry_for,
)


def _make_response(status: int, json_data: object, headers: dict[str, str] | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data)
    resp.raise_for_status = MagicMock()
    resp.headers = headers or {}
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _make_session(*responses: MagicMock) -> MagicMock:
    session = MagicMock()
    response_iter = iter(responses)

    def _next_response(*_args: object, **_kwargs: object) -> MagicMock:
        return next(response_iter)

    session.get = MagicMock(side_effect=_next_response)
    session.head = MagicMock(side_effect=_next_response)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


class TestDockerHubClient:
    @pytest.mark.asyncio
    async def test_get_tags_single_page(self) -> None:
        hub_resp = _make_response(
            200,
            {
                "results": [{"name": "0.20.4"}, {"name": "0.20.3"}, {"name": "latest"}],
                "next": None,
            },
        )
        with patch("aiohttp.ClientSession", return_value=_make_session(hub_resp)):
            client = DockerHubClient()
            tags = await client.get_tags("ollama/ollama")

        assert "0.20.4" in tags
        assert "0.20.3" in tags
        assert "latest" not in tags  # latest is excluded

    @pytest.mark.asyncio
    async def test_get_tags_pagination(self) -> None:
        page1 = _make_response(
            200,
            {
                "results": [{"name": f"0.{i}.0"} for i in range(100)],
                "next": "https://hub.docker.com/v2/repositories/ollama/ollama/tags/?page=2",
            },
        )
        page2 = _make_response(
            200,
            {
                "results": [{"name": f"1.{i}.0"} for i in range(50)],
                "next": None,
            },
        )
        with patch("aiohttp.ClientSession", return_value=_make_session(page1, page2)):
            client = DockerHubClient()
            tags = await client.get_tags("ollama/ollama")

        assert len(tags) == 150

    @pytest.mark.asyncio
    async def test_get_tags_stops_at_max(self) -> None:
        # Each page has 100 tags; cap is enforced after all fetching
        page1 = _make_response(
            200,
            {
                "results": [{"name": f"0.{i}.0"} for i in range(100)],
                "next": "https://hub.docker.com/page2",
            },
        )
        page2 = _make_response(
            200,
            {
                "results": [{"name": f"1.{i}.0"} for i in range(100)],
                "next": "https://hub.docker.com/page3",
            },
        )
        with patch("aiohttp.ClientSession", return_value=_make_session(page1, page2)):
            client = DockerHubClient()
            tags = await client.get_tags("ollama/ollama")

        assert len(tags) == 200

    @pytest.mark.asyncio
    async def test_get_tags_mid_pagination_failure_returns_pages_fetched_so_far(self) -> None:
        page1 = _make_response(
            200,
            {
                "results": [{"name": f"0.{i}.0"} for i in range(100)],
                "next": "https://hub.docker.com/page2",
            },
        )
        error_page = _make_response(503, {})
        with patch("aiohttp.ClientSession", return_value=_make_session(page1, error_page)):
            client = DockerHubClient()
            tags = await client.get_tags("ollama/ollama")

        assert len(tags) == 100

    @pytest.mark.asyncio
    async def test_get_tags_non_200_raises(self) -> None:
        error_resp = _make_response(429, {})
        with patch("aiohttp.ClientSession", return_value=_make_session(error_resp)):
            client = DockerHubClient()
            with pytest.raises(RegistryUnavailableError):
                await client.get_tags("ollama/ollama")

    @pytest.mark.asyncio
    async def test_get_tags_exception_raises(self) -> None:
        session = MagicMock()
        session.__aenter__ = AsyncMock(side_effect=OSError("connection refused"))
        session.__aexit__ = AsyncMock(return_value=False)
        with patch("aiohttp.ClientSession", return_value=session):
            client = DockerHubClient()
            with pytest.raises(RegistryUnavailableError):
                await client.get_tags("ollama/ollama")

    @pytest.mark.asyncio
    async def test_get_tags_includes_token_header(self) -> None:
        hub_resp = _make_response(200, {"results": [{"name": "0.20.4"}], "next": None})
        session = _make_session(hub_resp)
        with patch("aiohttp.ClientSession", return_value=session):
            client = DockerHubClient(token="mytoken")
            await client.get_tags("ollama/ollama")

        call_kwargs = session.get.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer mytoken"

    @pytest.mark.asyncio
    async def test_tag_exists_true(self) -> None:
        resp = _make_response(200, {"name": "0.9.0"})
        with patch("aiohttp.ClientSession", return_value=_make_session(resp)):
            client = DockerHubClient()
            assert await client.tag_exists("ollama/ollama", "0.9.0") is True

    @pytest.mark.asyncio
    async def test_tag_exists_includes_token_header(self) -> None:
        resp = _make_response(200, {"name": "0.9.0"})
        session = _make_session(resp)
        with patch("aiohttp.ClientSession", return_value=session):
            client = DockerHubClient(token="mytoken")
            assert await client.tag_exists("ollama/ollama", "0.9.0") is True

        call_kwargs = session.get.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer mytoken"

    @pytest.mark.asyncio
    async def test_tag_exists_false_when_not_found(self) -> None:
        resp = _make_response(404, {})
        with patch("aiohttp.ClientSession", return_value=_make_session(resp)):
            client = DockerHubClient()
            assert await client.tag_exists("ollama/ollama", "does-not-exist") is False

    @pytest.mark.asyncio
    async def test_tag_exists_raises_on_other_error(self) -> None:
        resp = _make_response(500, {})
        with patch("aiohttp.ClientSession", return_value=_make_session(resp)):
            client = DockerHubClient()
            with pytest.raises(RegistryUnavailableError):
                await client.tag_exists("ollama/ollama", "0.9.0")

    @pytest.mark.asyncio
    async def test_tag_exists_exception_raises(self) -> None:
        session = MagicMock()
        session.__aenter__ = AsyncMock(side_effect=OSError("connection refused"))
        session.__aexit__ = AsyncMock(return_value=False)
        with patch("aiohttp.ClientSession", return_value=session):
            client = DockerHubClient()
            with pytest.raises(RegistryUnavailableError):
                await client.tag_exists("ollama/ollama", "0.9.0")


class TestOciRegistryClient:
    @pytest.mark.asyncio
    async def test_get_tags_ghcr(self) -> None:
        token_resp = _make_response(200, {"token": "ghcr-anon-token"})
        tags_resp = _make_response(200, {"name": "ggml-org/llama.cpp", "tags": ["server-cuda-b7836", "server-b7836", "latest"]})
        with patch("aiohttp.ClientSession", return_value=_make_session(token_resp, tags_resp)):
            client = OciRegistryClient(
                registry_base="https://ghcr.io",
                token_url="https://ghcr.io/token",
                token_scope_template="repository:{image}:pull",
            )
            tags = await client.get_tags("ggml-org/llama.cpp")

        assert "server-cuda-b7836" in tags
        assert "server-b7836" in tags
        assert "latest" not in tags

    @pytest.mark.asyncio
    async def test_get_tags_ecr_public(self) -> None:
        token_resp = _make_response(200, {"token": "ecr-anon-token"})
        tags_resp = _make_response(200, {"tags": ["v0.19.1", "v0.19.0"]})
        with patch("aiohttp.ClientSession", return_value=_make_session(token_resp, tags_resp)):
            client = OciRegistryClient(
                registry_base="https://public.ecr.aws",
                token_url="https://public.ecr.aws/token/",
                token_scope_template=None,
            )
            tags = await client.get_tags("q9t5s3a7/vllm-cpu-release-repo")

        assert tags == ["v0.19.1", "v0.19.0"]

    @pytest.mark.asyncio
    async def test_get_tags_non_200_raises(self) -> None:
        token_resp = _make_response(200, {"token": "tok"})
        error_resp = _make_response(403, {})
        with patch("aiohttp.ClientSession", return_value=_make_session(token_resp, error_resp)):
            client = OciRegistryClient("https://ghcr.io", "https://ghcr.io/token", "repository:{image}:pull")
            with pytest.raises(RegistryUnavailableError):
                await client.get_tags("ggml-org/llama.cpp")

    @pytest.mark.asyncio
    async def test_get_tags_exception_raises(self) -> None:
        session = MagicMock()
        session.__aenter__ = AsyncMock(side_effect=OSError("connection refused"))
        session.__aexit__ = AsyncMock(return_value=False)
        with patch("aiohttp.ClientSession", return_value=session):
            client = OciRegistryClient("https://ghcr.io", "https://ghcr.io/token", "repository:{image}:pull")
            with pytest.raises(RegistryUnavailableError):
                await client.get_tags("ggml-org/llama.cpp")

    @pytest.mark.asyncio
    async def test_get_tags_follows_link_header_pagination(self) -> None:
        token_resp = _make_response(200, {"token": "ghcr-anon-token"})
        page1 = _make_response(
            200,
            {"tags": ["server-cuda-b4738"]},
            headers={"Link": '</v2/ggml-org/llama.cpp/tags/list?n=100&last=server-cuda-b4738>; rel="next"'},
        )
        page2 = _make_response(200, {"tags": ["server-cuda-b7836", "server-cuda-b5000"]})
        with patch("aiohttp.ClientSession", return_value=_make_session(token_resp, page1, page2)):
            client = OciRegistryClient("https://ghcr.io", "https://ghcr.io/token", "repository:{image}:pull")
            tags = await client.get_tags("ggml-org/llama.cpp")

        # The newest tag lives on the second page; a naive first-page-only fetch would miss it.
        assert tags[0] == "server-cuda-b7836"
        assert "server-cuda-b4738" in tags

    @pytest.mark.asyncio
    async def test_get_tags_sorted_numerically_not_by_list_order(self) -> None:
        token_resp = _make_response(200, {"token": "tok"})
        # Registry returns tags out of chronological order; the client must not trust list order.
        tags_resp = _make_response(200, {"tags": ["server-cuda-b100", "server-cuda-b9999", "server-cuda-b2"]})
        with patch("aiohttp.ClientSession", return_value=_make_session(token_resp, tags_resp)):
            client = OciRegistryClient("https://ghcr.io", "https://ghcr.io/token", "repository:{image}:pull")
            tags = await client.get_tags("ggml-org/llama.cpp")

        assert tags == ["server-cuda-b9999", "server-cuda-b100", "server-cuda-b2"]

    @pytest.mark.asyncio
    async def test_get_tags_link_header_without_next_rel_stops_pagination(self) -> None:
        token_resp = _make_response(200, {"token": "tok"})
        page1 = _make_response(
            200,
            {"tags": ["server-cuda-b1"]},
            headers={"Link": '</v2/x/tags/list?n=100>; rel="first"'},
        )
        with patch("aiohttp.ClientSession", return_value=_make_session(token_resp, page1)):
            client = OciRegistryClient("https://ghcr.io", "https://ghcr.io/token", "repository:{image}:pull")
            tags = await client.get_tags("ggml-org/llama.cpp")

        assert tags == ["server-cuda-b1"]

    @pytest.mark.asyncio
    async def test_get_tags_stops_after_max_pages(self) -> None:
        token_resp = _make_response(200, {"token": "tok"})
        pages = [
            _make_response(
                200,
                {"tags": [f"server-cuda-b{i}"]},
                headers={"Link": f'</v2/x/tags/list?n=100&last=b{i}>; rel="next"'},
            )
            for i in range(30)
        ]
        with patch("aiohttp.ClientSession", return_value=_make_session(token_resp, *pages)):
            client = OciRegistryClient("https://ghcr.io", "https://ghcr.io/token", "repository:{image}:pull")
            tags = await client.get_tags("ggml-org/llama.cpp")

        # Capped at _OCI_MAX_PAGES (20) pages fetched, then top _OCI_MAX_TAGS (25) kept.
        assert len(tags) == 20

    @pytest.mark.asyncio
    async def test_get_tags_mid_pagination_failure_returns_pages_fetched_so_far(self) -> None:
        token_resp = _make_response(200, {"token": "tok"})
        page1 = _make_response(
            200,
            {"tags": ["server-cuda-b1"]},
            headers={"Link": '</v2/x/tags/list?n=100&last=b1>; rel="next"'},
        )
        error_page = _make_response(503, {})
        with patch("aiohttp.ClientSession", return_value=_make_session(token_resp, page1, error_page)):
            client = OciRegistryClient("https://ghcr.io", "https://ghcr.io/token", "repository:{image}:pull")
            tags = await client.get_tags("ggml-org/llama.cpp")

        assert tags == ["server-cuda-b1"]

    @pytest.mark.asyncio
    async def test_get_tags_empty_token_response_raises(self) -> None:
        token_resp = _make_response(200, {"error": "no token here"})
        with patch("aiohttp.ClientSession", return_value=_make_session(token_resp)):
            client = OciRegistryClient("https://ghcr.io", "https://ghcr.io/token", "repository:{image}:pull")
            with pytest.raises(RegistryUnavailableError):
                await client.get_tags("ggml-org/llama.cpp")

    @pytest.mark.asyncio
    async def test_tag_exists_true(self) -> None:
        token_resp = _make_response(200, {"token": "tok"})
        manifest_resp = _make_response(200, {})
        with patch("aiohttp.ClientSession", return_value=_make_session(token_resp, manifest_resp)):
            client = OciRegistryClient("https://ghcr.io", "https://ghcr.io/token", "repository:{image}:pull")
            assert await client.tag_exists("ggml-org/llama.cpp", "server-b1") is True

    @pytest.mark.asyncio
    async def test_tag_exists_false_when_not_found(self) -> None:
        token_resp = _make_response(200, {"token": "tok"})
        manifest_resp = _make_response(404, {})
        with patch("aiohttp.ClientSession", return_value=_make_session(token_resp, manifest_resp)):
            client = OciRegistryClient("https://ghcr.io", "https://ghcr.io/token", "repository:{image}:pull")
            assert await client.tag_exists("ggml-org/llama.cpp", "does-not-exist") is False

    @pytest.mark.asyncio
    async def test_tag_exists_raises_on_other_error(self) -> None:
        token_resp = _make_response(200, {"token": "tok"})
        manifest_resp = _make_response(500, {})
        with patch("aiohttp.ClientSession", return_value=_make_session(token_resp, manifest_resp)):
            client = OciRegistryClient("https://ghcr.io", "https://ghcr.io/token", "repository:{image}:pull")
            with pytest.raises(RegistryUnavailableError):
                await client.tag_exists("ggml-org/llama.cpp", "server-b1")

    @pytest.mark.asyncio
    async def test_tag_exists_exception_raises(self) -> None:
        session = MagicMock()
        session.__aenter__ = AsyncMock(side_effect=OSError("connection refused"))
        session.__aexit__ = AsyncMock(return_value=False)
        with patch("aiohttp.ClientSession", return_value=session):
            client = OciRegistryClient("https://ghcr.io", "https://ghcr.io/token", "repository:{image}:pull")
            with pytest.raises(RegistryUnavailableError):
                await client.tag_exists("ggml-org/llama.cpp", "server-b1")

    @pytest.mark.asyncio
    async def test_tag_exists_true_when_registry_allows_anonymous_pulls(self) -> None:
        discovery_resp = _make_response(200, {}, headers={})
        manifest_resp = _make_response(200, {})
        session = _make_session(discovery_resp, manifest_resp)
        with patch("aiohttp.ClientSession", return_value=session):
            client = OciRegistryClient(registry_base="https://hub.example.com", token_scope_template="repository:{image}:pull")
            assert await client.tag_exists("deepfellow/doc-chunker-gpu", "v1.0.5") is True

        manifest_call_headers = session.head.call_args.kwargs["headers"]
        assert "Authorization" not in manifest_call_headers

    @pytest.mark.asyncio
    async def test_get_tags_discovers_token_endpoint_when_token_url_omitted(self) -> None:
        discovery_resp = _make_response(
            401, {}, headers={"WWW-Authenticate": 'Bearer realm="https://auth.example.com/token",service="Docker registry"'}
        )
        token_resp = _make_response(200, {"token": "anon-token"})
        tags_resp = _make_response(200, {"tags": ["v1.0.5", "v1.0.4"]})
        with patch("aiohttp.ClientSession", return_value=_make_session(discovery_resp, token_resp, tags_resp)):
            client = OciRegistryClient(registry_base="https://hub.example.com", token_scope_template="repository:{image}:pull")
            tags = await client.get_tags("deepfellow/doc-chunker-gpu")

        assert tags == ["v1.0.5", "v1.0.4"]

    @pytest.mark.asyncio
    async def test_get_tags_discovers_token_endpoint_without_scope_template(self) -> None:
        # Discovery path (no token_url configured) with no scope template: the scope param
        # must be omitted from the token request entirely.
        discovery_resp = _make_response(
            401, {}, headers={"WWW-Authenticate": 'Bearer realm="https://auth.example.com/token",service="Docker registry"'}
        )
        token_resp = _make_response(200, {"token": "anon-token"})
        tags_resp = _make_response(200, {"tags": ["v1.0.5"]})
        session = _make_session(discovery_resp, token_resp, tags_resp)
        with patch("aiohttp.ClientSession", return_value=session):
            client = OciRegistryClient(registry_base="https://hub.example.com", token_scope_template=None)
            tags = await client.get_tags("deepfellow/doc-chunker-gpu")

        assert tags == ["v1.0.5"]
        token_call_params = session.get.call_args_list[1].kwargs["params"]
        assert "scope" not in token_call_params

    @pytest.mark.asyncio
    async def test_discovered_auth_is_cached_across_calls(self) -> None:
        discovery_resp = _make_response(
            401, {}, headers={"WWW-Authenticate": 'Bearer realm="https://auth.example.com/token",service="Docker registry"'}
        )
        token_resp_1 = _make_response(200, {"token": "anon-token-1"})
        tags_resp_1 = _make_response(200, {"tags": ["v1.0.4"]})
        token_resp_2 = _make_response(200, {"token": "anon-token-2"})
        tags_resp_2 = _make_response(200, {"tags": ["v1.0.5"]})
        session = _make_session(discovery_resp, token_resp_1, tags_resp_1, token_resp_2, tags_resp_2)
        with patch("aiohttp.ClientSession", return_value=session):
            client = OciRegistryClient(registry_base="https://hub.example.com", token_scope_template="repository:{image}:pull")
            await client.get_tags("deepfellow/doc-chunker-gpu")
            await client.get_tags("deepfellow/doc-chunker-gpu")

        # Only 5 requests total (1 discovery + 2 * (token + tags)): the second get_tags() call reused
        # the cached realm/service instead of issuing a second discovery request.
        assert session.get.call_count == 5

    @pytest.mark.asyncio
    async def test_get_tags_discovery_without_service_param(self) -> None:
        # Some registries advertise only a realm, with no "service" attribute in the challenge.
        discovery_resp = _make_response(401, {}, headers={"WWW-Authenticate": 'Bearer realm="https://auth.example.com/token"'})
        token_resp = _make_response(200, {"token": "anon-token"})
        tags_resp = _make_response(200, {"tags": ["v1.0.5"]})
        with patch("aiohttp.ClientSession", return_value=_make_session(discovery_resp, token_resp, tags_resp)):
            client = OciRegistryClient(registry_base="https://hub.example.com", token_scope_template="repository:{image}:pull")
            tags = await client.get_tags("deepfellow/doc-chunker-gpu")

        assert tags == ["v1.0.5"]

    @pytest.mark.asyncio
    async def test_discover_auth_missing_realm_raises(self) -> None:
        # Bearer scheme, but the challenge advertises no realm param at all.
        discovery_resp = _make_response(401, {}, headers={"WWW-Authenticate": "Bearer"})
        with patch("aiohttp.ClientSession", return_value=_make_session(discovery_resp)):
            client = OciRegistryClient(registry_base="https://hub.example.com", token_scope_template="repository:{image}:pull")
            with pytest.raises(RegistryUnavailableError, match="realm"):
                await client.get_tags("deepfellow/doc-chunker-gpu")

    @pytest.mark.asyncio
    async def test_discover_auth_no_challenge_header_raises_unsupported_scheme(self) -> None:
        discovery_resp = _make_response(401, {}, headers={})
        with patch("aiohttp.ClientSession", return_value=_make_session(discovery_resp)):
            client = OciRegistryClient(registry_base="https://hub.example.com", token_scope_template="repository:{image}:pull")
            with pytest.raises(RegistryUnavailableError, match="unsupported auth scheme"):
                await client.get_tags("deepfellow/doc-chunker-gpu")

    @pytest.mark.asyncio
    async def test_get_tags_skips_token_when_registry_allows_anonymous_pulls(self) -> None:
        # A registry with anonymous read access answers 200 on the discovery probe with no
        # WWW-Authenticate challenge at all - tags should be fetched without ever requesting a token.
        discovery_resp = _make_response(200, {}, headers={})
        tags_resp = _make_response(200, {"tags": ["v1.0.5"]})
        session = _make_session(discovery_resp, tags_resp)
        with patch("aiohttp.ClientSession", return_value=session):
            client = OciRegistryClient(registry_base="https://hub.example.com", token_scope_template="repository:{image}:pull")
            tags = await client.get_tags("deepfellow/doc-chunker-gpu")

        assert tags == ["v1.0.5"]
        assert session.get.call_count == 2
        tags_call_headers = session.get.call_args_list[1].kwargs["headers"]
        assert "Authorization" not in tags_call_headers

    @pytest.mark.asyncio
    async def test_discover_auth_basic_scheme_raises_clear_error(self) -> None:
        discovery_resp = _make_response(401, {}, headers={"WWW-Authenticate": 'Basic realm="Registry Realm"'})
        with patch("aiohttp.ClientSession", return_value=_make_session(discovery_resp)):
            client = OciRegistryClient(registry_base="https://hub.example.com", token_scope_template="repository:{image}:pull")
            with pytest.raises(RegistryUnavailableError, match="Basic"):
                await client.get_tags("deepfellow/doc-chunker-gpu")


class TestRegistryFor:
    def test_docker_hub_image(self) -> None:
        client = registry_for("ollama/ollama")
        assert isinstance(client, DockerHubClient)

    def test_ghcr_image(self) -> None:
        client = registry_for("ghcr.io/ggml-org/llama.cpp")
        assert isinstance(client, OciRegistryClient)

    def test_ecr_public_image(self) -> None:
        client = registry_for("public.ecr.aws/q9t5s3a7/vllm-cpu-release-repo")
        assert isinstance(client, OciRegistryClient)

    def test_self_hosted_registry_uses_generic_oci_client(self) -> None:
        client = registry_for("hub.simplito.com/deepfellow/doc-chunker-gpu")
        assert isinstance(client, OciRegistryClient)
        assert not isinstance(client, DockerHubClient)

    def test_self_hosted_registry_client_is_cached_per_host(self) -> None:
        first = registry_for("hub.simplito.com/deepfellow/doc-chunker-gpu")
        second = registry_for("hub.simplito.com/deepfellow/doc-chunker-cpu")
        assert first is second

    @pytest.mark.asyncio
    async def test_docker_hub_token_forwarded(self) -> None:
        hub_resp = _make_response(200, {"results": [{"name": "0.20.4"}], "next": None})
        session = _make_session(hub_resp)
        with patch("aiohttp.ClientSession", return_value=session):
            client = registry_for("ollama/ollama", docker_hub_token="mytoken")
            assert isinstance(client, DockerHubClient)
            await client.get_tags("ollama/ollama")
        call_kwargs = session.get.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer mytoken"


class TestImageWithoutRegistryPrefix:
    def test_ghcr_prefix_stripped(self) -> None:
        assert image_without_registry_prefix("ghcr.io/ggml-org/llama.cpp") == "ggml-org/llama.cpp"

    def test_ecr_prefix_stripped(self) -> None:
        assert image_without_registry_prefix("public.ecr.aws/q9t5s3a7/vllm") == "q9t5s3a7/vllm"

    def test_docker_hub_unchanged(self) -> None:
        assert image_without_registry_prefix("ollama/ollama") == "ollama/ollama"

    def test_self_hosted_prefix_stripped(self) -> None:
        assert image_without_registry_prefix("hub.simplito.com/deepfellow/doc-chunker-gpu") == "deepfellow/doc-chunker-gpu"

    def test_self_hosted_single_segment_repo_prefix_stripped_without_library_namespace(self) -> None:
        # No namespace segment - must not fall back to DockerImageNameInfo.parse()'s Hub-flavored
        # "library" default, which would build a nonexistent /v2/library/myimage/tags/list URL.
        assert image_without_registry_prefix("registry.local/myimage") == "myimage"
