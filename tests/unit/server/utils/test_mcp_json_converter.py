# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for parsing a standard MCP client JSON config into DeepFellow's custom-model parameters."""

import pytest

from server.services.mcp_service import McpUserVariant
from server.utils.mcp_json_converter import McpJsonConvertError, detect_runtime, parse_mcp_json_config


def test_parse_proxy_config_with_server_url() -> None:
    result = parse_mcp_json_config({"mcpServers": {"deepwiki": {"serverUrl": "https://mcp.deepwiki.com/mcp"}}})

    assert result.kind == "proxy"
    assert result.name == "deepwiki"
    assert result.server_url == "https://mcp.deepwiki.com/mcp"
    assert result.transport == "streamable_http"
    assert result.headers == {}
    assert result.oauth is None


def test_parse_proxy_config_accepts_bare_url_field_and_no_mcp_servers_wrapper() -> None:
    result = parse_mcp_json_config({"deepwiki": {"url": "https://mcp.deepwiki.com/sse"}})

    assert result.kind == "proxy"
    assert result.server_url == "https://mcp.deepwiki.com/sse"
    assert result.transport == "sse"


def test_parse_proxy_config_explicit_transport_overrides_url_suffix() -> None:
    result = parse_mcp_json_config({"mcpServers": {"remote": {"serverUrl": "https://example.com/mcp", "transport": "sse"}}})

    assert result.kind == "proxy"
    assert result.transport == "sse"


def test_parse_proxy_config_collects_string_headers_only() -> None:
    result = parse_mcp_json_config(
        {"mcpServers": {"remote": {"serverUrl": "https://example.com/mcp", "headers": {"Authorization": "Bearer x", "n": 1}}}}
    )

    assert result.kind == "proxy"
    assert result.headers == {"Authorization": "Bearer x"}


def test_parse_proxy_config_extracts_oauth_camel_case_aliases() -> None:
    result = parse_mcp_json_config(
        {"mcpServers": {"remote": {"serverUrl": "https://example.com/mcp", "oauth": {"clientId": "abc", "scope": "tools:read"}}}}
    )

    assert result.kind == "proxy"
    assert result.oauth is not None
    assert result.oauth.client_id == "abc"
    assert result.oauth.scope == "tools:read"


def test_parse_proxy_config_empty_server_url_raises() -> None:
    with pytest.raises(McpJsonConvertError):
        parse_mcp_json_config({"mcpServers": {"remote": {"serverUrl": ""}}})


def test_parse_stdio_config_detects_node_runtime() -> None:
    result = parse_mcp_json_config(
        {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"],
                    "env": {"FOO": "bar"},
                }
            }
        }
    )

    assert result.kind == "user"
    assert result.name == "filesystem"
    assert result.command == "npx -y @modelcontextprotocol/server-filesystem /data"
    assert result.envs == {"FOO": "bar"}
    assert result.variant == McpUserVariant.node_headless


def test_parse_stdio_config_detects_python_runtime() -> None:
    result = parse_mcp_json_config({"mcpServers": {"srv": {"command": "uvx", "args": ["some-mcp-server"]}}})

    assert result.kind == "user"
    assert result.variant == McpUserVariant.python_headless


def test_parse_stdio_config_unknown_runtime_has_no_variant() -> None:
    result = parse_mcp_json_config({"mcpServers": {"srv": {"command": "/usr/local/bin/my-server"}}})

    assert result.kind == "user"
    assert result.variant is None


def test_parse_stdio_config_quotes_args_with_special_characters() -> None:
    result = parse_mcp_json_config({"mcpServers": {"srv": {"command": "npx", "args": ["--flag", "hello world"]}}})

    assert result.kind == "user"
    assert result.command == "npx --flag 'hello world'"


def test_parse_stdio_config_strips_windows_cmd_c_wrapper() -> None:
    result = parse_mcp_json_config({"mcpServers": {"srv": {"command": "cmd", "args": ["/c", "npx", "-y", "server"]}}})

    assert result.kind == "user"
    assert result.command == "npx -y server"


def test_parse_stdio_config_missing_command_raises() -> None:
    with pytest.raises(McpJsonConvertError):
        parse_mcp_json_config({"mcpServers": {"srv": {}}})


def test_parse_docker_run_config_extracts_image_command_and_volumes() -> None:
    result = parse_mcp_json_config(
        {
            "mcpServers": {
                "fs": {
                    "command": "docker",
                    "args": [
                        "run",
                        "-i",
                        "--rm",
                        "-v",
                        "/host/data:/data",
                        "-e",
                        "FOO=bar",
                        "mcp/filesystem",
                        "/data",
                    ],
                }
            }
        }
    )

    assert result.kind == "custom"
    assert result.image == "mcp/filesystem"
    assert result.command == "/data"
    assert result.volumes == ["/host/data:/data"]


def test_parse_docker_run_config_supports_mount_flag() -> None:
    result = parse_mcp_json_config(
        {
            "mcpServers": {
                "fs": {
                    "command": "docker",
                    "args": [
                        "run",
                        "--mount",
                        "type=bind,src=/host,dst=/container,readonly=true",
                        "mcp/filesystem",
                    ],
                }
            }
        }
    )

    assert result.kind == "custom"
    assert result.volumes == ["/host:/container:ro"]


def test_parse_docker_run_config_without_image_raises() -> None:
    with pytest.raises(McpJsonConvertError):
        parse_mcp_json_config({"mcpServers": {"fs": {"command": "docker", "args": ["run", "-i", "--rm"]}}})


def test_parse_docker_run_config_recognizes_restart_as_value_flag() -> None:
    result = parse_mcp_json_config(
        {"mcpServers": {"fs": {"command": "docker", "args": ["run", "--restart", "unless-stopped", "mcp/filesystem"]}}}
    )

    assert result.kind == "custom"
    assert result.image == "mcp/filesystem"
    assert result.command == ""


def test_parse_docker_run_config_unknown_flag_raises_instead_of_guessing() -> None:
    with pytest.raises(McpJsonConvertError):
        parse_mcp_json_config({"mcpServers": {"fs": {"command": "docker", "args": ["run", "--some-unknown-flag", "mcp/filesystem"]}}})


def test_parse_docker_run_config_volume_flag_without_value_at_end_raises() -> None:
    with pytest.raises(McpJsonConvertError):
        parse_mcp_json_config({"mcpServers": {"fs": {"command": "docker", "args": ["run", "-i", "-v"]}}})


def test_parse_docker_run_config_volume_flag_with_empty_inline_value_is_ignored() -> None:
    result = parse_mcp_json_config({"mcpServers": {"fs": {"command": "docker", "args": ["run", "-v=", "mcp/filesystem"]}}})

    assert result.kind == "custom"
    assert result.image == "mcp/filesystem"
    assert result.volumes == []


def test_parse_docker_run_config_mount_flag_rejects_non_bind_type() -> None:
    result = parse_mcp_json_config(
        {"mcpServers": {"fs": {"command": "docker", "args": ["run", "--mount", "type=volume,src=/host,dst=/data", "mcp/filesystem"]}}}
    )

    assert result.kind == "custom"
    assert result.volumes == []


def test_parse_docker_run_config_mount_flag_without_readonly() -> None:
    result = parse_mcp_json_config(
        {"mcpServers": {"fs": {"command": "docker", "args": ["run", "--mount", "src=/host,dst=/data", "mcp/filesystem"]}}}
    )

    assert result.kind == "custom"
    assert result.volumes == ["/host:/data"]


def test_parse_docker_run_config_mount_flag_missing_destination() -> None:
    result = parse_mcp_json_config({"mcpServers": {"fs": {"command": "docker", "args": ["run", "--mount", "src=/host", "mcp/filesystem"]}}})

    assert result.kind == "custom"
    assert result.volumes == []


def test_parse_docker_run_config_mount_flag_ignores_segment_without_equals() -> None:
    result = parse_mcp_json_config(
        {"mcpServers": {"fs": {"command": "docker", "args": ["run", "--mount", "bogus,src=/host,dst=/data", "mcp/filesystem"]}}}
    )

    assert result.kind == "custom"
    assert result.volumes == ["/host:/data"]


def test_parse_docker_run_config_mount_flag_with_empty_inline_value_is_ignored() -> None:
    result = parse_mcp_json_config({"mcpServers": {"fs": {"command": "docker", "args": ["run", "--mount=", "mcp/filesystem"]}}})

    assert result.kind == "custom"
    assert result.image == "mcp/filesystem"
    assert result.volumes == []


def test_parse_proxy_config_non_string_server_url_raises() -> None:
    with pytest.raises(McpJsonConvertError):
        parse_mcp_json_config({"mcpServers": {"remote": {"serverUrl": 123}}})


def test_parse_stdio_config_cmd_c_wrapper_with_no_command_left_raises() -> None:
    with pytest.raises(McpJsonConvertError):
        parse_mcp_json_config({"mcpServers": {"srv": {"command": "cmd", "args": ["/c"]}}})


def test_parse_config_no_servers_raises() -> None:
    with pytest.raises(McpJsonConvertError):
        parse_mcp_json_config({"mcpServers": {}})


def test_parse_config_invalid_server_entry_raises() -> None:
    with pytest.raises(McpJsonConvertError):
        parse_mcp_json_config({"mcpServers": {"srv": "not-an-object"}})


@pytest.mark.parametrize(
    ("cmd", "expected"),
    [
        ("npx -y @modelcontextprotocol/server-filesystem", "node"),
        ("uvx some-server", "python"),
        ("python3 -m server", "python"),
        ("/usr/local/bin/custom-server", None),
        ("", None),
    ],
)
def test_detect_runtime(cmd: str, expected: str | None) -> None:
    assert detect_runtime(cmd) == expected
