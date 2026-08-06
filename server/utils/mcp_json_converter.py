# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Convert a standard MCP client JSON config into DeepFellow's custom-model parameters.

This mirrors the parsing previously done client-side in the WebUI's "Auto-Import" tab
(`AddMcpServerModal.tsx`'s `parseMcpJsonConfig`), so it can be reused from the CLI and any other
client without reimplementing it.
"""

import shlex
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from server.services.mcp_service import McpUserVariant

# Keep in sync with PYTHON_BINS/NODE_BINS in webui/src/hooks/use-mcp-server-form.ts — duplicated
# there so the WebUI can auto-detect the variant locally while the user is still typing a manual
# command, without a round-trip to this endpoint on every keystroke.
_PYTHON_BINS = {"uvx", "python", "python3", "uv", "pipx"}
_NODE_BINS = {"node", "npx", "npm", "yarn", "pnpm", "bunx", "bun", "deno"}

# docker run flags that consume a following value argument. Flags NOT listed here (nor in
# _DOCKER_BOOLEAN_FLAGS below) are treated as unsupported and raise, rather than guessing —
# guessing wrong in either direction silently corrupts the image/command split (see
# _parse_docker_run_args).
_DOCKER_VALUE_FLAGS = {
    "-v",
    "--volume",
    "-e",
    "--env",
    "--env-file",
    "--name",
    "-p",
    "--publish",
    "--expose",
    "--network",
    "--net",
    "--network-alias",
    "--hostname",
    "-h",
    "-u",
    "--user",
    "-w",
    "--workdir",
    "--entrypoint",
    "-l",
    "--label",
    "--label-file",
    "--add-host",
    "--dns",
    "--dns-option",
    "--dns-opt",
    "--dns-search",
    "--domainname",
    "--link",
    "-m",
    "--memory",
    "--memory-swap",
    "--memory-reservation",
    "--memory-swappiness",
    "--kernel-memory",
    "--cpus",
    "--cpu-period",
    "--cpu-quota",
    "--cpu-rt-period",
    "--cpu-rt-runtime",
    "--cpu-shares",
    "--cpuset-cpus",
    "--cpuset-mems",
    "--blkio-weight",
    "--blkio-weight-device",
    "--device-read-bps",
    "--device-write-bps",
    "--device-read-iops",
    "--device-write-iops",
    "--device-cgroup-rule",
    "--runtime",
    "--platform",
    "--pull",
    "--ulimit",
    "--security-opt",
    "--cap-add",
    "--cap-drop",
    "--device",
    "--cidfile",
    "--volumes-from",
    "--volume-driver",
    "--storage-opt",
    "--log-driver",
    "--log-opt",
    "--health-cmd",
    "--health-interval",
    "--health-retries",
    "--health-timeout",
    "--health-start-period",
    "--health-start-interval",
    "--mount",
    "--tmpfs",
    "--shm-size",
    "--ipc",
    "--pid",
    "--uts",
    "--userns",
    "--cgroupns",
    "--cgroup-parent",
    "--isolation",
    "--restart",
    "--stop-signal",
    "--stop-timeout",
    "--gpus",
    "--group-add",
    "--mac-address",
    "--ip",
    "--ip6",
    "--link-local-ip",
    "--sysctl",
    "--annotation",
    "--pids-limit",
    "--oom-score-adj",
    "-a",
    "--attach",
    "--detach-keys",
}

# docker run flags that never take a value. Anything not in either set raises rather than being
# guessed, since a guessed-wrong flag silently misassigns the image name / container command.
_DOCKER_BOOLEAN_FLAGS = {
    "-d",
    "--detach",
    "-i",
    "--interactive",
    "-t",
    "--tty",
    "--rm",
    "--privileged",
    "--read-only",
    "--init",
    "-P",
    "--publish-all",
    "--oom-kill-disable",
    "--disable-content-trust",
    "-q",
    "--quiet",
    "--quiet-pull",
    "--sig-proxy",
    "--no-healthcheck",
}


class McpJsonConvertError(ValueError):
    """The input JSON is not a valid/supported MCP client config."""


class ConvertedMcpOAuth(BaseModel):
    client_id: str | None = None
    client_secret: str | None = None
    scope: str | None = None


class ConvertedUserMcpConfig(BaseModel):
    """kind == "user": stdio server, launched via the runtime bridge."""

    kind: Literal["user"] = "user"
    name: str
    command: str
    envs: dict[str, str]
    variant: McpUserVariant | None = None


class ConvertedProxyMcpConfig(BaseModel):
    """kind == "proxy": remote server, connected to directly."""

    kind: Literal["proxy"] = "proxy"
    name: str
    server_url: str
    transport: Literal["streamable_http", "sse"]
    headers: dict[str, str]
    oauth: ConvertedMcpOAuth | None = None


class ConvertedCustomMcpConfig(BaseModel):
    """kind == "custom": an existing `docker run`-style image."""

    kind: Literal["custom"] = "custom"
    name: str
    image: str
    command: str
    volumes: list[str]
    envs: dict[str, str]


ConvertedMcpConfig = Annotated[
    ConvertedUserMcpConfig | ConvertedProxyMcpConfig | ConvertedCustomMcpConfig,
    Field(discriminator="kind"),
]


def detect_runtime(cmd: str) -> Literal["python", "node"] | None:
    """Guess the runtime family from a command's binary name."""
    cmd = cmd.strip()
    if not cmd:
        return None
    base = cmd.split()[0].rsplit("/", 1)[-1].lower()
    if base in _PYTHON_BINS:
        return "python"
    if base in _NODE_BINS:
        return "node"
    return None


def _parse_mount_flag(val: str) -> str | None:
    """Convert a `--mount type=bind,src=...,dst=...[,readonly]` value into `-v`-style `src:dst[:ro]`.

    Returns None for non-bind mounts (e.g. `type=volume`, `type=tmpfs`) or if src/dst are missing,
    since those can't be expressed as a plain volume string.
    """
    fields = dict(segment.split("=", 1) for segment in val.split(",") if "=" in segment)

    mount_type = fields.get("type")
    if mount_type and mount_type != "bind":
        return None

    src = fields.get("src") or fields.get("source") or ""
    dst = fields.get("dst") or fields.get("destination") or fields.get("target") or ""
    if not src or not dst:
        return None

    is_readonly = fields.get("ro") == "true" or fields.get("readonly") == "true"
    return f"{src}:{dst}:ro" if is_readonly else f"{src}:{dst}"


def _flag_value(args: list[str], index: int, inline_value: str | None) -> tuple[str | None, int]:
    """Resolve a flag's value (either `--flag=value` or a separate `--flag value` token).

    Returns the value (or None if there isn't one) and the index of the next unconsumed arg.
    """
    if inline_value is not None:
        return inline_value, index + 1
    if index + 1 < len(args):
        return args[index + 1], index + 2
    return None, index + 1


def _parse_docker_run_args(args: list[str]) -> tuple[str, list[str], list[str]]:
    """Parse `docker run [flags...] image [cmd...]` into (image, cmd, volumes).

    Walks the argument list flag by flag, since flags (and their values, for the ones that take
    one) must be skipped to find where the image name starts. `-v`/`--volume` and `--mount` are
    additionally collected into `volumes`, normalized to `-v`-style `src:dst[:ro]` strings.
    """
    volumes: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]

        if not token.startswith("-"):
            # First non-flag argument is the image name; everything after it is the container command.
            return token, args[index + 1 :], volumes

        eq_pos = token.find("=")
        flag_name = token if eq_pos == -1 else token[:eq_pos]
        inline_value = None if eq_pos == -1 else token[eq_pos + 1 :]

        if flag_name in ("-v", "--volume"):
            volume, index = _flag_value(args, index, inline_value)
            if volume:
                volumes.append(volume)
        elif flag_name == "--mount":
            mount_spec, index = _flag_value(args, index, inline_value)
            mount = _parse_mount_flag(mount_spec) if mount_spec else None
            if mount:
                volumes.append(mount)
        elif flag_name in _DOCKER_VALUE_FLAGS and inline_value is None:
            # A flag with a value in a separate token (e.g. `-e FOO=bar`): skip both.
            _, index = _flag_value(args, index, inline_value)
        elif flag_name in _DOCKER_VALUE_FLAGS or flag_name in _DOCKER_BOOLEAN_FLAGS or inline_value is not None:
            # A known boolean flag with no value to skip, or a value already inlined via `=`
            # (unambiguous regardless of whether the flag is recognized).
            index += 1
        else:
            message = (
                f'Unsupported docker run flag "{flag_name}": cannot tell whether it takes a value, '
                "so the image name can't be reliably located."
            )
            raise McpJsonConvertError(message)

    raise McpJsonConvertError("Could not find image name in docker run arguments.")


def _parse_oauth_field(raw: object) -> ConvertedMcpOAuth | None:
    if not isinstance(raw, dict):
        return None
    client_id = raw.get("client_id") or raw.get("clientId")
    client_secret = raw.get("client_secret") or raw.get("clientSecret")
    scope = raw.get("scope")
    oauth = ConvertedMcpOAuth(
        client_id=client_id if isinstance(client_id, str) else None,
        client_secret=client_secret if isinstance(client_secret, str) else None,
        scope=scope if isinstance(scope, str) else None,
    )
    return oauth if (oauth.client_id or oauth.client_secret or oauth.scope) else None


def _string_map(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if isinstance(v, str)}


def _parse_proxy_config(name: str, server_url: object, server_config: dict[str, Any]) -> ConvertedProxyMcpConfig:
    if not isinstance(server_url, str) or not server_url:
        raise McpJsonConvertError('"serverUrl" or "url" must be a non-empty string.')
    raw_transport = server_config.get("transport")
    is_sse = raw_transport == "sse" or server_url.endswith("/sse")
    transport: Literal["streamable_http", "sse"] = "sse" if is_sse else "streamable_http"
    return ConvertedProxyMcpConfig(
        name=name,
        server_url=server_url,
        transport=transport,
        headers=_string_map(server_config.get("headers")),
        oauth=_parse_oauth_field(server_config.get("oauth")),
    )


def _strip_cmd_c(cmd: str, arg_list: list[str]) -> tuple[str, list[str]]:
    """Unwrap Windows-style `cmd /c <real command> [args...]` down to the real command."""
    if cmd.lower() != "cmd" or arg_list[:1] != ["/c"]:
        return cmd, arg_list
    arg_list = arg_list[1:]
    cmd = arg_list.pop(0) if arg_list else ""
    if not cmd:
        raise McpJsonConvertError('Empty command after stripping "cmd /c".')
    return cmd, arg_list


def _detect_variant(cmd: str) -> McpUserVariant | None:
    runtime = detect_runtime(cmd)
    if runtime == "python":
        return McpUserVariant.python_headless
    if runtime == "node":
        return McpUserVariant.node_headless
    return None


def parse_mcp_json_config(
    config: dict[str, Any],
) -> ConvertedUserMcpConfig | ConvertedProxyMcpConfig | ConvertedCustomMcpConfig:
    """Parse a standard `{"mcpServers": {...}}` (or bare `{name: config}`) MCP client config.

    Returns the normalized parameters for the first server entry found.
    """
    servers = config.get("mcpServers")
    entries = list((servers if isinstance(servers, dict) else config).items())
    if not entries:
        raise McpJsonConvertError("No servers found in config.")

    name, server_config = entries[0]
    if not isinstance(server_config, dict):
        raise McpJsonConvertError("Invalid server config.")

    server_url = server_config.get("serverUrl") or server_config.get("url")
    if server_url:
        return _parse_proxy_config(name, server_url, server_config)

    raw_command = server_config.get("command")
    if not isinstance(raw_command, str) or not raw_command:
        raise McpJsonConvertError('Missing "command" field.')

    raw_args = server_config.get("args")
    arg_list: list[str] = [str(a) for a in raw_args] if isinstance(raw_args, list) else []
    cmd, arg_list = _strip_cmd_c(raw_command, arg_list)

    envs = _string_map(server_config.get("env"))

    if cmd == "docker" and arg_list[:1] == ["run"]:
        image, cmd_args, volumes = _parse_docker_run_args(arg_list[1:])
        return ConvertedCustomMcpConfig(
            name=name,
            image=image,
            command=" ".join(shlex.quote(a) for a in cmd_args),
            volumes=volumes,
            envs=envs,
        )

    return ConvertedUserMcpConfig(
        name=name,
        command=" ".join(shlex.quote(a) for a in [cmd, *arg_list]),
        envs=envs,
        variant=_detect_variant(cmd),
    )
