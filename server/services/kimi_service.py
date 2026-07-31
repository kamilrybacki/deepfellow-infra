# SPDX-License-Identifier: MIT

"""Kimi service."""

from server.services.remote_service import DefaultRemoteServiceOptions, RemoteConst, RemoteModel, RemoteService

_const = RemoteConst(
    models={
        # Kimi's API only implements chat completions - no /v1/completions, no /v1/responses, no /v1/messages.
        "kimi-k2.6": RemoteModel(
            type="llm",
            context_length=262_144,
            max_context_length=262_144,
            legacy_completions=False,
            responses=False,
            messages=False,
        ),
        "kimi-k3": RemoteModel(
            type="llm",
            context_length=1_048_576,
            max_context_length=1_048_576,
            legacy_completions=False,
            responses=False,
            messages=False,
        ),
        # Context length inferred from the "256K" tier shared with kimi-k2.6 (confirmed exact at 262_144) -
        # docs never spell out the literal digit for these two models specifically.
        "kimi-k2.7-code": RemoteModel(
            type="llm",
            context_length=262_144,
            max_context_length=262_144,
            legacy_completions=False,
            responses=False,
            messages=False,
        ),
        "kimi-k2.7-code-highspeed": RemoteModel(
            type="llm",
            context_length=262_144,
            max_context_length=262_144,
            legacy_completions=False,
            responses=False,
            messages=False,
        ),
    }
)


class KimiService(RemoteService):
    is_cloud = True
    options_class = DefaultRemoteServiceOptions

    def get_type(self) -> str:
        """Return the service id."""
        return "kimi"

    def get_description(self) -> str:
        """Return the service description."""
        return "Remote access to Kimi models."

    def get_default_url(self) -> str:
        """Return the default url."""
        return "https://api.moonshot.ai"

    def get_models_registry(self) -> RemoteConst:
        """Return the models registry."""
        return _const
