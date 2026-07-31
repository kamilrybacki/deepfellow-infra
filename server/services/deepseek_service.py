# SPDX-License-Identifier: MIT

"""DeepSeek service."""

from server.services.remote_service import DefaultRemoteServiceOptions, RemoteConst, RemoteModel, RemoteService

_const = RemoteConst(
    models={
        # DeepSeek's API only implements chat completions - no /v1/completions (its FIM endpoint lives under a
        # separate /beta base path this proxy can't address per-model), no /v1/responses, no /v1/messages.
        "deepseek-v4-flash": RemoteModel(
            type="llm",
            context_length=1_048_576,
            max_context_length=1_048_576,
            legacy_completions=False,
            responses=False,
            messages=False,
        ),
        "deepseek-v4-pro": RemoteModel(
            type="llm",
            context_length=1_048_576,
            max_context_length=1_048_576,
            legacy_completions=False,
            responses=False,
            messages=False,
        ),
    }
)


class DeepSeekService(RemoteService):
    is_cloud = True
    options_class = DefaultRemoteServiceOptions

    def get_type(self) -> str:
        """Return the service id."""
        return "deepseek"

    def get_description(self) -> str:
        """Return the service description."""
        return "Remote access to DeepSeek models."

    def get_default_url(self) -> str:
        """Return the default url."""
        return "https://api.deepseek.com"

    def get_models_registry(self) -> RemoteConst:
        """Return the models registry."""
        return _const
