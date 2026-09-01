# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""OpenAI service."""

import json
import logging
from collections.abc import Sequence
from urllib.parse import urljoin

import aiohttp

from server.services.remote_service import DefaultRemoteServiceOptions, LiveModelEntry, RemoteConst, RemoteModel, RemoteService

logger = logging.getLogger("uvicorn.error")

_FETCH_TIMEOUT = aiohttp.ClientTimeout(total=15)

_const = RemoteConst(
    models={
        "davinci-002": RemoteModel(
            type="llm", context_length=16_385, max_context_length=16_385, completions=False, responses=False, messages=False
        ),
        "babbage-002": RemoteModel(
            type="llm", context_length=16_385, max_context_length=16_385, completions=False, responses=False, messages=False
        ),
        "gpt-3.5-turbo": RemoteModel(type="llm", context_length=16_385, max_context_length=16_385, messages=False),
        "gpt-3.5-turbo-instruct": RemoteModel(
            type="llm", context_length=4_096, max_context_length=4_096, completions=False, responses=False, messages=False
        ),
        "gpt-3.5-turbo-16k": RemoteModel(
            type="llm", context_length=16_385, max_context_length=16_385, legacy_completions=False, responses=False, messages=False
        ),
        "gpt-4-turbo": RemoteModel(type="llm", context_length=128_000, max_context_length=128_000, messages=False),
        "gpt-4.1": RemoteModel(type="llm", context_length=1_048_576, max_context_length=1_048_576, messages=False),
        "gpt-4.1-mini": RemoteModel(type="llm", context_length=1_048_576, max_context_length=1_048_576, messages=False),
        "gpt-4.1-nano": RemoteModel(type="llm", context_length=1_048_576, max_context_length=1_048_576, messages=False),
        "gpt-4o": RemoteModel(type="llm", context_length=128_000, max_context_length=128_000, messages=False),
        "gpt-4o-mini": RemoteModel(type="llm", context_length=128_000, max_context_length=128_000, messages=False),
        "gpt-4o-transcribe": RemoteModel(type="stt"),
        "gpt-4o-mini-transcribe": RemoteModel(type="stt"),
        "gpt-4o-mini-tts": RemoteModel(type="tts"),
        "gpt-4": RemoteModel(type="llm", context_length=8_196, max_context_length=8_196, messages=False),
        "gpt-5": RemoteModel(type="llm", context_length=400_000, max_context_length=400_000, messages=False),
        "gpt-5-mini": RemoteModel(type="llm", context_length=400_000, max_context_length=400_000, messages=False),
        "gpt-5-nano": RemoteModel(type="llm", context_length=400_000, max_context_length=400_000, messages=False),
        "gpt-5.1": RemoteModel(type="llm", context_length=400_000, max_context_length=400_000, messages=False),
        "gpt-5.1-codex": RemoteModel(
            type="llm",
            context_length=400_000,
            max_context_length=400_000,
            completions=False,
            legacy_completions=False,
            responses=True,
            messages=False,
        ),
        "gpt-5.1-codex-mini": RemoteModel(
            type="llm",
            context_length=400_000,
            max_context_length=400_000,
            completions=False,
            legacy_completions=False,
            responses=True,
            messages=False,
        ),
        "gpt-5.2": RemoteModel(type="llm", context_length=400_000, max_context_length=400_000, messages=False),
        "gpt-5.2-pro": RemoteModel(
            type="llm",
            context_length=400_000,
            max_context_length=400_000,
            completions=False,
            legacy_completions=False,
            responses=True,
            messages=False,
        ),
        "gpt-5.4": RemoteModel(type="llm", context_length=400_000, max_context_length=400_000, messages=False),
        "gpt-5.4-mini": RemoteModel(type="llm", context_length=400_000, max_context_length=400_000, messages=False),
        "gpt-5.4-nano": RemoteModel(type="llm", context_length=400_000, max_context_length=400_000, messages=False),
        "gpt-5.4-pro": RemoteModel(type="llm", context_length=1_050_000, max_context_length=1_050_000, messages=False),
        "gpt-5.5": RemoteModel(type="llm", context_length=1_050_000, max_context_length=1_050_000, messages=False),
        "gpt-5.5-pro": RemoteModel(type="llm", context_length=1_050_000, max_context_length=1_050_000, messages=False),
        "gpt-5.6-sol": RemoteModel(type="llm", context_length=1_050_000, max_context_length=1_050_000, messages=False),
        "gpt-5.6-terra": RemoteModel(type="llm", context_length=1_050_000, max_context_length=1_050_000, messages=False),
        "gpt-5.6-luna": RemoteModel(type="llm", context_length=1_050_000, max_context_length=1_050_000, messages=False),
        "o1": RemoteModel(type="llm", messages=False),
        "o1-pro": RemoteModel(
            type="llm",
            context_length=200_000,
            max_context_length=200_000,
            completions=False,
            legacy_completions=False,
            responses=True,
            messages=False,
        ),
        "o3": RemoteModel(type="llm", context_length=200_000, max_context_length=200_000, messages=False),
        "o3-mini": RemoteModel(type="llm", context_length=200_000, max_context_length=200_000, messages=False),
        "o4-mini": RemoteModel(type="llm", context_length=200_000, max_context_length=200_000, messages=False),
        "o4-mini-deep-research": RemoteModel(
            type="llm",
            context_length=200_000,
            max_context_length=200_000,
            completions=False,
            legacy_completions=False,
            responses=True,
            messages=False,
        ),
        "text-embedding-ada-002": RemoteModel(type="embedding"),
        "text-embedding-3-small": RemoteModel(type="embedding"),
        "text-embedding-3-large": RemoteModel(type="embedding"),
        "gpt-image-1": RemoteModel(type="txt2img"),
        "dall-e-2": RemoteModel(type="txt2img"),
        "dall-e-3": RemoteModel(type="txt2img"),
        "tts-1": RemoteModel(type="tts"),
        "tts-1-hd": RemoteModel(type="tts"),
        "whisper-1": RemoteModel(type="stt"),
    }
)


class OpenAIService(RemoteService):
    is_cloud = True
    options_class = DefaultRemoteServiceOptions

    def get_type(self) -> str:
        """Return the service id."""
        return "openai"

    def get_description(self) -> str:
        """Return the service description."""
        return "Remote access to OpenAI models."

    def get_default_url(self) -> str:
        """Return the default url."""
        return "https://api.openai.com"

    def get_models_registry(self) -> RemoteConst:
        """Return the models registry."""
        return _const

    async def _fetch_live_models(self, instance: str) -> Sequence[LiveModelEntry] | None:
        """Fetch the live model listing from OpenAI's `GET /v1/models`. Returns None on any failure."""
        try:
            info = self.get_instance_installed_info(instance)
            api_url = info.parsed_options.api_url
            api_key = info.parsed_options.api_key
            headers = {"Authorization": f"Bearer {api_key}"}
            models_url = urljoin(urljoin(api_url, self.api_version), "models")

            async with (
                aiohttp.ClientSession(timeout=_FETCH_TIMEOUT) as session,
                session.get(models_url, headers=headers) as response,
            ):
                if response.status != 200:
                    logger.warning("%s live model listing failed for instance %r: HTTP %d", self.get_type(), instance, response.status)
                    return None
                body = json.loads(await response.text())
                # Ollama's OpenAI-compatible /v1/models reports an empty catalog as
                # {"data": null} rather than {"data": []} — treat that as zero models,
                # not a malformed response.
                data = body["data"] if body["data"] is not None else []
                return [LiveModelEntry(id=model["id"]) for model in data]
        except Exception:
            logger.exception("%s live model listing failed for instance %r", self.get_type(), instance)
            return None
