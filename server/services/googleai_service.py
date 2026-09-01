# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""GoogleAI service."""

import json
import logging
from collections.abc import Sequence
from urllib.parse import urljoin

import aiohttp

from server.services.remote_service import (
    DefaultRemoteServiceOptions,
    LiveModelEntry,
    RemoteConst,
    RemoteModel,
    RemoteModelType,
    RemoteService,
)

logger = logging.getLogger("uvicorn.error")

_MAX_PAGES = 20
_FETCH_TIMEOUT = aiohttp.ClientTimeout(total=15)

# Google's native `GET /v1beta/models` reports each model's supported RPCs rather than a
# DeepFellow-style type -- map the ones we care about. A model backing several of these (or none)
# resolves to whichever comes first here; unmatched methods fall back to the RemoteConst overlay.
_GENERATION_METHOD_TYPE: dict[str, RemoteModelType] = {
    "generateContent": "llm",
    "embedContent": "embedding",
    "predict": "txt2img",
}

_const = RemoteConst(
    models={
        "gemini-1.5-flash-latest": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-1.5-flash": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-1.5-flash-002": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-1.5-flash-8b": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-1.5-flash-8b-001": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-1.5-flash-8b-latest": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-1.5-pro-latest": RemoteModel(
            type="llm", context_length=2_097_152, max_context_length=2_097_152, responses=False, messages=False
        ),
        "gemini-1.5-pro-002": RemoteModel(
            type="llm", context_length=2_097_152, max_context_length=2_097_152, responses=False, messages=False
        ),
        "gemini-1.5-pro": RemoteModel(type="llm", context_length=2_097_152, max_context_length=2_097_152, responses=False, messages=False),
        "gemini-2.0-flash-exp": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-2.0-flash": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-2.0-flash-001": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-2.0-flash-lite-001": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-2.0-flash-lite": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-2.0-flash-lite-preview-02-05": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-2.0-flash-lite-preview": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-2.0-flash-thinking-exp-01-21": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-2.0-flash-thinking-exp": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-2.0-flash-thinking-exp-1219": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-2.0-pro-exp": RemoteModel(
            type="llm", context_length=2_097_152, max_context_length=2_097_152, responses=False, messages=False
        ),
        "gemini-2.0-pro-exp-02-05": RemoteModel(
            type="llm", context_length=2_097_152, max_context_length=2_097_152, responses=False, messages=False
        ),
        "gemini-2.5-pro-preview-03-25": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-2.5-flash-preview-05-20": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-2.5-flash": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-2.5-flash-lite": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-2.5-flash-lite-preview-06-17": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-2.5-flash-image-preview": RemoteModel(type="txt2img"),  # NOTE: aka Nano Banana
        "gemini-2.5-pro-preview-05-06": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-2.5-pro-preview-06-05": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False
        ),
        "gemini-2.5-pro": RemoteModel(type="llm", context_length=1_048_576, max_context_length=1_048_576, responses=False, messages=False),
        "gemini-3.1-pro-preview": RemoteModel(type="llm", context_length=1_048_576, max_context_length=1_048_576, messages=False),
        "gemini-3.1-pro-preview-customtools": RemoteModel(
            type="llm", context_length=1_048_576, max_context_length=1_048_576, messages=False
        ),
        "gemini-3-flash-preview": RemoteModel(type="llm", context_length=1_048_576, max_context_length=1_048_576, messages=False),
        "gemini-3.1-flash-lite": RemoteModel(type="llm", context_length=1_048_576, max_context_length=1_048_576, messages=False),
        "gemini-3.5-flash": RemoteModel(type="llm", context_length=1_048_576, max_context_length=1_048_576, messages=False),
        "gemini-3.5-flash-lite": RemoteModel(type="llm", context_length=1_048_576, max_context_length=1_048_576, messages=False),
        "gemini-3.6-flash": RemoteModel(type="llm", context_length=1_048_576, max_context_length=1_048_576, messages=False),
        "gemini-3.1-flash-image": RemoteModel(type="txt2img"),  # NOTE: aka Nano Banana 2
        "gemini-3.1-flash-lite-image": RemoteModel(type="txt2img"),  # NOTE: aka Nano Banana 2 Lite
        "gemini-3-pro-image": RemoteModel(type="txt2img"),  # NOTE: aka Nano Banana Pro
        "gemini-exp-1206": RemoteModel(type="llm", context_length=2_097_152, max_context_length=2_097_152, responses=False, messages=False),
        "gemma-3-1b-it": RemoteModel(type="llm", context_length=32_768, max_context_length=32_768, responses=False, messages=False),
        "gemma-3-4b-it": RemoteModel(type="llm", context_length=131_072, max_context_length=131_072, responses=False, messages=False),
        "gemma-3-12b-it": RemoteModel(type="llm", context_length=131_072, max_context_length=131_072, responses=False, messages=False),
        "gemma-3-27b-it": RemoteModel(type="llm", context_length=131_072, max_context_length=131_072, responses=False, messages=False),
        "gemma-3n-e4b-it": RemoteModel(type="llm", context_length=32_768, max_context_length=32_768, responses=False, messages=False),
        "gemma-3n-e2b-it": RemoteModel(type="llm", context_length=32_768, max_context_length=32_768, responses=False, messages=False),
        # Gemma 4 - small variants 128K, medium variants (26B/31B) 256K per Google's own Gemma docs
        "gemma-4-26b-a4b-it": RemoteModel(type="llm", context_length=262_144, max_context_length=262_144, responses=False, messages=False),
        "gemma-4-31b-it": RemoteModel(type="llm", context_length=262_144, max_context_length=262_144, responses=False, messages=False),
        "embedding-001": RemoteModel(type="embedding"),
        "text-embedding-004": RemoteModel(type="embedding"),
        "gemini-embedding-exp-03-07": RemoteModel(type="embedding"),
        "gemini-embedding-exp": RemoteModel(type="embedding"),
        "gemini-embedding-001": RemoteModel(type="embedding"),
        "gemini-embedding-2": RemoteModel(type="embedding"),
        "imagen-3.0-generate-002": RemoteModel(type="txt2img"),
        "imagen-4.0-generate-preview-06-06": RemoteModel(type="txt2img"),
        "imagen-4.0-ultra-generate-preview-06-06": RemoteModel(type="txt2img"),
        "imagen-4.0-standard-generate-001": RemoteModel(type="txt2img"),
        "imagen-4.0-ultra-generate-001": RemoteModel(type="txt2img"),
        "learnlm-2.0-flash-experimental": RemoteModel(type="llm", context_length=1_048_576, responses=False, messages=False),
        # TTS Models are not open https://cloud.google.com/text-to-speech/docs/gemini-tts#curl ; speech is not listed here: https://ai.google.dev/gemini-api/docs/openai
        # "gemini-2.5-flash-preview-tts": RemoteModel(type="tts"),
        # "gemini-2.5-pro-preview-tts": RemoteModel(type="tts"),
    }
)


class GoogleAIService(RemoteService):
    is_cloud = True
    api_version = "v1beta/openai/"
    options_class = DefaultRemoteServiceOptions

    def get_type(self) -> str:
        """Return the service id."""
        return "google"

    def get_description(self) -> str:
        """Return the service description."""
        return "Remote access to Google AI models."

    def get_default_url(self) -> str:
        """Return the default url."""
        return "https://generativelanguage.googleapis.com"

    def get_models_registry(self) -> RemoteConst:
        """Return the models registry."""
        return _const

    async def _fetch_live_models(self, instance: str) -> Sequence[LiveModelEntry] | None:
        """Fetch the live model listing from Google AI's native, paginated `GET /v1beta/models`.

        Distinct from the OpenAI-compatible `v1beta/openai/` proxy path this service uses for chat
        (`api_version`) — Google's native listing has its own response shape (`models[].name` as
        `"models/<id>"`, `nextPageToken` pagination). Returns None on any failure, discarding any
        entries already accumulated from earlier pages, so a partial/incomplete listing never
        silently overwrites the hardcoded catalog.
        """
        entries: list[LiveModelEntry] = []

        try:
            info = self.get_instance_installed_info(instance)
            api_url = info.parsed_options.api_url
            # Google's native endpoint (unlike the OpenAI-compat proxy path this service uses for
            # chat) expects the key via x-goog-api-key, not an Authorization: Bearer header.
            headers = {"x-goog-api-key": info.parsed_options.api_key}
            models_url = urljoin(f"{api_url}/", "v1beta/models")

            page_token: str | None = None
            async with aiohttp.ClientSession(timeout=_FETCH_TIMEOUT) as session:
                for _ in range(_MAX_PAGES):
                    params = {"pageToken": page_token} if page_token else {}
                    async with session.get(models_url, headers=headers, params=params) as response:
                        if response.status != 200:
                            logger.warning(
                                "%s live model listing failed for instance %r: HTTP %d", self.get_type(), instance, response.status
                            )
                            return None
                        body = json.loads(await response.text())
                        for model in body["models"]:
                            name = model["name"]
                            if not name.startswith("models/"):
                                logger.warning(
                                    "%s live model listing returned an unexpected model name %r for instance %r",
                                    self.get_type(),
                                    name,
                                    instance,
                                )
                                return None
                            resolved_type: RemoteModelType | None = None
                            for method in model.get("supportedGenerationMethods", []):
                                if method in _GENERATION_METHOD_TYPE:
                                    resolved_type = _GENERATION_METHOD_TYPE[method]
                                    break
                            entries.append(LiveModelEntry(id=name.removeprefix("models/"), type=resolved_type))

                        next_page_token = body.get("nextPageToken")
                        if not next_page_token:
                            return entries
                        page_token = next_page_token
        except Exception:
            logger.exception("%s live model listing failed for instance %r", self.get_type(), instance)
            return None

        return entries
