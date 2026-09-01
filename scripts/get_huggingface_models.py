# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Get HuggingFace Models info.

Uses the HuggingFace Hub API (no API key required for public data).
Outputs a vllm-min.json compatible JSON to stdout.
"""

import argparse
import asyncio
import json
import re
import sys
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import aiohttp

from scripts.utils.hf_common import get_json_with_retry, is_embedding, is_gguf

# ruff: noqa: T201

HF_API = "https://huggingface.co/api"
LLM_TAGS = ["text-generation", "text2text-generation", "image-text-to-text"]
CONCURRENCY = 10  # parallel size-fetch requests
PAGE_SIZE = 100  # max models per API page (HF limit)
NON_LLM_TASK_WORDS = re.compile(r"\b(?:translation|summarization|classification|ner|qa)\b")


def fmt_size(n: int) -> str:
    """Format byte count as a human-readable string."""
    size: float = n
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def fmt_size_compact(size_str: str) -> str:
    """Convert '2.0 GB' to '2GB'."""
    if size_str == "N/A":
        return size_str
    parts = size_str.split()
    if len(parts) == 2:
        return f"{int(float(parts[0]))}{parts[1]}"
    return size_str


def is_llm(model: dict[str, Any]) -> bool:
    """Filter out non-LLM models using a basic heuristic.

    Task words are matched on word boundaries so that a substring buried in an unrelated name does
    not disqualify a real LLM — "ner" occurs inside "ForConditionalGeneration", which every
    multimodal repo carries. "ocr" is matched as a plain substring instead, because OCR models spell
    it glued to the rest of the name ("PaddleOCR-VL", "olmOCR", "GOT-OCR2_0"); those are
    single-purpose text extractors rather than chat models, so they don't belong in an LLM registry.
    """
    name = model.get("id", "").lower()
    return not NON_LLM_TASK_WORDS.search(name) and "ocr" not in name


def is_reranker(model: dict[str, Any]) -> bool:
    """Filter reranker models by name heuristic."""
    name = model.get("id", "").lower()
    return "rerank" in name


NON_RERANKER_CROSS_ENCODER_TASK_WORDS = re.compile(r"\b(?:sts|stsb|nli|qnli|mnli|quora|qqp|mrpc|paraphrase)\b", re.IGNORECASE)


def is_reranking_cross_encoder(model: dict[str, Any]) -> bool:
    """Exclude cross-encoders trained for a different scoring task than reranking.

    `library_name == "sentence-transformers"` plus a text-ranking/text-classification pipeline_tag
    only means "some CrossEncoder regression/classification head" — the same architecture is reused
    for semantic-textual-similarity (STS/STSB), natural-language-inference (NLI/QNLI/MNLI), and
    duplicate-question (Quora/QQP/MRPC) benchmarks, which score something other than relevance and
    would return meaningless rankings if served as a reranker. These tasks are conventionally named
    in the repo id (e.g. cross-encoder/stsb-*, cross-encoder/qnli-*, cross-encoder/quora-*), so
    they're excluded by name.
    """
    name = model.get("id", "").lower()
    return not NON_RERANKER_CROSS_ENCODER_TASK_WORDS.search(name)


# Each pipeline_tag maps to a validator that decides whether a candidate carrying that tag is kept.
# "sentence-similarity" is HF's dedicated tag for that task and is trusted as-is. "text-ranking" and
# "text-classification" both get HF-tagged onto any CrossEncoder-style model regardless of what it
# was actually trained to score — reranking relevance, but also semantic-textual-similarity (STS),
# natural-language-inference (NLI), or duplicate-question detection, which return meaningless
# rankings if served as a reranker — so every candidate from either tag is filtered through
# `is_reranking_cross_encoder` to exclude those by name. "text-classification" is additionally far
# broader still (sentiment classifiers, prompt-injection detectors, ...), so it also requires
# corroborating evidence: rerankers published under text-classification are consistently shipped
# with the sentence-transformers library (a survey of the top 100 by downloads found this held for
# all of them and none of the non-reranker classifiers). "feature-extraction" is embeddings' analog
# of the broad, noisy tag and is validated against the existing embedding name heuristic instead.
RERANKER_TAGS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "text-ranking": is_reranking_cross_encoder,
    "text-classification": lambda m: m.get("library_name") == "sentence-transformers" and is_reranking_cross_encoder(m),
}
EMBEDDING_TAGS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "sentence-similarity": lambda _m: True,
    "feature-extraction": lambda m: is_embedding(m),
}


def is_supported_cross_encoder(architectures: list[str]) -> bool:
    """Return True if any architecture is a standard HF sequence-classification head.

    vLLM's rerank/score task only serves this kind of cross-encoder. Models named "reranker"
    that use a different architecture (e.g. a causal LM doing generative ranking, or a custom
    multi-vector ranking head) fail to load in vLLM even though they match the name heuristic.
    """
    return any(a.endswith("ForSequenceClassification") for a in architectures)


async def has_chat_template(session: aiohttp.ClientSession, model_id: str, siblings: list[dict[str, Any]]) -> bool:
    """Return True if the model ships a chat template vLLM can use for chat completions.

    As of transformers v4.44, vLLM no longer falls back to a generic template, so serving a
    model whose tokenizer defines none raises a 400 on every chat request. Newer repos carry a
    standalone `chat_template.jinja` file; older ones embed a `chat_template` key in
    `tokenizer_config.json`. Base/pretrain models typically have neither.
    """
    filenames = {f.get("rfilename", "") for f in siblings}
    if "chat_template.jinja" in filenames:
        return True
    if "tokenizer_config.json" not in filenames:
        return False
    try:
        tokenizer_config = await get_json_with_retry(session, f"https://huggingface.co/{model_id}/raw/main/tokenizer_config.json", {})
        return bool(tokenizer_config.get("chat_template"))
    except Exception:
        return False


async def fetch_popular_models(session: aiohttp.ClientSession, tag: str, sort: str, limit: int) -> list[dict[str, Any]]:
    """Fetch popular LLMs from HuggingFace API sorted by the given criterion."""
    params = {
        "pipeline_tag": tag,
        "sort": sort,
        "direction": "-1",
        "limit": str(limit),
        "cardData": "true",
    }
    return await get_json_with_retry(session, f"{HF_API}/models", params)


async def fetch_popular_reranker_models(session: aiohttp.ClientSession, sort: str, limit: int) -> list[dict[str, Any]]:
    """Fetch popular reranker models from HuggingFace API sorted by the given criterion."""
    params = {
        "search": "rerank",
        "sort": sort,
        "direction": "-1",
        "limit": str(limit),
        "cardData": "true",
    }
    return await get_json_with_retry(session, f"{HF_API}/models", params)


async def fetch_popular_embedding_models(session: aiohttp.ClientSession, sort: str, limit: int) -> list[dict[str, Any]]:
    """Fetch popular embedding models from HuggingFace API sorted by the given criterion."""
    params = {
        "search": "embed",
        "sort": sort,
        "direction": "-1",
        "limit": str(limit),
        "cardData": "true",
    }
    return await get_json_with_retry(session, f"{HF_API}/models", params)


async def collect_llm_models(session: aiohttp.ClientSession, active: dict[str, int]) -> list[dict[str, Any]]:
    """Fetch and deduplicate LLM models across all sort/tag combinations."""
    fetch_tasks = [fetch_popular_models(session, tag, sort, limit) for sort, limit in active.items() for tag in LLM_TAGS]
    results = await asyncio.gather(*fetch_tasks)
    seen: set[str] = set()
    models: list[dict[str, Any]] = []
    for batch in results:
        for m in batch:
            mid = m["id"]
            if mid not in seen and is_llm(m) and not is_gguf(m):
                seen.add(mid)
                models.append(m)
    return models


async def _collect_by_tag_and_name(
    session: aiohttp.ClientSession,
    active: dict[str, int],
    tags: dict[str, Callable[[dict[str, Any]], bool]],
    fetch_by_name: Callable[[aiohttp.ClientSession, str, int], Coroutine[Any, Any, list[dict[str, Any]]]],
    is_named_match: Callable[[dict[str, Any]], bool],
) -> list[dict[str, Any]]:
    """Merge tag-driven and name-driven candidate discovery, deduplicated across all sources.

    A candidate qualifies if it satisfies *any* source: its pipeline_tag's own validator (see
    `RERANKER_TAGS`/`EMBEDDING_TAGS`), or the legacy name-search heuristic. This is a strict OR —
    the tag query surfaces models the name search can never find (its repo name doesn't contain
    the word being searched for), while the name search still catches candidates that carry
    neither pipeline_tag (e.g. generative rerankers tagged as plain text-generation models).
    """
    query_tags = [tag for _ in active for tag in tags]
    tag_fetch_tasks = [fetch_popular_models(session, tag, sort, limit) for sort, limit in active.items() for tag in tags]
    name_fetch_tasks = [fetch_by_name(session, sort, limit) for sort, limit in active.items()]
    tag_results = await asyncio.gather(*tag_fetch_tasks)
    name_results = await asyncio.gather(*name_fetch_tasks)

    seen: set[str] = set()
    models: list[dict[str, Any]] = []

    def add(m: dict[str, Any]) -> None:
        mid = m["id"]
        if mid not in seen and not is_gguf(m):
            seen.add(mid)
            models.append(m)

    for tag, batch in zip(query_tags, tag_results, strict=True):
        for m in batch:
            if tags[tag](m):
                add(m)
    for batch in name_results:
        for m in batch:
            if is_named_match(m):
                add(m)
    return models


async def collect_reranker_models(session: aiohttp.ClientSession, active: dict[str, int]) -> list[dict[str, Any]]:
    """Fetch and deduplicate reranker models across all sort combinations, by tag and by name."""
    return await _collect_by_tag_and_name(session, active, RERANKER_TAGS, fetch_popular_reranker_models, is_reranker)


async def collect_embedding_models(session: aiohttp.ClientSession, active: dict[str, int]) -> list[dict[str, Any]]:
    """Fetch and deduplicate embedding models across all sort combinations, by tag and by name."""
    return await _collect_by_tag_and_name(session, active, EMBEDDING_TAGS, fetch_popular_embedding_models, is_embedding)


async def fetch_model_details(
    session: aiohttp.ClientSession, model_id: str, sem: asyncio.Semaphore, check_chat_template: bool = False
) -> tuple[str, str, list[str] | None, bool]:
    """Return (model_id, human_readable_size, architectures, has_chat_template).

    `has_chat_template` is only checked when `check_chat_template` is set (LLM candidates); it is
    True by default for reranker/embedding candidates, for which the check is meaningless.

    `architectures` is `None` when the detail fetch itself failed (e.g. exhausted retries under
    rate limiting), as opposed to `[]` when it succeeded but the model genuinely reports none —
    `_drop_unsupported_models` needs to tell these apart so a fetch failure doesn't get treated as
    "confirmed not a cross-encoder" and silently kept with fabricated data.
    """
    async with sem:
        try:
            data = await get_json_with_retry(session, f"{HF_API}/models/{model_id}", {"blobs": "true"})
            siblings = data.get("siblings", [])
            total = sum(f.get("size", 0) for f in siblings if f.get("size"))
            architectures = data.get("config", {}).get("architectures") or []
            chat_ok = await has_chat_template(session, model_id, siblings) if check_chat_template else True
            return model_id, fmt_size(total) if total else "N/A", architectures, chat_ok
        except Exception:
            return model_id, "N/A", None, False


def _drop_unsupported_models(
    registry_key: str,
    models: list[dict[str, Any]],
    architectures: dict[str, list[str] | None],
    chat_capable: dict[str, bool],
    log: Callable[[str], None],
    keep_generative_rerankers: bool = False,
) -> list[dict[str, Any]]:
    """Drop candidates the target engine can't actually serve, logging what was dropped.

    `keep_generative_rerankers` opts a registry out of the cross-encoder-only filter: engines
    that can serve generative rerankers (e.g. SGLang, given the right startup flags) pass this so
    those candidates stay in and get tagged via `is_generative` instead of being dropped outright.
    """
    if registry_key == "rerankers":
        before = len(models)
        models = [m for m in models if architectures.get(m["id"]) is not None]
        if dropped := before - len(models):
            log(f"Dropped {dropped} reranker candidate(s) whose details couldn't be fetched from HuggingFace.")
        if not keep_generative_rerankers:
            before = len(models)
            models = [m for m in models if is_supported_cross_encoder(architectures.get(m["id"]) or [])]
            if dropped := before - len(models):
                log(f"Dropped {dropped} reranker candidate(s) whose architecture vLLM can't serve as a cross-encoder.")
    elif registry_key == "llms":
        before = len(models)
        models = [m for m in models if chat_capable.get(m["id"], False)]
        if dropped := before - len(models):
            log(f"Dropped {dropped} LLM candidate(s) without a usable chat template (base/pretrain models vLLM can't serve for chat).")
    return models


async def main(
    top_by_downloads: int,
    top_by_likes: int,
    top_by_trending: int,
    raw: bool,
    model_type: str,
    output: str | None = None,
    allow_generative_rerankers: bool = False,
) -> None:
    """Fetch models from HuggingFace and print (or write) a vllm-min.json compatible registry."""

    def log(msg: str) -> None:
        if not raw:
            print(msg, file=sys.stderr)

    active = {
        k: v
        for k, v in {
            "downloads": top_by_downloads,
            "likes": top_by_likes,
            "trendingScore": top_by_trending,
        }.items()
        if v > 0
    }

    if not active:
        print("Specify at least one of: --top-by-downloads, --top-by-likes, --top-by-trending", file=sys.stderr)
        return

    label = {"downloads": "downloads", "likes": "likes", "trendingScore": "trending"}
    summary = ", ".join(f"top {n} by {label[s]}" for s, n in active.items())
    log(f"Fetching {model_type} models from HuggingFace ({summary})...\n")

    async with aiohttp.ClientSession() as session:
        if model_type == "reranker":
            models = await collect_reranker_models(session, active)
            registry_key = "rerankers"
        elif model_type == "embedding":
            models = await collect_embedding_models(session, active)
            registry_key = "embeddings"
        else:
            models = await collect_llm_models(session, active)
            registry_key = "llms"

        log(f"Fetching disk sizes for {len(models)} unique models concurrently (max {CONCURRENCY} at a time)...")
        sem = asyncio.Semaphore(CONCURRENCY)
        check_chat_template = registry_key == "llms"
        details = await asyncio.gather(*[fetch_model_details(session, m["id"], sem, check_chat_template) for m in models])
        sizes = {mid: size for mid, size, _, _ in details}
        architectures = {mid: arch for mid, _, arch, _ in details}
        chat_capable = {mid: chat_ok for mid, _, _, chat_ok in details}

    models = _drop_unsupported_models(registry_key, models, architectures, chat_capable, log, allow_generative_rerankers)

    entries: list[dict[str, Any]] = []
    for m in models:
        mid = m["id"]
        size = fmt_size_compact(sizes.get(mid, "N/A"))
        entry: dict[str, Any] = {"name": mid, "size": size}
        if registry_key == "rerankers":
            entry["is_generative"] = not is_supported_cross_encoder(architectures.get(mid) or [])
        entries.append(entry)
    entries.sort(key=lambda e: e["name"].casefold())

    if output is None:
        print(json.dumps({registry_key: entries}, indent=4))
        return

    output_path = Path(output)
    registry: dict[str, Any] = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
    registry[registry_key] = entries
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(registry, indent=4), encoding="utf-8")
    log(f"Written {len(entries)} {registry_key} to {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Scrape popular models from HuggingFace")
    p.add_argument("--top-by-downloads", type=int, default=0, metavar="N", help="Fetch top N models sorted by downloads")
    p.add_argument("--top-by-likes", type=int, default=0, metavar="N", help="Fetch top N models sorted by likes")
    p.add_argument("--top-by-trending", type=int, default=0, metavar="N", help="Fetch top N models sorted by trending score")
    p.add_argument("--raw", action="store_true", help="Output JSON only, suppress progress messages")
    p.add_argument(
        "--type", choices=["llm", "reranker", "embedding"], default="llm", dest="model_type", help="Model type to fetch (default: llm)"
    )
    p.add_argument("--output", default=None, metavar="PATH", help="Write into PATH, merging into its existing keys, instead of stdout")
    p.add_argument(
        "--allow-generative-rerankers",
        action="store_true",
        help="Keep generative (non-cross-encoder) reranker candidates instead of dropping them; tags each entry with 'is_generative'",
    )
    args = p.parse_args()
    asyncio.run(
        main(
            args.top_by_downloads,
            args.top_by_likes,
            args.top_by_trending,
            args.raw,
            args.model_type,
            args.output,
            args.allow_generative_rerankers,
        )
    )
