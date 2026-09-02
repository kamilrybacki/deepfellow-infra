# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Get llama.cpp (GGUF) models info.

Uses the HuggingFace Hub API (no API key required for public data).
Outputs a llamacpp-min.json compatible JSON to stdout.
"""

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiohttp

from scripts.utils.hf_common import get_json_with_retry, is_embedding, is_gguf
from server.models.registries import LlamacppRegistryEntry

# ruff: noqa: T201

HF_API = "https://huggingface.co/api"
CONCURRENCY = 10  # parallel file-listing fetch requests
MAX_FETCH_FAILURE_RATE = 0.1  # abort instead of writing a possibly-degraded registry above this failure rate
SPLIT_GGUF_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)
QUANT_RE = re.compile(r"(?:^|[-_.])((?:i?q\d+(?:_[a-z0-9]+)*|f16|f32|bf16))\.gguf$", re.IGNORECASE)

NON_LLM_KEYWORDS = ("diffusion", "flux", "wan2", "video", "image", "tts", "whisper", "asr", "ocr", "vae", "audio")


def is_llm(model: dict[str, Any]) -> bool:
    """Filter out non-chat GGUF repos (image/video diffusion, TTS, ASR, OCR, ...) by name heuristic.

    "GGUF" search results aren't limited to servable chat models — image/video diffusion pipelines
    (Flux, Wan, Qwen-Image), speech models (Whisper, TTS, ASR), and OCR models are commonly
    packaged as GGUF too, but llama.cpp's chat completions endpoint can't serve any of them.
    Deliberately excludes "imatrix": that's a calibrated-quant technique many legitimate chat LLM
    repos advertise in their name (e.g. `...-Heretic-NEO-CODE-Imatrix-MAX-GGUF`), not a model kind.
    """
    name = model.get("id", "").lower()
    return not any(kw in name for kw in NON_LLM_KEYWORDS)


def fmt_size(n: int) -> str:
    """Format byte count as a human-readable string.

    Uses decimal (1000-based) units to match `convert_size_to_bytes` in server/utils/core.py,
    which decodes "GB"/"MB"/etc. as powers of 1000, not 1024.
    """
    size: float = n
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000:
            return f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} PB"


def fmt_size_compact(size_str: str) -> str:
    """Convert '2.0 GB' to '2.0GB'."""
    if size_str == "N/A":
        return size_str
    parts = size_str.split()
    if len(parts) == 2:
        return f"{float(parts[0]):.1f}{parts[1]}"
    return size_str


def is_split_gguf_file(filename: str) -> bool:
    """Return True for a multi-part GGUF shard (e.g. `model-00001-of-00005.gguf`)."""
    return bool(SPLIT_GGUF_RE.search(filename))


def is_mmproj_file(filename: str) -> bool:
    """Return True for a multimodal vision-projector companion file (e.g. `mmproj-F16.gguf`, `Bonsai-27B-mmproj-Q8_0.gguf`).

    These aren't standalone loadable models — they only pair with a separate base-model GGUF —
    so they'd otherwise show up in the model list as unusable, incomplete entries. Repos commonly
    prefix the filename with the model name, so `mmproj` isn't necessarily the first token.
    """
    return "mmproj" in filename.lower()


def is_gguf_file(filename: str) -> bool:
    """Return True if filename is a usable, standalone (non-split, non-mmproj) `.gguf` file.

    Excludes files nested in a subdirectory (e.g. `unet/Model-Q4_K_M.gguf`, `text_encoders/...`):
    that layout is used by multi-component diffusion/video pipelines (ComfyUI-style exports),
    never by a flat, directly-servable llama.cpp LLM checkpoint.
    """
    return filename.lower().endswith(".gguf") and "/" not in filename and not is_split_gguf_file(filename) and not is_mmproj_file(filename)


def sanitized_stem(filename: str) -> str:
    """Sanitize a GGUF filename's stem into a dash-separated suffix (e.g. `Model.Q8_0.gguf` -> `model-q8-0`)."""
    stem = re.sub(r"\.gguf$", "", filename, flags=re.IGNORECASE)
    return stem.lower().replace("_", "-").replace(".", "-")


def quant_suffix(filename: str) -> str:
    """Extract a dash-separated quant suffix from a GGUF filename (e.g. `Q4_K_M.gguf` -> `q4-k-m`).

    Falls back to a sanitized version of the whole filename stem when no recognized quant token
    is found, so every emitted entry still gets a distinguishing suffix.
    """
    match = QUANT_RE.search(filename)
    if match:
        return match.group(1).lower().replace("_", "-")
    return sanitized_stem(filename)


async def fetch_popular_gguf_models(session: aiohttp.ClientSession, sort: str, limit: int) -> list[dict[str, Any]]:
    """Fetch popular GGUF repos from HuggingFace API sorted by the given criterion."""
    params = {
        "search": "GGUF",
        "sort": sort,
        "direction": "-1",
        "limit": str(limit),
        "cardData": "true",
    }
    return await get_json_with_retry(session, f"{HF_API}/models", params)


async def collect_gguf_models(session: aiohttp.ClientSession, active: dict[str, int]) -> list[dict[str, Any]]:
    """Fetch and deduplicate GGUF repos across all sort combinations."""
    fetch_tasks = [fetch_popular_gguf_models(session, sort, limit) for sort, limit in active.items()]
    results = await asyncio.gather(*fetch_tasks)
    seen: set[str] = set()
    models: list[dict[str, Any]] = []
    for batch in results:
        for m in batch:
            mid = m["id"]
            if mid not in seen and is_gguf(m) and is_llm(m) and not is_embedding(m):
                seen.add(mid)
                models.append(m)
    return models


async def fetch_model_files(session: aiohttp.ClientSession, model_id: str, sem: asyncio.Semaphore) -> list[dict[str, Any]] | None:
    """Return the repo's file listing (siblings), or None if the fetch itself failed.

    `None` is deliberately distinct from an empty list: a repo can legitimately have no usable
    GGUF file (empty list), whereas a network error, exhausted rate-limit retries, or a malformed
    response means the repo was never actually checked — callers must not conflate the two.
    """
    async with sem:
        try:
            data = await get_json_with_retry(session, f"{HF_API}/models/{model_id}", {"blobs": "true"}, max_retries=3)
            return data.get("siblings", [])
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            print(f"Warning: failed to fetch file listing for {model_id}: {exc}", file=sys.stderr)
            return None


def split_fetch_results(
    models: list[dict[str, Any]], file_lists: list[list[dict[str, Any]] | None]
) -> tuple[list[LlamacppRegistryEntry], int, int]:
    """Split per-repo fetch results into built entries, plus counts of failed and empty-dropped repos."""
    entries: list[LlamacppRegistryEntry] = []
    dropped = 0
    failed = 0
    for m, siblings in zip(models, file_lists, strict=True):
        if siblings is None:
            failed += 1
            continue
        model_entries = build_entries(m["id"], siblings)
        if not model_entries:
            dropped += 1
            continue
        entries.extend(model_entries)
    return entries, failed, dropped


def build_entries(model_id: str, siblings: list[dict[str, Any]]) -> list[LlamacppRegistryEntry]:
    """Build one registry entry per non-split, non-mmproj `.gguf` file in `siblings`.

    Two files in the same repo can share a quant suffix (e.g. a base and an "-MTP-" variant both
    ending in `Q4_K_M`) — when that happens, both fall back to their full sanitized filename stem
    so neither entry silently overwrites the other once loaded into a name-keyed dict.

    Entries are built as `LlamacppRegistryEntry` (the same Pydantic model `llamacpp_service.py`
    validates the registry file against) so a field rename on either side is caught by pyright
    instead of silently drifting between the writer and the reader.
    """
    files = [f for f in siblings if is_gguf_file(f.get("rfilename", ""))]
    suffixes = [quant_suffix(f.get("rfilename", "")) for f in files]
    suffix_counts = Counter(suffixes)

    entries: list[LlamacppRegistryEntry] = []
    for f, suffix in zip(files, suffixes, strict=True):
        filename = f.get("rfilename", "")
        if suffix_counts[suffix] > 1:
            suffix = sanitized_stem(filename)
        size = f.get("size")
        entries.append(
            LlamacppRegistryEntry(
                name=f"{model_id}-{suffix}",
                url=f"https://huggingface.co/{model_id}/resolve/main/{filename}",
                size=fmt_size_compact(fmt_size(size)) if size else "N/A",
            )
        )
    return entries


class DegradedFetchError(Exception):
    """Raised when the HuggingFace file-listing fetch failure rate exceeds MAX_FETCH_FAILURE_RATE."""


async def fetch_llamacpp_entries(
    session: aiohttp.ClientSession,
    top_by_downloads: int,
    top_by_likes: int,
    top_by_trending: int,
    log: Callable[[str], None],
) -> list[LlamacppRegistryEntry]:
    """Fetch, filter, and build the full list of llama.cpp registry entries from HuggingFace.

    Shared by the CLI's `main()` and the server's in-process runtime refresh. Raises
    `DegradedFetchError` instead of writing a possibly-degraded registry when more than
    `MAX_FETCH_FAILURE_RATE` of the repo file-listing fetches failed.
    """
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
        raise ValueError("Specify at least one of: top_by_downloads, top_by_likes, top_by_trending")

    label = {"downloads": "downloads", "likes": "likes", "trendingScore": "trending"}
    summary = ", ".join(f"top {n} by {label[s]}" for s, n in active.items())
    log(f"Fetching GGUF models from HuggingFace ({summary})...\n")

    models = await collect_gguf_models(session, active)

    log(f"Fetching file listings for {len(models)} unique repos concurrently (max {CONCURRENCY} at a time)...")
    sem = asyncio.Semaphore(CONCURRENCY)
    file_lists = await asyncio.gather(*[fetch_model_files(session, m["id"], sem) for m in models])

    entries, failed, dropped = split_fetch_results(models, file_lists)

    if failed:
        log(f"Failed to fetch file listing for {failed} repo(s) (network/rate-limit/malformed response) — excluded, not counted as empty.")
    if dropped:
        log(f"Dropped {dropped} repo(s) with no usable (non-split) .gguf file.")

    failure_rate = failed / len(models) if models else 0.0
    if failure_rate > MAX_FETCH_FAILURE_RATE:
        msg = (
            f"{failed}/{len(models)} repo file-listing fetches failed ({failure_rate:.0%}, exceeds "
            f"{MAX_FETCH_FAILURE_RATE:.0%} threshold) — refusing to write a possibly-degraded registry."
        )
        raise DegradedFetchError(msg)

    entries.sort(key=lambda e: e.name.casefold())
    return entries


def preserve_jinja_flags(entries: list[LlamacppRegistryEntry], existing_registry: dict[str, Any]) -> None:
    """Copy the `jinja` flag forward from matching URLs in an existing registry dict, mutating entries in place.

    Whether a GGUF repo needs `--jinja` isn't derivable from the HuggingFace API response — it's set
    by hand after observing a model's chat template misbehave without it — so a refresh must not
    silently drop flags a maintainer (or a prior refresh) already set for URLs that still appear.
    """
    existing_entries = [LlamacppRegistryEntry.model_validate(e) for e in existing_registry.get("llms", [])]
    jinja_urls = {e.url for e in existing_entries if e.jinja}
    for entry in entries:
        if entry.url in jinja_urls:
            entry.jinja = True


async def main(top_by_downloads: int, top_by_likes: int, top_by_trending: int, raw: bool, output: str | None = None) -> None:
    """Fetch GGUF models from HuggingFace and print (or write) a llamacpp-min.json compatible registry."""

    def log(msg: str) -> None:
        if not raw:
            print(msg, file=sys.stderr)

    if top_by_downloads <= 0 and top_by_likes <= 0 and top_by_trending <= 0:
        print("Specify at least one of: --top-by-downloads, --top-by-likes, --top-by-trending", file=sys.stderr)
        return

    async with aiohttp.ClientSession() as session:
        try:
            entries = await fetch_llamacpp_entries(session, top_by_downloads, top_by_likes, top_by_trending, log)
        except DegradedFetchError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    if output is None:
        print(json.dumps({"llms": [e.model_dump() for e in entries]}, indent=4))
        return

    output_path = Path(output)
    registry: dict[str, Any] = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}

    preserve_jinja_flags(entries, registry)

    registry["llms"] = [e.model_dump() for e in entries]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(registry, indent=4), encoding="utf-8")
    log(f"Written {len(entries)} llms to {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Scrape popular GGUF models from HuggingFace")
    p.add_argument("--top-by-downloads", type=int, default=0, metavar="N", help="Fetch top N models sorted by downloads")
    p.add_argument("--top-by-likes", type=int, default=0, metavar="N", help="Fetch top N models sorted by likes")
    p.add_argument("--top-by-trending", type=int, default=0, metavar="N", help="Fetch top N models sorted by trending score")
    p.add_argument("--raw", action="store_true", help="Output JSON only, suppress progress messages")
    p.add_argument("--output", default=None, metavar="PATH", help="Write into PATH, merging into its existing keys, instead of stdout")
    args = p.parse_args()
    asyncio.run(main(args.top_by_downloads, args.top_by_likes, args.top_by_trending, args.raw, args.output))
