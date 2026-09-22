#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Verify example.env documents every bootstrap environment variable AppSettings can be populated from."""

# ruff: noqa: T201

import re
from collections.abc import Collection
from pathlib import Path
from typing import Annotated

import typer
from pydantic import AliasChoices
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings

from server.config import AppSettings
from server.dynamic_config import DynamicSettings

app = typer.Typer(add_completion=False)

# Only the variable name matters for this check, never the value, so a line-anchored key
# pattern is enough - no need for a full `.env` parser dependency. Anchored at line start
# (re.MULTILINE), so a commented-out line ("# FOO=bar") never counts as documented - a comment
# marker never matches the leading `[ \t]*`, since `#` isn't in the key character class. Leading
# and surrounding whitespace ("  FOO = bar") is allowed because python-dotenv (what
# pydantic-settings actually uses to parse `.env`) accepts it too; matching it strictly would
# make this check report a working line as "missing".
ENV_KEY_PATTERN = re.compile(r"^[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*=", re.MULTILINE)


def expected_names_for_field(field_name: str, field_info: FieldInfo, env_prefix: str) -> set[str]:
    """Compute the real environment variable name(s) a field is sourced from.

    A `validation_alias` (e.g. the `ollama_*` fields' `AliasChoices("OLLAMA_...")`) gives the
    field's real env name(s), bypassing `env_prefix` entirely - unlike df-server-new's equivalent
    check, that alias is genuinely reachable here and must be checked under its own name, not
    skipped. An `AliasPath` never appears among AppSettings's current fields; skip it explicitly
    (it addresses a nested structure via indices, not a top-level env var name).
    """
    alias = field_info.validation_alias
    if alias is None:
        return {f"{env_prefix}{field_name}".upper()}
    if isinstance(alias, str):
        return {alias}
    if isinstance(alias, AliasChoices):
        return {choice for choice in alias.choices if isinstance(choice, str)}
    return set()  # AliasPath or another non-name alias form - nothing to check by name


def get_expected_env_names(
    settings_cls: type[BaseSettings] = AppSettings,
    dynamic_field_names: Collection[str] | None = None,
) -> set[str]:
    """Compute the env var names every *bootstrap* field of `settings_cls` is expected to document.

    A field is "bootstrap" when its name is not in `dynamic_field_names` (default: the real
    `DynamicSettings.model_fields`) - fields in `DynamicSettings` are config.json-managed and
    mutable via `PUT /admin/config`, so documenting them in `example.env` would be misleading:
    their env var is only ever consulted once, to seed config.json before it first exists.
    """
    env_prefix = settings_cls.model_config.get("env_prefix") or ""
    excluded = set(DynamicSettings.model_fields) if dynamic_field_names is None else set(dynamic_field_names)

    names: set[str] = set()
    for field_name, field_info in settings_cls.model_fields.items():
        if field_name in excluded:
            continue
        names |= expected_names_for_field(field_name, field_info, env_prefix)
    return names


def parse_existing_keys(content: str) -> set[str]:
    """Collect every variable name already actively (uncommented) defined in example.env's content."""
    return set(ENV_KEY_PATTERN.findall(content))


def find_missing(expected: set[str], existing: set[str], *, case_sensitive: bool) -> list[str]:
    """Return the sorted expected names with no match in `existing`.

    `case_sensitive` must come from the settings class's own `model_config["case_sensitive"]`
    (see `main()`), not be assumed - pydantic-settings only ignores case when that's False, so a
    differently-cased but otherwise-documented name (e.g. `df_docker_subnet=`) only counts as
    satisfied under that same setting. Hardcoding case-insensitivity here would silently keep
    accepting the wrong case if `AppSettings` (or another settings class this is reused for) ever
    sets `case_sensitive=True`, and stay green while pydantic-settings treats the var as unset.
    """
    if case_sensitive:
        return sorted(expected - existing)
    existing_upper = {name.upper() for name in existing}
    return sorted(name for name in expected if name.upper() not in existing_upper)


@app.command()
def main(
    example_env_path: Annotated[
        Path,
        typer.Argument(help="Path to the example.env file to check", exists=True),
    ] = Path("example.env"),
) -> None:
    """Fail if example.env is missing any bootstrap environment variable expected by AppSettings."""
    expected = get_expected_env_names()
    # "utf-8-sig" transparently strips a leading BOM if present (and behaves exactly like "utf-8"
    # when absent), so a file saved with one doesn't shift ENV_KEY_PATTERN's `^` anchor off the
    # first line and cause a false "missing" report for whatever variable is declared first.
    existing = parse_existing_keys(example_env_path.read_text(encoding="utf-8-sig"))
    case_sensitive = bool(AppSettings.model_config.get("case_sensitive", False))
    missing = find_missing(expected, existing, case_sensitive=case_sensitive)

    if missing:
        print(f"{example_env_path} is missing {len(missing)} bootstrap variable(s) expected by AppSettings:")
        for name in missing:
            print(f"  {name}")
        raise typer.Exit(1)

    print(f"{example_env_path}: all bootstrap AppSettings variables are documented.")


if __name__ == "__main__":
    app()
