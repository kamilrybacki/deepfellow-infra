# 1. Pin pre-1.0 direct dependencies with `~=`

## Status

Accepted

## Context

`pyproject.toml` declares most dependencies with a lower-bound constraint
only, e.g. `uvicorn>=0.35.0`. For packages that have reached a stable `1.0`
release this is reasonable: semantic versioning guarantees that `1.x`
releases don't break the public API, and `2.x` would only land via a new
lower bound we explicitly review.

Packages still in the `0.x` series don't carry that guarantee. Per SemVer,
anything in `0.x` is allowed to break compatibility on a **minor** version
bump (`0.34.x` -> `0.35.0`), not just on a major one. With only a `>=` floor,
`uv lock` is free to resolve any later `0.x` release the next time the lock
file is regenerated, silently pulling in a breaking change between minor
versions of a pre-1.0 dependency. `df-server-new`'s
[DFSERVER-53](https://gitlab2.simplito.com/df/df-server-new/-/merge_requests/414)
and `df-cli`'s DFCLI-18 established this policy after a post-mortem showed a
Typer minor version bump (`0.16` -> `0.26`) silently changing internal
behavior and breaking installs.

## Decision

For every direct (non-dev) dependency in `pyproject.toml` whose currently
resolved version is in the `0.x.y` form, replace the `>=` lower bound with a
compatible-release pin, `~=X.Y.Z`. This restricts resolution to patch-level
updates only (`>=X.Y.Z, ==X.Y.*`) and blocks any `0.(Y+1).0` bump from being
picked up without an explicit change to `pyproject.toml`.

Dev dependencies (`[dependency-groups].dev`) are unaffected — they don't
ship to users and a stricter local dev-tool version isn't worth the extra
maintenance, consistent with `df-server-new`'s version of this policy.

The pinned version is the version already resolved in `uv.lock` at the time
of this change (not necessarily the old `>=` floor, which had in most cases
drifted behind), so this change does not by itself change what gets
installed.

Affected direct dependencies:

| Package | Before | After |
|---|---|---|
| `fastapi` | `>=0.112.1` | `~=0.128.0` |
| `tomlkit` | `>=0.13.3` | `~=0.14.0` |
| `uvicorn` | `>=0.35.0` | `~=0.40.0` |
| `python-multipart` | `>=0.0.20` | `~=0.0.21` |
| `aiodocker` | `>=0.24.0` | `~=0.25.0` |
| `prometheus-client` | `>=0.24.1` | `~=0.24.1` |

## Consequences

- A minor bump in a pinned `0.x` dependency (e.g. `fastapi` `0.128.x` ->
  `0.129.0`) now requires an explicit `pyproject.toml` edit instead of being
  picked up silently by `uv lock` / `uv sync`.
- Patch releases (`0.x.Y` -> `0.x.(Y+1)`) still update automatically, so bug
  and security fixes aren't blocked.
- Any newly added direct dependency starting out in the `0.x` range should
  follow the same convention (`~=` instead of `>=`) to stay consistent with
  this decision.
- Dependencies that later cross `1.0` can drop back to a `>=` floor, since
  SemVer's stability guarantee starts to apply.
