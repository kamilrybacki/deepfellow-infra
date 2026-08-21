# webui2 - the DeepFellow Dashboard as this service's web UI

This directory is a **proof of concept**. It is not what the image serves today.

`webui/` still builds the UI that DF Infra ships. `webui2/` installs the DeepFellow Dashboard instead, built to manage
exactly one instance of one service. The two sit side by side on purpose, so the legacy UI stays intact while the
replacement is tried out.

## Why replace `webui` at all

The Dashboard already manages DF Infra when it runs against the DeepFellow Dashboard backend. In single-instance mode
it drops that backend and talks to this service directly, which makes it a drop-in replacement for the UI in `webui/`.
One UI then serves both purposes, and this repository maintains no second frontend.

Nothing here is compiled. The package carries the built files, so this directory holds six committed files: a
`package.json`, a `package-lock.json`, an `.npmrc`, a `.gitignore` and two documents.

## What you need first

An access token for the Modular Frontend project's GitLab package registry, exported as `GITLAB_NPM_TOKEN`.

The artifact is not on the public npm registry yet, because the product is not public yet. It lives in an `internal`
GitLab project, so the VPN and a signed-in account are what protect it.

**You do not need to be a member of that project.** Its package registry follows the project's `internal` visibility,
so any signed-in account on that GitLab instance can read it. There is nothing to request.

**The fastest way, when `glab` is already signed in to that instance:**

```bash
export GITLAB_NPM_TOKEN="$(glab config get token --host gitlab2.simplito.com)"
```

Do not ask `glab config get host` for that host name. It answers `gitlab.com`, which is glab's own default and not the
instance you signed in to. A wrong host gives an empty token, and the install then fails with a registry 404 that reads
like a missing package. `glab auth status` names the instance.

To make a token by hand instead, make one of these, and export it:

- A personal access token with the `read_api` scope. This is the usual case.
- A deploy token on that project with the `read_package_registry` scope. Use this when a personal token will not do.

**Two cases where a personal token is not enough.**

- **An account marked "external" on that GitLab instance cannot see an `internal` project at all.** A personal token
  of such an account fails with a 404, which reads like a missing package rather than a permission problem. Ask the
  owner of the artifact project for a deploy token instead: a deploy token carries its own access and needs no account.
- **A pipeline of this repository** cannot use its own `CI_JOB_TOKEN`, because that token belongs to this project and
  the registry belongs to another one. The artifact project has to list this project in its inbound job-token
  allowlist first, or the pipeline needs a deploy token in a CI/CD variable.

```bash
export GITLAB_NPM_TOKEN=<your token>
```

Node 20 or newer. The installed package declares that in its own `engines` field, so npm warns on an older one.

## Run the Dashboard against this service

Start this service in one terminal. It listens on port 8086.

```bash
just dev
```

Then, in a second terminal:

```bash
export GITLAB_NPM_TOKEN="$(glab config get token --host gitlab2.simplito.com)"
cd webui2 && npm ci   # the token has to be in this same shell
just dashboard        # from anywhere in the repository, once the install is done
```

The command serves the Dashboard and forwards everything it does not answer itself to the service, so both share one
origin and no CORS rule is needed. It listens on this machine only, and it defaults to port 5180. `--service` names a
service at another address, and `--port` selects another port.

**Open http://localhost:5180 when it starts.** The command prints the address that it listens on.

Every service has its own Dashboard, and each one defaults to port 5180. A Dashboard for another service therefore
takes that port first. The command then stops, and it says so. `just dashboard --port 5181` selects another port,
because the recipe forwards its arguments.

**A deep route answers the page to a browser, and answers 500 to `curl` while `static/` holds no UI build.**
This command answers a request that asks for `Accept: text/html`, and `curl` sends `*/*` instead, so the request
goes to the service. The service has no page of its own until a UI build lands there. Use a browser, or send the
header: `curl -H 'Accept: text/html' ...`.

Sign in with the administrator API key, the same `DF_INFRA_ADMIN_API_KEY` this service already uses. The `example.env`
of this repository sets that key to `ADMIN`. The Dashboard has no accounts of its own in this mode.

## Serve it from this service, the way production would

```bash
export GITLAB_NPM_TOKEN="$(glab config get token --host gitlab2.simplito.com)"
cd webui2
npm ci
npm run buildx      # builds into webui2/dist, then copies it into ../static/
```

The service then serves the Dashboard itself, at its own origin, exactly as a real deployment would.

`just ui2-rebuild` runs that last command from anywhere in the repository, once the install is done.

**Undo it with one of two recipes.**

- `just ui2-clean` removes every file the Dashboard build wrote, and nothing else. Use it when `static/` held no UI
  before, which is the state of a fresh clone. It reads the file names from the installed package, which states them in
  its own `deepfellowDashboard.emittedFiles` field, so the list is never written out in this repository.
- `just ui-restore` puts the legacy UI back. It builds the legacy UI first, then removes the Dashboard's files, then
  copies the legacy build in. A legacy build that fails therefore leaves `static/` exactly as it was.

A plain `just ui-rebuild` is not enough on its own. `buildx` copies without deleting, so the legacy build overwrites
only the names that the two UIs share, and every other Dashboard file stays behind.

**`just ui2-check-ignores` verifies that `.gitignore` covers every name the Dashboard build emits.** Git reads
`.gitignore` and cannot compute it, so that one list is written by hand. This compares it against the package's own
field, and the `check_dashboard_ui_ignores` CI job runs it on every pipeline.

## Build the image with it

```bash
glab config get token --host gitlab2.simplito.com > /tmp/gitlab_npm_token   # or write the token there yourself
DOCKER_BUILDKIT=1 docker build \
  --build-arg UI_VARIANT=dashboard \
  --secret id=gitlab_npm_token,src=/tmp/gitlab_npm_token \
  -t deepfellow-infra:dashboard .
rm -f /tmp/gitlab_npm_token
```

**The `--secret` is not optional.** The install inside the image reads the token from that mount, and `GITLAB_NPM_TOKEN`
in your own shell does not cross into a build. Without it `npm ci` runs unauthenticated against an `internal` project
and the build fails on a registry 404.

Without that argument the image is exactly what it was: `UI_VARIANT` defaults to `legacy`, and the Dashboard stage
never runs. The stage graph is in the `Dockerfile`, and `MIGRATION.md` explains how to make the Dashboard the default.

## Two things to know before this is adopted

- **The registry is a personal namespace today.** `.npmrc` points at project 529, `wpazderski/modular-frontend`. That
  is where the artifact is published while the work is a proof of concept. It moves to the company registry when the
  product is public, and `MIGRATION.md` carries that step.
- **A registry move changes two lines, not one.** An npm authentication line is keyed to the host and the path, so it
  travels with the registry line. A change to the registry line alone gives a 401 on every install.

## Finishing the migration

`MIGRATION.md`, beside this file, is the step-by-step guide. It marks every decision that is still open.
