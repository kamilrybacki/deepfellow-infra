# Finishing the move from `webui` to the DeepFellow Dashboard

This is the step-by-step guide for the day the Dashboard becomes this service's real web UI.

Until then `webui2/` is a proof of concept, and `webui/` still builds what the image serves.

**Every decision below is open.** This guide states the options and their consequences. It settles none of them,
because they belong to the team that owns this repository.

## Before you start

- [ ] Decide **which release carries the swap**, and how an operator hears that the UI changed. The Dashboard looks
      nothing like the current UI, so somebody will file a bug about it if nobody says so first.
- [ ] Decide **what the rollback is**. While the `UI_VARIANT` build argument is in the `Dockerfile`, a rollback is one
      word at build time. If step 6 removes it, the rollback becomes a revert instead.
- [ ] Confirm the Dashboard covers what operators actually use here. Anything the legacy UI does and the Dashboard
      does not is a gap that has to be closed first, in the Modular Frontend repository.

## 1. Decide where the artifact comes from

- [ ] **Registry.** The proof of concept installs from the Modular Frontend project's own GitLab registry, which is
      `internal` and reachable on the VPN. The alternatives are the company npm registry once the product is public,
      or another registry entirely.
- [ ] **Change both lines in `webui2/.npmrc`, not one.** An npm authentication line is keyed to the host **and the
      path**. A registry change with the old authentication line gives a 401 on every install.
- [ ] **Regenerate `webui2/package-lock.json` in the same change.** It records the full download URL of the package,
      host included, and `npm ci` in the image build installs strictly from the lock. Change only the `.npmrc` and the
      build still fetches from the old registry. Delete the lock, run `npm install`, and commit the result.
- [ ] **Check the `integrity` field of the new lock.** The GitLab registry hands back a `sha1-` hash today, and npm
      keeps whatever the registry gives. SHA-1 is a weak check for the file that becomes the whole admin UI. A registry
      that answers with `sha512-` fixes this for free, so confirm it after the move.

For the company registry the pair becomes:

```ini
@simplito:registry=https://<the company npm registry>/
//<the company npm registry>/:_authToken=${NPM_TOKEN}
```

- [ ] **Decide how a developer with no access authenticates.** The artifact project is `internal` today, so any
      signed-in account can read it and nothing is needed. An account marked "external" cannot, and its token fails
      with a 404 that looks like a missing package. If anybody on either team is in that position, ask the artifact
      project's owner for a deploy token with the `read_package_registry` scope, which carries its own access.
      A move to a public registry removes the question entirely.
- [ ] **Decide whether `.npmrc` stays committed at all.** If the target registry allows an anonymous read, the file
      can go, and with it the token every developer currently needs.
- [ ] **Pin a version that exists in the target registry.** The proof of concept pins a version published to GitLab.
      That exact version has to be published to the new registry too, or the pin has to move in the same commit.

## 2. Decide what happens to the two directories

- [ ] **`webui/`**: delete it, rename it, or leave it. Deleting it is the point of the exercise, and it is also the
      step that makes a rollback a revert. Leaving it for one release keeps the rollback cheap.
- [ ] **`webui2/`**: rename it to `webui`, give it another name, or leave the name alone. A rename touches the
      `Dockerfile`, the `justfile`, `.dockerignore` and `.gitignore` together.
- [ ] If `webui/` goes, remove what only it needed: its `.gitlab-ci.yml` job, its `justfile` recipes, and its
      entries in `.dockerignore` and `.gitignore`.
- [ ] **Four `justfile` recipes belong to `webui2`, and two of them hold its path.** `ui2-rebuild` builds the
      Dashboard, `ui2-clean` removes what that build wrote, `dashboard` runs the development server, and `ui-restore`
      puts the legacy UI back. `ui2-rebuild` and `dashboard` are the two that hold the path, so a rename reaches both
      of them. A comment above `ui2-rebuild` names the directory as well. If `webui/` goes, `ui-restore` has nothing to restore, so delete it. Then rename `ui2-rebuild` to
      `ui-rebuild` and `ui2-clean` to `ui-clean`, so that no recipe carries a `2` that names nothing. `dashboard`
      keeps its name.
- [ ] **`.gitignore` holds the Dashboard build's file names, and it is the only place that still does.**
      `ui2-clean` reads them from the installed package instead, which states them in `deepfellowDashboard.emittedFiles`.
      Git reads `.gitignore` and cannot compute it, so that list stays hand-written. `just ui2-check-ignores` compares
      the two, and the `check_dashboard_ui_ignores` CI job runs it. Keep that job when this directory is renamed.

## 3. Make the Dashboard the default build

In the `Dockerfile`, change the default of the build argument:

```dockerfile
ARG UI_VARIANT=dashboard
```

- [ ] **Decide whether to keep the argument at all.** Keeping it leaves the legacy UI selectable, which is the cheap
      rollback. Removing it, along with the `ui-legacy` stage, leaves one path and less to read.

## 4. Tidy the ignore files

- [ ] `.gitignore` lists the output names of BOTH UIs today, because both exist. When the legacy UI goes, delete its
      names and keep the Dashboard's. Leave everything else in `static/` tracked, as this repository always has.
- [ ] `.dockerignore` still names the legacy UI's output files one by one, at its `static/...` lines. Rewrite those
      the same way, or delete them once the legacy UI is gone.

## 5. Fix what the proof of concept could not

- [ ] Nothing in this repository. `webui2/` does not reach the runtime image: the builder strips it, so no file of it
      discloses the internal registry. DF Server's copy of this guide has a step here, because its runner copies the
      whole tree and its `webui2/` does ship.

## 6. Check it before the release

- [ ] `docker build .` produces an image that serves the Dashboard at the origin root.
      **Until the registry is a public one, that build still needs `--secret id=gitlab_npm_token,src=<a file holding a token>`.**
      This step is only true without it once step 1 has moved the registry.
- [ ] Sign in with the administrator API key, and load real data.
- [ ] Hard-reload a deep route, for example `/dfInfra/current`. The single-page fallback has to answer it.
      `curl` is enough for this one. The service's own fallback reads no header, so it answers any request for an
      unknown path with `static/index.html`. A 500 means that file is absent, and the UI build therefore did not
      land. The service opens the file without a check that it exists.
- [ ] `GET /health` still answers. The Dashboard uses it to report an outage.
- [ ] **The eleven committed files under `static/` are still reachable**, for example `/ollama.json`. They are not part
      of any UI, and a UI build must never remove them. Check this against the running service.
- [ ] The image carries a license file. It did not when this guide was written.

## 6b. Decide what happens to the CI job this change added

`build_dashboard_variant_image` builds the Dashboard variant and pushes nothing. It is manual, it is
`allow_failure: true`, and it drops itself while the `GITLAB_NPM_TOKEN` CI/CD variable is unset. It exists so the
alternative UI has a gate rather than only a description.

- [ ] **Somebody has to set that CI/CD variable** before the job can run at all. Nothing sets it today.
- [ ] Decide whether the job becomes the real image build at swap time, or whether the ordinary
      `build_and_publish_docker_image` simply starts producing the Dashboard once step 3 changes the default.
- [ ] If the ordinary job takes over, delete this one. Two jobs that build the same image is a trap.

## 7. Clean up

- [ ] Remove `webui2/README.md`, because it describes a proof of concept that has become the real thing.
- [ ] **Delete this file.** Its work is finished, and a guide that describes a completed migration only misleads.
