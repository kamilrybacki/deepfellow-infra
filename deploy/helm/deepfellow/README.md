# deepfellow

The Kubernetes distribution of the DeepFellow Suite (Infra + Server + Workspace).

The chart runs the Suite **socket-free**: model backends are served natively by
[llama.cpp](https://github.com/ggml-org/llama.cpp) Deployments and registered into Infra as
external `openai` services, so no Docker-in-Docker and no Docker socket are required. Only the
components you enable are rendered.

> **Status: templates complete; pre-smoke.** The chart renders the whole Suite — Infra,
> native model backends, Server, Mongo, vector DBs (Qdrant/Milvus), FalkorDB, Workspace, the
> provisioning Job, Ingress, and NetworkPolicies — with schema + fail-closed render-time
> validation. The `DF_EXTERNAL_ONLY` Infra patch ships alongside it. Not yet smoke-tested on a
> cluster; the separate `deepfellow-server` image's login/health/Mongo-auth contracts are
> verified during the cluster smoke, not by rendering.

## Contract philosophy: fail-closed

The chart does **not** auto-generate credentials or run against mutable image tags by default.
A default `helm template` intentionally **fails** until you supply, per credential, either an
existing Secret, an explicit inline value, or opt-in generation — and, per image, either a
digest or an explicit `allowMutableTag: true`. This is the safe default for GitOps/ArgoCD, where
render-time `randAlphaNum` drifts on every reconcile and a repo-server cannot look up live
cluster Secrets. See [`ci/minimal-values.yaml`](./ci/minimal-values.yaml) for a valid minimum.

### Credential shape

Every credential uses one shape with an explicit `source` discriminator (no empty-string XOR):

```yaml
<cred>:
  source: existingSecret        # existingSecret | value | generate | none
  existingSecret: { name: "", key: "<fixed-key>" }   # source=existingSecret
  value: ""                                            # source=value
  generate: { retain: true }                           # source=generate (opt-in; NOT GitOps-safe)
```

- `existingSecret` — reference a Secret you manage (ESO, sops, manual). `name` required.
- `value` — inline (fine for non-GitOps or non-sensitive). Non-empty required.
- `generate` — chart generates once and `lookup`s thereafter on **live** `helm` operations.
  Rejected/unsafe under pure GitOps rendering; use `existingSecret` there.
- `none` — only for optional credentials (e.g. HuggingFace token, endpoint API keys).

### Image shape (digest-first)

```yaml
image:
  repository: ...
  tag: latest              # human-readable provenance only
  digest: ""               # sha256:… — THE deploy identity, precedence over tag
  allowMutableTag: false   # must be true to deploy on a tag with no digest
```

Rendering fails unless `digest` is set **or** `allowMutableTag: true`. Component images publish
`:latest` upstream — resolve and record a digest per chart release for production.

### `infra.externalOnly`

```yaml
infra:
  externalOnly:
    enabled: true              # socket-free placement + exports DF_EXTERNAL_ONLY=true
    enforcement: compatibility # compatibility | patched
```

- `enabled` — no docker socket/hostPath; native/external model paths only. Required whenever
  native `modelBackends` are enabled.
- `enforcement: compatibility` — the current upstream image ignores `DF_EXTERNAL_ONLY` but boots
  socket-free anyway (it bundles the Docker CLI; daemon probes are non-fatal). The env is set but
  not enforced upstream.
- `enforcement: patched` — a later image that truly honors `DF_EXTERNAL_ONLY`; requires a pinned
  `infra.image.digest`.

The chart enforces **socket-free placement** today; a patched image enforces **Infra behavior**.

## Architecture (request path)

```
consumer -> Server (:8000, project key)
              -> Infra (:8086, OpenAI proxy)
                   -> native llama.cpp Deployment(s)   (modelBackends, registered as external openai)
                   -> or a bring-your-own endpoint     (externalBackends)
Workspace (:3000) -> Server
```

## Model backends

Two keyed maps keep native-vs-BYO explicit:

```yaml
modelBackends:            # native llama.cpp Deployments this chart runs
  llama:
    enabled: true
    modelId: gate-llm     # defaults to the map key
    gguf: { url: "https://.../model.gguf" }
    image: { allowMutableTag: true }

externalBackends:         # endpoints you host elsewhere
  remote:
    enabled: true
    serviceType: openai
    apiUrl: http://host:8080   # endpoint ROOT — Infra appends v1/ (not .../v1)
    modelId: my-model
```

A map (not a list) means an overlay adds or disables one backend without restating the rest. Each
backend is registered as its own Infra `openai` service **instance** named by its map key
(`openai|<key>`), so two backends never collide on one endpoint; keys must be unique across
`modelBackends` and `externalBackends`. `modelId` stays free-form (it may contain dots) — the map
key, not `modelId`, carries Infra's instance-name charset constraint.
An external endpoint's `GET /v1/models` must **not** list its custom `modelId` (a live-listed id
seeds `type=None` in Infra and collides with custom-model install — Gate A finding).

## Prerequisites

- Kubernetes 1.23+ and Helm 3.8+
- A default StorageClass (or set `*.storage.storageClass` / `*.storage.existingClaim`)
- Secrets for the credentials you reference (or use `source: value` / `generate`)
- Model weights reachable as GGUF URLs (downloaded by an initContainer per native backend)

### Storage for stateful components

MongoDB, the vector DBs, and FalkorDB store data on a PVC. Give the stateful components a
StorageClass that meets the engine's requirements — in particular **MongoDB (WiredTiger) does not
support NFS** (file operations fail with `Operation not permitted`), so back `server.mongo` and
`workspace.mongo` with block or node-local storage (e.g. a `local-path` class), not an NFS class.
The embedded MongoDB runs as its own non-root user and relies on `fsGroup`, so the StorageClass
must honor `fsGroup` for volume ownership. Set `*.storage.storageClass` per component, and use
`*.nodeSelector` / `*.tolerations` / `*.affinity` (including on `server.mongo` and
`workspace.mongo`) to pin data pods when using node-local storage.

## Installing

```console
helm install my-deepfellow deploy/helm/deepfellow -f my-values.yaml
```

Render / lint (a valid minimum is in `ci/`):

```console
helm lint     deploy/helm/deepfellow -f deploy/helm/deepfellow/ci/minimal-values.yaml
helm template t deploy/helm/deepfellow -f deploy/helm/deepfellow/ci/minimal-values.yaml
```

## Validation (render-time, fail-closed)

`templates/_validate.tpl` enforces what the JSON schema cannot (Go RE2 has no lookahead; schema
cannot compare fields). Every message is path-prefixed:

- **Images** — a mutable tag without a digest fails unless `allowMutableTag: true`.
- **Credentials** — `source` must be valid; `existingSecret` needs a `name`; `value` must be
  non-empty; `generate` takes neither; required credentials cannot be `none`.
- **Passwords** — an inline `server.admin.password.value` must be ≥10 chars with lower, upper,
  digit, and special. An inline `workspace.mongo.auth.rootPassword.value` must additionally be
  URL-safe (letters, digits, `!$&'()*+,._~-`): it is placed in the Workspace `MONGO_URL` userinfo
  unencoded. A password with other characters must come from an `existingSecret` (already URL-safe
  or percent-encoded); `generate` is URL-safe by construction.
- **Infra** — `mesh.key` non-empty (image requires `DF_MESH_KEY`); `externalOnly.enforcement`
  enum; `patched` requires an infra digest; native backends require `externalOnly.enabled`.
- **Mongo** — `mode` enum; `embedded` needs its passwords; `external` needs `uriSecret.name`.
- **Backends** — native needs `gguf.url` and `type: llamaCpp`; external needs `apiUrl` + `modelId`.
- **Provisioning** — `enabled` with no enabled backend fails unless `allowNoModels: true`.

## Values (selected)

| Key | Description | Default |
| --- | --- | --- |
| `infra.image.{repository,tag,digest,allowMutableTag}` | Infra image (digest-first). | `…/deepfellow-infra`, `latest`, `""`, `false` |
| `infra.externalOnly.{enabled,enforcement}` | Socket-free mode; see above. | `true`, `compatibility` |
| `infra.auth.adminApiKey` / `apiKey` | Infra admin / server-facing keys (credential shape). | `source: existingSecret` |
| `infra.mesh.{enabled,connectUrl,key}` | Mesh; `key` must be non-empty. | `false`, `""`, `unused-…` |
| `modelBackends.<name>` | Native llama.cpp backends (keyed map). | `{}` |
| `externalBackends.<name>` | BYO external endpoints (keyed map). | `{}` |
| `server.admin.password` | Bootstrap admin password (credential shape; complexity when inline). | `source: existingSecret` |
| `server.mongo.mode` | `embedded` or `external`. | `embedded` |
| `server.mongo.external.uriSecret.{name,key}` | External Mongo URI Secret ref. | `""`, `uri` |
| `server.vectorDb.{active,type}` | RAG store; `qdrant` or `milvus`. | `false`, `qdrant` |
| `server.graph.enabled` | FalkorDB graph store. | `false` |
| `workspace.enabled` | Deploy the Workspace panel. | `true` |
| `provisioning.enabled` | Run provisioning (helm hooks). | `false` |
| `provisioning.{revision,allowNoModels,stateSecret.name}` | Rerun key / no-model escape / dfproj Secret. | `"1"`, `false`, `""` |

> **State Secret.** With `provisioning.stateSecret.name` empty the chart creates and owns the
> dfproj state Secret (`resource-policy: keep`, so the one-time project key survives uninstall).
> If you set `provisioning.stateSecret.name` to your own Secret, the chart neither creates it nor
> grants RBAC to create it — that Secret **must already exist** in the release namespace, otherwise
> provisioning fails (fail-closed) when it tries to persist the project key.
| `ingress.enabled` / `networkPolicy.enabled` | Ingress / NetworkPolicies. | `false` / `false` |

See [`values.yaml`](./values.yaml) for the complete, commented set.
