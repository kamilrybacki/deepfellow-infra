{{/*
Cross-field validation, invoked once from templates/validation.yaml so `helm template`
and `helm lint` exercise it. Enforces the fail-closed contract and everything Helm's
gojsonschema cannot express (Go RE2 has no lookahead; schema cannot compare two fields).
Each failed check aborts rendering with a clear, path-prefixed message.
*/}}

{{/* Validate an image sub-map ({digest, allowMutableTag}). Fail-closed on mutable tags. */}}
{{- define "deepfellow.validateImage" -}}
{{- $img := .image -}}
{{- $path := .path -}}
{{- if $img.digest -}}
{{- if not (regexMatch "^sha256:[a-f0-9]{64}$" $img.digest) -}}
{{- fail (printf "%s.digest must be a valid sha256 digest (sha256:<64 hex>)." $path) -}}
{{- end -}}
{{- else -}}
{{- if not $img.allowMutableTag -}}
{{- fail (printf "%s: set an image digest, or acknowledge the mutable tag with %s.allowMutableTag=true (fail-closed)." $path $path) -}}
{{- end -}}
{{- if not $img.tag -}}
{{- fail (printf "%s.tag is required when no digest is set (otherwise the image ref is malformed)." $path) -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/* Validate a credential sub-map ({source, existingSecret{name}, value}). */}}
{{- define "deepfellow.validateSecret" -}}
{{- $c := .cred -}}
{{- $path := .path -}}
{{- $required := .required -}}
{{- $src := $c.source -}}
{{- if not (has $src (list "existingSecret" "value" "generate" "none")) -}}
{{- fail (printf "%s.source must be one of: existingSecret, value, generate, none." $path) -}}
{{- end -}}
{{- if eq $src "none" -}}
{{- if $required -}}
{{- fail (printf "%s is required: set source to existingSecret, value, or generate (fail-closed)." $path) -}}
{{- end -}}
{{- else if eq $src "existingSecret" -}}
{{- if not $c.existingSecret.name -}}
{{- fail (printf "%s.existingSecret.name is required when source=existingSecret." $path) -}}
{{- end -}}
{{- if $c.value -}}
{{- fail (printf "%s.value must be empty when source=existingSecret." $path) -}}
{{- end -}}
{{- else if eq $src "value" -}}
{{- if not $c.value -}}
{{- fail (printf "%s.value is required when source=value." $path) -}}
{{- end -}}
{{- if $c.existingSecret.name -}}
{{- fail (printf "%s.existingSecret.name must be empty when source=value." $path) -}}
{{- end -}}
{{- else if eq $src "generate" -}}
{{- if or $c.value $c.existingSecret.name -}}
{{- fail (printf "%s: source=generate takes neither a value nor an existingSecret." $path) -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/* Password complexity — only when the credential is an inline value. */}}
{{- define "deepfellow.validatePasswordComplexity" -}}
{{- $c := .cred -}}
{{- $path := .path -}}
{{- if eq $c.source "value" -}}
{{- $p := $c.value -}}
{{- $ok := and (ge (len $p) 10) (regexMatch "[a-z]" $p) (regexMatch "[A-Z]" $p) (regexMatch "[0-9]" $p) (regexMatch "[^a-zA-Z0-9]" $p) -}}
{{- if not $ok -}}
{{- fail (printf "%s.value policy: >=10 chars incl a lowercase, an uppercase, a digit, and a special character." $path) -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
An inline `value` credential that gets interpolated into a connection-string userinfo
(mongodb://user:PASS@host) must be URL-safe unencoded — the chart does not percent-encode it.
Allow unreserved chars plus the URL-safe sub-delims; reject %@:/?#[]... which break the URL.
existingSecret/generate are not checked here (generate is URL-safe by construction; an
existingSecret's content is opaque at render — its password must be URL-safe or pre-encoded).
*/}}
{{- define "deepfellow.validateUrlSafeValue" -}}
{{- $c := .cred -}}
{{- $path := .path -}}
{{- if eq $c.source "value" -}}
{{- if not (regexMatch "^[A-Za-z0-9!$&'()*+,._~-]+$" $c.value) -}}
{{- fail (printf "%s.value must be URL-safe (it is placed in a connection-string userinfo unencoded): allowed characters are letters, digits, and !$&'()*+,._~- . Use source=existingSecret for a password with other characters." $path) -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/* Whether a keyed backend entry is enabled (missing `enabled` => true). */}}
{{- define "deepfellow.backendEnabled" -}}
{{- if hasKey .b "enabled" -}}{{- if .b.enabled -}}true{{- end -}}{{- else -}}true{{- end -}}
{{- end -}}

{{- define "deepfellow.validate" -}}

{{/* ---------- Images (fail-closed on mutable tags) ---------- */}}
{{- include "deepfellow.validateImage" (dict "image" .Values.infra.image "path" "infra.image") -}}
{{- include "deepfellow.validateImage" (dict "image" .Values.server.image "path" "server.image") -}}
{{- if eq .Values.server.mongo.mode "embedded" -}}
{{- include "deepfellow.validateImage" (dict "image" .Values.server.mongo.image "path" "server.mongo.image") -}}
{{- end -}}
{{- if .Values.server.graph.enabled -}}
{{- include "deepfellow.validateImage" (dict "image" .Values.server.graph.image "path" "server.graph.image") -}}
{{- end -}}
{{- if .Values.server.vectorDb.active -}}
{{- if eq .Values.server.vectorDb.type "qdrant" -}}
{{- if not .Values.server.vectorDb.qdrant.url -}}
{{- include "deepfellow.validateImage" (dict "image" .Values.server.vectorDb.qdrant.image "path" "server.vectorDb.qdrant.image") -}}
{{- end -}}
{{- else if eq .Values.server.vectorDb.type "milvus" -}}
{{- include "deepfellow.validateImage" (dict "image" .Values.server.vectorDb.milvus.etcdImage "path" "server.vectorDb.milvus.etcdImage") -}}
{{- include "deepfellow.validateImage" (dict "image" .Values.server.vectorDb.milvus.minioImage "path" "server.vectorDb.milvus.minioImage") -}}
{{- include "deepfellow.validateImage" (dict "image" .Values.server.vectorDb.milvus.milvusImage "path" "server.vectorDb.milvus.milvusImage") -}}
{{- end -}}
{{- end -}}
{{- if .Values.workspace.enabled -}}
{{- include "deepfellow.validateImage" (dict "image" .Values.workspace.image "path" "workspace.image") -}}
{{- include "deepfellow.validateImage" (dict "image" .Values.workspace.mongo.image "path" "workspace.mongo.image") -}}
{{- end -}}
{{- if and .Values.provisioning.enabled .Values.provisioning.image.repository -}}
{{- include "deepfellow.validateImage" (dict "image" .Values.provisioning.image "path" "provisioning.image") -}}
{{- end -}}

{{/* ---------- Infra credentials + mesh + externalOnly ---------- */}}
{{- include "deepfellow.validateSecret" (dict "cred" .Values.infra.auth.adminApiKey "path" "infra.auth.adminApiKey" "required" true) -}}
{{- include "deepfellow.validateSecret" (dict "cred" .Values.infra.auth.apiKey "path" "infra.auth.apiKey" "required" true) -}}
{{- include "deepfellow.validateSecret" (dict "cred" .Values.infra.huggingFaceToken "path" "infra.huggingFaceToken" "required" false) -}}
{{- if not .Values.infra.mesh.key -}}
{{- fail "infra.mesh.key must be non-empty: the Infra image requires DF_MESH_KEY at boot even when no mesh is joined." -}}
{{- end -}}
{{- if and .Values.infra.mesh.enabled (not .Values.infra.mesh.connectUrl) -}}
{{- fail "infra.mesh.connectUrl is required when infra.mesh.enabled=true." -}}
{{- end -}}
{{- if not (has .Values.infra.externalOnly.enforcement (list "compatibility" "patched")) -}}
{{- fail "infra.externalOnly.enforcement must be one of: compatibility, patched." -}}
{{- end -}}
{{- if and (eq .Values.infra.externalOnly.enforcement "patched") (not .Values.infra.image.digest) -}}
{{- fail "infra.externalOnly.enforcement=patched requires a pinned infra.image.digest (the patched image identity)." -}}
{{- end -}}

{{/* ---------- Native + external backends ---------- */}}
{{- $nativeEnabled := 0 -}}
{{- $modelIds := dict -}}
{{- $instances := dict -}}
{{- range $name, $bRaw := .Values.modelBackends -}}
{{- if include "deepfellow.backendEnabled" (dict "b" $bRaw) -}}
{{- $b := mergeOverwrite (include "deepfellow.modelBackendDefaults" $ | fromYaml) (deepCopy $bRaw) -}}
{{- $nativeEnabled = add1 $nativeEnabled -}}
{{- if ne $b.type "llamaCpp" -}}
{{- fail (printf "modelBackends.%s.type must be llamaCpp." $name) -}}
{{- end -}}
{{- if not $b.gguf.url -}}
{{- fail (printf "modelBackends.%s.gguf.url is required (the GGUF weights to download)." $name) -}}
{{- end -}}
{{- include "deepfellow.validateImage" (dict "image" $b.image "path" (printf "modelBackends.%s.image" $name)) -}}
{{- include "deepfellow.validateImage" (dict "image" $b.downloadImage "path" (printf "modelBackends.%s.downloadImage" $name)) -}}
{{- if $b.gguf.hfToken -}}
{{- include "deepfellow.validateSecret" (dict "cred" $b.gguf.hfToken "path" (printf "modelBackends.%s.gguf.hfToken" $name) "required" false) -}}
{{- end -}}
{{- $mid := default $name $b.modelId -}}
{{- if hasKey $modelIds $mid -}}
{{- fail (printf "duplicate modelId %q (modelBackends.%s and %s): model ids must be unique across modelBackends and externalBackends." $mid $name (index $modelIds $mid)) -}}
{{- end -}}
{{- $modelIds = merge $modelIds (dict $mid $name) -}}
{{- $instances = merge $instances (dict $name "modelBackends") -}}
{{- end -}}
{{- end -}}
{{- if and (gt $nativeEnabled 0) (not .Values.infra.externalOnly.enabled) -}}
{{- fail "native modelBackends are enabled but infra.externalOnly.enabled is false: native llama.cpp backends are registered as EXTERNAL services and require infra.externalOnly.enabled=true." -}}
{{- end -}}
{{- $externalEnabled := 0 -}}
{{- range $name, $b := .Values.externalBackends -}}
{{- if include "deepfellow.backendEnabled" (dict "b" $b) -}}
{{- $externalEnabled = add1 $externalEnabled -}}
{{- if not (has (default "openai" $b.serviceType) (list "openai")) -}}
{{- fail (printf "externalBackends.%s.serviceType must be one of: openai." $name) -}}
{{- end -}}
{{- if not $b.apiUrl -}}
{{- fail (printf "externalBackends.%s.apiUrl is required (the endpoint ROOT URL; Infra appends v1/)." $name) -}}
{{- end -}}
{{- if not $b.modelId -}}
{{- fail (printf "externalBackends.%s.modelId is required." $name) -}}
{{- end -}}
{{- if $b.apiKey -}}
{{- include "deepfellow.validateSecret" (dict "cred" $b.apiKey "path" (printf "externalBackends.%s.apiKey" $name) "required" false) -}}
{{- end -}}
{{- if hasKey $modelIds $b.modelId -}}
{{- fail (printf "duplicate modelId %q (externalBackends.%s and %s): model ids must be unique across modelBackends and externalBackends." $b.modelId $name (index $modelIds $b.modelId)) -}}
{{- end -}}
{{- $modelIds = merge $modelIds (dict $b.modelId $name) -}}
{{- if hasKey $instances $name -}}
{{- fail (printf "duplicate backend key %q (externalBackends.%s and modelBackends.%s): the map key is the Infra service instance name and must be unique across modelBackends and externalBackends." $name $name $name) -}}
{{- end -}}
{{- $instances = merge $instances (dict $name "externalBackends") -}}
{{- end -}}
{{- end -}}

{{/* ---------- Server ---------- */}}
{{- include "deepfellow.validateSecret" (dict "cred" .Values.server.admin.password "path" "server.admin.password" "required" true) -}}
{{- include "deepfellow.validatePasswordComplexity" (dict "cred" .Values.server.admin.password "path" "server.admin.password") -}}
{{- if not (has .Values.server.mongo.mode (list "embedded" "external")) -}}
{{- fail "server.mongo.mode must be one of: embedded, external." -}}
{{- end -}}
{{- if eq .Values.server.mongo.mode "embedded" -}}
{{- include "deepfellow.validateSecret" (dict "cred" .Values.server.mongo.auth.rootPassword "path" "server.mongo.auth.rootPassword" "required" true) -}}
{{- include "deepfellow.validateSecret" (dict "cred" .Values.server.mongo.auth.password "path" "server.mongo.auth.password" "required" true) -}}
{{- else -}}
{{- if not .Values.server.mongo.external.uriSecret.name -}}
{{- fail "server.mongo.external.uriSecret.name is required when server.mongo.mode=external." -}}
{{- end -}}
{{- end -}}
{{- if .Values.server.vectorDb.active -}}
{{- if not (has .Values.server.vectorDb.type (list "qdrant" "milvus")) -}}
{{- fail "server.vectorDb.type must be one of: qdrant, milvus." -}}
{{- end -}}
{{- end -}}
{{- if .Values.server.graph.enabled -}}
{{- include "deepfellow.validateSecret" (dict "cred" .Values.server.graph.password "path" "server.graph.password" "required" true) -}}
{{- end -}}
{{- if .Values.server.otel.enabled -}}
{{- if not .Values.server.otel.endpoint -}}
{{- fail "server.otel.endpoint is required when server.otel.enabled=true." -}}
{{- end -}}
{{- end -}}
{{- if .Values.server.metrics.enabled -}}
{{- include "deepfellow.validateSecret" (dict "cred" .Values.server.metrics.password "path" "server.metrics.password" "required" true) -}}
{{- end -}}
{{- if and .Values.server.vectorDb.active (eq .Values.server.vectorDb.type "milvus") -}}
{{- include "deepfellow.validateSecret" (dict "cred" .Values.server.vectorDb.milvus.minio.accessKey "path" "server.vectorDb.milvus.minio.accessKey" "required" true) -}}
{{- include "deepfellow.validateSecret" (dict "cred" .Values.server.vectorDb.milvus.minio.secretKey "path" "server.vectorDb.milvus.minio.secretKey" "required" true) -}}
{{- end -}}

{{/* ---------- Ingress ---------- */}}
{{- if .Values.ingress.enabled -}}
{{- $hasServer := and .Values.ingress.server.host true -}}
{{- $hasWorkspace := and .Values.workspace.enabled .Values.ingress.workspace.host -}}
{{- if not (or $hasServer $hasWorkspace) -}}
{{- fail "ingress.enabled=true but no host is set: set ingress.server.host and/or ingress.workspace.host (an Ingress with no rules is invalid)." -}}
{{- end -}}
{{- end -}}

{{/* ---------- Workspace ---------- */}}
{{- if .Values.workspace.enabled -}}
{{- if and (not .Values.provisioning.enabled) (not .Values.provisioning.stateSecret.name) -}}
{{- fail "workspace.enabled=true needs the provisioning state Secret (DF_SERVER_* dfproj identity): enable provisioning.enabled, or set provisioning.stateSecret.name to a Secret you manage. Otherwise the Workspace pod never starts." -}}
{{- end -}}
{{- include "deepfellow.validateSecret" (dict "cred" .Values.workspace.auth.betterAuthSecret "path" "workspace.auth.betterAuthSecret" "required" true) -}}
{{- include "deepfellow.validateSecret" (dict "cred" .Values.workspace.auth.bootstrapAdmin.password "path" "workspace.auth.bootstrapAdmin.password" "required" true) -}}
{{- include "deepfellow.validateSecret" (dict "cred" .Values.workspace.mongo.auth.rootPassword "path" "workspace.mongo.auth.rootPassword" "required" true) -}}
{{- include "deepfellow.validateUrlSafeValue" (dict "cred" .Values.workspace.mongo.auth.rootPassword "path" "workspace.mongo.auth.rootPassword") -}}
{{- end -}}

{{/* ---------- Provisioning gate ---------- */}}
{{- if .Values.provisioning.enabled -}}
{{- if and (eq (add $nativeEnabled $externalEnabled) 0) (not .Values.provisioning.allowNoModels) -}}
{{- fail "provisioning.enabled=true but no native modelBackends or externalBackends are enabled: the suite would provision a project with no usable model. Enable a backend or set provisioning.allowNoModels=true." -}}
{{- end -}}
{{- end -}}

{{- end -}}
