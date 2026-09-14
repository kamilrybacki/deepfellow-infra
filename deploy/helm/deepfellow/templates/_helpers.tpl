{{/*
Chart name / fullname helpers.
*/}}
{{- define "deepfellow.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "deepfellow.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "deepfellow.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels applied to every resource.
*/}}
{{- define "deepfellow.labels" -}}
helm.sh/chart: {{ include "deepfellow.chart" . }}
app.kubernetes.io/name: {{ include "deepfellow.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: deepfellow
{{- end -}}

{{/*
Selector labels for a given component. Call as:
  {{ include "deepfellow.selectorLabels" (dict "context" . "component" "infra") }}
*/}}
{{- define "deepfellow.selectorLabels" -}}
app.kubernetes.io/name: {{ include "deepfellow.name" .context }}
app.kubernetes.io/instance: {{ .context.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{/*
Component resource name: <fullname>-<component>. Call as:
  {{ include "deepfellow.componentName" (dict "context" . "component" "server") }}
*/}}
{{- define "deepfellow.componentName" -}}
{{- printf "%s-%s" (include "deepfellow.fullname" .context) .component | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Resolve an image ref from an image sub-map ({repository, tag, digest}), honoring
global.imageRegistry. Digest wins over tag when present. Call as:
  {{ include "deepfellow.image" (dict "context" . "image" .Values.infra.image) }}
*/}}
{{- define "deepfellow.image" -}}
{{- $registry := .context.Values.global.imageRegistry -}}
{{- $repo := .image.repository -}}
{{- if $registry -}}
{{- $repo = printf "%s/%s" $registry $repo -}}
{{- end -}}
{{- if .image.digest -}}
{{- printf "%s@%s" $repo .image.digest -}}
{{- else -}}
{{- printf "%s:%s" $repo (.image.tag | toString) -}}
{{- end -}}
{{- end -}}

{{/*
Render global.imagePullSecrets as a `imagePullSecrets:` list value (may be empty).
*/}}
{{- define "deepfellow.imagePullSecrets" -}}
{{- with .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- range . }}
  - name: {{ .name | default . }}
{{- end }}
{{- end -}}
{{- end -}}

{{/*
Default pod / container security contexts (non-root, hardened). Components spread these.
*/}}
{{- define "deepfellow.podSecurityContext" -}}
runAsNonRoot: true
runAsUser: 1000
runAsGroup: 1000
fsGroup: 1000
seccompProfile:
  type: RuntimeDefault
{{- end -}}

{{- define "deepfellow.containerSecurityContext" -}}
allowPrivilegeEscalation: false
capabilities:
  drop:
    - ALL
{{- end -}}

{{/*
Name of the chart-managed Secret holding a component's value/generate credentials.
  {{ include "deepfellow.credSecretName" (dict "context" . "component" "infra") }}
*/}}
{{- define "deepfellow.credSecretName" -}}
{{- printf "%s-creds" (include "deepfellow.componentName" (dict "context" .context "component" .component)) -}}
{{- end -}}

{{/*
Deterministic generated-credential value that satisfies the create_admin password policy
(>=10 chars incl lower/upper/digit/special). randAlphaNum alone has no special character,
which create_admin rejects — so append one of each class.
*/}}
{{- define "deepfellow.genSecretValue" -}}
{{- printf "%sAa1!" (randAlphaNum 28) -}}
{{- end -}}

{{/*
Emit ONE env-var list entry for a credential sub-map, sourcing it from the right Secret.
Nothing is emitted when source=none (caller decides whether that's allowed).
  {{ include "deepfellow.credEnv" (dict "name" "DF_X" "cred" .Values....x "secretName" $s) }}
*/}}
{{- define "deepfellow.credEnv" -}}
{{- $c := .cred -}}
{{- if and .required (eq $c.source "none") -}}
{{- fail (printf "credEnv %s: source=none but this env var is required." .name) -}}
{{- end -}}
{{- if ne $c.source "none" -}}
- name: {{ .name }}
  valueFrom:
    secretKeyRef:
{{- if eq $c.source "existingSecret" }}
      name: {{ $c.existingSecret.name }}
      key: {{ $c.existingSecret.key }}
{{- else }}
      name: {{ .secretName }}
      key: {{ $c.existingSecret.key }}
{{- end }}
{{- end -}}
{{- end -}}

{{/*
Render a chart-managed Secret carrying every value/generate credential in `creds`
(a dict of label->credSubmap). Renders nothing when all creds are existingSecret/none.
generate: reuse the live Secret's value (no upgrade drift); on a fresh install mint a
policy-compliant value; fail on upgrade/GitOps where lookup is empty (use existingSecret).
  {{ include "deepfellow.chartSecret" (dict "context" . "secretName" $s "creds" (dict "a" .Values....a)) }}
*/}}
{{- define "deepfellow.chartSecret" -}}
{{- $ctx := .context -}}
{{- $secretName := .secretName -}}
{{- $data := dict -}}
{{- $keep := false -}}
{{- range $label, $c := .creds -}}
{{- if or (eq $c.source "value") (eq $c.source "generate") -}}
{{- $key := $c.existingSecret.key -}}
{{- $val := "" -}}
{{- if eq $c.source "value" -}}
{{- $val = $c.value -}}
{{- else -}}
{{- if dig "generate" "retain" false $c -}}{{- $keep = true -}}{{- end -}}
{{- $existing := (lookup "v1" "Secret" $ctx.Release.Namespace $secretName) -}}
{{- if and $existing (hasKey ($existing.data | default dict) $key) -}}
{{- $val = (index $existing.data $key | b64dec) -}}
{{- else if $ctx.Release.IsInstall -}}
{{- $val = (include "deepfellow.genSecretValue" $ctx) -}}
{{- else -}}
{{- fail (printf "generated credential %q not found in Secret %q and this is not a fresh install: GitOps/upgrade must use source=existingSecret." $key $secretName) -}}
{{- end -}}
{{- end -}}
{{- $data = merge $data (dict $key ($val | b64enc)) -}}
{{- end -}}
{{- end -}}
{{- if gt (len $data) 0 -}}
apiVersion: v1
kind: Secret
metadata:
  name: {{ $secretName }}
  labels:
    {{- include "deepfellow.labels" $ctx | nindent 4 }}
{{- if $keep }}
  annotations:
    helm.sh/resource-policy: keep
{{- end }}
type: Opaque
data:
{{- range $k, $v := $data }}
  {{ $k }}: {{ $v }}
{{- end }}
{{- end -}}
{{- end -}}

{{/*
Per-entry defaults for a native modelBackends map entry (map entries do NOT inherit the
commented example in values.yaml). Merge with `mergeOverwrite (defaults) (deepCopy $b)` so
user-supplied fields win. Used by BOTH _validate.tpl and model-backends.yaml — keep it the
single source of these defaults.
*/}}
{{- define "deepfellow.modelBackendDefaults" -}}
enabled: true
type: llamaCpp
modelId: ""
image:
  repository: ghcr.io/ggml-org/llama.cpp
  tag: server
  digest: ""
  allowMutableTag: false
  pullPolicy: IfNotPresent
gguf:
  url: ""
  sha256: ""
downloadImage:
  repository: curlimages/curl
  tag: "8.11.1"
  digest: ""
  allowMutableTag: false
  pullPolicy: IfNotPresent
parallel: 1
threads: 4
ctxSize: 8192
extraArgs: []
gpu:
  enabled: false
  count: 1
storage:
  size: 20Gi
  medium: ""
  storageClass: ""
service:
  port: 8080
{{- end -}}

{{/*
Name of the Secret holding the GET-ONCE dfproj provisioning state (org_id/project_id/key).
Keys: organization-id, project-id, project-api-key. Minted by the Phase-3 provisioning Job.
*/}}
{{- define "deepfellow.provisioning.stateSecretName" -}}
{{- default (printf "%s-provisioning-state" (include "deepfellow.fullname" .)) .Values.provisioning.stateSecret.name -}}
{{- end -}}

{{/*
Provisioning ServiceAccount name.
*/}}
{{- define "deepfellow.provisioning.serviceAccountName" -}}
{{- if .Values.provisioning.serviceAccount.create -}}
{{- default (include "deepfellow.componentName" (dict "context" . "component" "provisioning")) .Values.provisioning.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.provisioning.serviceAccount.name -}}
{{- end -}}
{{- end -}}
