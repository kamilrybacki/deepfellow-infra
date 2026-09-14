#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""DeepFellow Suite provisioning — reconcile | verify | status.

Idempotent, fail-closed, and secret-safe:
  * The Kubernetes state Secret is the source of truth. On every run it is read
    FIRST; if it already carries organization_id/project_id/project_api_key the
    workspace/key creation is skipped and later steps reconcile desired state.
  * The one-time project API key is persisted to the state Secret BEFORE any grant.
    If the workspace was created remotely but the Secret write failed, the run fails
    closed for operator recovery rather than minting a second key.
  * No credential value is ever printed. Only ids and step outcomes are logged.

Runs inside the Server image (has python + the create_admin module). The Kubernetes
API is reached with the pod ServiceAccount token; a Role scopes writes to one Secret.
Only stdlib is used so the script does not depend on the image's extra packages.
"""

import base64
import json
import os
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request

SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"
K8S_API = "https://kubernetes.default.svc"


def log(msg):
    print(f"[provision] {msg}", flush=True)


def env(name, default=None, required=False):
    val = os.environ.get(name, default)
    if required and not val:
        raise SystemExit(f"missing required env {name}")
    return val


# --------------------------------------------------------------------------- HTTP


def _request(method, url, headers=None, body=None, ctx=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            parsed = json.loads(raw) if raw else {}
        except Exception:
            parsed = {"raw": raw.decode("utf-8", "replace")}
        return e.code, parsed
    except urllib.error.URLError as e:
        return 0, {"error": str(e.reason)}


# ---------------------------------------------------------------------- Kubernetes


def _k8s_ctx():
    return ssl.create_default_context(cafile=f"{SA_DIR}/ca.crt")


def _k8s_headers():
    with open(f"{SA_DIR}/token") as f:
        token = f.read().strip()
    return {"Authorization": f"Bearer {token}"}


def read_state_secret(ns, name):
    ctx = _k8s_ctx()
    status, body = _request(
        "GET", f"{K8S_API}/api/v1/namespaces/{ns}/secrets/{name}", _k8s_headers(), ctx=ctx
    )
    if status == 404:
        return None
    if status != 200:
        raise SystemExit(f"reading state secret failed: HTTP {status}")
    return {k: base64.b64decode(v).decode() for k, v in (body.get("data") or {}).items()}


def write_state_secret(ns, name, data):
    ctx = _k8s_ctx()
    encoded = {k: base64.b64encode(v.encode()).decode() for k, v in data.items()}
    existing = read_state_secret(ns, name)
    payload = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": name, "namespace": ns},
        "type": "Opaque",
        "data": encoded,
    }
    if existing is None:
        status, _ = _request(
            "POST", f"{K8S_API}/api/v1/namespaces/{ns}/secrets", _k8s_headers(), payload, ctx=ctx
        )
    else:
        hdr = dict(_k8s_headers())
        hdr["Content-Type"] = "application/merge-patch+json"
        status, _ = _request(
            "PATCH",
            f"{K8S_API}/api/v1/namespaces/{ns}/secrets/{name}",
            hdr,
            {"data": encoded},
            ctx=ctx,
        )
    if status not in (200, 201):
        raise SystemExit(f"persisting state secret failed: HTTP {status}")


# ------------------------------------------------------------------- DeepFellow


def wait_healthy(url, label, attempts=60, delay=5, accept_any=False):
    # accept_any: treat any HTTP response (connection accepted) as up. Use it for the
    # separate Server image, whose HTTP /health shape is not established here (verified
    # only in the Gate A / Phase 6 smoke); the Infra /health returns 200.
    for _ in range(attempts):
        status, _ = _request("GET", f"{url}/health")
        if status == 200 or (accept_any and status != 0):
            log(f"{label} reachable")
            return
        time.sleep(delay)
    raise SystemExit(f"{label} not reachable at {url}")


def create_admin(name, email, password):
    # existing email is a successful no-op; any other non-zero exit is fatal.
    proc = subprocess.run(
        [sys.executable, "-m", "server.scripts.create_admin", name, email, password],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        log("admin created")
        return
    combined = (proc.stdout + proc.stderr).lower()
    if "exist" in combined or "already" in combined:
        log("admin already exists (no-op)")
        return
    raise SystemExit(f"create_admin failed (rc={proc.returncode})")


def login(server_url, email, password):
    status, body = _request(
        "POST", f"{server_url}/auth/login", body={"email": email, "password": password}
    )
    token = body.get("token") or body.get("access_token") or body.get("jwt")
    if status != 200 or not token:
        raise SystemExit(f"login failed: HTTP {status}")
    return token


def ensure_workspace(server_url, jwt, org_name, project_name, key_name, state, ns, state_secret):
    if state and state.get("organization-id") and state.get("project-id") and state.get("project-api-key"):
        log("workspace/key already provisioned (from state secret)")
        return state["organization-id"], state["project-id"], state["project-api-key"]

    auth = {"Authorization": f"Bearer {jwt}"}
    status, body = _request("POST", f"{server_url}/admin/organization/", auth, {"name": org_name})
    if status not in (200, 201):
        raise SystemExit(f"create organization failed: HTTP {status}")
    org_id = body["organization"]["id"]

    org_auth = dict(auth)
    org_auth["OpenAI-Organization"] = org_id
    status, body = _request(
        "POST", f"{server_url}/v1/organization/projects", org_auth, {"name": project_name}
    )
    if status not in (200, 201):
        raise SystemExit(f"create project failed: HTTP {status}")
    project_id = body["id"]

    status, body = _request(
        "POST",
        f"{server_url}/v1/organization/projects/{project_id}/api_keys",
        org_auth,
        {"name": key_name},
    )
    if status not in (200, 201) or not body.get("value"):
        raise SystemExit(f"create project api key failed: HTTP {status}")
    project_key = body["value"]

    # Persist the GET-ONCE key immediately, before any grant. If this write fails the
    # workspace/key exists remotely but is unrecoverable here: fail closed for operator
    # recovery rather than let a later run mint a second key.
    try:
        write_state_secret(ns, state_secret, {
            "organization-id": org_id, "project-id": project_id, "project-api-key": project_key,
        })
    except SystemExit as e:
        raise SystemExit(
            "workspace/key created remotely but persisting the state secret failed "
            f"({e}); the GET-ONCE key is unrecoverable — manual recovery required "
            "(delete the remote project/key, then re-run)."
        )
    return org_id, project_id, project_key


def register_backend(infra_url, admin_key, model_id, api_url, api_key, context_length):
    auth = {"Authorization": f"Bearer {admin_key}"}
    status, _ = _request(
        "POST",
        f"{infra_url}/admin/services/openai",
        auth,
        {"spec": {"api_url": api_url, "api_key": api_key or "none"}, "stream": False, "ignore_warnings": True},
    )
    if status not in (200, 201, 409):
        log(f"service register for {model_id} returned HTTP {status} (continuing to reconcile)")

    status, body = _request(
        "POST",
        f"{infra_url}/admin/services/openai/models/custom",
        auth,
        {"spec": {
            "id": model_id, "type": "llm", "completions": True, "legacy_completions": False,
            "responses": False, "messages": False,
            "context_length": context_length, "max_context_length": context_length,
        }},
    )
    already = status == 400 and "exist" in json.dumps(body).lower()
    if status not in (200, 201) and not already:
        raise SystemExit(f"custom model add for {model_id} failed: HTTP {status}")

    status, _ = _request(
        "POST",
        f"{infra_url}/admin/services/openai/models/_?model_id={model_id}",
        auth,
        {"stream": False, "ignore_warnings": True},
    )
    if status not in (200, 201, 409):
        raise SystemExit(f"model install for {model_id} failed: HTTP {status}")
    log(f"backend {model_id} registered")


def infra_model_ready(infra_url, admin_key, model_id, attempts=30, delay=5):
    auth = {"Authorization": f"Bearer {admin_key}"}
    for _ in range(attempts):
        status, body = _request("GET", f"{infra_url}/v1/models", auth)
        if status == 200:
            ids = [m.get("id") for m in body.get("data", [])]
            if model_id in ids:
                return
        time.sleep(delay)
    raise SystemExit(f"infra model {model_id} not ready")


def grant_models(server_url, jwt, org_id, project_id, model_ids):
    auth = {"Authorization": f"Bearer {jwt}", "OpenAI-Organization": org_id}
    status, _ = _request(
        "POST",
        f"{server_url}/v1/organization/projects/{project_id}",
        auth,
        {"models": model_ids},
    )
    if status not in (200, 201):
        raise SystemExit(f"grant models failed: HTTP {status}")
    log(f"granted {len(model_ids)} model(s) to project")


def verify(server_url, project_key, model_ids):
    auth = {"Authorization": f"Bearer {project_key}"}
    status, body = _request("GET", f"{server_url}/v1/models", auth)
    if status != 200:
        raise SystemExit(f"verify: server /v1/models HTTP {status}")
    listed = {m.get("id") for m in body.get("data", [])}
    missing = [m for m in model_ids if m not in listed]
    if missing:
        raise SystemExit(f"verify: models not exposed via project key: {missing}")
    log("verify: all granted models exposed via the project key")


# ------------------------------------------------------------------------- main


def load_config():
    return {
        "ns": env("POD_NAMESPACE", required=True),
        "state_secret": env("STATE_SECRET_NAME", required=True),
        "infra_url": env("DF_INFRA_URL", required=True),
        "infra_admin_key": env("DF_INFRA_ADMIN_API_KEY", required=True),
        "server_url": env("DF_SERVER_URL", required=True),
        "admin_name": env("ADMIN_NAME", "admin"),
        "admin_email": env("ADMIN_EMAIL", required=True),
        "admin_password": env("ADMIN_PASSWORD", required=True),
        "org_name": env("PROJECT_ORG_NAME", "Workspace"),
        "project_name": env("PROJECT_NAME", "Default"),
        "key_name": env("PROJECT_KEY_NAME", "app"),
        "backends": json.loads(env("BACKENDS_JSON", "[]")),
    }


def do_reconcile(c):
    wait_healthy(c["infra_url"], "infra")
    wait_healthy(c["server_url"], "server", accept_any=True)
    for b in c["backends"]:
        if b.get("native"):
            wait_healthy(b["api_url"], f"backend {b['id']}")

    create_admin(c["admin_name"], c["admin_email"], c["admin_password"])
    jwt = login(c["server_url"], c["admin_email"], c["admin_password"])

    state = read_state_secret(c["ns"], c["state_secret"])
    org_id, project_id, project_key = ensure_workspace(
        c["server_url"], jwt, c["org_name"], c["project_name"], c["key_name"],
        state, c["ns"], c["state_secret"],
    )

    model_ids = [b["id"] for b in c["backends"]]
    for b in c["backends"]:
        register_backend(
            c["infra_url"], c["infra_admin_key"], b["id"], b["api_url"],
            b.get("api_key", ""), int(b.get("context_length", 8192)),
        )
    for b in c["backends"]:
        infra_model_ready(c["infra_url"], c["infra_admin_key"], b["id"])

    if model_ids:
        grant_models(c["server_url"], jwt, org_id, project_id, model_ids)
        verify(c["server_url"], project_key, model_ids)
    log("reconcile complete")


def do_verify(c):
    state = read_state_secret(c["ns"], c["state_secret"])
    if not state or not state.get("project-api-key"):
        raise SystemExit("verify: no provisioned state secret")
    verify(c["server_url"], state["project-api-key"], [b["id"] for b in c["backends"]])


def do_status(c):
    state = read_state_secret(c["ns"], c["state_secret"])
    provisioned = bool(state and state.get("project-api-key"))
    log(f"state secret present: {provisioned}; backends: {len(c['backends'])}")


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "reconcile"
    c = load_config()
    if action == "reconcile":
        do_reconcile(c)
    elif action == "verify":
        do_verify(c)
    elif action == "status":
        do_status(c)
    else:
        raise SystemExit(f"unknown action {action}: use reconcile | verify | status")


if __name__ == "__main__":
    main()
