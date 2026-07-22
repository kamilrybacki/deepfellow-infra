# container-crash-recovery Specification

## Purpose
Guarantee that the llama.cpp backend's in-memory VRAM bookkeeping (installed-models registry, `is_loaded` state, cached VRAM estimates) stays consistent with the Docker container that actually holds the GPU memory, so a crashed/killed container releases its reserved VRAM without requiring a backend restart — matching the crash-recovery behavior already provided for vLLM.

Ollama is intentionally out of scope: it runs every model inside a single per-instance container (no per-model container to strand), has no reserved VRAM budget counter to leak, and derives `is_loaded`/`vram_estimate_gb` live from Ollama's `/api/ps`. When its container dies those values already read as not-loaded on the next listing, so a reconciliation pass would only risk destructively de-registering still-installed models.

## Requirements
### Requirement: llama.cpp releases a crashed model's VRAM bookkeeping
Each llama.cpp model runs in its own Docker container. Once a model is successfully installed and its container later stops running for good (crash, OOM-kill, manual `docker stop`, or a process reported `unhealthy` — not a transient restart-in-progress), the system SHALL detect this and release the model's VRAM bookkeeping (drop it from the installed-models registry, clear its cached VRAM estimate, unregister its chat-completion endpoint, and tear down the dead container) without requiring a backend restart.

#### Scenario: Installed model's container crashes for good
- **WHEN** a successfully installed llama.cpp model's container stops running and does not come back within the reconciliation threshold
- **THEN** the model is removed from the installed-models registry (so it reports `is_loaded: false` with no VRAM estimate), its chat-completion endpoint is unregistered, and its container is torn down, without any backend restart

#### Scenario: Installed model's container restarts transiently
- **WHEN** a successfully installed llama.cpp model's container restarts (e.g. via its `unless-stopped` policy) and recovers within the reconciliation threshold
- **THEN** the model remains marked as loaded and its bookkeeping is unchanged

#### Scenario: Teardown of the dead container fails
- **WHEN** the dead container is detected but tearing it down raises an error
- **THEN** the model's bookkeeping is left intact and teardown is retried on the next reconciliation tick, rather than releasing VRAM for a container that was not confirmed gone

### Requirement: Crash detection does not disturb in-flight installs
While a model is being installed, the reconciliation pass SHALL NOT release or tear down that model, and it SHALL skip any model whose container reference has changed (uninstalled or reinstalled) between polls.

#### Scenario: Model being installed is skipped
- **WHEN** a reconciliation tick runs while a model of the instance is currently installing
- **THEN** that model's container is not inspected and its bookkeeping is left untouched

#### Scenario: Model reinstalled during polling is skipped
- **WHEN** a model that was accumulating bad polls is uninstalled or reinstalled before the threshold is reached
- **THEN** its stale crash-poll state is discarded and no release is performed against the new installation
