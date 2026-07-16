# vllm-error-cleanup Specification

## Purpose
Guarantee that the vLLM backend's in-memory VRAM bookkeeping (`gpu_memory_utilization` counter, installed-models registry) stays consistent with the physical Docker container that actually holds the GPU memory, on every error path around installing a model.
## Requirements
### Requirement: VRAM bookkeeping matches container state on install failure
When installing a vLLM model fails at any point after the GPU utilization counter was incremented, the system SHALL release the reserved `gpu_memory_utilization` fraction, stop the model's Docker container, and remove the model from the installed-models registry before propagating the error. Additionally, once a model is successfully installed and its container later stops running for good (crash, OOM-kill, or exhausted restart policy — not a transient restart-in-progress), the system SHALL detect this and release the reserved `gpu_memory_utilization` fraction without requiring a backend restart.

#### Scenario: Container start fails
- **WHEN** `install_and_run_docker` raises an exception during model install
- **THEN** the GPU utilization counter is decremented by the model's reserved fraction and the container is stopped before the error propagates

#### Scenario: Failure after container started
- **WHEN** the container starts successfully but a subsequent install step (e.g. endpoint registration) raises an exception
- **THEN** the GPU utilization counter is decremented, the container is stopped, the model is absent from the installed-models registry, and the error propagates

#### Scenario: Install succeeds
- **WHEN** the model install completes without error
- **THEN** the GPU utilization counter includes the model's reserved fraction and the container remains running

#### Scenario: Installed model's container crashes for good
- **WHEN** a successfully installed model's container stops running and does not come back within the reconciliation threshold (not just mid-restart)
- **THEN** the GPU utilization counter is decremented by the model's reserved fraction and the model's bookkeeping reflects that it is no longer running, without any backend restart

#### Scenario: Installed model's container restarts transiently
- **WHEN** a successfully installed model's container restarts (e.g. via its `unless-stopped` policy) and recovers within the reconciliation threshold
- **THEN** the GPU utilization counter is unchanged and the model remains marked as running

### Requirement: Cancelled install leaves no orphan container
When a model install is cancelled (`asyncio.CancelledError`), the system SHALL release the reserved GPU utilization fraction AND stop the model's Docker container, even though the surrounding task is being cancelled.

#### Scenario: Cancellation during container start
- **WHEN** an install is cancelled while `install_and_run_docker` is in progress
- **THEN** the GPU utilization counter returns to its prior value and `stop_docker` is invoked for the model's container before the cancellation propagates

