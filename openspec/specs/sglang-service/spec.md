# sglang-service Specification

## Purpose
Provide a self-hosted SGLang model runner, alongside the existing vLLM/Ollama/llama.cpp services, exposing chat/completions, embedding, and reranking models through the endpoint registry, with per-model Docker lifecycle management, GPU-only hardware support, and a versioned Docker image selector.

## Requirements
### Requirement: SGLang service installs on GPU hardware only

The system SHALL offer an SGLang service that can be installed only on hardware with GPU support (NVIDIA GPUs recognized by the existing hardware detection). Attempting to select CPU-only hardware for the SGLang service SHALL be rejected with a clear error, the same way `VllmService` rejects unsupported hardware.

#### Scenario: Install on a machine with a supported NVIDIA GPU
- **WHEN** a user installs the SGLang service selecting a detected GPU as the hardware target
- **THEN** the service installs successfully and the container is launched with GPU access

#### Scenario: Install attempted with CPU-only hardware
- **WHEN** a user attempts to install the SGLang service with `hardware` resolved to `"CPU"` (or a machine with no supported GPU)
- **THEN** the system raises an HTTP 400 error explaining the hardware is unsupported, and no container is created

### Requirement: SGLang model catalog is derived from the existing vLLM model registry

The system SHALL populate the SGLang service's available-models list from the same underlying HuggingFace checkpoint catalog vLLM uses (`static/vllm-min.json`), tagging each entry with its `model_type` (`llm`, `reranker`, or `embedding`) exactly as the source registry does, without introducing a separate generation script or catalog file.

#### Scenario: Listing available SGLang models
- **WHEN** a user lists installable models for the SGLang service
- **THEN** the returned list matches the `llm`, `reranker`, and `embedding` entries from the existing vLLM model registry, each with its `model_type` preserved

#### Scenario: Custom model addition
- **WHEN** a user adds a custom model to the SGLang service by HuggingFace ID, size, and model type
- **THEN** the custom model is added to that instance's catalog the same way `VllmService` supports custom models, and can be installed like a catalog entry

### Requirement: SGLang container command targets the correct engine invocation and endpoint

Installing any SGLang model SHALL launch a Docker container running `python3 -m sglang.launch_server` (not a bare flag list), binding to port `30000` inside the container, and passing `--model-path`, `--served-model-name`, `--host 0.0.0.0`, `--port 30000`, plus any resolved quantization, context length, and memory-utilization flags mapped to SGLang's flag names (`--mem-fraction-static`, `--context-length`, `--quantization`).

#### Scenario: Installing an LLM model
- **WHEN** a user installs an `llm`-type model on the SGLang service with GPU hardware
- **THEN** the resulting Docker container's command starts with `python3 -m sglang.launch_server`, includes `--model-path` pointing at the downloaded checkpoint, `--served-model-name` set to the model id (or alias), `--host 0.0.0.0`, and `--port 30000`

#### Scenario: GPU memory utilization option applied
- **WHEN** a user sets `gpu_memory_utilization` on a model's install options
- **THEN** the container command includes `--mem-fraction-static` set to the resolved value, honoring the same aggregate-utilization cap (sum across models on this service instance never exceeds 1) that `VllmService` enforces

#### Scenario: Max context length option applied
- **WHEN** a user sets `max_model_length` on a model's install options
- **THEN** the container command includes `--context-length` set to the resolved value

### Requirement: SGLang reranker install resolves the `--is-embedding` flag per model family

When installing a `reranker`-type model, the system SHALL decide whether to pass `--is-embedding` based on the model's identity: models recognized as belonging to the Qwen3-Reranker family SHALL be launched without `--is-embedding`; all other reranker models SHALL be launched with `--is-embedding` by default. If the container nonetheless fails to start and its logs report the flag is incompatible with the loaded model, the system SHALL surface an actionable error rather than a generic Docker failure.

#### Scenario: Installing a cross-encoder reranker
- **WHEN** a user installs a reranker model not recognized as Qwen3-Reranker family
- **THEN** the container command includes `--is-embedding`

#### Scenario: Installing a Qwen3-Reranker model
- **WHEN** a user installs a reranker model recognized as belonging to the Qwen3-Reranker family
- **THEN** the container command omits `--is-embedding`

#### Scenario: Container rejects the resolved flag choice at startup
- **WHEN** the SGLang container fails to start and its logs contain the engine's "relaunch without --is-embedding" error message
- **THEN** the system surfaces an error identifying the `--is-embedding` mismatch as the cause, instead of a bare Docker startup failure

### Requirement: SGLang models register OpenAI-compatible proxy endpoints matching their type

Once an SGLang model container is running, the system SHALL register endpoint-registry proxies pointing at the model's SGLang server: `llm`-type models register chat completions, completions, responses, and messages proxies; `reranker`-type models register a rerank proxy against `/v1/rerank`; `embedding`-type models register an embeddings proxy against `/v1/embeddings`. Uninstalling the model SHALL unregister the corresponding proxy/proxies.

#### Scenario: LLM model registration
- **WHEN** an `llm`-type SGLang model finishes installing
- **THEN** chat completion, completions, responses, and messages endpoints are registered against that container, rewriting the requested model to the SGLang-served model id

#### Scenario: Reranker model registration
- **WHEN** a `reranker`-type SGLang model finishes installing
- **THEN** a rerank endpoint is registered pointing at that container's `/v1/rerank` path

#### Scenario: Embedding model registration
- **WHEN** an `embedding`-type SGLang model finishes installing
- **THEN** an embeddings endpoint is registered pointing at that container's `/v1/embeddings` path

#### Scenario: Uninstall removes registration
- **WHEN** an installed SGLang model is uninstalled
- **THEN** its registered endpoint(s) are unregistered and its Docker container is stopped

### Requirement: SGLang container healthcheck uses generation-based liveness

The SGLang service's Docker container healthcheck SHALL probe the `/health_generate` endpoint (which exercises an actual forward pass) rather than a bare liveness endpoint, with a startup grace period tolerant of model load time.

#### Scenario: Healthcheck configuration
- **WHEN** an SGLang model container is created
- **THEN** its healthcheck test targets `/health_generate` on the container's SGLang port, with a start period long enough to accommodate model loading

### Requirement: SGLang Docker image version is selectable from live registry tags

The system SHALL expose a Docker image version selector field on the SGLang service specification, backed by live tag data from the `lmsysorg/sglang` Docker Hub repository, filtered to GPU-capable (CUDA-suffixed) tags, with a specific pinned version (not a floating `latest`/`dev` tag) as the default.

#### Scenario: Fetching available tags
- **WHEN** a user opens the SGLang service install/edit form
- **THEN** the image version field is populated from live `lmsysorg/sglang` tags, filtered to CUDA-suffixed (GPU) variants

#### Scenario: Default version
- **WHEN** a user installs the SGLang service without explicitly choosing an image version
- **THEN** the service installs using a specific pinned `lmsysorg/sglang` version tag, not a `latest` or unversioned `dev`/`nightly` tag

### Requirement: SGLang VRAM utilization bookkeeping mirrors vLLM's crash-safety guarantees

The SGLang service SHALL maintain its own per-instance GPU memory utilization counter (independent of any other installed service), releasing the reserved fraction when a model install fails after the counter was incremented, when an install is cancelled, and when an installed model's container is confirmed dead by reconciliation — without requiring a backend restart.

#### Scenario: Install failure releases reserved utilization
- **WHEN** installing an SGLang model fails after its GPU memory fraction was reserved
- **THEN** the reserved fraction is released and the model does not appear in the installed-models registry

#### Scenario: Dead container releases reserved utilization
- **WHEN** an installed SGLang model's container stops running for good and is confirmed dead by the reconciliation loop
- **THEN** the reserved GPU utilization fraction is released and the model's bookkeeping reflects it is no longer running
