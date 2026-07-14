## Purpose

Lets clients control reasoning/thinking behavior (enable, disable, or tune effort)
for reasoning-capable models on `/v1/chat/completions`, by accepting and forwarding
`reasoning_effort`/`reasoning.effort` to the backend instead of silently dropping
them. Backends behind this endpoint include both Ollama and DeepFellow's OpenAI
proxy, so accepted values cover both vocabularies.

## Requirements

### Requirement: Chat completions request accepts reasoning effort fields
The `/v1/chat/completions` endpoint SHALL accept an optional `reasoning_effort` field
and an optional `reasoning` object field (with a nested `effort` field) on the request
body, each independently accepting one of `none`, `minimal`, `low`, `medium`, `high`,
`xhigh`, or `max` - the union of values documented by Ollama and OpenAI, since this
endpoint proxies to both. Requests omitting both fields SHALL behave exactly as before
this change.

#### Scenario: Client sends flat reasoning_effort field
- **WHEN** a client POSTs to `/v1/chat/completions` with
  `{"model": "deepseek-r1", "messages": [...], "reasoning_effort": "medium"}`
- **THEN** the request SHALL pass validation and `reasoning_effort` SHALL be included
  in the JSON body forwarded to the backend

#### Scenario: Client sends nested reasoning.effort field
- **WHEN** a client POSTs to `/v1/chat/completions` with
  `{"model": "deepseek-r1", "messages": [...], "reasoning": {"effort": "high"}}`
- **THEN** the request SHALL pass validation and `reasoning` SHALL be included in the
  JSON body forwarded to the backend as `{"effort": "high"}`

#### Scenario: Client sends an invalid effort value
- **WHEN** a client POSTs to `/v1/chat/completions` with
  `{"model": "deepseek-r1", "messages": [...], "reasoning_effort": "extreme"}`
- **THEN** the request SHALL fail validation with a 422 error, since `"extreme"` is
  not one of the accepted values

#### Scenario: Client omits both reasoning fields
- **WHEN** a client POSTs to `/v1/chat/completions` without `reasoning_effort` or
  `reasoning` in the body
- **THEN** the request SHALL be processed exactly as it was before this change, with
  neither field present in the JSON body forwarded to the backend

#### Scenario: Client sends both fields with different values
- **WHEN** a client POSTs to `/v1/chat/completions` with both
  `"reasoning_effort": "low"` and `"reasoning": {"effort": "high"}` set to different
  values
- **THEN** both fields SHALL be forwarded as-is in the JSON body sent to the backend,
  with no disambiguation or precedence applied by DeepFellow

### Requirement: Reasoning fields are forwarded without altering streaming behavior
Enabling `reasoning_effort` or `reasoning` SHALL NOT change how DeepFellow proxies or
streams the response. Any reasoning-related content returned by the backend in the
response body (streaming or non-streaming) SHALL reach the client unmodified, since
DeepFellow forwards backend responses as an unparsed byte stream.

#### Scenario: Backend returns reasoning content in a streamed response
- **WHEN** a client requests `stream: true` together with `reasoning_effort: "medium"`
  against a reasoning-capable model, and the backend emits reasoning content in its
  streamed chunks
- **THEN** DeepFellow SHALL relay those chunks to the client unmodified, without
  filtering, parsing, or stripping any reasoning-related fields
- **Verified** (2026-07-13, direct call to Ollama with `deepseek-r1:1.5b`, bypassing
  DeepFellow): streamed chunks carry `choices[0].delta.reasoning` (a plain string,
  distinct from `delta.content`) while the model is thinking, e.g.
  `{"delta":{"role":"assistant","content":"","reasoning":"First"}}`; `content` stays
  empty until reasoning is complete, then the final answer streams normally in
  `delta.content`. Since DeepFellow forwards bytes unmodified, this exact shape is
  what SHALL reach the client once `reasoning_effort` survives request validation.
