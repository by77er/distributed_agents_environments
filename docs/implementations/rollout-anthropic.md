# Anthropic Messages endpoint

For whoever evaluates Claude models, or binds a judge to one: the model endpoint for Anthropic's Messages API, how a
cluster declares it, what each model takes, and what a reply says.

**Read first:** [the cluster config's hosted APIs](../guide/cluster.md#hosted-apis). **Next:** [the gateway's hosted
APIs](../libraries/rollout-train/gateway.md#hosted-apis).

Code: `rollout_anthropic` · See [`MessagesEndpoint`](../guide/reference.md#messagesendpoint),
[the model endpoint contract](../libraries/rollout/contracts/model-endpoint.md),
[use a hosted model](../guide/models.md), [evals](../libraries/rollout-train/evals.md#an-eval)

`MessagesEndpoint` implements the [`ModelEndpoint`](../guide/reference.md#modelendpoint) protocol over Anthropic's
Messages API, through the `anthropic` SDK. It is what a cluster's `api` provider samples Claude models with
(`endpoint = "rollout_anthropic:hosted"`), beside `rollout_openai`'s endpoint for OpenAI's Responses API. Nothing it
serves is trained on: the API returns text, not tokens.

## Installing

`implementations/rollout-anthropic` is a member of the workspace and a dependency of the platform: every `uv sync`
installs it, with the `anthropic` SDK. The platform image has it.

```bash
uv run pytest tests/rollout_anthropic   # against a fake Messages API on this machine: no key, no network
```

## Declaring it

```toml
[inference.anthropic]
kind = "api"
endpoint = "rollout_anthropic:hosted"
api_key_env = "ANTHROPIC_API_KEY"
concurrency = 16
[inference.anthropic.models."claude-opus-5-5"]
context = 1000000
cost = { input = 4.0, cached_input = 0.20, output = 20.0 }
options = { max_output_tokens = 128000, thinking = "adaptive", sampling = false, forced_tool_choice = false }
[inference.anthropic.models."claude-haiku-4-5-20251001"]
context = 200000
cost = { input = 1.0, cached_input = 0.10, output = 5.0 }
options = { max_output_tokens = 64000, thinking = "budget" }
```

`hosted(model, api_key=…, context_limit=…, max_output_tokens=…, options=…, base_url=…, blobs=…, timeout=…)`
makes the endpoint from the provider's key and the model's catalog entry, with the blob store it reads images and PDFs
from and the SDK's timeout for a request (600 s by default); the gateway calls it the first time a channel on the
provider samples ([hosted APIs](../libraries/rollout-train/gateway.md#hosted-apis)). Without a key it raises
`PermissionError`, which the gateway reports with the variable's name. `base_url` reaches another server that speaks the
API (none: Anthropic's own). The SDK retries nothing (`max_retries=0`): the gateway's channel backs off and asks again.

## What each model takes

What a model takes differs, and its catalog entry's `options` say it (`MessagesOptions`):

| Option | Values | Says |
|---|---|---|
| `thinking` | `adaptive`, `budget`, `none` (default) | How the model thinks. `adaptive`: the request asks for adaptive thinking, summarized, steered by the binding's reasoning effort, else the model's `effort` (`output_config.effort`). `budget`: a thinking budget of the binding's thinking tokens, where there are at least 1024 and room for them under `max_tokens`. `none`: no thinking is asked for |
| `effort` | `low` … `max` | The effort adaptive thinking is steered by where the binding says none |
| `sampling` | `true` (default), `false` | Whether the model takes temperature and top-p. Where it does, each is sent when it differs from 1 (temperature first: the API takes one of the two), never beside a thinking budget |
| `forced_tool_choice` | `true` (default), `false` | Whether the model takes a tool choice that forces a call; where it does not, `required` and a named tool are sent as `auto` |
| `max_output_tokens` | tokens | The most one reply writes (the gateway reads it; by default the model's context) |

`max_tokens` is the request's `max_output_tokens`, else the binding's thinking and answer budgets added up, else the
model's most output. A request's sampling parameters are taken per request (`sample(request, sampling=…)`), as the
gateway samples one channel for many bindings.

## Messages and tools

A request is the whole context, every time:

- **system messages** become the request's `system`, joined;
- **user messages** become user turns: text, and images and PDFs as base64 blocks read from the blob store (other
  media is replaced by a line saying it was left out);
- **the assistant's replies** become assistant turns: text, and each tool call a `tool_use` block. Reasoning is left
  out: a thinking block is replayed only with the signature the API gave it, which canonical content does not keep;
- **a tool's results** become `tool_result` blocks, opening the user turn that follows the call (consecutive results
  in one turn); an error result is marked `is_error`;
- **tool specifications** become tools (`name`, `description`, `input_schema`); a call's id is kept to the letters,
  digits, `_` and `-` the API takes, the same for the call and its result.

Consecutive messages of one side share one turn, and empty ones are left out.

## A reply

A reply is streamed and read whole. Its text, tool calls and summarized thinking (`Reasoning`) come back as canonical
content. It finishes with `length` where the API stopped at `max_tokens` or the context's end, `tool_use` where it
calls tools, else `stop`. Its usage says:

| Field | From the API's usage |
|---|---|
| `input_tokens` | every input token: uncached, read from the cache and written to it |
| `cached_input_tokens` | those read from the cache |
| `output_tokens` | every output token |
| `thinking_tokens` | those spent thinking (`output_tokens_details.thinking_tokens`) |

The gateway counts a turn's spend from them at the model's catalog prices.

## Errors

| The API answers | Raised |
|---|---|
| 429, 500, 502, 503, 504, 529; an `overloaded_error`, `rate_limit_error` or `api_error` in the stream | `Overloaded`, with `retry-after` where the API says it |
| a prompt too long for the context (400 or 413) | `ContextOverflow` |
| credentials refused (401, 403) | `PermissionError` |
| any other refusal (a request the API rejects) | `ModelEndpointError`: asking again would not change the answer |
| a connection that fails | `InternalError` |
