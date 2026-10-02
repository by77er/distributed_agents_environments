# Models

A run's model slots are served by **model endpoints**. Which endpoint serves a slot is decided by the run's binding,
never by task or agent code. A `ModelBinding` is either `direct` (a provider's API, on this page) or `recorded` (a
trainable channel served through the [recorder](../libraries/rollout-train/recorder.md)). This page covers the
adapter for the OpenAI Responses API.

## Binding a slot to a provider

A `DirectModel` binding names a provider, a model and sampling parameters. The runner creates the endpoint with the
factory registered for that provider:

```python fragment
from rollout_openai import codex_provider
from rollout.harness import DirectModel, ModelBinding, RunBinding, SamplingParameters
from rollout.local import LocalRunner

runner = LocalRunner(providers={"codex": codex_provider()})
binding = RunBinding(models={
    "policy": ModelBinding(direct=DirectModel(
        provider="codex",
        model="gpt-6-astra",
        sampling=SamplingParameters(reasoning_effort="low"),   # low | medium | high: what the adapter sends
    )),
})
```

Everything else stays the same: the specification is `RunSpecification(program=agent_program(...), binding=binding)`.
A task that declares more slots gets one binding per slot, and slots may use different providers.

## Credentials

| Credentials | Requests go to | Use it for |
|---|---|---|
| `CodexLogin()` | `https://chatgpt.com/backend-api/codex/responses` | development on a machine logged in with `codex login` (a ChatGPT account) |
| `ApiKey(key)` | `https://api.openai.com/v1/responses` | an OpenAI API key |

`CodexLogin` reads `~/.codex/auth.json` on every request. When the access token is about to expire it refreshes it
and writes the new tokens back in Codex's own format, so the Codex CLI and this adapter keep working side by side.
`codex_provider()` uses it; for an API key, register a factory yourself:

```python fragment
from rollout_openai import ApiKey, ResponsesEndpoint

key = ApiKey(os.environ["OPENAI_API_KEY"])
runner = LocalRunner(providers={
    "openai": lambda model: ResponsesEndpoint(key, model.model, sampling=model.sampling),
})
```

## What the adapter does

- **Rendering.** SYSTEM messages become the request's `instructions`; USER and ASSISTANT text becomes message items;
  tool calls and tool results become `function_call` and `function_call_output` items; tool specifications become
  function tools.
- **Images.** A `Media` block with an `image/*` type is sent as `input_image`, in user messages and in tool results.
  The endpoint reads the bytes from the blob store it was given (`ResponsesEndpoint(..., blobs=...)`,
  `codex_provider(blobs=...)`; see [content](content.md#media-and-blobs)); a context with media and no blob store
  raises `ValueError`. Media of other types is replaced by a line of text saying it was omitted.
- **Stateless requests.** Every request carries the whole context (`store: false`). The model's reasoning is not
  carried from one turn to the next.
- **Sampling parameters.** Of a binding's `SamplingParameters`, `reasoning_effort` is sent when it is set.
  `temperature` and `top_p` are sent only when they differ from the API's own default: reasoning models reject the
  two parameters whatever their value.
- **Output cap.** A request's `max_output_tokens` is sent where the backend accepts it, which its credentials say
  (`Credentials.accepts_max_output_tokens`). The public API accepts it, and counts reasoning tokens against it. The
  Codex backend rejects the parameter, so `CodexLogin` leaves it out.
- **Tool choice.** A request's `tool_choice` is sent to the provider.
- **Results.** Text and tool calls come back as canonical blocks. The finish reason is `TOOL_USE` when the reply
  makes tool calls and `LENGTH` when the response was cut off. Usage reports the provider's token counts.
- **Errors.** HTTP 429 raises `Overloaded`, a context that is too long raises `ContextOverflow`, rejected
  credentials raise `PermissionError` (after one refresh and retry on a 401), and other failures raise
  `InternalError`. `Model.sample` [retries](agents.md#the-model-interface) `Overloaded` and `InternalError`.
- **Limits.** The provider does not report its limits, so the endpoint advertises them from `ResponsesContract`
  (a context of 200,000 tokens and 32,000 output tokens by default).
- **No recording.** Direct adapters are for models you do not train: nothing about tokens or logprobs is kept, and a
  retried effect samples again.

## Trying it

```bash
ROLLOUT_LIVE=1 uv run pytest tests/rollout_openai -k live   # one real round trip; skipped unless ROLLOUT_LIVE=1
```
