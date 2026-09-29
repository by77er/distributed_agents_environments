# Models

A run's model slots are served by **model endpoints**. Which endpoint serves a slot is decided by the run's binding,
never by task or agent code. This page covers the one real-model adapter so far, for the OpenAI Responses API.

## Binding a slot to a provider

A `DirectModel` binding names a provider, a model and sampling parameters. The runner creates the endpoint with the
factory registered for that provider:

```python fragment
from rollout.adapters.responses import codex_provider
from rollout.core.harness import DirectModel, ModelBinding, RunBinding, SamplingParameters
from rollout.core.local import LocalRunner

runner = LocalRunner(providers={"codex": codex_provider()})
binding = RunBinding(models={
    "policy": ModelBinding(direct=DirectModel(
        provider="codex",
        model="gpt-6-astra",
        sampling=SamplingParameters(reasoning_effort="low"),   # low | medium | high
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
from rollout.adapters.responses import ApiKey, ResponsesEndpoint

key = ApiKey(os.environ["OPENAI_API_KEY"])
runner = LocalRunner(providers={
    "openai": lambda model: ResponsesEndpoint(key, model.model, sampling=model.sampling),
})
```

## What the adapter does

- **Rendering.** SYSTEM messages become the request's `instructions`; USER and ASSISTANT text becomes message items;
  tool calls and tool results become `function_call` and `function_call_output` items; tool specifications become
  function tools. Media blocks are not sent yet.
- **Stateless requests.** Every request carries the whole context (`store: false`). The model's reasoning is not
  carried from one turn to the next.
- **Results.** Text and tool calls come back as canonical blocks. The finish reason is `TOOL_USE` when the reply
  makes tool calls and `LENGTH` when the response was cut off. Usage reports the provider's token counts.
- **Errors.** HTTP 429 raises `Overloaded`, a context that is too long raises `ContextOverflow`, and other failures
  raise `InternalError`. `Model.sample` retries `Overloaded` and `InternalError` with exponential backoff (three
  retries by default) before giving up.
- **Limits.** The provider does not report a context limit, so the endpoint advertises one from
  `ResponsesContract` (200,000 tokens by default).
- **No recording.** Direct adapters are for models you do not train: nothing about tokens or logprobs is kept, and a
  retried effect samples again.

## Trying it

```bash
ROLLOUT_LIVE=1 uv run pytest tests/adapters -k live   # one real round trip; skipped unless ROLLOUT_LIVE=1
```
