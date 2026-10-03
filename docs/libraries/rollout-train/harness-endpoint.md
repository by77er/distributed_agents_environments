# The recorder over HTTP

Code: `rollout_train.recorder.compat` · See [recorder](recorder.md),
[endpoints with an address](../rollout/contracts/model-endpoint.md#endpoints-with-an-address)

For a harness that brings its own loop: a coding agent running inside an environment, or any program that already
knows how to talk to a model. It needs no agent loop from this library. The program that launches it asks its model
slot for an address and hands over a base URL and a key:

```python
address = run.model.address()          # ModelAddress(base_url, api_key, model)
# launch the harness with OPENAI_BASE_URL=address.base_url and OPENAI_API_KEY=address.api_key
```

What the harness samples there is recorded for the run's slot like any other sample, reaches the runner's hooks, and
ends up in the episode's trajectory.

| Path | |
|---|---|
| `POST {base_url}/chat/completions` | OpenAI's Chat Completions: one reply, or the same as a stream of server-sent events |
| `GET {base_url}/models` | the recorder's channels |

`create_app(recorder)` serves both under `SERVED_UNDER`, and a recorder's `base_url` ends with that path. A
recorder has a `base_url` when its deployment serves it: a [profile](../../guide/deploying.md)'s `serve`, reached
at its `address`. `Model.address()` raises `RuntimeError` otherwise.

## What a request means

- **The key names the session.** Each call of `address()` makes a key for one run's slot. It is valid until the
  recorder forgets the run.
- **The model name and sampling parameters a client sends are ignored**: a trainable channel samples as its binding
  says, so that the trainer can reproduce the distribution.
- **A cap on the output** (`max_completion_tokens`, or `max_tokens`) is honoured, up to what the slot's capability
  contract allows.
- **`Idempotency-Key`** is honoured as the sample's `effect_id`: a request repeated under one key returns the
  recorded reply. Without one, a repeated request is a new sample that replaces the earlier one in
  [what the session exports](recorder.md#what-a-session-exports).
- **Messages** are read into canonical content. A `developer` message is a system message, and an assistant
  message's `reasoning_content` is its reasoning. A reply carries its reasoning the same way.
- **A stream** carries the whole message in one chunk, then how it ended, then usage. The reply is sampled before
  the first event is sent, so a recorded turn is kept whole or not at all.

## Errors

| Status | `code` | When |
|---|---|---|
| 401 | `invalid_api_key` | the key names no session |
| 400 | `invalid_request_error` | the request could not be read |
| 400 | `context_length_exceeded` | the context is too long for the model. Harnesses compact on it. |
| 503 | `server_error` | the model endpoint failed |
