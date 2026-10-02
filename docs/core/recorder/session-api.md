# The recorder over HTTP

Status: **Working** (2026-10-02) · Code: `rollout.recorder.compat`

For a harness that brings its own loop: a coding agent running inside an environment, or any program that already
knows how to talk to a model. It needs no agent loop from this library. The program that launches it asks its model
slot for an address and hands over a base URL and a key:

```python
address = run.model.address()          # ModelAddress(base_url, api_key, model)
# launch the harness with OPENAI_BASE_URL=address.base_url and OPENAI_API_KEY=address.api_key
```

What the harness samples there is recorded for the run's slot like any other sample, reaches the runner's hooks, and
ends up in the episode's trace.

| Path | |
|---|---|
| `POST {base_url}/chat/completions` | OpenAI's Chat Completions: one reply, or the same as a stream of server-sent events |
| `GET {base_url}/models` | The one model there is |

- **The key names the session.** It is valid for one run's slot, until the run is forgotten.
- **The model name and sampling parameters a client sends are ignored**: a trainable channel samples as its binding
  says, so that the trainer can reproduce the distribution.
- **`Idempotency-Key`** is honoured as the sample's effect id; without one, a repeated request is a new sample that
  replaces the earlier one in what the session exports.
- **A context too long** is refused with `context_length_exceeded`, which harnesses compact on.
- **A stream** carries the whole message in one chunk: the reply is sampled before the first event is sent, so a
  recorded turn is kept whole or not at all.

A recorder is served when it has a `base_url` (a [profile](../../guide/perspectives.md#deploying)'s `serve`);
`Model.address()` raises otherwise.
