# The recorder over HTTP

Code: `rollout_train.recorder.compat` · See [recorder](recorder.md),
[endpoints with an address](../rollout/contracts/model-endpoint.md#endpoints-with-an-address)

For a harness that brings its own loop: a coding agent running inside an environment, or any program that already
knows how to talk to a model. It needs no agent loop from this library. The program that launches it asks its model
slot for an address and hands over a base URL and a key:

```python
address = run.model.address()          # ModelAddress(base_url, api_key, model)
# an OpenAI client:     OPENAI_BASE_URL=address.base_url     OPENAI_API_KEY=address.api_key
# an Anthropic client:  ANTHROPIC_BASE_URL=address.base_url.removesuffix("/v1")  ANTHROPIC_API_KEY=address.api_key
```

A harness inside a sandbox is handed its address without any of that: a sandbox's spec names the slots it samples,
and the runner puts each one's address in the sandbox's environment ([sandboxes](../rollout/sandboxes.md#harnesses-inside-a-sandbox)).

What the harness samples there is recorded for the run's slot like any other sample, reaches the runner's hooks, and
ends up in the episode's trajectory. Whichever API it speaks, each request is rendered with the channel's renderer
and sampled by the channel, so the recorded tokens and logprobs are exactly what the policy sampled.

| Path | API |
|---|---|
| `POST {base_url}/chat/completions` | OpenAI's Chat Completions |
| `POST {base_url}/responses` | OpenAI's Responses (what Codex speaks) |
| `POST {base_url}/messages` | Anthropic's Messages (what Claude Code speaks) |
| `GET {base_url}/models` | the recorder's channels, in a list both OpenAI's and Anthropic's clients read |

Each answers with one reply, or with `"stream": true` the same reply as server-sent events in that API's own event
shapes. `create_app(recorder)` serves them under `SERVED_UNDER`, and a recorder's `base_url` ends with that path
(Anthropic's clients add `/v1` themselves). A recorder has a `base_url` when its deployment serves it: a
[profile](../../guide/deploying.md)'s `serve`, reached at its `address`. `Model.address()` raises `RuntimeError`
otherwise.

The [gateway](gateway.md) serves the same paths, read and answered by the same functions (`requested`, `replied`,
`refused`), with two differences: its keys are signed and name their session by themselves, and every turn is
recorded in the ledger and the blob store before its reply is sent. What follows holds for both.

## What a request means

- **The key names the session.** Each call of `address()` makes a key for one run's slot. The recorder's is valid
  until the recorder forgets the run; the gateway's until it expires ([keys](gateway.md#keys)). OpenAI's clients send it as a bearer token, Anthropic's as `x-api-key`; either is read on
  every path.
- **The model name and sampling parameters a client sends are ignored**: a trainable channel samples as its binding
  says, so that the trainer can reproduce the distribution. So is Anthropic's `thinking.budget_tokens`: the channel's
  [thinking budget](recorder.md) applies.
- **A cap on the output** is honoured, up to what the slot's capability contract allows: `max_completion_tokens` or
  `max_tokens` in Chat Completions, `max_output_tokens` in Responses, `max_tokens` in Messages.
- **`Idempotency-Key`** is honoured as the sample's `effect_id`: a request repeated under one key returns the
  recorded reply. Without one, a repeated request is a new sample that replaces the earlier one in
  [what the session exports](recorder.md#what-a-session-exports).
- **Content is read as text.** Tools are offered as their name, description and schema: functions (OpenAI) and
  custom tools (Anthropic). A provider's own tools (web search, say) are not offered to the model.
- **Reasoning goes both ways.** A reply carries the model's reasoning, and a request that sends it back has it
  rendered as the renderer renders reasoning:

  | API | In a reply | Read back from |
  |---|---|---|
  | Chat Completions | the message's `reasoning_content` | an assistant message's `reasoning_content` |
  | Responses | a `reasoning` item with `reasoning_text` content | a `reasoning` item's content, or its summary |
  | Messages | a `thinking` block, signed with a digest of its text | `thinking` blocks |

- **Messages are canonical messages.** A `developer` message (or Responses' `instructions`, or Anthropic's `system`)
  is a system message. The items of one Responses turn (its reasoning, message and function calls) are one assistant
  message, and so are the blocks of one Anthropic assistant message. A tool's result is a tool message.
- **Responses are stateless**: a request carries its whole input, as Codex sends it (`store: false`).
  `previous_response_id` is refused.
- **A stream is sampled before its first event**, so a recorded turn is kept whole or not at all. Chat Completions
  sends the whole message in one chunk, then how it ended, then usage. Responses sends `response.created`, each
  output item with its content, then `response.completed` (or `response.incomplete` when the output was cut short).
  Messages sends `message_start`, each content block, `message_delta` and `message_stop`.

## Errors

Each API's errors come in its own shape: `{"error": {"message", "type", "code"}}` for OpenAI's, `{"type": "error",
"error": {"type", "message"}}` for Anthropic's.

| When | Chat Completions, Responses | Messages |
|---|---|---|
| the key names no session | 401 `invalid_api_key` | 401 `authentication_error` |
| the request could not be read | 400 `invalid_request_error` | 400 `invalid_request_error` |
| the context is too long for the model. Harnesses compact on it | 400 `context_length_exceeded` | 400 `invalid_request_error`, "prompt is too long: ..." |
| the model endpoint failed | 503 `server_error` | 500 `api_error` |
