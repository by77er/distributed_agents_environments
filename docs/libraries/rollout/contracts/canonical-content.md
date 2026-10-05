# Canonical content

For whoever implements a model endpoint or a renderer: the model-agnostic form of everything a model reads or writes.

**Read first:** [Messages, files and digests](../../../guide/content.md). **Next:** [Run events](run-events.md).

Code: `rollout.contracts.content`, `rollout.contracts.digests` · See [guide: content](../../../guide/content.md),
[API reference](../../../guide/reference.md#message)

Canonical content is the model-agnostic form of everything a model reads or writes. Task and agent code, run events,
tool sets and the model endpoint use only this form. Rendering to a model's tokens happens in the
[gateway](../../rollout-train/gateway.md)'s renderer ([recording](../../rollout-train/recorder.md#renderers)) or inside
a direct adapter, and nowhere else.

Building and reading messages is shown in [messages, files and digests](../../../guide/content.md). This page gives the
rules.

## Messages and blocks

A [`Message`](../../../guide/reference.md#message) is a role and a sequence of blocks. A block is one of
[`Text`](../../../guide/reference.md#text), [`Media`](../../../guide/reference.md#media),
[`ToolCall`](../../../guide/reference.md#toolcall), [`ToolResultBlock`](../../../guide/reference.md#toolresultblock)
and [`Reasoning`](../../../guide/reference.md#reasoning), told apart by its `type`.

| Rule | Enforced |
|---|---|
| A TOOL message contains only tool results, and a tool result appears only in a TOOL message | when the message is constructed |
| A tool call appears only in an ASSISTANT message | when the message is constructed |
| An observation contains only USER and TOOL messages | by the loop ([validation](../../../guide/tasks.md#validation)) |
| Every tool call of a reply is answered by a tool result with the same `call_id`, and a tool result answers a call of that reply | by the loop |
| A sample result is an ASSISTANT message | when the result is constructed |

- **`meta`** is a string map that no model sees. It is never rendered, and no digest covers it. The harness uses
  `meta["effect_id"]` to tie a sampled reply to the sample that produced it.
- **`call_id`** is unique within a context. Whatever parses a model's output makes it; the tool result repeats it.
- **Reasoning** is portable: plain text that a renderer may render or drop. The gateway's renderer parses a model's
  thinking into a reasoning block and renders it back for the model family's template. The Responses adapter does
  not carry reasoning from one turn to the next.
- **Media** holds a [`BlobReference`](../../../guide/reference.md#blobreference), never bytes. The reference depends
  only on the bytes, so storing a blob is not an effect ([media and blobs](../../../guide/content.md#media-and-blobs)).

## Tool specifications

A [`ToolSpecification`](../../../guide/reference.md#toolspecification) is what a model sees about a tool, and one
field it never sees.

- **Model-visible**: the name, the description and the input schema. `model_visible()` returns exactly these, an
  endpoint renders only these, and the spec hash covers only these.
- **Not model-visible**: the `retry_class`, which says whether a call is safe to make again
  ([effects](effects.md#receivers-that-deduplicate)).
- A name matches `^[a-zA-Z0-9_-]{1,64}$`, and an input schema is a JSON Schema of type `object`. Both are checked
  when the specification is constructed. The names of the tools in one sample request are unique.

## Tool results

A [`ToolResult`](../../../guide/reference.md#toolresult) is what a tool produces.

- **`is_error`** marks a failure of the tool that the model should see and reason about: a non-zero exit, a file
  that is not there. The result is still a result.
- **A platform failure is not a result.** A tool set that cannot be reached raises, and the `tool.call` effect
  completes as `failed`. The task loop then answers the call with an error result of its own, so that every call
  has an answer ([failures](../README.md#failures)).
- **`truncated`** says that `content` is only part of what the tool produced.
- **`structured`** is the result as JSON, for code that reads it. A `@tool` body that returns a value other than
  text or blocks fills it, with the same JSON as text for the model.

## Digests

A digest is the SHA-256, in lowercase hexadecimal, of the canonical JSON of a value
([RFC 8785](https://www.rfc-editor.org/rfc/rfc8785)). A contract model is put in canonical form with the fields
whose value is `None` left out, so a new optional field does not change the digest of a value that does not set it.
A `null` inside a JSON value, such as a tool argument, is kept.

| Digest | Function | Covers |
|---|---|---|
| of a value | `digest` | the value's canonical JSON |
| arguments digest | `arguments_digest` | the arguments of an effect, as its `effect.requested` event records them |
| message digest | `message_digest` | a message without its `meta`: what a model can see |
| spec hash | `spec_hash` | the model-visible fields of a tool specification |
| context digest | `context_digests` | the messages of a context, in order |

A context digest is the last value of a chain. `d₀` is the SHA-256 of the empty string (`EMPTY_DIGEST`), and
`dᵢ = sha256(dᵢ₋₁ ‖ sha256(JCS(messageᵢ)))`, over the raw bytes of the digests. `dₖ` identifies the first `k`
messages, so two contexts that share a beginning share the chain up to it.
