# Content

Everything a model reads or writes is **canonical content**: model-agnostic messages made of typed blocks. Task
and agent code use only this form; rendering to a model's tokens happens in the recorder or inside a provider.
The types live in `rollout.core.contracts` and are defined once, in [canonical content](../contracts/canonical-content.md).

## Messages and blocks

A `Message` has a `Role` and a sequence of content blocks:

| Block | `type` | Holds | Allowed in |
|---|---|---|---|
| `Text` | `text` | `text` | any role |
| `Media` | `media` | `media_type` and a `BlobReference` (image, audio, document) | any role |
| `ToolCall` | `tool_call` | `call_id`, `name`, `arguments` (a JSON object) | ASSISTANT only |
| `ToolResultBlock` | `tool_result` | `call_id` and a `ToolResult` | TOOL only, which holds nothing else |
| `Reasoning` | `reasoning` | portable text, or opaque policy-scoped reasoning | ASSISTANT |

```python
from rollout.core.contracts import Message, Role, Text, ToolCall, ToolResult, ToolResultBlock

question = Message.user("What is the weather in Lisbon?")
call = Message(
    role=Role.ASSISTANT,
    content=[Text(text="Let me check."), ToolCall(call_id="c1", name="weather", arguments={"city": "Lisbon"})],
)
answer = Message(
    role=Role.TOOL,
    content=[ToolResultBlock(call_id="c1", result=ToolResult(content=[Text(text="18 °C, clear")]))],
)

assert call.text == "Let me check."                 # the concatenated text blocks
assert [c.name for c in call.tool_calls] == ["weather"]
```

`Message.user(text)`, `Message.assistant(text)` and `Message.system(text)` build one-block messages. Constructing a
message that breaks the role rules raises a pydantic `ValidationError`:

```python
import pytest
from pydantic import ValidationError

with pytest.raises(ValidationError):
    Message(role=Role.USER, content=[ToolCall(call_id="c1", name="weather", arguments={})])
```

`name` optionally names the speaker (for multi-agent conversations). `meta` is a string map that is never shown to
the model and never affects digests; the core uses `meta["effect_id"]` to link a reply to the sample that produced
it.

## Tool specifications and results

`ToolSpecification` is what the model sees about a tool (`name`, `description`, `input_schema`, `output_schema`)
plus fields it never sees (`annotations`, `retry_class`, `timeout_ms`, `max_result_bytes`). `@tool` builds these
for you ([tools](tools.md)).

`ToolResult` is what a tool produces:

| Field | Meaning |
|---|---|
| `content` | text and media blocks shown to the model |
| `structured` | an optional JSON value (validated against `output_schema` when there is one) |
| `is_error` | a tool-level error the model should see and reason about |
| `truncated`, `overflow` | output cut to `max_result_bytes`, with the full output in blob storage |
| `provenance` | whether the result came from untrusted input, and from what kind of binding |

## Immutability and JSON

Contract types are frozen pydantic models. Sequences are stored as tuples. To change a value, build a new one
(`model_copy(update=...)` works). Every type round-trips through JSON, and unknown fields are kept, so a
component can read and re-write records written by newer code.

```python
restored = Message.model_validate_json(call.model_dump_json())
assert restored == call
assert isinstance(restored.content, tuple)

from_newer_code = Message.model_validate({"role": "user", "content": [], "priority_hint": "high"})
assert from_newer_code.model_dump()["priority_hint"] == "high"
```

## Digests

Digests are lowercase hexadecimal SHA-256 over [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785) canonical JSON.
Fields that are `None` are left out, so adding an optional field later does not change existing digests.

```python
from rollout.core.contracts import arguments_digest, context_digests, message_digest

assert arguments_digest({"b": 1, "a": 2}) == arguments_digest({"a": 2, "b": 1})    # key order does not matter
assert message_digest(question) == message_digest(question.model_copy(update={"meta": {"trace": "x"}}))

chain = context_digests([question, call, answer])   # d0 … d3
assert context_digests([question, call])[-1] == chain[2]   # a prefix is recognizable by its own digest
```

The context digest chain lets an endpoint recognize that a new request extends a previous one, so only the new
messages need to be sent. Requests carry the chain's last value today; sending only the new messages arrives with
the model adapters (M0 task 6).
