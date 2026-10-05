# Content

For anyone who builds or reads messages: canonical content, tool calls and results, media kept as blobs, and digests.

**Read first:** [Choose what the model sees](agents.md). **Next:** [What a run records](runs-and-events.md).

Everything a model reads or writes is **canonical content**: model-agnostic messages made of typed blocks. Task and
agent code use only this form; rendering to a model's tokens happens in the
[gateway](../libraries/rollout-train/gateway.md) or inside a provider. The types live in `rollout.contracts` and are
specified in [canonical content](../libraries/rollout/contracts/canonical-content.md).

## Messages and blocks

A `Message` has a `Role` and a sequence of content blocks:

| Block | `type` | Holds | Allowed in |
|---|---|---|---|
| `Text` | `text` | `text` | any role |
| `Media` | `media` | `media_type` and a `BlobReference` (image, audio, document) | any role |
| `ToolCall` | `tool_call` | `call_id`, `name`, `arguments` (a JSON object) | ASSISTANT only |
| `ToolResultBlock` | `tool_result` | `call_id` and a `ToolResult` | TOOL only, which holds nothing else |
| `Reasoning` | `reasoning` | the model's reasoning as portable text, which a renderer may render or drop | ASSISTANT |

```python
from rollout.contracts import Message, Role, Text, ToolCall, ToolResult, ToolResultBlock

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

`meta` is a string map that is never shown to the model and never affects digests; the core uses
`meta["effect_id"]` to link a reply to the sample that produced it.

## Tool specifications and results

`ToolSpecification` is what the model sees about a tool (`name`, `description`, `input_schema`) plus one field it
never sees (`retry_class`). `@tool` builds one from a method ([tools](tools.md)).

`ToolResult` is what a tool produces:

| Field | Meaning |
|---|---|
| `content` | text and media blocks shown to the model |
| `structured` | an optional JSON value |
| `is_error` | a tool-level error the model should see and reason about |
| `truncated` | set by a tool that cut its output: `content` is only part of what the tool produced |

## Media and blobs

A `Media` block holds a media type and a `BlobReference` (`uri`, `sha256`, `size`, `media_type`), never the bytes.
Task code stores bytes with `run.blobs.put(data, media_type)` and puts the reference it gets in a `Media` block;
model adapters read the bytes back when they render the block ([models](models.md)). Storing is not an effect: in
one store the reference depends only on the bytes, so storing them again returns the same reference.

```python
import asyncio
import tempfile
from pathlib import Path

from rollout.contracts import Media
from rollout.harness import FileBlobStore


async def store_an_image() -> None:
    with tempfile.TemporaryDirectory() as directory:
        blobs = FileBlobStore(Path(directory))               # what a task reaches as run.blobs
        reference = await blobs.put(b"not really a PNG", "image/png")
        assert reference == await blobs.put(b"not really a PNG", "image/png")
        picture = Message(role=Role.USER, content=[Text(text="What is this?"),
                                                   Media(media_type="image/png", source=reference)])
        assert await blobs.read(picture.content[1].source) == b"not really a PNG"


asyncio.run(store_an_image())
```

| Store | Keeps |
|---|---|
| `FileBlobStore(directory)` | one file per blob, named by its SHA-256 |
| `rollout_s3.S3BlobStore` | one object per blob, in S3 or an S3-compatible store |

Both implement `Blobs` (`put`, `read`, `delete`). A runner takes one as `blobs=`. Blobs are read by hash, so stores are
interchangeable.

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
from rollout.contracts import arguments_digest, context_digests, message_digest

assert arguments_digest({"b": 1, "a": 2}) == arguments_digest({"a": 2, "b": 1})    # key order does not matter
assert message_digest(question) == message_digest(question.model_copy(update={"meta": {"note": "x"}}))

chain = context_digests([question, call, answer])   # d0 … d3
assert context_digests([question, call])[-1] == chain[2]   # a prefix is recognizable by its own digest
```

Every sample request carries the whole context in `request.context.append`, and the chain's last value in
`request.context.digest`. A run's events name a sample's context by that digest.
