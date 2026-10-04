"""OpenAI's Responses: `POST {base_url}/responses`.

Stateless: a request carries its whole input (`store: false`, as Codex sends it); `previous_response_id` is refused.
`instructions`, and `system` and `developer` messages, are system messages. The items of one assistant turn
(reasoning, its message, its function calls) are one assistant message. Reasoning is read from a reasoning item's
`content` (or its summary), and a reply carries it as `reasoning_text` content. Tools other than functions are not
offered to the model.
"""

import itertools
import json
import time
from collections.abc import Iterator
from typing import Any

from rollout.contracts import (
    FinishReason,
    Message,
    Reasoning,
    ReasoningScope,
    Role,
    SampleResult,
    Text,
    ToolCall,
    ToolResult,
    ToolResultBlock,
    ToolSpecification,
)
from rollout_train.recorder.compat.wire import Format, Prompt, event, identifier, openai_error, reasoning_of, text_of

ASSISTANT_ITEMS = ("reasoning", "function_call")


def read(body: dict[str, Any]) -> Prompt:
    if body.get("previous_response_id"):
        raise ValueError("previous_response_id is not supported: send the whole input (store: false)")
    messages: list[Message] = []
    if body.get("instructions"):
        messages.append(Message.system(str(body["instructions"])))
    given: str | list[dict[str, Any]] = body["input"]
    items: list[dict[str, Any]] = (
        [{"type": "message", "role": "user", "content": given}] if isinstance(given, str) else given
    )
    turn: list[Any] = []  # the blocks of the assistant turn being read
    for item in items:
        kind = item.get("type", "message")
        role = item.get("role")
        if kind not in ASSISTANT_ITEMS and not (kind == "message" and role == "assistant") and turn:
            messages.append(Message(role=Role.ASSISTANT, content=turn))
            turn = []
        if kind == "reasoning":
            parts: list[dict[str, Any]] = item.get("content") or item.get("summary") or []
            if text := "".join(str(part.get("text", "")) for part in parts):
                turn.append(Reasoning(scope=ReasoningScope.PORTABLE, text=text))
        elif kind == "function_call":
            arguments = json.loads(item.get("arguments") or "{}")
            turn.append(ToolCall(call_id=str(item["call_id"]), name=str(item["name"]), arguments=arguments))
        elif kind == "function_call_output":
            result = ToolResult(content=[Text(text=text_of(item.get("output")))])
            messages.append(
                Message(role=Role.TOOL, content=[ToolResultBlock(call_id=str(item["call_id"]), result=result)])
            )
        elif kind == "message" and role == "assistant":
            if text := text_of(item.get("content")):
                turn.append(Text(text=text))
        elif kind == "message":
            said = Role.SYSTEM if role in ("system", "developer") else Role(role)
            messages.append(Message(role=said, content=[Text(text=text_of(item.get("content")))]))
        else:
            raise ValueError(f"an input item of type {kind!r} cannot be read")
    if turn:
        messages.append(Message(role=Role.ASSISTANT, content=turn))
    offered: list[dict[str, Any]] = body.get("tools") or []
    limit = body.get("max_output_tokens")
    return Prompt(
        messages=messages,
        tools=[specification(entry) for entry in offered if entry.get("type") == "function"],
        max_output_tokens=int(limit) if limit else None,
    )


def specification(entry: dict[str, Any]) -> ToolSpecification:
    schema: dict[str, Any] = entry.get("parameters") or {"type": "object", "properties": {}}
    return ToolSpecification(name=entry["name"], description=entry.get("description") or "", input_schema=schema)


def reply(result: SampleResult, effect: str, body: dict[str, Any]) -> dict[str, Any]:
    """A sample as a Responses response: its reasoning, its message and its function calls, as output items."""
    output: list[dict[str, Any]] = []
    if reasoning := reasoning_of(result):
        content = [{"type": "reasoning_text", "text": reasoning}]
        output.append({"type": "reasoning", "id": identifier("rs_", effect), "summary": [], "content": content})
    if text := result.message.text:
        part: dict[str, Any] = {"type": "output_text", "text": text, "annotations": [], "logprobs": []}
        said = {"type": "message", "id": identifier("msg_", effect), "status": "completed", "role": "assistant"}
        output.append(said | {"content": [part]})
    for index, call in enumerate(result.message.tool_calls):
        output.append(
            {
                "type": "function_call",
                "id": identifier(f"fc_{index}_", effect),
                "call_id": call.call_id,
                "name": call.name,
                "arguments": json.dumps(dict(call.arguments)),
                "status": "completed",
            }
        )
    cut = result.finish_reason is FinishReason.LENGTH
    usage = result.usage
    return {
        "id": identifier("resp_", effect),
        "object": "response",
        "created_at": int(time.time()),
        "status": "incomplete" if cut else "completed",
        "incomplete_details": {"reason": "max_output_tokens"} if cut else None,
        "error": None,
        "model": str(body.get("model", "")),
        "instructions": body.get("instructions"),
        "output": output,
        "parallel_tool_calls": bool(body.get("parallel_tool_calls", True)),
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": body.get("tools") or [],
        "temperature": None,
        "top_p": None,
        "max_output_tokens": body.get("max_output_tokens"),
        "previous_response_id": None,
        "metadata": {},
        "store": False,
        "usage": {
            "input_tokens": usage.input_tokens or 0,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": usage.output_tokens or 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage.context_used,
        },
    }


def events(response: dict[str, Any]) -> Iterator[str]:
    """The response's events, as OpenAI streams them: created, each output item with its content, then completed
    (or incomplete)."""
    numbers = itertools.count()

    def emit(kind: str, **fields: Any) -> str:
        return event({"type": kind, "sequence_number": next(numbers), **fields}, kind)

    started: dict[str, Any] = response | {
        "status": "in_progress",
        "output": [],
        "usage": None,
        "incomplete_details": None,
    }
    yield emit("response.created", response=started)
    yield emit("response.in_progress", response=started)
    for index, item in enumerate(response["output"]):
        at = {"item_id": item["id"], "output_index": index}
        if item["type"] == "message":
            part = item["content"][0]
            yield emit(
                "response.output_item.added", output_index=index, item=item | {"status": "in_progress", "content": []}
            )
            yield emit("response.content_part.added", **at, content_index=0, part=part | {"text": ""})
            yield emit("response.output_text.delta", **at, content_index=0, delta=part["text"], logprobs=[])
            yield emit("response.output_text.done", **at, content_index=0, text=part["text"], logprobs=[])
            yield emit("response.content_part.done", **at, content_index=0, part=part)
        elif item["type"] == "reasoning":
            part = item["content"][0]
            yield emit("response.output_item.added", output_index=index, item=item | {"content": []})
            yield emit("response.content_part.added", **at, content_index=0, part=part | {"text": ""})
            yield emit("response.reasoning_text.delta", **at, content_index=0, delta=part["text"])
            yield emit("response.reasoning_text.done", **at, content_index=0, text=part["text"])
            yield emit("response.content_part.done", **at, content_index=0, part=part)
        else:
            yield emit(
                "response.output_item.added", output_index=index, item=item | {"arguments": "", "status": "in_progress"}
            )
            yield emit("response.function_call_arguments.delta", **at, delta=item["arguments"])
            yield emit("response.function_call_arguments.done", **at, name=item["name"], arguments=item["arguments"])
        yield emit("response.output_item.done", output_index=index, item=item)
    finished = "response.completed" if response["status"] == "completed" else "response.incomplete"
    yield emit(finished, response=response)


FORMAT = Format(read=read, reply=reply, events=events, error=openai_error)
