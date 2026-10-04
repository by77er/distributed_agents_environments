"""Anthropic's Messages: `POST {base_url}/messages`.

The key comes as `x-api-key` (or as a bearer token). `system` is a system message; a user message's `tool_result`
blocks are tool messages, read before its text. `thinking` blocks are reasoning, and a reply carries its reasoning as
a `thinking` block whose signature is a digest of its text. `max_tokens` caps the output; `thinking.budget_tokens` is
ignored, like every sampling parameter. Server tools (those with a `type` other than `custom`) are not offered to the
model. A context too long for the model is refused with "prompt is too long", which Claude Code compacts on.
"""

import hashlib
import json
from collections.abc import Iterator
from typing import Any

from starlette.responses import JSONResponse, Response

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
from rollout_train.recorder.compat.wire import Failure, Format, Prompt, event, identifier, reasoning_of, text_of

STOP = {FinishReason.TOOL_USE: "tool_use", FinishReason.LENGTH: "max_tokens", FinishReason.STOP: "end_turn"}


def read(body: dict[str, Any]) -> Prompt:
    messages: list[Message] = []
    if system := text_of(body.get("system")):
        messages.append(Message.system(system))
    entries: list[dict[str, Any]] = body["messages"]
    for entry in entries:
        content: str | list[dict[str, Any]] = entry["content"]
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
        if entry["role"] == "assistant":
            messages.append(Message(role=Role.ASSISTANT, content=[known for block in blocks if (known := said(block))]))
            continue
        if entry["role"] != "user":
            raise ValueError(f"a message's role is user or assistant, not {entry['role']!r}")
        for block in blocks:
            if block.get("type") == "tool_result":
                result = ToolResult(
                    content=[Text(text=text_of(block.get("content")))], is_error=bool(block.get("is_error"))
                )
                answer = ToolResultBlock(call_id=str(block["tool_use_id"]), result=result)
                messages.append(Message(role=Role.TOOL, content=[answer]))
        if text := text_of([block for block in blocks if block.get("type") == "text"]):
            messages.append(Message.user(text))
    offered: list[dict[str, Any]] = body.get("tools") or []
    return Prompt(
        messages=messages,
        tools=[specification(entry) for entry in offered if entry.get("type", "custom") == "custom"],
        max_output_tokens=int(body["max_tokens"]),
    )


def said(block: dict[str, Any]) -> Reasoning | Text | ToolCall | None:
    """An assistant's content block in canonical content (None: a kind not read, such as redacted thinking)."""
    kind = block.get("type")
    if kind == "thinking" and block.get("thinking"):
        return Reasoning(scope=ReasoningScope.PORTABLE, text=str(block["thinking"]))
    if kind == "text" and block.get("text"):
        return Text(text=str(block["text"]))
    if kind == "tool_use":
        return ToolCall(call_id=str(block["id"]), name=str(block["name"]), arguments=block.get("input") or {})
    return None


def specification(entry: dict[str, Any]) -> ToolSpecification:
    schema: dict[str, Any] = entry.get("input_schema") or {"type": "object", "properties": {}}
    return ToolSpecification(name=entry["name"], description=entry.get("description") or "", input_schema=schema)


def reply(result: SampleResult, effect: str, body: dict[str, Any]) -> dict[str, Any]:
    """A sample as a Messages response."""
    content: list[dict[str, Any]] = []
    if reasoning := reasoning_of(result):
        content.append({"type": "thinking", "thinking": reasoning, "signature": signature(reasoning)})
    if text := result.message.text:
        content.append({"type": "text", "text": text})
    for call in result.message.tool_calls:
        content.append({"type": "tool_use", "id": call.call_id, "name": call.name, "input": dict(call.arguments)})
    usage = result.usage
    return {
        "id": identifier("msg_", effect),
        "type": "message",
        "role": "assistant",
        "model": str(body.get("model", "")),
        "content": content,
        "stop_reason": STOP[result.finish_reason],
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.input_tokens or 0,
            "output_tokens": usage.output_tokens or 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def signature(reasoning: str) -> str:
    return hashlib.sha256(reasoning.encode()).hexdigest()


def events(response: dict[str, Any]) -> Iterator[str]:
    """The response's events, as Anthropic streams them: the message, each content block, then how it ended."""
    usage: dict[str, Any] = response["usage"]
    started: dict[str, Any] = response | {"content": [], "stop_reason": None, "usage": usage | {"output_tokens": 0}}
    yield event({"type": "message_start", "message": started}, "message_start")
    for index, block in enumerate(response["content"]):
        if block["type"] == "thinking":
            opened = {"type": "thinking", "thinking": "", "signature": ""}
            deltas = [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": block["signature"]},
            ]
        elif block["type"] == "text":
            opened, deltas = {"type": "text", "text": ""}, [{"type": "text_delta", "text": block["text"]}]
        else:
            opened = block | {"input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
        yield event({"type": "content_block_start", "index": index, "content_block": opened}, "content_block_start")
        for delta in deltas:
            yield event({"type": "content_block_delta", "index": index, "delta": delta}, "content_block_delta")
        yield event({"type": "content_block_stop", "index": index}, "content_block_stop")
    ended = {"stop_reason": response["stop_reason"], "stop_sequence": None}
    yield event({"type": "message_delta", "delta": ended, "usage": usage}, "message_delta")
    yield event({"type": "message_stop"}, "message_stop")


ERRORS = {
    Failure.KEY: (401, "authentication_error"),
    Failure.REQUEST: (400, "invalid_request_error"),
    Failure.CONTEXT: (400, "invalid_request_error"),
    Failure.ENDPOINT: (500, "api_error"),
}


def error(failure: Failure, message: str) -> Response:
    """An error as Anthropic's API sends one."""
    status, kind = ERRORS[failure]
    if failure is Failure.CONTEXT:
        message = f"prompt is too long: {message}"
    return JSONResponse({"type": "error", "error": {"type": kind, "message": message}}, status_code=status)


FORMAT = Format(read=read, reply=reply, events=events, error=error)
