"""OpenAI's Chat Completions: `POST {base_url}/chat/completions`.

A `developer` message is a system message, and an assistant message's `reasoning_content` is its reasoning; a reply
carries its reasoning the same way. A stream carries the whole message in one chunk, then how it ended, then usage.
"""

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

FINISH = {FinishReason.TOOL_USE: "tool_calls", FinishReason.LENGTH: "length"}


def read(body: dict[str, Any]) -> Prompt:
    entries: list[dict[str, Any]] = body["messages"]
    offered: list[dict[str, Any]] = body.get("tools") or []
    limit = body.get("max_completion_tokens") or body.get("max_tokens")
    return Prompt(
        messages=[canonical(entry) for entry in entries],
        tools=[specification(entry) for entry in offered if entry.get("type", "function") == "function"],
        max_output_tokens=int(limit) if limit else None,
    )


def canonical(entry: dict[str, Any]) -> Message:
    """A Chat Completions message in canonical content."""
    role = "system" if entry["role"] == "developer" else entry["role"]
    text = text_of(entry.get("content"))
    if role == "tool":
        result = ToolResult(content=[Text(text=text)])
        return Message(role=Role.TOOL, content=[ToolResultBlock(call_id=str(entry["tool_call_id"]), result=result)])
    blocks: list[Any] = []
    if role == "assistant" and entry.get("reasoning_content"):
        blocks.append(Reasoning(scope=ReasoningScope.PORTABLE, text=str(entry["reasoning_content"])))
    if text:
        blocks.append(Text(text=text))
    calls: list[dict[str, Any]] = entry.get("tool_calls") or []
    for call in calls:
        function: dict[str, Any] = call["function"]
        arguments = json.loads(function.get("arguments") or "{}")
        blocks.append(ToolCall(call_id=str(call["id"]), name=str(function["name"]), arguments=arguments))
    return Message(role=Role(role), content=blocks)


def specification(entry: dict[str, Any]) -> ToolSpecification:
    function: dict[str, Any] = entry["function"]
    schema: dict[str, Any] = function.get("parameters") or {"type": "object", "properties": {}}
    return ToolSpecification(name=function["name"], description=function.get("description") or "", input_schema=schema)


def reply(result: SampleResult, effect: str, body: dict[str, Any]) -> dict[str, Any]:
    """A sample as a Chat Completions response."""
    message = result.message
    reasoning = reasoning_of(result)
    calls = [
        {
            "id": call.call_id,
            "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(dict(call.arguments))},
        }
        for call in message.tool_calls
    ]
    said: dict[str, Any] = {"role": "assistant", "content": message.text or None}
    if reasoning:
        said["reasoning_content"] = reasoning
    if calls:
        said["tool_calls"] = calls
    usage = result.usage
    return {
        "id": identifier("chatcmpl-", effect),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(body.get("model", "")),
        "choices": [{"index": 0, "message": said, "finish_reason": FINISH.get(result.finish_reason, "stop")}],
        "usage": {
            "prompt_tokens": usage.input_tokens or 0,
            "completion_tokens": usage.output_tokens or 0,
            "total_tokens": usage.context_used,
        },
    }


def events(response: dict[str, Any]) -> Iterator[str]:
    """The whole message in one chunk, then how it ended and what it used."""
    choice: dict[str, Any] = response["choices"][0]
    delta: dict[str, Any] = dict(choice["message"])
    calls: list[dict[str, Any]] = delta.pop("tool_calls", [])
    delta["tool_calls"] = [{"index": index, **call} for index, call in enumerate(calls)]
    head = {key: response[key] for key in ("id", "created", "model")} | {"object": "chat.completion.chunk"}
    yield event({**head, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
    yield event({**head, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]})
    yield event({**head, "choices": [], "usage": response["usage"]})
    yield "data: [DONE]\n\n"


FORMAT = Format(read=read, reply=reply, events=events, error=openai_error)
