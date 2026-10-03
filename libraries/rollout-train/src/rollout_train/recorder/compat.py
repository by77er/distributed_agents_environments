"""The recorder over HTTP, for harnesses that bring their own loop: OpenAI's Chat Completions protocol.

A coding agent running inside an environment, or any other program that already knows how to talk to a model, needs
no agent loop from this library: it is given a base URL and a key (`Model.address()`), and what it samples there is
recorded for the run's slot like any other sample. The key names the session; the model name a client sends is
ignored, and so are its sampling parameters (a trainable channel samples as its binding says).

    POST {base_url}/chat/completions      one reply, or the same as a stream of server-sent events
    GET  {base_url}/models                every channel of the recorder, as a model  (`base_url` ends in `/v1`)

A context too long for the model is refused with OpenAI's `context_length_exceeded`, which harnesses compact on.
"""

import itertools
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from rollout.contracts import (
    ContextDelta,
    ContextOverflow,
    FinishReason,
    Message,
    ModelEndpointError,
    Reasoning,
    ReasoningScope,
    Role,
    SampleRequest,
    SampleResult,
    Text,
    ToolCall,
    ToolResult,
    ToolResultBlock,
    ToolSpecification,
    arguments_digest,
    context_digests,
)
from rollout_train.recorder.recorder import SERVED_UNDER, Recorder

FINISH = {FinishReason.TOOL_USE: "tool_calls", FinishReason.LENGTH: "length"}


def create_app(recorder: Recorder) -> Starlette:
    """Serve `recorder` (whose `base_url` says where this app is reachable, path included)."""
    counter = itertools.count()

    async def models(request: Request) -> Response:
        return JSONResponse({"object": "list", "data": [{"id": name, "object": "model"} for name in recorder.channels]})

    async def chat(request: Request) -> Response:
        served = recorder.served(request.headers.get("authorization", "").removeprefix("Bearer ").strip())
        if served is None:
            return _error(401, "invalid_api_key", "this key names no session")
        session_id, endpoint = served
        body: dict[str, Any] = await request.json()
        try:
            entries: list[dict[str, Any]] = body["messages"]
            offered: list[dict[str, Any]] = body.get("tools") or []
            messages = [canonical(entry) for entry in entries]
            tools = [specification(entry) for entry in offered]
        except (KeyError, TypeError, ValueError) as error:
            return _error(400, "invalid_request_error", f"the request could not be read: {error}")
        effect = request.headers.get("idempotency-key") or f"{session_id}:harness:{next(counter)}"
        limit = body.get("max_completion_tokens") or body.get("max_tokens")
        allowed = endpoint.describe(session_id).max_output_tokens
        sample = SampleRequest(
            effect_id=effect,
            arguments_digest=arguments_digest(body),
            session_id=session_id,
            context=ContextDelta(append=messages, digest=context_digests(messages)[-1]),
            tools=tools,
            max_output_tokens=min(int(limit), allowed) if limit else None,
        )
        try:
            result = await endpoint.sample(sample)
        except ContextOverflow as error:
            return _error(400, "context_length_exceeded", str(error))
        except ModelEndpointError as error:
            return _error(503, "server_error", str(error))
        reply = completion(result, effect, str(body.get("model", "")))
        if body.get("stream"):
            return StreamingResponse(events(reply), media_type="text/event-stream")
        return JSONResponse(reply)

    return Starlette(
        routes=[
            Route(f"{SERVED_UNDER}/models", models),
            Route(f"{SERVED_UNDER}/chat/completions", chat, methods=["POST"]),
        ]
    )


def canonical(entry: dict[str, Any]) -> Message:
    """A Chat Completions message in canonical content."""
    role = "system" if entry["role"] == "developer" else entry["role"]
    content: str | list[dict[str, Any]] | None = entry.get("content")
    text = content if isinstance(content, str) else "".join(str(part.get("text", "")) for part in content or [])
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


def completion(result: SampleResult, effect: str, model: str) -> dict[str, Any]:
    """A sample as a Chat Completions response."""
    message = result.message
    reasoning = "".join(block.text or "" for block in message.content if isinstance(block, Reasoning))
    calls = [
        {
            "id": call.call_id,
            "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(dict(call.arguments))},
        }
        for call in message.tool_calls
    ]
    reply: dict[str, Any] = {"role": "assistant", "content": message.text or None}
    if reasoning:
        reply["reasoning_content"] = reasoning
    if calls:
        reply["tool_calls"] = calls
    usage = result.usage
    return {
        "id": f"chatcmpl-{effect}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": reply, "finish_reason": FINISH.get(result.finish_reason, "stop")}],
        "usage": {
            "prompt_tokens": usage.input_tokens or 0,
            "completion_tokens": usage.output_tokens or 0,
            "total_tokens": usage.context_used,
        },
    }


async def events(reply: dict[str, Any]) -> AsyncIterator[str]:
    """The same reply as a stream: the whole message in one chunk, then how it ended and what it used. (The reply
    is sampled before the first event is sent: a recorded turn is kept whole or not at all.)"""
    choice: dict[str, Any] = reply["choices"][0]
    delta: dict[str, Any] = dict(choice["message"])
    calls: list[dict[str, Any]] = delta.pop("tool_calls", [])
    delta["tool_calls"] = [{"index": index, **call} for index, call in enumerate(calls)]
    head: dict[str, Any] = {key: reply[key] for key in ("id", "created", "model")} | {"object": "chat.completion.chunk"}
    chunks: list[dict[str, Any]] = [
        {**head, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {**head, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]},
        {**head, "choices": [], "usage": reply["usage"]},
    ]
    for chunk in chunks:
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


def _error(status: int, code: str, message: str) -> Response:
    return JSONResponse({"error": {"message": message, "type": code, "code": code}}, status_code=status)
