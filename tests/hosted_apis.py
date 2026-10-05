"""Fake hosted APIs for tests: OpenAI's Responses and Anthropic's Messages, each a real HTTP server on this machine
that answers in the API's own event stream with the replies a test scripts, and keeps every request it was sent. No
request leaves the machine, and nothing is paid for.

A test scripts replies in order (`FakeApi.replies`): a `Said` (text, tool calls, thinking and usage) or a `Refusal`
(a status, a body, headers). Once they run out, every request is answered with `default` (by default the text `ok`).
`replying` answers a request by a function of it instead (what its last user message says, say)."""

import asyncio
import json
import socket
import threading
import time
from collections.abc import AsyncIterator, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

__all__ = ["FakeApi", "Refusal", "Said", "anthropic_api", "openai_api"]


@dataclass(frozen=True)
class Said:
    """A reply: its text, its tool calls (`(id, name, arguments)`), its thinking, and its usage."""

    text: str = "ok"
    calls: tuple[tuple[str, str, dict[str, Any]], ...] = ()
    thinking: str = ""
    input_tokens: int = 100
    cached_input_tokens: int = 0
    output_tokens: int = 20
    thinking_tokens: int = 0
    cut: bool = False
    """Stopped at the most output it may write."""


@dataclass(frozen=True)
class Refusal:
    """An error the API answers with."""

    status: int
    body: str = '{"error": {"type": "error", "message": "refused"}}'
    headers: dict[str, str] = field(default_factory=dict[str, str])


type Reply = Said | Refusal


@dataclass
class FakeApi:
    """A fake API at `url` (its base URL, as a client is given it), with the requests it was sent (`requests`: each
    one's body and headers)."""

    url: str = ""
    replies: list[Reply] = field(default_factory=list[Reply])
    default: Reply = field(default_factory=Said)
    replying: Callable[[dict[str, Any]], Reply] | None = None
    requests: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    headers: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    delay: float = 0.0
    """Seconds each reply takes (to see how many requests are in flight at once)."""
    in_flight: int = 0
    most_in_flight: int = 0

    def next(self, body: dict[str, Any]) -> Reply:
        if self.replies:
            return self.replies.pop(0)
        return self.replying(body) if self.replying is not None else self.default


def _events(lines: list[str]) -> StreamingResponse:
    async def stream() -> AsyncIterator[str]:
        for line in lines:
            yield line

    return StreamingResponse(stream(), media_type="text/event-stream")


def _openai(api: FakeApi) -> Starlette:
    async def responses(request: Request) -> Response:
        body = await request.json()
        api.requests.append(body)
        api.headers.append(dict(request.headers))
        api.in_flight += 1
        api.most_in_flight = max(api.most_in_flight, api.in_flight)
        try:
            await asyncio.sleep(api.delay)
            reply = api.next(body)
        finally:
            api.in_flight -= 1
        if isinstance(reply, Refusal):
            return Response(reply.body, status_code=reply.status, headers=reply.headers, media_type="application/json")
        items: list[dict[str, Any]] = []
        if reply.thinking:
            items.append({"type": "reasoning", "summary": [{"type": "summary_text", "text": reply.thinking}]})
        if reply.text:
            items.append({"type": "message", "role": "assistant",
                          "content": [{"type": "output_text", "text": reply.text}]})  # fmt: skip
        for call_id, name, arguments in reply.calls:
            items.append(
                {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(arguments)}
            )
        usage = {
            "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
            "input_tokens_details": {"cached_tokens": reply.cached_input_tokens},
            "output_tokens_details": {"reasoning_tokens": reply.thinking_tokens},
        }  # fmt: skip
        done = {"status": "incomplete" if reply.cut else "completed", "output": items, "usage": usage}
        lines = [f"data: {json.dumps({'type': 'response.output_item.done', 'item': item})}\n\n" for item in items]
        kind = "response.incomplete" if reply.cut else "response.completed"
        lines.append(f"data: {json.dumps({'type': kind, 'response': done})}\n\n")
        return _events(lines)

    return Starlette(routes=[Route("/v1/responses", responses, methods=["POST"])])


def _sse(kind: str, data: dict[str, Any]) -> str:
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n"


def _anthropic(api: FakeApi) -> Starlette:
    async def messages(request: Request) -> Response:
        body = await request.json()
        api.requests.append(body)
        api.headers.append(dict(request.headers))
        api.in_flight += 1
        api.most_in_flight = max(api.most_in_flight, api.in_flight)
        try:
            await asyncio.sleep(api.delay)
            reply = api.next(body)
        finally:
            api.in_flight -= 1
        if isinstance(reply, Refusal):
            return Response(reply.body, status_code=reply.status, headers=reply.headers, media_type="application/json")
        blocks: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        if reply.thinking:
            blocks.append(({"type": "thinking", "thinking": "", "signature": ""},
                           [{"type": "thinking_delta", "thinking": reply.thinking},
                            {"type": "signature_delta", "signature": "sig"}]))  # fmt: skip
        if reply.text:
            blocks.append(({"type": "text", "text": ""}, [{"type": "text_delta", "text": reply.text}]))
        for call_id, name, arguments in reply.calls:
            blocks.append(({"type": "tool_use", "id": call_id, "name": name, "input": {}},
                           [{"type": "input_json_delta", "partial_json": json.dumps(arguments)}]))  # fmt: skip
        uncached = reply.input_tokens - reply.cached_input_tokens
        start: dict[str, Any] = {
            "id": "msg_fake", "type": "message", "role": "assistant", "model": body.get("model"), "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": uncached, "cache_read_input_tokens": reply.cached_input_tokens,
                      "cache_creation_input_tokens": 0, "output_tokens": 1},
        }  # fmt: skip
        lines = [_sse("message_start", {"message": start})]
        for index, (block, deltas) in enumerate(blocks):
            lines.append(_sse("content_block_start", {"index": index, "content_block": block}))
            lines += [_sse("content_block_delta", {"index": index, "delta": delta}) for delta in deltas]
            lines.append(_sse("content_block_stop", {"index": index}))
        stop = "max_tokens" if reply.cut else "tool_use" if reply.calls else "end_turn"
        usage = {"output_tokens": reply.output_tokens, "input_tokens": uncached,
                 "cache_read_input_tokens": reply.cached_input_tokens, "cache_creation_input_tokens": 0,
                 "output_tokens_details": {"thinking_tokens": reply.thinking_tokens}}  # fmt: skip
        lines.append(_sse("message_delta", {"delta": {"stop_reason": stop, "stop_sequence": None}, "usage": usage}))
        lines.append(_sse("message_stop", {}))
        return _events(lines)

    async def models(request: Request) -> Response:
        return JSONResponse({"data": []})

    return Starlette(routes=[Route("/v1/messages", messages, methods=["POST"]), Route("/v1/models", models)])


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@contextmanager
def _served(app: Starlette) -> Generator[int]:
    """`app` served on this machine, on a free port, in a thread of its own, until the block ends."""
    import uvicorn

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("the fake API did not start")
        time.sleep(0.01)
    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@contextmanager
def openai_api(api: FakeApi | None = None) -> Generator[FakeApi]:
    """A fake Responses API; its `url` ends in `/v1`, as OpenAI's base URL does."""
    api = api or FakeApi()
    with _served(_openai(api)) as port:
        api.url = f"http://127.0.0.1:{port}/v1"
        yield api


@contextmanager
def anthropic_api(api: FakeApi | None = None) -> Generator[FakeApi]:
    """A fake Messages API; its `url` is the host, as Anthropic's base URL is (the SDK adds `/v1/messages`)."""
    api = api or FakeApi()
    with _served(_anthropic(api)) as port:
        api.url = f"http://127.0.0.1:{port}"
        yield api
