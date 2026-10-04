"""What the API formats share: what a request asks for, the ways a request fails, and server-sent events."""

import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from starlette.responses import JSONResponse, Response

from rollout.contracts import Message, Reasoning, SampleResult, ToolSpecification


class Failure(StrEnum):
    """How a request fails, whatever its format."""

    KEY = "key"
    """The key names no session."""
    REQUEST = "request"
    """The request could not be read."""
    CONTEXT = "context"
    """The context is too long for the model: harnesses compact on it."""
    ENDPOINT = "endpoint"
    """The model endpoint failed."""


@dataclass(frozen=True)
class Prompt:
    """What a request asks of the model, in canonical content."""

    messages: list[Message]
    tools: list[ToolSpecification]
    max_output_tokens: int | None
    """The cap the client put on the output, if any."""


@dataclass(frozen=True)
class Format:
    """One API's wire format."""

    read: Callable[[dict[str, Any]], Prompt]
    """A request body as a prompt. Raises `KeyError`, `TypeError` or `ValueError` when it cannot be read."""
    reply: Callable[[SampleResult, str, dict[str, Any]], dict[str, Any]]
    """A sample as the response to a request: the result, its effect id, the request's body."""
    events: Callable[[dict[str, Any]], Iterator[str]]
    """The same response as server-sent events."""
    error: Callable[[Failure, str], Response]


def event(data: dict[str, Any], name: str | None = None) -> str:
    """One server-sent event; `name` is its `event:` field."""
    head = f"event: {name}\n" if name else ""
    return f"{head}data: {json.dumps(data)}\n\n"


def identifier(prefix: str, effect: str) -> str:
    """An id for what a sample produced, derived from its effect id (so a replayed effect has the same ids)."""
    return f"{prefix}{hashlib.sha256(effect.encode()).hexdigest()[:24]}"


def reasoning_of(result: SampleResult) -> str:
    return "".join(block.text for block in result.message.content if isinstance(block, Reasoning))


def text_of(content: str | list[dict[str, Any]] | None) -> str:
    """Content given as a string or as a list of parts: its text (parts of any other kind are not read)."""
    if content is None or isinstance(content, str):
        return content or ""
    return "".join(str(part.get("text", "")) for part in content if isinstance(part.get("text"), str))


OPENAI_ERRORS = {
    Failure.KEY: (401, "invalid_request_error", "invalid_api_key"),
    Failure.REQUEST: (400, "invalid_request_error", "invalid_request_error"),
    Failure.CONTEXT: (400, "invalid_request_error", "context_length_exceeded"),
    Failure.ENDPOINT: (503, "server_error", "server_error"),
}


def openai_error(failure: Failure, message: str) -> Response:
    """An error as OpenAI's APIs send one."""
    status, kind, code = OPENAI_ERRORS[failure]
    return JSONResponse({"error": {"message": message, "type": kind, "param": None, "code": code}}, status_code=status)
