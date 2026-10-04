"""OpenAI's Chat Completions and Responses, and Anthropic's Messages, read and answered: how the gateway
(`rollout_train.gateway`) serves harnesses that bring their own loop.

A coding agent running inside an environment, or any other program that already knows how to talk to a model, needs
no agent loop from this library: it is given a base URL and a key (`Model.address()`), and what it samples there is
recorded for the run's slot like any other sample. The key names the session; the model name a client sends is
ignored, and so are its sampling parameters (a trainable channel samples as its binding says).

    POST {base_url}/chat/completions      OpenAI's Chat Completions  (`chat`)
    POST {base_url}/responses             OpenAI's Responses  (`responses`)
    POST {base_url}/messages              Anthropic's Messages  (`messages`)

Each answers with one reply, or the same reply as server-sent events in its API's own shapes. The reply is sampled
before the first event is sent, so a recorded turn is kept whole or not at all. A context too long for the model is
refused the way each API refuses one, which harnesses compact on.
"""

from collections.abc import Mapping
from typing import Any, cast

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from rollout.contracts import (
    ContextDelta,
    ContextOverflow,
    ModelEndpointError,
    SampleRequest,
    SampleResult,
    arguments_digest,
    context_digests,
)
from rollout_train.recorder.compat.wire import Failure, Format

__all__ = ["SERVED_UNDER", "key", "refused", "replied", "requested"]

SERVED_UNDER = "/v1"
"""The path the APIs are served under: a base URL handed to a harness ends with it."""


def key(request: Request) -> str:
    """The key a client sent: OpenAI's clients send it as a bearer token, Anthropic's as `x-api-key`."""
    bearer = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    return request.headers.get("x-api-key") or bearer


async def requested(
    request: Request, format: Format, session_id: str, allowed: int, effect: str
) -> tuple[dict[str, Any], SampleRequest]:
    """A request's body, and the sample it asks for in a session: under its `Idempotency-Key` if it has one, else
    under `effect`, with its cap on the output kept within `allowed`. Raises `KeyError`, `TypeError` or `ValueError`
    when it cannot be read."""
    body = await request.json()
    if not isinstance(body, dict):
        raise TypeError("the body is not a JSON object")
    body = cast(dict[str, Any], body)
    prompt = format.read(body)
    if not prompt.messages:
        raise ValueError("the request has no messages")
    cap = prompt.max_output_tokens
    sample = SampleRequest(
        effect_id=request.headers.get("idempotency-key") or effect,
        arguments_digest=arguments_digest(body),
        session_id=session_id,
        context=ContextDelta(append=prompt.messages, digest=context_digests(prompt.messages)[-1]),
        tools=prompt.tools,
        max_output_tokens=min(cap, allowed) if cap else None,
    )
    return body, sample


def replied(
    format: Format, result: SampleResult, effect: str, body: dict[str, Any], headers: Mapping[str, str] | None = None
) -> Response:
    """A sample as the reply to a request: one response, or with `"stream": true` its server-sent events."""
    reply = format.reply(result, effect, body)
    if body.get("stream"):
        return StreamingResponse(format.events(reply), media_type="text/event-stream", headers=headers)
    return JSONResponse(reply, headers=headers)


def refused(format: Format, error: ModelEndpointError) -> Response:
    """A sample that failed, as the API refuses it: a context too long for the model the way harnesses compact on."""
    if isinstance(error, ContextOverflow):
        return format.error(Failure.CONTEXT, str(error))
    return format.error(Failure.ENDPOINT, str(error))
