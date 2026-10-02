"""A `ModelEndpoint` for the OpenAI Responses API (docs/guide/models.md).

Two ways to authenticate:

- `CodexLogin`: the developer's local Codex login (`~/.codex/auth.json`, ChatGPT account tokens) against the Codex
  backend. The file is re-read on every request; an expired access token is refreshed and written back in Codex's
  own format, so the Codex CLI and this adapter never invalidate each other's tokens.
- `ApiKey`: an API key against `https://api.openai.com/v1`.

Requests are stateless (`store: false`): the whole context is sent every time, and reasoning is not carried between
turns.

Of a binding's sampling parameters, `reasoning_effort` is sent when set, and `temperature` and `top_p` only when they
differ from the API's own default (1.0): reasoning models reject the two parameters whatever their value. A
request's `max_output_tokens` is sent where the backend accepts it: the public API does, and counts reasoning tokens
against it; the Codex backend rejects the parameter.
"""

import base64
import json
import os
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import httpx
from pydantic import JsonValue

from rollout.contracts import (
    CapabilityContract,
    ContextOverflow,
    FinishReason,
    InternalError,
    Media,
    Message,
    NamedToolChoice,
    Overloaded,
    Role,
    SampleRequest,
    SampleResult,
    Text,
    ToolCall,
    ToolChoice,
    ToolResultBlock,
    ToolSpecification,
    Usage,
)
from rollout.harness.blobs import Blobs
from rollout.harness.runner import DirectModel, SamplingParameters

__all__ = ["ApiKey", "CodexLogin", "Credentials", "ResponsesContract", "ResponsesEndpoint", "codex_provider"]


class Credentials(Protocol):
    """Where requests go and how they authenticate."""

    async def headers(self, client: httpx.AsyncClient, *, force_refresh: bool = False) -> dict[str, str]: ...
    @property
    def url(self) -> str: ...
    @property
    def accepts_max_output_tokens(self) -> bool:
        """Whether the backend accepts `max_output_tokens`."""
        ...


@dataclass
class ApiKey:
    """An OpenAI API key against the public Responses API."""

    key: str
    base_url: str = "https://api.openai.com/v1"
    accepts_max_output_tokens: bool = True

    async def headers(self, client: httpx.AsyncClient, *, force_refresh: bool = False) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.key}"}

    @property
    def url(self) -> str:
        return f"{self.base_url}/responses"


def _claims(token: str) -> dict[str, Any]:
    part = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))


@dataclass
class CodexLogin:
    """ChatGPT account tokens from a local Codex login."""

    path: Path = Path.home() / ".codex" / "auth.json"
    url: str = "https://chatgpt.com/backend-api/codex/responses"
    refresh_margin_seconds: int = 300
    accepts_max_output_tokens: bool = False

    async def headers(self, client: httpx.AsyncClient, *, force_refresh: bool = False) -> dict[str, str]:
        auth = json.loads(self.path.read_text())
        tokens = auth["tokens"]
        expires = _claims(tokens["access_token"]).get("exp", 0)
        if force_refresh or expires - time.time() < self.refresh_margin_seconds:
            tokens = await self._refresh(client, auth)
        return {
            "Authorization": f"Bearer {tokens['access_token']}",
            "chatgpt-account-id": tokens["account_id"],
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",
        }

    async def _refresh(self, client: httpx.AsyncClient, auth: dict[str, Any]) -> dict[str, Any]:
        tokens = auth["tokens"]
        claims = _claims(tokens["access_token"])
        response = await client.post(
            f"{claims['iss'].rstrip('/')}/oauth/token",
            json={
                "client_id": claims["client_id"],
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
                "scope": "openid profile email",
            },
        )
        if response.status_code != 200:
            raise PermissionError(f"refreshing the Codex login failed ({response.status_code}); run `codex login`")
        refreshed = response.json()
        tokens = {
            **tokens,
            **{key: refreshed[key] for key in ("access_token", "id_token", "refresh_token") if key in refreshed},
        }
        auth = {**auth, "tokens": tokens, "last_refresh": datetime.now(UTC).isoformat().replace("+00:00", "Z")}
        # Write back atomically, keeping the file private, so the Codex CLI sees the rotated tokens.
        with tempfile.NamedTemporaryFile("w", dir=self.path.parent, delete=False) as file:
            json.dump(auth, file, indent=2)
        os.chmod(file.name, 0o600)
        os.replace(file.name, self.path)
        return tokens


@dataclass(frozen=True)
class ResponsesContract:
    """The capability contract the endpoint advertises; the provider does not report it."""

    context_limit: int = 200_000
    max_output_tokens: int = 32_000


class ResponsesEndpoint:
    """Serves one model through the Responses API. Direct adapters do not deduplicate: a retried effect re-samples."""

    def __init__(
        self,
        credentials: Credentials,
        model: str,
        *,
        sampling: SamplingParameters | None = None,
        contract: ResponsesContract | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 600.0,
        blobs: Blobs | None = None,
    ) -> None:
        """`blobs` reads the bytes of `Media` blocks (images); without it, a context with media cannot be sent."""
        self._credentials = credentials
        self._blobs = blobs
        self._model = model
        self._sampling = sampling or SamplingParameters()
        limits = contract or ResponsesContract()
        self._contract = CapabilityContract(
            context_limit=limits.context_limit, max_output_tokens=limits.max_output_tokens
        )
        self._client = client or httpx.AsyncClient(timeout=timeout)

    def describe(self, session_id: str) -> CapabilityContract:
        return self._contract

    async def cancel(self, effect_id: str) -> None:
        """Nothing to do: the request stops when the task awaiting `sample` is cancelled."""

    async def sample(self, request: SampleRequest) -> SampleResult:
        body = self.request_body(request, await self._read_media(request.context.append))
        for attempt in (1, 2):
            headers = await self._credentials.headers(self._client, force_refresh=attempt == 2)
            headers["Accept"] = "text/event-stream"
            async with self._client.stream("POST", self._credentials.url, headers=headers, json=body) as response:
                if response.status_code == 401 and attempt == 1:
                    continue
                if response.status_code != 200:
                    body_text = (await response.aread()).decode(errors="replace")
                    raise _error(response, body_text, self._contract.context_limit)
                return _result(await _completed_response(response.aiter_lines()), self._contract)
        raise PermissionError("the model API rejected the credentials after a refresh")

    def request_body(self, request: SampleRequest, media: Mapping[str, bytes] | None = None) -> dict[str, JsonValue]:
        """The Responses API request for a sample request (public for tests and debugging). `media` holds the bytes
        of the context's `Media` blocks by SHA-256."""
        instructions, items = _render(request.context.append, media or {})
        body: dict[str, JsonValue] = {
            "model": self._model,
            "instructions": instructions,
            "input": items,
            "stream": True,
            "store": False,
            "parallel_tool_calls": True,
        }
        if request.tools:
            body["tools"] = [_tool(specification) for specification in request.tools]
        if request.tool_choice is not None:
            body["tool_choice"] = _tool_choice(request.tool_choice)
        if request.max_output_tokens is not None and self._credentials.accepts_max_output_tokens:
            body["max_output_tokens"] = request.max_output_tokens
        if self._sampling.temperature != 1.0:
            body["temperature"] = self._sampling.temperature
        if self._sampling.top_p != 1.0:
            body["top_p"] = self._sampling.top_p
        if self._sampling.reasoning_effort is not None:
            body["reasoning"] = {"effort": self._sampling.reasoning_effort}
        return body

    async def _read_media(self, messages: Sequence[Message]) -> dict[str, bytes]:
        media: dict[str, bytes] = {}
        for block in _media_blocks(messages):
            if block.source.sha256 not in media:
                if self._blobs is None:
                    raise ValueError("the context has media, but this endpoint was given no blob store to read it")
                media[block.source.sha256] = await self._blobs.read(block.source)
        return media


def codex_provider(
    contract: ResponsesContract | None = None, *, blobs: Blobs | None = None
) -> Callable[[DirectModel], ResponsesEndpoint]:
    """An endpoint factory for `LocalRunner(providers={"codex": codex_provider()})`, using the local Codex login."""
    credentials = CodexLogin()
    client = httpx.AsyncClient(timeout=600.0)

    def factory(model: DirectModel) -> ResponsesEndpoint:
        return ResponsesEndpoint(
            credentials, model.model, sampling=model.sampling, contract=contract, client=client, blobs=blobs
        )

    return factory


# Rendering canonical content to Responses API items ---------------------------------------------------------------


def _render(messages: Sequence[Message], media: Mapping[str, bytes]) -> tuple[str, list[JsonValue]]:
    instructions: list[str] = []
    items: list[JsonValue] = []
    for message in messages:
        if message.role is Role.SYSTEM:
            instructions.append(message.text)
        elif message.role is Role.USER:
            content = [_input(block, media) for block in message.content if isinstance(block, Text | Media)]
            items.append({"type": "message", "role": "user", "content": content})
        elif message.role is Role.ASSISTANT:
            text = message.text
            if text:
                items.append(
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
                )
            for call in message.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": json.dumps(dict(call.arguments)),
                    }
                )
        else:
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    items.append(
                        {"type": "function_call_output", "call_id": block.call_id, "output": _output(block, media)}
                    )
    return "\n\n".join(instructions) or "You are a helpful assistant.", items


def _output(block: ToolResultBlock, media: Mapping[str, bytes]) -> JsonValue:
    """A tool's output: a string, or a list of content items when it has media."""
    parts = block.result.content
    if not any(isinstance(part, Media) for part in parts):
        output = "".join(part.text for part in parts if isinstance(part, Text))
        return f"Error: {output}" if block.result.is_error else output
    items = [_input(part, media) for part in parts]
    if block.result.is_error:
        items.insert(0, {"type": "input_text", "text": "Error:"})
    return items


def _input(block: Text | Media, media: Mapping[str, bytes]) -> JsonValue:
    if isinstance(block, Text):
        return {"type": "input_text", "text": block.text}
    if not block.media_type.startswith("image/"):
        return {"type": "input_text", "text": f"[{block.media_type} content omitted: this model reads only images]"}
    data = base64.b64encode(media[block.source.sha256]).decode()
    return {"type": "input_image", "image_url": f"data:{block.media_type};base64,{data}"}


def _media_blocks(messages: Sequence[Message]) -> list[Media]:
    found: list[Media] = []
    for message in messages:
        for block in message.content:
            if isinstance(block, Media):
                found.append(block)
            elif isinstance(block, ToolResultBlock):
                found.extend(part for part in block.result.content if isinstance(part, Media))
    return found


def _tool(specification: ToolSpecification) -> JsonValue:
    return {
        "type": "function",
        "name": specification.name,
        "description": specification.description,
        "parameters": cast(JsonValue, dict(specification.input_schema)),
        "strict": False,
    }


def _tool_choice(choice: ToolChoice) -> JsonValue:
    if isinstance(choice, NamedToolChoice):
        return {"type": "function", "name": choice.name}
    return choice.value


# Reading the event stream -----------------------------------------------------------------------------------------


async def _completed_response(lines: AsyncIterator[str]) -> Mapping[str, Any]:
    """The final response object of a server-sent event stream, with the output items collected along the way."""
    items: list[Mapping[str, Any]] = []
    async for line in lines:
        if not line.startswith("data: "):
            continue
        event = json.loads(line[6:])
        kind = event.get("type")
        if kind == "response.output_item.done":
            items.append(event["item"])
        elif kind in ("response.completed", "response.incomplete"):
            response = dict(event["response"])
            response["output"] = response.get("output") or items
            return response
        elif kind in ("response.failed", "error"):
            detail = event.get("response", {}).get("error") or event.get("error") or event
            raise InternalError(f"the model API failed: {detail}")
    raise InternalError("the model API closed the stream before the response completed")


def _result(response: Mapping[str, Any], contract: CapabilityContract) -> SampleResult:
    blocks: list[Text | ToolCall] = []
    for item in response["output"]:
        if item.get("type") == "message":
            text = "".join(
                part.get("text", "") for part in item.get("content", []) if part.get("type") == "output_text"
            )
            if text:
                blocks.append(Text(text=text))
        elif item.get("type") == "function_call":
            arguments = json.loads(item.get("arguments") or "{}")
            blocks.append(ToolCall(call_id=item["call_id"], name=item["name"], arguments=arguments))
    usage: Mapping[str, Any] = response.get("usage") or {}
    input_tokens = cast(int | None, usage.get("input_tokens"))
    output_tokens = cast(int | None, usage.get("output_tokens"))
    if response.get("status") == "incomplete":
        finish = FinishReason.LENGTH
    elif any(isinstance(block, ToolCall) for block in blocks):
        finish = FinishReason.TOOL_USE
    else:
        finish = FinishReason.STOP
    return SampleResult(
        message=Message(role=Role.ASSISTANT, content=blocks),
        finish_reason=finish,
        usage=Usage(
            context_used=(input_tokens or 0) + (output_tokens or 0),
            context_limit=contract.context_limit,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _error(response: httpx.Response, body: str, context_limit: int) -> Exception:
    status = response.status_code
    if status == 429:
        retry_after = response.headers.get("retry-after")
        return Overloaded(float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else None)
    if status == 400 and "context_length_exceeded" in body:
        return ContextOverflow(context_limit)
    if status in (401, 403):
        return PermissionError(f"the model API rejected the credentials ({status}): {body[:300]}")
    return InternalError(f"the model API returned {status}: {body[:300]}")
