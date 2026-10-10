"""A `ModelEndpoint` for Anthropic's Messages API, through the `anthropic` SDK
(docs/implementations/rollout-anthropic.md).

A request is the whole context, every time: system messages become the request's `system`, user messages and tool
results its user turns (a tool's result a `tool_result` block, consecutive ones in one turn), and the assistant's
replies its assistant turns (text, and each tool call a `tool_use` block). Reasoning in the context is left out: a
thinking block is replayed only with the signature the API gave it, which canonical content does not keep. Images
(and PDFs) are sent as base64 blocks, read from the blob store.

What each model takes differs, and its catalog entry says it (`MessagesOptions`, from the model's `options`): how it
thinks (`adaptive`, steered by an effort; `budget`, a number of thinking tokens; or not at all), whether it takes
temperature and top-p, and whether it takes a forced tool choice. A request asks for the thinking the model takes:
adaptive thinking with the binding's reasoning effort (else the model's own `effort`); a budget of the binding's
thinking tokens, where there are at least 1024 and room for them under `max_tokens`. Temperature and top-p are sent
where the model takes them and they differ from 1, never beside a thinking budget (the API refuses both). `max_tokens`
is the request's `max_output_tokens`, else the binding's thinking and answer budgets added up, else the model's most
output.

A reply is streamed and read whole. Its text, tool calls and summarized thinking (as `Reasoning`) come back as
canonical content; it finishes with `length` where the API stopped at `max_tokens` or the context's end, `tool_use`
where it calls tools, else `stop`. Usage counts every input token (uncached, read from the cache, written to it), those
read from the cache, every output token and those spent thinking.

Errors map to the contract: rate limits and an API that is down or overloaded (429, 500, 502, 503, 504, 529, and an
`overloaded_error`, `rate_limit_error` or `api_error` in the stream) are `Overloaded`, with the API's `retry-after`; a
prompt too long for the context is `ContextOverflow`; credentials refused raise `PermissionError`; any other refusal (a
request the API rejects) is a `ModelEndpointError`, which asking again does not change; a connection that fails is an
`InternalError`. The SDK retries nothing itself: whoever holds the endpoint decides (the gateway's API channels back off
and ask again).
"""

import base64
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import anthropic
from pydantic import JsonValue

from rollout.contracts import (
    CapabilityContract,
    ContextOverflow,
    FinishReason,
    InternalError,
    Media,
    Message,
    ModelEndpointError,
    NamedToolChoice,
    Overloaded,
    Reasoning,
    ReasoningScope,
    Role,
    SampleRequest,
    SampleResult,
    Text,
    ToolCall,
    ToolChoice,
    ToolChoiceMode,
    ToolResultBlock,
    ToolSpecification,
    Usage,
)
from rollout.harness.blobs import Blobs
from rollout.harness.runner import SamplingParameters

__all__ = ["MessagesEndpoint", "MessagesOptions", "hosted"]

THINKING = ("adaptive", "budget", "none")
"""How a model thinks: adaptively (steered by an effort), within a budget of tokens, or not at all."""
LEAST_BUDGET = 1024
"""The fewest thinking tokens the API takes as a budget."""
OVERLOADED = frozenset({429, 500, 502, 503, 504, 529})
"""Statuses that say the API cannot take the request now: asked again later, it may."""
SAMPLING = ("temperature", "top_p")
"""The sampling parameters a request may carry, which the SDK takes as extra body fields."""
STREAM_OVERLOADED = frozenset({"overloaded_error", "rate_limit_error", "api_error"})
"""Error types an event in the stream says that asking again may cure."""
_TOOL_ID = re.compile(r"[^a-zA-Z0-9_-]")


@dataclass(frozen=True)
class MessagesOptions:
    """What one model takes, as its catalog entry's `options` say."""

    thinking: Literal["adaptive", "budget", "none"] = "none"
    effort: str | None = None
    """The effort adaptive thinking is steered by where a binding says none (none: the model's default)."""
    sampling: bool = True
    """Whether it takes temperature and top-p."""
    forced_tool_choice: bool = True
    """Whether it takes a tool choice that forces a call (`any`, a named tool); where it does not, such a choice is
    sent as `auto`."""

    @classmethod
    def of(cls, options: Mapping[str, JsonValue] | None) -> "MessagesOptions":
        """The options of a model's catalog entry (those not about the Messages API are left)."""
        said = dict(options or {})
        thinking = str(said.get("thinking", "none"))
        if thinking not in THINKING:
            raise ValueError(f"thinking is one of {', '.join(THINKING)}, not {thinking!r}")
        effort = said.get("effort")
        return cls(
            thinking=thinking,
            effort=str(effort) if effort else None,
            sampling=bool(said.get("sampling", True)),
            forced_tool_choice=bool(said.get("forced_tool_choice", True)),
        )


class MessagesEndpoint:
    """Serves one model through the Messages API. It does not deduplicate: a retried effect samples again."""

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        model: str,
        *,
        contract: CapabilityContract,
        options: MessagesOptions | None = None,
        sampling: SamplingParameters | None = None,
        blobs: Blobs | None = None,
    ) -> None:
        """`client` is made with `max_retries=0` (the holder retries); `blobs` reads the bytes of `Media` blocks:
        without it, a context with media cannot be sent."""
        self._client = client
        self._model = model
        self._contract = contract
        self._options = options or MessagesOptions()
        self._sampling = sampling or SamplingParameters()
        self._blobs = blobs

    def describe(self, session_id: str) -> CapabilityContract:
        return self._contract

    async def cancel(self, effect_id: str) -> None:
        """Nothing to do: the request stops when the task awaiting `sample` is cancelled."""

    async def sample(self, request: SampleRequest, *, sampling: SamplingParameters | None = None) -> SampleResult:
        """One reply; `sampling` in place of the endpoint's own sampling parameters, for this request."""
        body = self.request_body(request, await self._read_media(request.context.append), sampling=sampling)
        extra = {key: body.pop(key) for key in SAMPLING if key in body}  # (sent as they are: the SDK names neither)
        try:
            async with self._client.messages.stream(**body, extra_body=extra or None) as stream:
                message = await stream.get_final_message()
        except anthropic.APIStatusError as error:
            raise _error(error, self._contract.context_limit) from None
        except anthropic.APIConnectionError as error:
            raise InternalError(f"the Messages API could not be reached: {error}") from None
        return _result(message.model_dump(mode="json"), self._contract)

    def request_body(
        self,
        request: SampleRequest,
        media: Mapping[str, bytes] | None = None,
        *,
        sampling: SamplingParameters | None = None,
    ) -> dict[str, Any]:
        """The Messages API request for a sample request (public for tests and debugging). `media` holds the bytes of
        the context's `Media` blocks by SHA-256; `sampling`, the sampling parameters in place of the endpoint's."""
        said = sampling or self._sampling
        options = self._options
        system, messages = _render(request.context.append, media or {})
        room = request.max_output_tokens
        if room is None and said.thinking_tokens is not None and said.answer_tokens is not None:
            room = said.thinking_tokens + said.answer_tokens
        room = min(room or self._contract.max_output_tokens, self._contract.max_output_tokens)
        body: dict[str, Any] = {"model": self._model, "max_tokens": room, "messages": messages}
        if system:
            body["system"] = system
        if request.tools:
            body["tools"] = [_tool(specification) for specification in request.tools]
            if request.tool_choice is not None:
                body["tool_choice"] = _tool_choice(request.tool_choice, options.forced_tool_choice)
        budgeted = False
        if options.thinking == "adaptive":
            body["thinking"] = {"type": "adaptive", "display": "summarized"}
            effort = said.reasoning_effort or self._sampling.reasoning_effort or options.effort
            if effort is not None:
                body["output_config"] = {"effort": effort}
        elif options.thinking == "budget" and said.thinking_tokens is not None:
            budget = min(said.thinking_tokens, room - 1)
            if budget >= LEAST_BUDGET:
                body["thinking"] = {"type": "enabled", "budget_tokens": budget}
                budgeted = True
        if options.sampling and not budgeted:
            if said.temperature != 1.0:
                body["temperature"] = said.temperature
            elif said.top_p != 1.0:  # (the API takes one of the two)
                body["top_p"] = said.top_p
        return body

    async def _read_media(self, messages: Sequence[Message]) -> dict[str, bytes]:
        media: dict[str, bytes] = {}
        for block in _media_blocks(messages):
            if block.source.sha256 not in media:
                if self._blobs is None:
                    raise ValueError("the context has media, but this endpoint was given no blob store to read it")
                media[block.source.sha256] = await self._blobs.read(block.source)
        return media


def hosted(
    model: str,
    *,
    api_key: str | None,
    context_limit: int,
    max_output_tokens: int,
    options: Mapping[str, JsonValue] | None = None,
    base_url: str | None = None,
    blobs: Blobs | None = None,
    timeout: float = 600.0,
) -> MessagesEndpoint:
    """The endpoint of a cluster's `api` provider for one of its models (`endpoint = "rollout_anthropic:hosted"`): its
    API key, the model's context and most output and its `options` (`MessagesOptions`), from its catalog entry, and the
    API's base URL (none: Anthropic's own). Raises `PermissionError` without a key."""
    if not api_key:
        raise PermissionError("the provider has no API key: set the environment variable its api_key_env names")
    client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url, max_retries=0, timeout=timeout)
    contract = CapabilityContract(context_limit=context_limit, max_output_tokens=max_output_tokens)
    return MessagesEndpoint(client, model, contract=contract, options=MessagesOptions.of(options), blobs=blobs)


# Rendering canonical content to Messages API turns ----------------------------------------------------------------


def _render(messages: Sequence[Message], media: Mapping[str, bytes]) -> tuple[str, list[dict[str, Any]]]:
    """The system prompt, and the turns: consecutive messages of one side in one turn (a tool's results open the user
    turn that follows the call), empty ones left out."""
    system: list[str] = []
    turns: list[dict[str, Any]] = []
    for message in messages:
        if message.role is Role.SYSTEM:
            system.append(message.text)
            continue
        if message.role is Role.ASSISTANT:
            role, content = "assistant", _assistant(message)
        elif message.role is Role.TOOL:
            role = "user"
            content = [_tool_result(block, media) for block in message.content if isinstance(block, ToolResultBlock)]
        else:
            role = "user"
            content = [_input(block, media) for block in message.content if isinstance(block, Text | Media)]
        if not content:
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(content)
        else:
            turns.append({"role": role, "content": content})
    return "\n\n".join(part for part in system if part), turns


def _assistant(message: Message) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, Text) and block.text:
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolCall):
            content.append({"type": "tool_use", "id": _tool_id(block.call_id), "name": block.name,
                            "input": dict(block.arguments)})  # fmt: skip
    return content


def _tool_id(call_id: str) -> str:
    """A call's id as the API takes it (letters, digits, `_` and `-`), the same for the call and its result."""
    return _TOOL_ID.sub("_", call_id) or "call"


def _tool_result(block: ToolResultBlock, media: Mapping[str, bytes]) -> dict[str, Any]:
    parts = [_input(part, media) for part in block.result.content]
    said: dict[str, Any] = {"type": "tool_result", "tool_use_id": _tool_id(block.call_id), "content": parts}
    if block.result.is_error:
        said["is_error"] = True
    return said


def _input(block: Text | Media, media: Mapping[str, bytes]) -> dict[str, Any]:
    if isinstance(block, Text):
        return {"type": "text", "text": block.text}
    if block.media_type.startswith("image/"):
        kind = "image"
    elif block.media_type == "application/pdf":
        kind = "document"
    else:
        return {"type": "text", "text": f"[{block.media_type} content omitted: this model reads images and PDFs]"}
    data = base64.b64encode(media[block.source.sha256]).decode()
    return {"type": kind, "source": {"type": "base64", "media_type": block.media_type, "data": data}}


def _media_blocks(messages: Sequence[Message]) -> list[Media]:
    found: list[Media] = []
    for message in messages:
        for block in message.content:
            if isinstance(block, Media):
                found.append(block)
            elif isinstance(block, ToolResultBlock):
                found.extend(part for part in block.result.content if isinstance(part, Media))
    return found


def _tool(specification: ToolSpecification) -> dict[str, Any]:
    return {
        "name": specification.name,
        "description": specification.description,
        "input_schema": dict(specification.input_schema),
    }


def _tool_choice(choice: ToolChoice, forced: bool) -> dict[str, Any]:
    if isinstance(choice, NamedToolChoice):
        return {"type": "tool", "name": choice.name} if forced else {"type": "auto"}
    if choice is ToolChoiceMode.REQUIRED:
        return {"type": "any"} if forced else {"type": "auto"}
    return {"type": choice.value}


# Reading the reply -------------------------------------------------------------------------------------------------


def _result(message: Mapping[str, Any], contract: CapabilityContract) -> SampleResult:
    blocks: list[Text | ToolCall | Reasoning] = []
    content: list[Mapping[str, Any]] = message.get("content") or []
    for block in content:
        kind = block.get("type")
        if kind == "text" and block.get("text"):
            blocks.append(Text(text=block["text"]))
        elif kind == "tool_use":
            blocks.append(ToolCall(call_id=block["id"], name=block["name"], arguments=block.get("input") or {}))
        elif kind == "thinking" and block.get("thinking"):
            blocks.append(Reasoning(scope=ReasoningScope.PORTABLE, text=block["thinking"]))
    stop = message.get("stop_reason")
    if stop in ("max_tokens", "model_context_window_exceeded"):
        finish = FinishReason.LENGTH
    elif any(isinstance(block, ToolCall) for block in blocks):
        finish = FinishReason.TOOL_USE
    else:
        finish = FinishReason.STOP
    usage: Mapping[str, Any] = message.get("usage") or {}
    cached = int(usage.get("cache_read_input_tokens") or 0)
    written = int(usage.get("cache_creation_input_tokens") or 0)
    input_tokens = int(usage.get("input_tokens") or 0) + cached + written
    output_tokens = int(usage.get("output_tokens") or 0)
    details: Mapping[str, Any] = usage.get("output_tokens_details") or {}
    thinking = details.get("thinking_tokens")
    return SampleResult(
        message=Message(role=Role.ASSISTANT, content=blocks),
        finish_reason=finish,
        usage=Usage(
            context_used=input_tokens + output_tokens,
            context_limit=contract.context_limit,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached,
            thinking_tokens=int(thinking) if thinking is not None else None,
        ),
    )


def _error(error: anthropic.APIStatusError, context_limit: int) -> Exception:
    status, said = error.status_code, str(error)[:300]
    if status in OVERLOADED or error.type in STREAM_OVERLOADED:
        retry_after = error.response.headers.get("retry-after")
        seconds = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else None
        return Overloaded(seconds)
    if status in (400, 413) and any(each in said.lower() for each in ("prompt is too long", "context window")):
        return ContextOverflow(context_limit)
    if status in (401, 403):
        return PermissionError(f"the Messages API rejected the credentials ({status}): {said}")
    return ModelEndpointError(f"the Messages API refused the request ({status}): {said}")
