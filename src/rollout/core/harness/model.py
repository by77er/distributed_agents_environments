"""`Model.sample` as an effect on a `ModelEndpoint` (docs/contracts/model-endpoint.md)."""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from pydantic import JsonValue
from pydantic_core import to_jsonable_python

from rollout.core.contracts import (
    CapabilityContract,
    ContextDelta,
    ContractViolation,
    EffectKind,
    InternalError,
    Message,
    ModelEndpoint,
    Overloaded,
    SampleRequest,
    SampleResult,
    ToolChoice,
    ToolSpecification,
    Usage,
    context_digests,
    spec_hash,
)

EFFECT_ID_META = "effect_id"
"""`Message.meta` key under which a sampled reply carries the `effect_id` of the sample that produced it."""


class Effects(Protocol):
    """How a run performs effects. The runner decides what performing means: a direct call, or a durable step."""

    async def perform[T](
        self,
        kind: EffectKind,
        arguments: JsonValue,
        execute: Callable[[str, str], Awaitable[T]],
        *,
        completion: Callable[[T], JsonValue],
    ) -> T:
        """Assign the next `effect_id`, digest `arguments`, and run `execute(effect_id, arguments_digest)`.

        `completion` renders the result for the run's events.
        """
        ...


class EndpointModel:
    """A model slot bound to an endpoint. Sends the full context; context deltas come with the direct adapters."""

    def __init__(
        self, endpoint: ModelEndpoint, session_id: str, effects: Effects, *, retries: int = 3, backoff: float = 1.0
    ) -> None:
        self._retries = retries
        self._backoff = backoff
        self._endpoint = endpoint
        self._session_id = session_id
        self._effects = effects
        self._capabilities = endpoint.describe(session_id)
        self._usage: Usage | None = None

    @property
    def capabilities(self) -> CapabilityContract:
        return self._capabilities

    @property
    def usage(self) -> Usage | None:
        return self._usage

    async def sample(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpecification] = (),
        max_output_tokens: int | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Message:
        if max_output_tokens is not None and max_output_tokens > self._capabilities.max_output_tokens:
            raise ContractViolation(
                f"max_output_tokens {max_output_tokens} exceeds the contract's {self._capabilities.max_output_tokens}"
            )
        context = ContextDelta(append=messages, digest=context_digests(messages)[-1])
        arguments: JsonValue = {
            "session_id": self._session_id,
            "context_digest": context.digest,
            "tools": [spec_hash(specification) for specification in tools],
            "max_output_tokens": max_output_tokens,
            "tool_choice": to_jsonable_python(tool_choice),
        }

        sampled_effect_id = ""

        async def execute(effect_id: str, arguments_digest: str) -> SampleResult:
            nonlocal sampled_effect_id
            sampled_effect_id = effect_id
            request = SampleRequest(
                effect_id=effect_id,
                arguments_digest=arguments_digest,
                session_id=self._session_id,
                context=context,
                tools=tools,
                max_output_tokens=max_output_tokens,
                tool_choice=tool_choice,
            )
            try:
                return await self._sample_with_retries(request)
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    await self._endpoint.cancel(effect_id)
                raise

        result = await self._effects.perform(
            EffectKind.MODEL_SAMPLE,
            arguments,
            execute,
            completion=lambda result: result.model_dump(mode="json", exclude_none=True),
        )
        self._usage = result.usage
        return result.message.model_copy(update={"meta": {**result.message.meta, EFFECT_ID_META: sampled_effect_id}})

    async def _sample_with_retries(self, request: SampleRequest) -> SampleResult:
        """Retry overloaded and failing endpoints with exponential backoff; other errors propagate."""
        for attempt in range(self._retries + 1):
            try:
                return await self._endpoint.sample(request)
            except (Overloaded, InternalError) as error:
                if attempt == self._retries:
                    raise
                delay = self._backoff * 2**attempt
                if isinstance(error, Overloaded) and error.retry_after is not None:
                    delay = max(delay, error.retry_after)
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")
