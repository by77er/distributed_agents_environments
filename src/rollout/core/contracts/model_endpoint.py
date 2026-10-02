"""The model endpoint contract: the only thing code knows about models (docs/contracts/model-endpoint.md).

Implemented by the recorder and by direct adapters; code cannot tell which serves a model slot.
"""

from enum import StrEnum
from typing import Protocol, Self, runtime_checkable

from pydantic import model_validator

from rollout.core.contracts.base import ContractModel, FrozenSequence
from rollout.core.contracts.content import Message, Role, ToolSpecification


class ToolChoiceMode(StrEnum):
    """Whether the model may, must not, or must call a tool."""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


class NamedToolChoice(ContractModel):
    """The model must call this tool."""

    name: str


type ToolChoice = ToolChoiceMode | NamedToolChoice


class FinishReason(StrEnum):
    """Why a sample stopped."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_USE = "tool_use"


class CapabilityContract(ContractModel):
    """What a model slot guarantees. It must not weaken during a run."""

    context_limit: int
    """Minimum guaranteed."""
    max_output_tokens: int


class ContextDelta(ContractModel):
    """The context of a request: its messages, and the digest that names them."""

    append: FrozenSequence[Message] = ()
    """The whole context, in order."""
    digest: str
    """The last value of the chain `rollout.core.contracts.digests.context_digests` computes over `append`."""


class SampleRequest(ContractModel):
    """A request for one reply. Sampling parameters are not here: they belong to the policy."""

    effect_id: str
    """Required: the idempotency key."""
    arguments_digest: str
    session_id: str
    context: ContextDelta
    tools: FrozenSequence[ToolSpecification] = ()
    """The subset exposed this turn; endpoints use only the model-visible fields."""
    max_output_tokens: int | None = None
    """Must not exceed the contract's `max_output_tokens`."""
    tool_choice: ToolChoice | None = None

    @model_validator(mode="after")
    def _unique_tool_names(self) -> Self:
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError(f"tool names must be unique within a request: {names}")
        return self


class Usage(ContractModel):
    """Context use after a sample."""

    context_used: int
    """Drives the agent's compaction decisions."""
    context_limit: int
    input_tokens: int | None = None
    output_tokens: int | None = None


class SampleResult(ContractModel):
    """Nothing here identifies the policy, weights version or engine."""

    message: Message
    finish_reason: FinishReason
    usage: Usage

    @model_validator(mode="after")
    def _assistant_message(self) -> Self:
        if self.message.role is not Role.ASSISTANT:
            raise ValueError("a sample result is an ASSISTANT message")
        return self


class ModelAddress(ContractModel):
    """Where a harness that brings its own loop reaches a model slot: an OpenAI-compatible endpoint. Whatever
    answers there is the slot's model; the harness only sets its base URL and key."""

    base_url: str
    api_key: str
    """Names the session: valid for this run's slot only."""
    model: str
    """What to send as the model's name (the endpoint ignores it)."""


class ModelEndpoint(Protocol):
    """Serves model slots: implemented by the recorder and by direct adapters. An endpoint that can also be reached
    over HTTP is an `AddressableEndpoint`."""

    def describe(self, session_id: str) -> CapabilityContract:
        """The capability contract of the session's model slot."""
        ...

    async def sample(self, request: SampleRequest) -> SampleResult:
        """One reply. Where the endpoint deduplicates, a repeated `effect_id` returns the recorded result."""
        ...

    async def cancel(self, effect_id: str) -> None:
        """Best-effort."""
        ...


@runtime_checkable
class AddressableEndpoint(ModelEndpoint, Protocol):
    """A model endpoint that also serves its slots over HTTP, to a harness that brings its own loop."""

    def address(self, session_id: str, *, through: ModelEndpoint | None = None) -> ModelAddress:
        """Where such a harness reaches the session's slot. What it samples there goes `through` an endpoint
        wrapping this one, if one is given (a runner's, which reports samples to its hooks)."""
        ...


def address_of(endpoint: ModelEndpoint, session_id: str, *, through: ModelEndpoint | None = None) -> ModelAddress:
    """`AddressableEndpoint.address` of an endpoint; raises if the endpoint has no address."""
    if not isinstance(endpoint, AddressableEndpoint):
        raise RuntimeError("this model slot is not served over HTTP: a harness cannot be given an address")
    return endpoint.address(session_id, through=through)


class ModelEndpointError(Exception):
    """Errors an endpoint raises; see the table in the contract for how the core handles each."""


class Overloaded(ModelEndpointError):
    """Admission control: retry after `retry_after` seconds."""

    def __init__(self, retry_after: float | None = None) -> None:
        super().__init__(f"overloaded; retry after {retry_after}s")
        self.retry_after = retry_after


class ContextOverflow(ModelEndpointError):
    """The context exceeds the contract's limit; agents compact and retry."""

    def __init__(self, context_limit: int) -> None:
        super().__init__(f"context exceeds the limit of {context_limit} tokens")
        self.context_limit = context_limit


class ContractViolation(ModelEndpointError):
    """The request exceeds the capability contract."""


class InternalError(ModelEndpointError):
    """The endpoint failed."""
