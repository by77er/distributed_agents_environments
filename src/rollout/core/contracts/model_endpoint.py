"""The model endpoint contract: the only thing code knows about models (docs/contracts/model-endpoint.md).

Implemented by the recorder and by direct adapters; code cannot tell which serves a model slot.
"""

from datetime import datetime
from enum import StrEnum
from typing import Protocol, Self

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
    CONTENT_FILTER = "content_filter"


class ReasoningSupport(StrEnum):
    """Which kinds of reasoning blocks a model slot produces."""

    NONE = "none"
    PORTABLE = "portable"
    POLICY_SCOPED = "policy_scoped"


class CapabilityContract(ContractModel):
    """What a model slot guarantees. It must not weaken during a run."""

    contract_version: str = "1"
    context_limit: int
    """Minimum guaranteed."""
    max_output_tokens: int
    modalities_in: frozenset[str] = frozenset({"text"})
    tool_calling: bool = True
    parallel_tool_calls: bool = True
    reasoning: ReasoningSupport = ReasoningSupport.NONE
    accepts_context_delta: bool = False


class ContextDelta(ContractModel):
    """The context of a request as an edit of the previous request's context in the same slot.

    Digests are values of the chain computed by `rollout.core.contracts.digests.context_digests`.
    """

    parent_digest: str | None = None
    """Digest of the previous request's context; `None` means `append` is the full context."""
    keep_prefix: int = 0
    """Number of parent items retained (the parent's length for a pure append)."""
    append: FrozenSequence[Message] = ()
    digest: str


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
    deadline: datetime | None = None

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
    over HTTP has `address(session_id) -> ModelAddress`."""

    def describe(self, session_id: str) -> CapabilityContract:
        """The capability contract of the session's model slot."""
        ...

    async def sample(self, request: SampleRequest) -> SampleResult:
        """One reply. Where the endpoint deduplicates, a repeated `effect_id` returns the recorded result."""
        ...

    async def cancel(self, effect_id: str) -> None:
        """Best-effort."""
        ...


class ModelEndpointError(Exception):
    """Errors an endpoint raises; see the table in the contract for how the core handles each."""


class Overloaded(ModelEndpointError):
    """Admission control: retry after `retry_after` seconds."""

    def __init__(self, retry_after: float | None = None) -> None:
        super().__init__(f"overloaded; retry after {retry_after}s")
        self.retry_after = retry_after


class NeedFullContext(ModelEndpointError):
    """The delta's parent is unknown to the endpoint; the core resends the full context."""


class ContextOverflow(ModelEndpointError):
    """The context exceeds the contract's limit; agents compact and retry."""

    def __init__(self, context_limit: int) -> None:
        super().__init__(f"context exceeds the limit of {context_limit} tokens")
        self.context_limit = context_limit


class ContractViolation(ModelEndpointError):
    """The request exceeds the capability contract."""


class Conflict(ModelEndpointError):
    """A known `effect_id` arrived with a different arguments digest."""


class DeadlineExceeded(ModelEndpointError):
    """The request's deadline passed."""


class InternalError(ModelEndpointError):
    """The endpoint failed."""
