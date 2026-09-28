"""Effects: operations that reach outside task, agent or program code (docs/contracts/effects.md)."""

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum

from pydantic import Field, JsonValue

from rollout.core.contracts.base import ContractModel
from rollout.core.contracts.content import RetryClass


class EffectKind(StrEnum):
    MODEL_SAMPLE = "model.sample"
    TOOL_CALL = "tool.call"
    ENVIRONMENT_CALL = "environment.call"
    ENVIRONMENT_LIFECYCLE = "environment.lifecycle"
    MESSAGE_SEND = "message.send"
    MESSAGE_WAIT = "message.wait"
    RUN_SPAWN = "run.spawn"
    TIMER_SLEEP = "timer.sleep"
    OUTPUT_EMIT = "output.emit"


class EffectStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"
    OUTCOME_UNKNOWN = "outcome_unknown"
    """A possible duplicate of a side effect that could not be deduplicated; never silently retried."""


class CallContext(ContractModel):
    """Assembled by the runner, never by task code."""

    labels: Mapping[str, str] = Field(default_factory=dict[str, str])
    tenant: str | None = None


class EffectRequest(ContractModel):
    """What an executor receives. Transport is per implementation; these fields are mandatory everywhere."""

    effect_id: str
    """`{run_id}:{generation}:{ordinal}`; identical on every attempt."""
    arguments_digest: str
    kind: EffectKind
    run_id: str
    attempt: int = 1
    """1-based; informational."""
    deadline: datetime
    """Absolute."""
    payload: JsonValue
    retry_class: RetryClass = RetryClass.UNKNOWN
    context: CallContext = CallContext()


class EffectCompletion(ContractModel):
    """The first completion recorded for an `effect_id` wins; later ones are dropped."""

    effect_id: str
    status: EffectStatus
    payload: JsonValue = None
    error_class: str | None = None
