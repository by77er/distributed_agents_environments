"""Effects: operations that reach outside task, agent or program code (docs/libraries/rollout/contracts/effects.md)."""

from enum import StrEnum


class EffectKind(StrEnum):
    """The catalog of effects."""

    MODEL_SAMPLE = "model.sample"
    TOOL_CALL = "tool.call"
    ENVIRONMENT_CALL = "environment.call"
    ENVIRONMENT_LIFECYCLE = "environment.lifecycle"
    OUTPUT_EMIT = "output.emit"


class EffectStatus(StrEnum):
    """How an effect completed."""

    OK = "ok"
    FAILED = "failed"
    OUTCOME_UNKNOWN = "outcome_unknown"
    """A possible duplicate of a side effect that could not be deduplicated; never silently retried."""


class OutcomeUnknown(Exception):
    """A guarded effect was interrupted by a crash in an earlier attempt: it may or may not have happened."""

    def __init__(self, effect_id: str) -> None:
        super().__init__(f"effect {effect_id} may or may not have happened: a crash interrupted it")
        self.effect_id = effect_id


class Conflict(Exception):
    """A receiver that deduplicates by `effect_id` was sent a known `effect_id` with a different arguments digest."""
