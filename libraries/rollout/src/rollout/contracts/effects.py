"""Effects: operations that reach outside task, agent or program code (docs/libraries/rollout/contracts/effects.md)."""

from enum import StrEnum


class EffectKind(StrEnum):
    """The catalog of effects."""

    MODEL_SAMPLE = "model.sample"
    TOOL_CALL = "tool.call"
    OUTPUT_EMIT = "output.emit"


class EffectStatus(StrEnum):
    """How an effect completed."""

    OK = "ok"
    FAILED = "failed"


class Conflict(Exception):
    """A receiver that deduplicates by `effect_id` was sent a known `effect_id` with a different arguments digest."""
