"""What task hooks return: `Observation`, `End` and `WaitFor` (docs/core/harness/task.md)."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from typing import Any

from rollout.core.contracts import Message

type ObservationContent = str | Message | Sequence[Message]


class Ending(StrEnum):
    TERMINATED = "terminated"
    """A real end state (value methods do not bootstrap)."""
    TRUNCATED = "truncated"
    """Stopped by a limit: turns, time, budget (value methods may bootstrap)."""


@dataclass(frozen=True, init=False)
class Observation:
    """What the model is shown next, with the reward for the reply it answers.

    `Observation("text")` builds one USER text message; `Observation(message)` and `Observation([messages])` take
    canonical messages.
    """

    messages: tuple[Message, ...]
    reward: float | None
    """Bound to the reply this observation answers."""
    end: Ending | None
    """`None` means the episode continues."""
    info: Mapping[str, Any]
    """Logged; never shown to the model."""

    def __init__(
        self,
        messages: ObservationContent = (),
        *,
        reward: float | None = None,
        end: Ending | None = None,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(messages, str):
            messages = (Message.user(messages),)
        elif isinstance(messages, Message):
            messages = (messages,)
        object.__setattr__(self, "messages", tuple(messages))
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "info", dict(info or {}))


def End(reward: float | None = None, *, truncated: bool = False, info: Mapping[str, Any] | None = None) -> Observation:
    """A terminal observation."""
    return Observation(reward=reward, end=Ending.TRUNCATED if truncated else Ending.TERMINATED, info=info)


@dataclass(frozen=True)
class WaitFor:
    """Suspend the run until a message of `kind` arrives or `timeout` passes."""

    kind: str = "message"
    timeout: timedelta | None = None
    on_timeout: Observation = field(default_factory=lambda: End(truncated=True))


class InvalidObservation(Exception):
    """A hook returned an observation that breaks the validation rules; the run fails with `INVALID_OBSERVATION`."""
