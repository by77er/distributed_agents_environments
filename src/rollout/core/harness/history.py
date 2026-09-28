"""The episode as the agent sees it: observations and replies (docs/core/harness/agent.md)."""

from dataclasses import dataclass
from enum import StrEnum

from rollout.core.contracts import Message, Role
from rollout.core.harness.observation import Observation


@dataclass(frozen=True)
class Turn:
    """One step of the episode: a reply and the observation that answers it.

    The start observation, and observations produced by `resume`, have no reply. A reply answered by a `WaitFor`
    has no observation until the run resumes.
    """

    reply: Message | None
    observation: Observation | None


class HistoryShape(StrEnum):
    """How much of the history the model should see."""

    FULL = "full"
    """Everything."""
    LATEST_OBSERVATION = "latest_observation"
    """Only the latest observation: observations are complete states."""
    WINDOW = "window"
    """The start observation and the most recent `window` turns."""


@dataclass(frozen=True)
class ContextHints:
    """Advisory for the agent: how much history the task needs the model to see."""

    history: HistoryShape = HistoryShape.FULL
    window: int | None = None
    """For `WINDOW`: the number of recent turns."""


class History:
    """Read-only for agents and tasks; the loop appends to it through the run context."""

    def __init__(self) -> None:
        self._turns: list[Turn] = []

    @property
    def turns(self) -> tuple[Turn, ...]:
        """Every turn, the start observation first."""
        return tuple(self._turns)

    def append(self, turn: Turn) -> None:
        """For run contexts only."""
        self._turns.append(turn)

    def messages(self, hints: ContextHints | None = None) -> list[Message]:
        """The episode's messages in order, shaped by `hints`."""
        turns = self._turns
        shape = hints.history if hints else HistoryShape.FULL
        if shape is HistoryShape.LATEST_OBSERVATION:
            turns = _from_latest_observation(turns)
        elif shape is HistoryShape.WINDOW and hints and hints.window is not None:
            turns = turns[:1] + turns[1:][-hints.window :] if hints.window > 0 else turns[:1]
        messages: list[Message] = []
        for turn in turns:
            if turn.reply is not None:
                messages.append(turn.reply)
            if turn.observation is not None:
                messages.extend(turn.observation.messages)
        return messages


def _from_latest_observation(turns: list[Turn]) -> list[Turn]:
    for index in range(len(turns) - 1, -1, -1):
        observation = turns[index].observation
        if observation is not None and observation.messages:
            turn = turns[index]
            # Tool results are only meaningful next to the reply whose calls they answer.
            answers_calls = any(message.role is Role.TOOL for message in observation.messages)
            return [turn if answers_calls else Turn(reply=None, observation=observation)]
    return []
