"""An episode: one finished run as whoever trains on it sees it.

Its labels say which group and task it came from; its outcome and result say how it went; its traces hold, for each
model slot, the token sequences the policy saw and continued, with the logprobs it sampled them at. Nothing else
about the run is needed to compute a loss, and nothing here says where the run executed.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import JsonValue, TypeAdapter

from rollout.core.contracts import RunEvent, RunEventType
from rollout.recorder import Epoch

POLICY = "policy"
"""The model slot an observation's reward belongs to: the one an agent acts through."""
DEFAULT = "default"
"""The key of a reward assigned without one."""


class Outcome(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    """The program raised: its `detail` says what."""
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Trace:
    """One model slot's part of an episode."""

    epochs: list[Epoch]
    rewards: Mapping[str, float]
    """By key; a program that assigns one reward uses the key `default`."""

    @property
    def reward(self) -> float:
        return self.rewards.get(DEFAULT, 0.0)


@dataclass(frozen=True)
class Episode:
    cursor: int
    """Its place in the job's log: episodes are numbered from 1 in the order they ended."""
    job: str
    ticket: str
    run_id: str
    labels: Mapping[str, str]
    parameters: JsonValue
    """The row the run was given."""
    outcome: Outcome
    detail: str | None = None
    info: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """What the program reported as its result (`run.emit("result", {...})`). `solved`, `saturated` and `duration`
    read the three entries training knows about."""
    excluded: str | None = None
    """Why the program asked for the run to be left out of training, if it did."""
    traces: Mapping[str, Trace] = field(default_factory=dict[str, Trace])

    @property
    def reward(self) -> float:
        """The mean of the slots' rewards (a team that is rewarded together has one reward)."""
        return sum(trace.reward for trace in self.traces.values()) / len(self.traces) if self.traces else 0.0

    @property
    def trainable(self) -> bool:
        return self.outcome is Outcome.COMPLETED and self.excluded is None

    @property
    def solved(self) -> bool:
        """Whether the program said its task was solved (`info["solved"]`)."""
        return self.info.get("solved") is True

    @property
    def saturated(self) -> bool:
        """Whether the program said nothing was left to earn (`info["saturated"]`)."""
        return self.info.get("saturated") is True

    @property
    def duration(self) -> float | None:
        """How long the program said it took, in the task's own units (`info["duration"]`), if it said."""
        took = self.info.get("duration")
        return float(took) if isinstance(took, int | float) and not isinstance(took, bool) else None

    def to_json(self) -> dict[str, Any]:
        return _EPISODE.dump_python(self, mode="json")

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Episode":
        return _EPISODE.validate_python(data)


_EPISODE = TypeAdapter(Episode)


def rewards(events: Iterable[tuple[str, Mapping[str, Any]]]) -> dict[str, dict[str, float]]:
    """The rewards a run's events assign, by model slot and key: each event as its type and payload. A reward on an
    observation belongs to the slot the agent acts through."""
    total: dict[str, dict[str, float]] = {}

    def add(slot: str, key: str, value: float) -> None:
        of_slot = total.setdefault(slot, {})
        of_slot[key] = of_slot.get(key, 0.0) + value

    for kind, payload in events:
        if kind == RunEventType.REWARD_ASSIGNED:
            add(str(payload["slot"]), str(payload.get("key", DEFAULT)), float(payload["value"]))
        elif kind == RunEventType.OBSERVATION_RECORDED and payload.get("reward") is not None:
            add(POLICY, DEFAULT, float(payload["reward"]))
    return total


def assemble(
    events: Sequence[RunEvent],
    epochs: Mapping[str, list[Epoch]],
    *,
    cursor: int,
    job: str,
    ticket: str,
    parameters: JsonValue,
) -> Episode:
    """An episode from a run's events (its labels, rewards, result and ending) and what the recorder kept of each
    of its model slots."""
    labels: Mapping[str, str] = {}
    info: Mapping[str, JsonValue] = {}
    excluded: str | None = None
    outcome, detail = Outcome.CANCELLED, None
    payloads: list[tuple[str, Mapping[str, Any]]] = [
        (event.type, event.payload if isinstance(event.payload, dict) else {}) for event in events
    ]
    for kind, payload in payloads:
        match kind:
            case RunEventType.RUN_CREATED:
                labels = dict(payload.get("labels") or {})
            case RunEventType.OUTPUT_EMITTED if payload.get("kind") == "result":
                info = dict(payload.get("payload") or {})
            case RunEventType.TRAINING_EXCLUDED:
                excluded = str(payload.get("reason"))
            case RunEventType.RUN_COMPLETED:
                outcome = Outcome.COMPLETED
            case RunEventType.RUN_FAILED:
                outcome, detail = Outcome.FAILED, str(payload.get("detail"))
            case _:
                pass
    assigned = rewards(payloads)
    traces = {slot: Trace(epochs.get(slot, []), assigned.get(slot, {})) for slot in dict.fromkeys([*epochs, *assigned])}
    run_id = events[0].run_id if events else ""
    return Episode(cursor, job, ticket, run_id, labels, parameters, outcome, detail, info, excluded, traces)
