"""An episode: one finished run as whoever trains on it sees it.

Its labels say which group and task it came from; its outcome and result say how it went; its traces hold, for each
model slot, the token sequences the policy saw and continued, with the logprobs it sampled them at. Nothing else
about the run is needed to compute a loss, and nothing here says where the run executed.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import JsonValue

from rollout.core.contracts import RunEvent, RunEventType
from rollout.recorder import Epoch, Span


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
        return self.rewards.get("default", 0.0)


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
    """What the program reported as its result (`run.emit("result", {...})`). By convention `solved` and `saturated`
    (nothing was left to earn) are booleans and `duration` is a number in the task's own units."""
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

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Episode":
        traces = {
            str(slot): Trace(
                [Epoch(e["tokens"], [Span(**span) for span in e["spans"]], e["logprobs"]) for e in trace["epochs"]],
                trace["rewards"],
            )
            for slot, trace in data["traces"].items()
        }
        return cls(
            cursor=data["cursor"],
            job=data["job"],
            ticket=data["ticket"],
            run_id=data["run_id"],
            labels=data["labels"],
            parameters=data["parameters"],
            outcome=Outcome(data["outcome"]),
            detail=data["detail"],
            info=data["info"],
            excluded=data["excluded"],
            traces=traces,
        )


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
    rewards: dict[str, dict[str, float]] = {slot: {} for slot in epochs}
    info: Mapping[str, JsonValue] = {}
    excluded: str | None = None
    outcome, detail = Outcome.CANCELLED, None

    def add(slot: str, key: str, value: float) -> None:
        of_slot = rewards.setdefault(slot, {})
        of_slot[key] = of_slot.get(key, 0.0) + value

    for event in events:
        payload: Any = event.payload if isinstance(event.payload, dict) else {}
        match event.type:
            case RunEventType.RUN_CREATED:
                labels = dict(payload.get("labels") or {})
            case RunEventType.REWARD_ASSIGNED:
                add(str(payload["slot"]), str(payload.get("key", "default")), float(payload["value"]))
            case RunEventType.OBSERVATION_RECORDED if payload.get("reward") is not None:
                add("policy", "default", float(payload["reward"]))
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
    traces = {slot: Trace(epochs.get(slot, []), rewards[slot]) for slot in rewards}
    run_id = events[0].run_id if events else ""
    return Episode(cursor, job, ticket, run_id, labels, parameters, outcome, detail, info, excluded, traces)
