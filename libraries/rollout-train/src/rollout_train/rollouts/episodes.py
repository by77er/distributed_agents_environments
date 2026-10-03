"""An episode: one finished run as whoever trains on it sees it.

Its labels say which group and task it came from; its outcome and result say how it went; its trajectories hold, for
each model slot (each rollout), the segments the policy saw and continued, with the logprobs it sampled them at.
Nothing else about the run is needed to compute a loss, and nothing here says where the run executed.

An episode is kept as a `Record`: one line, small enough for a log, that names two blobs. One holds the
trajectories; the other the run's events (its tool calls and their results, observations, rewards), which a span's
`effect_id` joins a trajectory to. `stored` writes them and `loaded` reads them back, so an episode outlives the
process that ran it for as long as the blob store keeps it.
"""

import asyncio
import json
import lzma
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from pydantic import JsonValue, TypeAdapter

from rollout.contracts import BlobReference, RunEvent, RunEventType
from rollout.harness.blobs import Blobs
from rollout_train.recorder import Segment

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
class Trajectory:
    """What one model slot's rollout leaves to train on: its segments, and its rewards."""

    segments: list[Segment]
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
    outcome: Outcome
    detail: str | None = None
    info: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """What the program reported as its result (`run.emit("result", {...})`). `solved`, `saturated` and `duration`
    read the three entries training knows about."""
    excluded: str | None = None
    """Why the program asked for the run to be left out of training, if it did."""
    trajectories: Mapping[str, Trajectory] = field(default_factory=dict[str, Trajectory])

    @property
    def reward(self) -> float:
        """The mean of the slots' rewards (a team that is rewarded together has one reward)."""
        return (
            sum(trajectory.reward for trajectory in self.trajectories.values()) / len(self.trajectories)
            if self.trajectories
            else 0.0
        )

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


@dataclass(frozen=True)
class Record:
    """An episode as it is logged and sent: everything but its trajectories, and where those and its events are kept."""

    episode: Episode
    """With no segments in its trajectories: their rewards only."""
    trajectories: BlobReference | None = None
    events: BlobReference | None = None
    sampled: Mapping[str, int] = field(default_factory=dict[str, int])
    """Tokens the policy sampled, by model slot."""

    def to_json(self) -> dict[str, Any]:
        return _RECORD.dump_python(self, mode="json")

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Record":
        return _RECORD.validate_python(data)


_RECORD = TypeAdapter(Record)
_SEGMENTS = TypeAdapter(dict[str, list[Segment]])
COMPRESSED = "application/x-xz"
"""Blobs are JSON, compressed: an episode's segments repeat their prompts, and shrink to a few percent."""


async def stored(episode: Episode, events: Sequence[RunEvent], blobs: Blobs) -> Record:
    """Keep an episode's trajectories and its run's events in `blobs`; returns the record that names them."""
    segments = {slot: trajectory.segments for slot, trajectory in episode.trajectories.items()}
    lines = "".join(event.model_dump_json() + "\n" for event in events).encode()
    trajectories, kept = await asyncio.gather(
        asyncio.to_thread(lzma.compress, _SEGMENTS.dump_json(segments), preset=1),
        asyncio.to_thread(lzma.compress, lines, preset=1),
    )
    return Record(
        episode=_without_segments(episode),
        trajectories=await blobs.put(trajectories, COMPRESSED),
        events=await blobs.put(kept, COMPRESSED),
        sampled={
            slot: sum(segment.sampled for segment in trajectory.segments)
            for slot, trajectory in episode.trajectories.items()
        },
    )


async def loaded(record: Record, blobs: Blobs) -> Episode:
    """The episode a record names, with its trajectories read back from `blobs`."""
    if record.trajectories is None:
        return record.episode
    packed = await blobs.read(record.trajectories)
    segments = _SEGMENTS.validate_json(await asyncio.to_thread(lzma.decompress, packed))
    trajectories = {
        slot: Trajectory(segments.get(slot, []), trajectory.rewards)
        for slot, trajectory in record.episode.trajectories.items()
    }
    return replace(record.episode, trajectories=trajectories)


async def events_of(record: Record, blobs: Blobs) -> list[RunEvent]:
    """The events of the run a record names, as its runner recorded them."""
    if record.events is None:
        return []
    lines = (await asyncio.to_thread(lzma.decompress, await blobs.read(record.events))).decode().splitlines()
    return [RunEvent.model_validate(json.loads(line)) for line in lines]


def _without_segments(episode: Episode) -> Episode:
    return replace(
        episode,
        trajectories={slot: Trajectory([], trajectory.rewards) for slot, trajectory in episode.trajectories.items()},
    )


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
    segments: Mapping[str, list[Segment]],
    *,
    cursor: int,
    job: str,
    ticket: str,
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
    trajectories = {
        slot: Trajectory(segments.get(slot, []), assigned.get(slot, {}))
        for slot in dict.fromkeys([*segments, *assigned])
    }
    run_id = events[0].run_id if events else ""
    return Episode(cursor, job, ticket, run_id, labels, outcome, detail, info, excluded, trajectories)
