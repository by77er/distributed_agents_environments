"""What training asks of a trainer, in terms that say nothing of where it runs or what it trains.

An algorithm decides what to train on: segments, and how much each should count (`Weighted`), for a policy gradient
or a likelihood; pairs of episodes' segments, one preferred to the other (`Pair`), or episodes' segments labelled
desirable or undesirable (`Labelled`), for a preference loss; segments with a teacher's scores of their sampled tokens
(`Distilled`), for distillation. Which the trainer's objective takes is its family's
(`rollout_train.objectives`), which a trainer says (`objective`; the `default` preset if it says none). A trainer takes
a batch, moves the policy, and says where the new weights are (`Step`). It also says what it can take (`Budget`): the
longest segment, and how many a step can afford. Those come from its hardware, and nothing above it chooses them. A
trainer may take some of its settings between steps (`Changeable`): a run's settings page changes them for the run's
next step (`rollout_train.settings`).

A trainer on another machine (`Remote`) takes its parent and gives what it made as manifests of the blob store, so
the loop records, serves and thins its checkpoints without reading their files. One that keeps a step's full state
itself after the step returns (`Keeps`) has its weights served first, and its checkpoint's state completed once kept
(`rollout_train.checkpoints.Checkpoints.completed`).

A trainer may say how far the step it is taking has got, as it goes (`Progressing`, with `Progress`): the loop hands
it to its hooks, and a training pod puts it in its beats and its answer about the step.

The loop marks the weighted and distilled segments sampled wholly on the weights a step starts from
(`sampled_at_start`, `marked_at_start`): their behaviour logprobs are where the step starts, up to how the engine and
the trainer compute differently, so a trainer may take them for its own instead of computing them
(`rollout_objectives.settings.StepSettings.old_logprobs`).
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import JsonValue

from rollout_train.checkpoints import Checkpoint, Manifest
from rollout_train.objectives import DEFAULT, Objective
from rollout_train.recorder import Segment, TeacherScores


@dataclass(frozen=True)
class Weighted:
    """A segment to train on, and its advantage: every token the policy sampled in it counts by that much."""

    segment: Segment
    advantage: float
    source: str = ""
    """Where the segment is from, for the record of what a step trained on: `RUN/GROUP/EPISODE/SLOT/INDEX`."""
    sampled_at_start: bool = False
    """Whether every token the policy sampled in it was sampled on the weights the step starts from."""


@dataclass(frozen=True)
class Pair:
    """Two sides over a shared context (one start), the chosen preferred to the rejected: each the segments of one
    episode (every turn the policy sampled in it), of which only the tokens the policy sampled count."""

    chosen: tuple[Segment, ...]
    rejected: tuple[Segment, ...]
    source: str = ""
    """Where the sides are from, for the record of what a step trained on: `RUN/GROUP/CHOSEN>REJECTED` (the episodes'
    numbers)."""

    @property
    def segments(self) -> tuple[Segment, ...]:
        return (*self.chosen, *self.rejected)


@dataclass(frozen=True)
class Labelled:
    """The segments of one episode, labelled desirable or not (KTO's unpaired examples)."""

    side: tuple[Segment, ...]
    desirable: bool
    source: str = ""
    """`RUN/GROUP/EPISODE`."""

    @property
    def segments(self) -> tuple[Segment, ...]:
        return self.side


@dataclass(frozen=True)
class Distilled:
    """A segment and a teacher's scores of its sampled tokens (for the distillation family), and its episode's
    advantage (for a policy gradient with a distillation term; 0 for a distillation alone)."""

    segment: Segment
    scores: TeacherScores
    """One for each sampled token, in the order of the segment's spans, and which teacher gave them."""
    advantage: float = 0.0
    source: str = ""
    """`RUN/GROUP/EPISODE/SLOT/INDEX`."""
    sampled_at_start: bool = False
    """Whether every token the policy sampled in it was sampled on the weights the step starts from."""


type Item = Weighted | Pair | Labelled | Distilled
"""What a batch holds: weighted segments, pairs, labelled examples or distilled segments (never a mixture)."""


def segments_of(item: Item) -> tuple[Segment, ...]:
    """The segments an item holds."""
    return (item.segment,) if isinstance(item, Weighted | Distilled) else item.segments


def marked_at_start(items: Sequence[Item], depth: int) -> list[Item]:
    """`items`, each weighted or distilled segment every span of which was sampled at `depth` (the depth of the weights
    a step starts from: 0, the base model's) marked `sampled_at_start`."""
    return [
        replace(item, sampled_at_start=True)
        if isinstance(item, Weighted | Distilled)
        and item.segment.spans
        and all(span.version == depth for span in item.segment.spans)
        else item
        for item in items
    ]


def weight_of(item: Item) -> float:
    """What the record of a step says an item counted for: a segment's advantage (a distilled segment's 0 without a
    policy gradient), a pair's 1, an example's 1 or -1."""
    if isinstance(item, Weighted | Distilled):
        return item.advantage
    if isinstance(item, Labelled):
        return 1.0 if item.desirable else -1.0
    return 1.0


@dataclass(frozen=True)
class Budget:
    segment_tokens: int | None = None
    """The longest segment the trainer can train on (None: any)."""
    segments: int | None = None
    """How many segments a step can afford (None: any number), counting each of a pair's or an example's."""


@dataclass(frozen=True)
class Files:
    """A checkpoint's files on this machine: what a step starts from."""

    weights: Path
    state: Path | None = None
    """What the trainer left for itself beside the weights (an optimizer's state, say), if it left any."""


@dataclass(frozen=True)
class Step:
    metrics: Mapping[str, float]
    keeping: bool = False
    """Whether the trainer keeps the rest of the step's state itself after returning (`Keeps.kept`): `into/state` then
    holds only what it wrote before it returned."""


@dataclass(frozen=True)
class Made:
    """What a step of a trainer elsewhere made (`Remote`), kept in the blob store: its metrics, the new weights, and the
    state, whole or (`complete` false) only what was kept with the weights so far."""

    metrics: Mapping[str, float]
    weights: Manifest
    state: Manifest | None = None
    complete: bool = True


WEIGHTS = "weights"
"""Under a step's directory: the new weights, as engines load them."""
STATE = "state"
"""Under a step's directory: what the trainer goes on from, and what the step did."""
HELD = "held.txt"
"""In a step's state, from a trainer that keeps its policy in memory between steps (`Resident`): the name it gave what
it holds after the step. A later step whose parent's state names what the trainer holds goes on from its memory, and
reads none of the parent's other files."""


class StepFailed(Exception):
    """A step did not produce weights: the policy is as it was, and a later step may succeed."""


class Trainer(Protocol):
    """A trainer keeps nothing between steps that it cannot be given again: a step says what it starts from and
    where its files go, so any trainer can take any step of any policy."""

    budget: Budget
    weights: str
    """What its steps make: `lora` (an adapter over the weights the engines hold) or `full` (all the weights)."""

    async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step:
        """Train on the batch, starting from `parent` (None: from the base model). The new weights are left in
        `into/weights`, and what a later step starts from in `into/state`. Raises `StepFailed` if the step
        produced no weights."""
        ...


def objective_of(trainer: object) -> Objective:
    """The objective a trainer trains with: its `objective`, else the `default` preset."""
    found = getattr(trainer, "objective", None)
    return found if isinstance(found, Objective) else DEFAULT


@runtime_checkable
class Resident(Protocol):
    """A trainer that keeps its policy and optimizer in memory between steps (`rollout_lora`'s, with its GPUs to
    itself): a step from the checkpoint it made last goes on from them. Its steps still leave every file a later step
    needs, so any trainer can take any step, unless it is told to leave its full state out of some (`rollout_lora`'s
    `state_every`): a step from one of those needs the trainer that holds it, and fails (`StepFailed`) without it."""

    @property
    def holding(self) -> str | None:
        """The name it gave what it holds (written to that step's state as `HELD`); none while it holds nothing."""
        ...

    def close(self) -> None:
        """End what it keeps running (its processes, and what they hold)."""
        ...


@runtime_checkable
class Remote(Protocol):
    """A trainer on another machine, whose steps' files go through the blob store (`rollout_train.pods.RemoteTrainer`):
    given its parent as the checkpoint, it says what it made as manifests (`Made`), so nothing is read to the loop's
    machine. The rest of a step's state may be kept after `made` returns (`Made.complete` false): `state` waits for
    it."""

    async def made(self, batch: Sequence[Item], *, seed: int, parent: Checkpoint | None, into: str) -> Made:
        """`Trainer.step`, from `parent`'s files in the blob store, making the checkpoint `into` (its id)."""
        ...

    async def state(self, into: str) -> Manifest:
        """The whole state of the step that made `into`, once it is kept. Raises `StateLost` if it never will be."""
        ...


@runtime_checkable
class OnDemand(Protocol):
    """A trainer whose machine is leased only once a step is coming (a training pod of the run's own:
    `rollout_train.pods.LeasedTrainer`), not while the run waits for its first groups: told that a step is coming, it
    starts getting its machine ready, and a step waits for it."""

    def wanted(self) -> None:
        """A step is coming: lease the machine now, unless it is leased already."""
        ...


@runtime_checkable
class Keeps(Protocol):
    """A trainer that can keep the full state of its steps in a blob store itself, after each step returns, so a step's
    weights are kept and served while its state is still being kept (`rollout_lora`'s, with its processes kept between
    steps). Its steps say which they keep so (`Step.keeping`)."""

    def keep_in(self, blobs: Mapping[str, JsonValue]) -> None:
        """Keep its steps' full state from now on in the blob store at this location (`rollout_train.stores.opened`),
        rather than in `into/state` before a step returns."""
        ...

    async def kept(self, into: str) -> Manifest:
        """The files it kept of the full state of the step that wrote into `into` (by its name), by their paths within
        the state, once they are kept. Raises `StateLost` if they never will be."""
        ...


START, MINIBATCH = "start", "minibatch"
"""A step's phases, as `Progress` says them: computing the logprobs it starts from, then its minibatches' updates."""


@dataclass(frozen=True)
class Progress:
    """How far a step being taken has got, as its trainer says it after each pack it runs and each minibatch it steps
    on (`rollout_objectives.step.StepProgress`). Shared among processes, it counts every process's packs and tokens,
    and its memory is the most any process's GPU holds."""

    phase: str
    """`start` (the logprobs the step starts from) or `minibatch`."""
    minibatch: int = 0
    """The minibatch being taken, from 1 (0 in the start)."""
    minibatches: int = 0
    """The minibatches of the whole step, every pass's (as planned: until the start is done, as many in each pass as
    the first takes)."""
    packs: int = 0
    """Packs run so far, in every phase (a trainer that runs no packs counts its passes over the model)."""
    packs_total: int = 0
    """Packs the whole step runs, as planned now: each minibatch's are counted again once it has run them."""
    fraction: float = 0.0
    """Of the step's work done: each pack's tokens, three times over where it is run with a gradient."""
    seconds: float = 0.0
    """Since the step began."""
    tokens_per_second: float = 0.0
    """Segments' tokens run through the model a second, so far."""
    eta_seconds: float | None = None
    """Seconds left, at the pace so far."""
    loss: float | None = None
    """The mean loss of the minibatches stepped on so far."""
    kl: float | None = None
    """How far the last minibatch stepped on found the policy from where the step began (what the stop reads)."""
    max_kl: float | None = None
    """Where the step stops (none: it does not)."""
    clip_fraction: float | None = None
    """Of the tokens stepped on so far, those whose ratio was clipped."""
    gpu_gib: float | None = None
    """GPU memory held now, in GiB."""
    peak_gpu_gib: float | None = None
    """The most GPU memory held during the step, in GiB."""
    gpu_utilization: tuple[float, ...] = field(default_factory=tuple[float, ...])
    """How busy each process's GPU is, in percent, as NVML says (none where it cannot be read)."""

    @property
    def percent(self) -> int:
        """The fraction done, in whole percent."""
        return min(100, max(0, math.floor(100 * self.fraction)))

    def line(self) -> str:
        """In one line: `minibatch 23/58 · 41% · 5.9k tok/s · KL 0.012/0.05 · ETA 34 min`."""
        said = [f"minibatch {self.minibatch}/{self.minibatches}" if self.phase == MINIBATCH else self.phase]
        said += [f"{self.percent}%", f"{_thousands(self.tokens_per_second)} tok/s"]
        if self.kl is not None:
            said.append(f"KL {self.kl:.2g}" + (f"/{self.max_kl:g}" if self.max_kl is not None else ""))
        if self.eta_seconds is not None:
            said.append(f"ETA {_span(self.eta_seconds)}")
        return " · ".join(said)

    def to_json(self) -> dict[str, JsonValue]:
        said: dict[str, JsonValue] = {}
        for each in fields(self):
            value = getattr(self, each.name)
            if isinstance(value, tuple):
                value = list[JsonValue](round(float(used), 1) for used in cast(tuple[float, ...], value))
            said[each.name] = round(value, 6) if isinstance(value, float) else value
        return said

    @classmethod
    def from_json(cls, said: Any) -> "Progress | None":
        """What `to_json` wrote (none for anything else)."""
        if not isinstance(said, dict):
            return None
        found = cast(dict[str, Any], said)
        if not isinstance(found.get("phase"), str):
            return None
        names = {each.name for each in fields(cls)}
        given: dict[str, Any] = {key: value for key, value in found.items() if key in names}
        used = given.get("gpu_utilization")
        given["gpu_utilization"] = (
            tuple(float(each) for each in cast(list[Any], used)) if isinstance(used, list) else ()
        )
        try:
            return cls(**given)
        except (TypeError, ValueError):
            return None


def _thousands(value: float) -> str:
    return f"{value / 1000:.1f}k" if value >= 1000 else f"{value:.0f}"


def _span(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


@runtime_checkable
class Progressing(Protocol):
    """A trainer that says how far the step it is taking has got, as it goes (`rollout_lora`'s, Tinker's, a training
    pod's): each `Progress` to what it was told to tell (none: to nothing), from whatever thread it learns it in."""

    def watch(self, told: Callable[[Progress], None] | None) -> None: ...


class StateLost(Exception):
    """A step's full state will never be kept: its checkpoint stays incomplete, and a step from it needs the trainer
    that holds it."""


@runtime_checkable
class Changeable(Protocol):
    """A trainer that takes some of its settings between steps (its learning rate, say): those that change neither
    what its weights are nor what it can take (`Budget`)."""

    @property
    def changeable(self) -> Mapping[str, JsonValue]:
        """The settings it takes between steps, by its name for each, with their values now: a component of its
        objective by its run setting's key (`objective.kl.coefficient`)."""
        ...

    def change(self, settings: Mapping[str, JsonValue]) -> None:
        """Take these settings (some of `changeable`) from its next step on. Raises `ValueError` for one it does not
        take, or a value it cannot."""
        ...
