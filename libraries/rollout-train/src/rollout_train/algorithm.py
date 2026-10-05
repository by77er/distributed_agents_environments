"""Group-relative policy optimisation over episodes: what to compare, how much each episode counts, what to train on.

- **Groups.** Episodes of one row from one start are compared with each other: the loop asks for a group's episodes
  together, and reads them once all have ended (`rollout_train.rollouts.scheduler`).
- **Advantages** (Dr. GRPO): an episode's score minus its group's mean, without dividing by the group's spread
  (which favours groups that are nearly solved or nearly hopeless). Every token the policy sampled in the episode
  gets it: with several model slots, every slot's, so a team is rewarded together.
- **Dynamic sampling** (DAPO): a group whose scores are all equal has nothing to teach and is skipped.
- **The fastest of the saturated.** Episodes that reached everything their task has to give earned the same; the
  one that took the least (`Episode.duration`, in the task's own units) played better, and scores a point more.
  The task says what saturated means and how long it took; comparing across the group is done here.
- **What is trained on**: every segment of the episodes whose advantage is not zero, up to what the trainer can
  afford in a step; beyond that, segments are taken at even steps through the group, so that each episode and
  slot keeps its share, spread over its whole game.
- **What is never trained on**: a segment of a slot that is not trained (a judge's, a fixed opponent's:
  `Segment.trained`).
- **What cannot be trained on**: a segment whose turns were sampled without their exact tokens or their behaviour
  logprobs (`Segment.lacks`) has no importance weight. A group with one is not trained on, and its result says why.
"""

import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import JsonValue

from rollout_train.recorder.segments import Segment
from rollout_train.rollouts import Episode
from rollout_train.trainer import Budget, Weighted


@dataclass(frozen=True)
class Batch:
    """What an algorithm makes of a group of episodes."""

    segments: Sequence[Weighted] = ()
    """What to train on."""
    skipped: str | None = None
    """Why there is nothing to train on, if there is not."""
    notes: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """What the algorithm wants logged with the group."""


class Algorithm(Protocol):
    """What the training loop asks of an algorithm."""

    @property
    def group_size(self) -> int:
        """How many episodes of one start it compares."""
        ...

    def batch(self, group: Sequence[Episode], budget: Budget, rng: random.Random) -> Batch:
        """What to train on from a group's episodes (of every outcome), within what the trainer can afford."""
        ...


def group_advantages(scores: Sequence[float]) -> list[float] | None:
    """Each score minus the group's mean; None when all are equal (no signal)."""
    if len(scores) < 2 or max(scores) == min(scores):
        return None
    mean = statistics.fmean(scores)
    return [score - mean for score in scores]


def fastest_of_the_saturated(group: Sequence[Episode]) -> list[float]:
    """A point for the fastest of the episodes that saturated their task, when more than one did; episodes that tie
    for fastest all get it. An episode that does not say how long it took is not compared."""
    took = {
        index: episode.duration
        for index, episode in enumerate(group)
        if episode.saturated and episode.duration is not None
    }
    if len(took) < 2:
        return [0.0] * len(group)
    fastest = min(took.values())
    return [1.0 if took.get(index) == fastest else 0.0 for index in range(len(group))]


_LACKING = {"token_exact": "their exact tokens", "sampled_logprobs": "behaviour logprobs"}


def unweighable(segments: Sequence[Segment]) -> str | None:
    """Why some of `segments` cannot be trained on with an importance weight, if they cannot: what their turns were
    sampled without, by channel."""
    lacking: dict[str, set[str]] = {}
    for segment in segments:
        lacking.setdefault(segment.channel, set()).update(segment.lacks)
    said = [
        f"turns of channel `{channel or 'unnamed'}` were sampled without "
        + " or ".join(_LACKING[each] for each in _LACKING if each in missing)
        for channel, missing in sorted(lacking.items())
        if missing
    ]
    return "; ".join(said) or None


def spread[Item](items: Sequence[Item], limit: int | None, rng: random.Random) -> list[Item]:
    """At most `limit` of the items, taken at even steps through them from a random start."""
    if limit is None or len(items) <= limit:
        return list(items)
    step = len(items) / limit
    start = rng.random() * step
    return [items[int(start + index * step)] for index in range(limit)]


@dataclass(frozen=True)
class Grpo:
    group_size: int = 4
    tie_break: bool = True
    """Whether the fastest of a group's saturated episodes scores a point more."""

    def batch(self, group: Sequence[Episode], budget: Budget, rng: random.Random) -> Batch:
        """The segments of the group's episodes that are fit to train on (completed, and not excluded), each with
        its episode's advantage; none, if one of them cannot be weighed (`unweighable`)."""
        good = [episode for episode in group if episode.trainable]
        if len(good) < 2:
            return Batch(skipped=f"{len(good)} of {len(group)} episodes completed")
        bonus = fastest_of_the_saturated(good) if self.tie_break else [0.0] * len(good)
        notes: dict[str, JsonValue] = {"speed_bonus": list(bonus)} if any(bonus) else {}
        advantages = group_advantages([episode.reward + extra for episode, extra in zip(good, bonus, strict=True)])
        if advantages is None:
            return Batch(skipped="every episode scored the same", notes=notes)
        weighted = [
            Weighted(segment, advantage, f"{episode.run}/{episode.group}/{episode.number}/{slot}/{index}")
            for episode, advantage in zip(good, advantages, strict=True)
            if advantage != 0.0
            for slot, trajectory in episode.trajectories.items()
            for index, segment in enumerate(trajectory.segments)
            if segment.trained  # (a judge's turns, or a fixed opponent's, are never trained on)
        ]
        if (why := unweighable([each.segment for each in weighted])) is not None:
            return Batch(skipped=why, notes=notes)
        return Batch(spread(weighted, budget.segments, rng), notes=notes)
