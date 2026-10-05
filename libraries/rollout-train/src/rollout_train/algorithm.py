"""Group-relative optimisation over episodes: what to compare, how much each episode counts, what to train on.

- **Groups.** Episodes of one row from one start are compared with each other: the loop asks for a group's episodes
  together, and reads them once all have ended (`rollout_train.rollouts.scheduler`).
- **Advantages**, by the objective's advantage components (`rollout_train.objectives.Advantage`): an episode's score
  less a baseline, the group's mean (`group_mean`, Dr. GRPO), the mean of the others' (`leave_one_out`, RLOO) or none;
  then divided by the standard deviation of the group's scores (`group_std`, GRPO) or not. Every token the policy
  sampled in the episode gets it: with several model slots, every slot's, so a team is rewarded together.
- **Dynamic sampling** (DAPO, `equal_scores`): a group whose scores are all equal is skipped.
- **The fastest of the saturated.** Episodes that reached everything their task has to give earned the same; the
  one that took the least (`Episode.duration`, in the task's own units) played better, and scores a point more.
  The task says what saturated means and how long it took; comparing across the group is done here.
- **What is trained on**: every segment of the episodes whose advantage is not zero, up to what the trainer can
  afford in a step; beyond that, segments are taken at even steps through the group, so that each episode and
  slot keeps its share, spread over its whole game.
- **What is never trained on**: a segment of a slot that is not trained (a judge's, a fixed opponent's:
  `Segment.trained`).
- **Preferences** (`Preferences`, for the preference family): a group's best episode is preferred to its worst, a
  `Pair` whose shared context is the start (for a multi-turn episode, the start's first observation: each side's
  later turns differ, and only the tokens the policy sampled count); or, for labelled examples (KTO), each episode
  scoring above the group's mean is desirable and each below it undesirable.
- **What cannot be trained on**: a segment whose turns were sampled without their exact tokens or their behaviour
  logprobs (`Segment.lacks`). A policy gradient trains on no group with a turn sampled without its exact tokens, nor,
  with an importance correction, without its behaviour logprobs, and the group's result says why; a preference loss
  and a likelihood read neither (`needs_of`).
"""

import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

from pydantic import JsonValue

from rollout_train.objectives import LIKELIHOOD, PREFERENCE, Advantage, Objective
from rollout_train.recorder.segments import Segment
from rollout_train.rollouts import Episode
from rollout_train.trainer import Budget, Item, Labelled, Pair, Weighted, segments_of

Each = TypeVar("Each", bound=Item, covariant=True, default=Item)
"""The kind of item a batch holds."""


@dataclass(frozen=True)
class Batch(Generic[Each]):
    """What an algorithm makes of a group of episodes: items of one kind (`Each`)."""

    items: Sequence[Each] = ()
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

    def batch(self, group: Sequence[Episode], budget: Budget, rng: random.Random) -> Batch[Item]:
        """What to train on from a group's episodes (of every outcome), within what the trainer can afford."""
        ...


def equal_scores(scores: Sequence[float]) -> bool:
    """Whether a group has nothing to compare: fewer than two scores, or all equal (DAPO's dynamic sampling skips
    it)."""
    return len(scores) < 2 or max(scores) == min(scores)


def group_mean(scores: Sequence[float]) -> list[float]:
    """Each score less the group's mean (Dr. GRPO)."""
    mean = statistics.fmean(scores)
    return [score - mean for score in scores]


def leave_one_out(scores: Sequence[float]) -> list[float]:
    """Each score less the mean of the group's other scores (RLOO)."""
    total, others = sum(scores), len(scores) - 1
    return [score - (total - score) / others for score in scores]


def no_baseline(scores: Sequence[float]) -> list[float]:
    """Each score as it is (REINFORCE without a baseline)."""
    return [float(score) for score in scores]


def group_std(advantages: Sequence[float], scores: Sequence[float]) -> list[float]:
    """Advantages divided by the standard deviation of the group's scores (the sample's, as GRPO's implementations
    take it); all zero where the scores do not vary."""
    spread = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return [advantage / spread if spread > 0 else 0.0 for advantage in advantages]


BASELINES = {"group_mean": group_mean, "leave_one_out": leave_one_out, "none": no_baseline}
"""Each `advantage.baseline`, by name."""


DEFAULT_ADVANTAGE = Advantage()
"""The `default` preset's: each score less the group's mean, groups of equal scores skipped."""


def advantages_of(scores: Sequence[float], advantage: Advantage = DEFAULT_ADVANTAGE) -> list[float] | None:
    """The advantages of a group's scores by the objective's advantage components; None for a group the filter skips,
    or of fewer than two scores (nothing to compare them with)."""
    if len(scores) < 2 or (advantage.filter == "equal_scores" and equal_scores(scores)):
        return None
    found = BASELINES[advantage.baseline](scores)
    return group_std(found, scores) if advantage.scale == "group_std" else found


def group_advantages(scores: Sequence[float]) -> list[float] | None:
    """Each score minus the group's mean; None when all are equal (no signal): the `default` preset's advantages."""
    return advantages_of(scores)


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


def unweighable(segments: Sequence[Segment], needs: Sequence[str] = tuple(_LACKING)) -> str | None:
    """Why some of `segments` cannot be trained on, if they cannot: what their turns were sampled without, by channel,
    of what the objective `needs` (by default exact tokens and behaviour logprobs, as an importance weight does)."""
    lacking: dict[str, set[str]] = {}
    for segment in segments:
        lacking.setdefault(segment.channel, set()).update(each for each in segment.lacks if each in needs)
    said = [
        f"turns of channel `{channel or 'unnamed'}` were sampled without "
        + " or ".join(_LACKING[each] for each in _LACKING if each in missing)
        for channel, missing in sorted(lacking.items())
        if missing
    ]
    return "; ".join(said) or None


def needs_of(objective: Objective) -> tuple[str, ...]:
    """What a segment's turns must have been sampled with for `objective` to train on them: exact tokens for a policy
    gradient, and behaviour logprobs too for one with an importance correction; nothing for the other families."""
    if objective.family in (PREFERENCE, LIKELIHOOD):
        return ()
    return tuple(_LACKING) if objective.needs_behaviour else ("token_exact",)


def spread[Each](items: Sequence[Each], limit: int | None, rng: random.Random) -> list[Each]:
    """At most `limit` of the items, taken at even steps through them from a random start."""
    if limit is None or len(items) <= limit:
        return list(items)
    step = len(items) / limit
    start = rng.random() * step
    return [items[int(start + index * step)] for index in range(limit)]


def within[Kind: Item](items: Sequence[Kind], segments: int | None, rng: random.Random) -> list[Kind]:
    """Items holding at most about `segments` segments in all (a pair holds both sides'), taken at even steps through
    them (`spread`) where they hold more."""
    held = sum(len(segments_of(item)) for item in items)
    if segments is None or held <= segments:
        return list(items)
    return spread(items, max(1, segments * len(items) // held), rng)


def scores_of(good: Sequence[Episode], tie_break: bool) -> tuple[list[float], dict[str, JsonValue]]:
    """A group's scores (each reward, and with `tie_break` a point for the fastest of the saturated), and what to log
    of the bonus."""
    bonus = fastest_of_the_saturated(good) if tie_break else [0.0] * len(good)
    notes: dict[str, JsonValue] = {"speed_bonus": list(bonus)} if any(bonus) else {}
    return [episode.reward + extra for episode, extra in zip(good, bonus, strict=True)], notes


def side_of(episode: Episode) -> tuple[Segment, ...]:
    """Every segment of an episode that is trained, slot by slot, in order."""
    return tuple(
        segment for trajectory in episode.trajectories.values() for segment in trajectory.segments if segment.trained
    )


@dataclass(frozen=True)
class Grpo:
    """Weighted segments, each with its episode's advantage (for the policy-gradient and likelihood families)."""

    group_size: int = 4
    tie_break: bool = True
    """Whether the fastest of a group's saturated episodes scores a point more."""
    advantage: Advantage = DEFAULT_ADVANTAGE
    needs: tuple[str, ...] = tuple(_LACKING)
    """What every segment's turns must have been sampled with (`needs_of`)."""

    def batch(self, group: Sequence[Episode], budget: Budget, rng: random.Random) -> Batch[Weighted]:
        """The segments of the group's episodes that are fit to train on (completed, and not excluded), each with
        its episode's advantage; none, if one of them cannot be weighed (`unweighable`)."""
        good = [episode for episode in group if episode.trainable]
        if len(good) < 2:
            return Batch(skipped=f"{len(good)} of {len(group)} episodes completed")
        scores, notes = scores_of(good, self.tie_break)
        advantages = advantages_of(scores, self.advantage)
        if advantages is None:
            return Batch(skipped="every episode scored the same", notes=notes)
        if not any(advantages):
            return Batch(skipped="every advantage is zero", notes=notes)
        weighted = [
            Weighted(segment, advantage, f"{episode.run}/{episode.group}/{episode.number}/{slot}/{index}")
            for episode, advantage in zip(good, advantages, strict=True)
            if advantage != 0.0
            for slot, trajectory in episode.trajectories.items()
            for index, segment in enumerate(trajectory.segments)
            if segment.trained  # (a judge's turns, or a fixed opponent's, are never trained on)
        ]
        if (why := unweighable([each.segment for each in weighted], self.needs)) is not None:
            return Batch(skipped=why, notes=notes)
        return Batch(spread(weighted, budget.segments, rng), notes=notes)


@dataclass(frozen=True)
class Preferences:
    """Pairs, a group's best episode preferred to its worst; or, `labelled`, examples, each episode above the group's
    mean desirable and each below it undesirable (for the preference family)."""

    group_size: int = 4
    tie_break: bool = True
    labelled: bool = False

    def batch(self, group: Sequence[Episode], budget: Budget, rng: random.Random) -> Batch[Pair | Labelled]:
        """A pair of the group's best and worst completed episodes (the first of each where several tie), or every
        completed episode not at the group's mean, labelled; none where every score is the same."""
        good = [episode for episode in group if episode.trainable]
        if len(good) < 2:
            return Batch(skipped=f"{len(good)} of {len(group)} episodes completed")
        scores, notes = scores_of(good, self.tie_break)
        if equal_scores(scores):
            return Batch(skipped="every episode scored the same", notes=notes)
        items: list[Pair | Labelled]
        if self.labelled:
            mean = statistics.fmean(scores)
            items = [
                Labelled(side_of(episode), score > mean, f"{episode.run}/{episode.group}/{episode.number}")
                for episode, score in zip(good, scores, strict=True)
                if score != mean
            ]
        else:
            best = max(range(len(good)), key=lambda index: (scores[index], -index))
            worst = min(range(len(good)), key=lambda index: (scores[index], index))
            chosen, rejected = good[best], good[worst]
            source = f"{chosen.run}/{chosen.group}/{chosen.number}>{rejected.number}"
            items = [Pair(side_of(chosen), side_of(rejected), source)]
        return Batch(within(items, budget.segments, rng), notes=notes)


def algorithm_for(objective: Objective, group_size: int = 4) -> Grpo | Preferences:
    """The algorithm that makes the batch items an objective's family takes."""
    if objective.family == PREFERENCE:
        return Preferences(group_size, labelled=objective.labelled)
    return Grpo(group_size, advantage=objective.advantage, needs=needs_of(objective))
