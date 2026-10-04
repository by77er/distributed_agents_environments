"""Datasets: examples to imitate, chosen from runs' episodes by a rule (rejection sampling), and made once.

A **dataset** is a record and a manifest. The record, in the ledger's table `datasets` under a random id (appended once,
under the fence `datasets/ID`, and never changed), says how its examples were chosen (the runs, the episode rule and
its parameters, the turn filters, the kinds of guidance cut), what came of it (episodes, groups, tasks, turns, sampled
and context tokens, and the turns left out, by why), the checkpoints that sampled its examples, where its manifest is,
and who made it and when. The manifest, a blob, has one JSON line per example: its source
(`RUN/GROUP/EPISODE/SLOT/INDEX`), its task and its episode's reward, the depth it was sampled at and the checkpoint
served at that depth, its tokens, the guidance to cut from it, and what the turn filters saw of it. The examples are
made from the episodes' blobs when a step trains on them (`examples`), so a dataset holds no copy of its trajectories.
A name for one is optional, in the registry (`rollout_train.registry`).

Its examples are chosen in two parts.

- **An episode rule** (`RULES`) picks among the episodes the runs completed: `solved-all`, every solved one;
  `best-of-group`, of each group's solved episodes the one with the highest reward, then the shortest (`duration`),
  then the first; `capped-per-task`, solved episodes by the same order, at most `per_task` of each task.
- **Turn filters** pick among those episodes' turns (their segments). `all` keeps every turn that sampled something;
  an environment supplies others, named `module:name` (`minecraft_team.datasets:worked`, say). A filter is given an
  episode's turns, its run's events and its result, and says for each turn what it saw; a turn it gives a `left_out`
  (why) is no example. A turn is an example if every filter keeps it.

The guidance an episode's prompts carried, of the kinds the dataset cuts, is cut from its examples when they are made,
as `rollout_train.imitation.without` cuts it; a turn where the cut cannot be made exactly is left out then.
"""

import asyncio
import getpass
import json
import lzma
import math
import socket
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import unquote, urlparse

from pydantic import JsonValue, TypeAdapter

from rollout.contracts import BlobReference, RunEvent
from rollout.harness.blobs import Blobs
from rollout.names import named
from rollout_train.checkpoints import SHORTEST, checkpoints_in, new_id
from rollout_train.imitation import GUIDANCE, Examples, without
from rollout_train.ledger import Ledger
from rollout_train.record import GROUPS, STARTS, table
from rollout_train.recorder.renderers import Renderer
from rollout_train.registry import Registry
from rollout_train.rollouts.episodes import COMPRESSED, Episode, Record, events_of, loaded
from rollout_train.rollouts.scheduler import EPISODES
from rollout_train.stores import FILES, opened
from rollout_train.trainer import Weighted

DATASETS = "datasets"
"""The ledger's table of datasets, by id."""
LEFT_OUT = "left_out"
"""What a turn filter says of a turn that is no example: why."""
ALL = "all"
"""The turn filter that keeps every turn that sampled something."""


@dataclass(frozen=True)
class Turn:
    """One turn of an episode, as a turn filter sees it: a segment of a model slot's trajectory."""

    slot: str
    index: int
    """The segment's place in the slot's trajectory, from 0."""
    samples: tuple[str, ...]
    """The samples its sampled tokens came from, by the `effect_id`s the run's events know them by."""
    tokens: int
    sampled: int
    depth: int | None
    """The depth of the checkpoint that sampled it (its spans' newest); None if it sampled nothing."""


class TurnFilter(Protocol):
    def __call__(
        self, turns: Sequence[Turn], events: Sequence[RunEvent], info: Mapping[str, JsonValue]
    ) -> Sequence[Mapping[str, JsonValue]]:
        """What it saw of each of an episode's `turns` (in order), given its run's `events` and its result (`info`):
        a turn whose entry has `left_out` (why) is no example."""
        ...


def every_turn(
    turns: Sequence[Turn], events: Sequence[RunEvent], info: Mapping[str, JsonValue]
) -> list[dict[str, JsonValue]]:
    """Every turn that sampled something."""
    return [{} if turn.sampled else {LEFT_OUT: "nothing sampled"} for turn in turns]


TURN_FILTERS: dict[str, TurnFilter] = {ALL: every_turn}
"""The turn filters any dataset can use, by name; an environment's are named `module:name`."""


def turn_filter(name: str) -> TurnFilter:
    """The turn filter a name says: one of `TURN_FILTERS`, or `module:name`."""
    if name in TURN_FILTERS:
        return TURN_FILTERS[name]
    if ":" not in name:
        raise ValueError(f"there is no turn filter {name!r}: one of {', '.join(TURN_FILTERS)}, or module:name")
    return cast(TurnFilter, named(name))


@dataclass(frozen=True)
class Candidate:
    """An episode a run completed, as an episode rule sees it."""

    run: str
    group: int
    number: int
    task: str
    """The row's key."""
    record: Record

    @property
    def episode(self) -> Episode:
        return self.record.episode


def _best(candidate: Candidate) -> tuple[float, float, int, int]:
    """Highest reward first, then the shortest, then the first."""
    duration = candidate.episode.duration
    return (-candidate.episode.reward, duration if duration is not None else math.inf, candidate.group,
            candidate.number)  # fmt: skip


def solved_all(candidates: Sequence[Candidate], per_task: int | None) -> list[Candidate]:
    """Every solved episode."""
    return [each for each in candidates if each.episode.solved]


def best_of_group(candidates: Sequence[Candidate], per_task: int | None) -> list[Candidate]:
    """Of each group's solved episodes, the one with the highest reward, then the shortest, then the first."""
    groups: dict[tuple[str, int], list[Candidate]] = {}
    for each in solved_all(candidates, per_task):
        groups.setdefault((each.run, each.group), []).append(each)
    return [min(solved, key=_best) for solved in groups.values()]


PER_TASK = 3
"""Solved episodes of each task `capped-per-task` takes, unless it is told otherwise."""


def capped_per_task(candidates: Sequence[Candidate], per_task: int | None) -> list[Candidate]:
    """Solved episodes by highest reward, then the shortest, then the first: at most `per_task` of each task."""
    taken: Counter[str] = Counter()
    chosen: list[Candidate] = []
    for each in sorted(solved_all(candidates, per_task), key=_best):
        if taken[each.task] < (per_task if per_task is not None else PER_TASK):
            taken[each.task] += 1
            chosen.append(each)
    return chosen


RULES: dict[str, Callable[[Sequence[Candidate], int | None], list[Candidate]]] = {
    "solved-all": solved_all,
    "best-of-group": best_of_group,
    "capped-per-task": capped_per_task,
}
"""The episode rules, by name: each picks episodes among those the runs completed."""


@dataclass(frozen=True)
class Dataset:
    """A dataset's record (`DATASETS`): how its examples were chosen, what came of it, and where its manifest is."""

    id: str
    rule: str
    """The episode rule, by name (`RULES`)."""
    runs: list[str]
    """The runs its episodes are from, by id."""
    turns: list[str]
    """The turn filters, by name."""
    cut: list[str]
    """The kinds of guidance cut from its examples' prompts when they are made."""
    manifest: BlobReference
    blobs: Mapping[str, JsonValue]
    """Where the manifest is kept, as any process opens it (`rollout_train.stores`)."""
    per_task: int | None = None
    """For `capped-per-task`: the most episodes of each task."""
    counts: Mapping[str, int] = field(default_factory=dict[str, int])
    """Of its examples: `episodes`, `groups`, `tasks`, `turns`, `sampled_tokens` and `context_tokens`; and
    `turns_seen`, every turn of its episodes."""
    left_out: Mapping[str, int] = field(default_factory=dict[str, int])
    """Turns of its episodes that are no examples, by why."""
    checkpoints: list[str] = field(default_factory=list[str])
    """The checkpoints that sampled its examples, by id, by depth (examples sampled by the base model name none)."""
    made: float = 0.0
    by: str = ""
    """Who made it: `user@host`."""


_DATASET = TypeAdapter(Dataset)


async def make(
    ledger: Ledger,
    rule: str,
    runs: Sequence[str],
    *,
    into: Blobs,
    at: Mapping[str, JsonValue],
    turns: Sequence[str] = (ALL,),
    cut: Sequence[str] = ("way",),
    per_task: int | None = None,
    by: str | None = None,
) -> Dataset:
    """Make a dataset of the episodes `runs` (by id) completed: those `rule` picks, and of them the turns every filter
    of `turns` keeps. Its manifest is kept in `into`, which is at `at` (as `rollout_train.stores.opened` reads it).
    Episodes are read one at a time, from where each run's blobs are. Raises `ValueError` for a rule or filter that
    does not exist, or a dataset of no examples."""
    if rule not in RULES:
        raise ValueError(f"there is no episode rule {rule!r}: one of {', '.join(RULES)}")
    filters = [turn_filter(name) for name in turns]
    candidates: list[Candidate] = []
    for run in runs:
        groups: Any = await ledger.read(table(run, GROUPS))
        for key, line in (await ledger.read(table(run, EPISODES))).items():
            record = Record.from_json(cast(dict[str, Any], line))
            if record.episode.trainable:
                group, _, number = key.partition("/")
                decided: Any = groups.get(group) or {}
                task = str(decided.get("task", ""))
                candidates.append(Candidate(run, int(group), int(number), task, record))
    order = {run: index for index, run in enumerate(runs)}
    chosen = sorted(RULES[rule](candidates, per_task), key=lambda each: (order[each.run], each.group, each.number))
    served = {run: await served_at(ledger, run) for run in runs}
    lines: list[dict[str, JsonValue]] = []
    left_out: Counter[str] = Counter()
    seen = 0
    for candidate in chosen:
        store = opened(await where_blobs_are(ledger, candidate.run, candidate.record.trajectories))
        episode = await loaded(candidate.record, store)
        events = await events_of(candidate.record, store)
        listed = turns_of(episode)
        said = [list(each(listed, events, episode.info)) for each in filters]
        if any(len(verdicts) != len(listed) for verdicts in said):
            raise ValueError("a turn filter said something of each turn but not of every one")
        guidance: Any = episode.info.get(GUIDANCE)
        carried = [kind for kind in cut if isinstance(guidance, dict) and kind in guidance]
        seen += len(listed)
        for position, turn in enumerate(listed):
            noted: dict[str, JsonValue] = {}
            why: str | None = None
            for verdicts in said:
                verdict = verdicts[position]
                if why is None and LEFT_OUT in verdict:
                    why = str(verdict[LEFT_OUT])
                noted |= {key: value for key, value in verdict.items() if key != LEFT_OUT}
            if why is not None:
                left_out[why] += 1
                continue
            source = f"{candidate.run}/{candidate.group}/{candidate.number}/{turn.slot}/{turn.index}"
            lines.append({
                **noted, "source": source, "task": candidate.task, "reward": episode.reward, "depth": turn.depth,
                "checkpoint": served[candidate.run].get(turn.depth) if turn.depth is not None else None,
                "tokens": turn.tokens, "sampled": turn.sampled, "guidance": list[JsonValue](carried),
            })  # fmt: skip
    if not lines:
        raise ValueError(f"no turn of the episodes {rule} picks is an example ({len(chosen)} episodes)")
    packed = await asyncio.to_thread(lzma.compress, "".join(json.dumps(line) + "\n" for line in lines).encode())
    manifest = await into.put(packed, COMPRESSED)
    depths = {str(line["checkpoint"]): int(cast(int, line["depth"])) for line in lines if line["checkpoint"]}
    dataset = Dataset(
        new_id(),
        rule,
        list(runs),
        list(turns),
        list(cut),
        manifest,
        dict(at),
        per_task=per_task if rule == "capped-per-task" else None,
        counts={
            "episodes": len(chosen),
            "groups": len({(each.run, each.group) for each in chosen}),
            "tasks": len({str(line["task"]) for line in lines}),
            "turns": len(lines),
            "sampled_tokens": sum(int(cast(int, line["sampled"])) for line in lines),
            "context_tokens": sum(int(cast(int, line["tokens"])) for line in lines),
            "turns_seen": seen,
        },
        left_out=dict(left_out.most_common()),
        checkpoints=sorted(depths, key=lambda each: (depths[each], each)),
        made=round(time.time(), 1),
        by=by or f"{getpass.getuser()}@{socket.gethostname()}",
    )
    fence = await ledger.take(f"{DATASETS}/{dataset.id}")
    record: Any = _DATASET.dump_python(dataset, mode="json")
    await ledger.append(DATASETS, dataset.id, record, fence)
    return dataset


def turns_of(episode: Episode) -> list[Turn]:
    """Every turn of an episode, slot by slot, in order."""
    return [
        Turn(
            slot,
            index,
            tuple(span.effect_id for span in segment.spans),
            len(segment.tokens),
            segment.sampled,
            max((span.version for span in segment.spans), default=None),
        )
        for slot, trajectory in episode.trajectories.items()
        for index, segment in enumerate(trajectory.segments)
    ]


async def served_at(ledger: Ledger, run: str) -> dict[int, str]:
    """The checkpoint a run served at each depth, by id: the line of first parents back from each checkpoint it made
    and each it started from (the base model, at depth 0, is none)."""
    every = {checkpoint.id: checkpoint for checkpoint in await checkpoints_in(ledger)}
    starts: Any = await ledger.read(table(run, STARTS))
    tips = [each.id for each in every.values() if each.run == run]
    tips += [str(start["from"]) for start in starts.values() if start.get("from")]
    served: dict[int, str] = {}
    for tip in tips:
        at = every.get(tip)
        while at is not None and at.depth not in served:
            served[at.depth] = at.id
            at = every.get(at.parent) if at.parent else None
    return served


async def where_blobs_are(ledger: Ledger, run: str, reference: BlobReference | None = None) -> dict[str, JsonValue]:
    """Where a run's blobs are: as its newest start says; else (a run recorded before starts said so) the directory of
    files that holds `reference`, one of its blobs."""
    starts: Any = await ledger.read(table(run, STARTS))
    kept: Any = starts[max(starts, key=int)].get("blobs") if starts else None
    if isinstance(kept, dict):
        return cast(dict[str, JsonValue], kept)
    if reference is not None and reference.uri.startswith("file://"):
        return {"kind": FILES, "directory": str(Path(unquote(urlparse(reference.uri).path)).parent.parent)}
    raise ValueError(f"the run {run} does not say where its blobs are")


async def dataset_of(ledger: Ledger, id: str) -> Dataset:
    """The dataset an id says."""
    found = (await ledger.read(DATASETS)).get(id)
    if found is None:
        raise KeyError(f"there is no dataset {id}")
    return _DATASET.validate_python(found)


async def datasets_in(ledger: Ledger) -> list[Dataset]:
    """Every dataset, oldest first."""
    return [_DATASET.validate_python(record) for record in (await ledger.read(DATASETS)).values()]


async def resolved_dataset(ledger: Ledger, registry: Registry | None, reference: str) -> str:
    """The dataset a reference says, by id: its name, its id, or the start of its id (at least `SHORTEST` characters)
    that no other id begins with. Raises `KeyError` for one that says none, or more than one."""
    called = {each.name: each.dataset for each in await registry.datasets()} if registry else {}
    if reference in called:
        return called[reference]
    ids = list(await ledger.read(DATASETS))
    if reference in ids:
        return reference
    starting = [each for each in ids if each.startswith(reference)]
    if len(reference) >= SHORTEST and len(starting) == 1:
        return starting[0]
    raise KeyError(f"{reference!r} says {'more than one dataset' if starting else 'no dataset'}")


async def manifest_of(dataset: Dataset) -> list[dict[str, Any]]:
    """A dataset's manifest: a line for each example."""
    packed = await opened(dataset.blobs).read(dataset.manifest)
    text = (await asyncio.to_thread(lzma.decompress, packed)).decode()
    return [json.loads(line) for line in text.splitlines() if line]


async def examples(ledger: Ledger, dataset: Dataset, renderer: Renderer) -> Examples:
    """A dataset's examples, made from its episodes' blobs (each episode read once), each weighted 1, with the
    guidance its manifest names cut from its prompt; a turn whose cut cannot be made exactly is left out."""
    lines = await manifest_of(dataset)
    by_episode: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for line in lines:
        run, group, number, _, _ = str(line["source"]).split("/")
        by_episode.setdefault((run, f"{group}/{number}"), []).append(line)
    found = Examples([], dataset=dataset.id)
    records: dict[str, dict[str, JsonValue]] = {}
    for (run, key), chosen in by_episode.items():
        if run not in records:
            records[run] = await ledger.read(table(run, EPISODES))
        record = Record.from_json(cast(dict[str, Any], records[run][key]))
        episode = await loaded(record, opened(await where_blobs_are(ledger, run, record.trajectories)))
        guidance: Any = episode.info.get(GUIDANCE) or {}
        found.episodes += 1
        for line in chosen:
            source = str(line["source"])
            slot, index = source.split("/")[3:]
            segment = episode.trajectories[slot].segments[int(index)]
            kinds: list[str] = line.get("guidance") or []
            texts = [str(guidance[kind]) for kind in kinds]
            shorter = without(segment, texts, renderer) if texts else segment
            if shorter is None:
                found.left_out += 1
                continue
            found.segments.append(Weighted(shorter, 1.0, source))
            if line.get("checkpoint"):
                found.sampled_by[source] = str(line["checkpoint"])
    return found
