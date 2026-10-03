"""Imitation: train a policy to do, without being told how, what it did when it was told.

An environment may guide its agents (tell them the way to a goal, say) and report, in an episode's result, the
guidance its prompts carried, word for word and by kind (`info["guidance"]`). The episodes that succeeded under
guidance show the policy doing the task; taking the guidance back out of their prompts makes them examples of doing
it unguided. `examples` reads such episodes from a job's log and cuts the guidance out of every segment; `imitate`
takes a supervised step on them (the trainer's likelihood objective) and commits the version it makes.

A segment is cut by its tokens: what came before its first sampled token is decoded, the guidance is taken out of
that text, and the text is encoded again; what the policy sampled is kept token for token, and its spans move with
it. A prompt that does not encode back to its own tokens is left out, since cutting it could not be exact.
"""

import asyncio
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from rollout.harness.blobs import Blobs
from rollout_train.ledger import Fence
from rollout_train.policies import Policies, Version
from rollout_train.recorder.recorder import Segment, Span
from rollout_train.recorder.renderers import Renderer
from rollout_train.rollouts.episodes import Episode, Record, loaded
from rollout_train.rollouts.jobs import EPISODES
from rollout_train.trainer import STATE, WEIGHTS, Checkpoint, Trainer, Weighted

GUIDANCE = "guidance"
"""The entry of an episode's result that holds the guidance its prompts carried: by kind, word for word."""


def without(segment: Segment, texts: Sequence[str], renderer: Renderer) -> Segment | None:
    """The segment as if its prompt had never held `texts`: each is cut from the text before the first sampled
    token, with the blank line that set it apart, and that text encoded again; the sampled tokens stay as they were.
    None if a text is not there, or the prompt does not encode back to its own tokens."""
    first = min((span.start for span in segment.spans), default=len(segment.tokens))
    prompt = segment.tokens[:first]
    text = renderer.decode(prompt)
    if renderer.encode(text) != prompt:
        return None
    for cut in texts:
        if cut not in text:
            return None
        text = text.replace(f"\n\n{cut}", "", 1) if f"\n\n{cut}" in text else text.replace(f" {cut}", "", 1)
    shorter = renderer.encode(text)
    moved = len(shorter) - len(prompt)
    spans = [Span(span.start + moved, span.end + moved, span.version, span.effect_id) for span in segment.spans]
    return replace(segment, tokens=[*shorter, *segment.tokens[first:]], spans=spans)


@dataclass
class Examples:
    """Segments to imitate, each weighted 1, with what was left out and why."""

    segments: list[Weighted]
    episodes: int = 0
    """Episodes they came from."""
    left_out: int = 0
    """Segments whose prompt could not be cut exactly."""


async def examples(
    log: Path, blobs: Blobs, renderer: Renderer, *, kinds: Sequence[str], solved_only: bool = True
) -> Examples:
    """The segments of every episode in a job's log (`log`, its directory) whose prompts carried guidance of each of
    `kinds` and (with `solved_only`) that solved its task, with that guidance cut out."""
    lines = await asyncio.to_thread(lambda: (log / EPISODES).read_text().splitlines())
    found = Examples([])
    for line in lines:
        record = Record.from_json(json.loads(line))
        said: Any = record.episode.info.get(GUIDANCE)
        guidance = cast(dict[str, Any], said) if isinstance(said, dict) else {}
        if not all(kind in guidance for kind in kinds):
            continue
        if solved_only and not record.episode.solved:
            continue
        episode: Episode = await loaded(record, blobs)
        cut = [str(guidance[kind]) for kind in kinds]
        found.episodes += 1
        for slot, trajectory in episode.trajectories.items():
            for index, segment in enumerate(trajectory.segments):
                shorter = without(segment, cut, renderer)
                if shorter is None:
                    found.left_out += 1
                    continue
                found.segments.append(Weighted(shorter, 1.0, f"{episode.cursor}/{slot}/{index}"))
    return found


async def imitate(
    policies: Policies,
    trainer: Trainer,
    taught: Examples,
    *,
    fence: Fence,
    policy: str,
    directory: Path,
    limit: int | None = None,
    seed: int = 0,
) -> Version:
    """One supervised step of `trainer` (whose objective is likelihood) on `taught`, from `policy`'s newest version,
    committed as its next. `limit` takes that many segments at random. `directory` holds the versions' files on
    this machine. `fence` is the policy's writer's."""
    chosen = list(taught.segments)
    if limit is not None and len(chosen) > limit:
        chosen = random.Random(seed).sample(chosen, limit)
    head = await policies.head(policy)
    parent: Checkpoint | None = None
    if head is not None:
        here = directory / head.name
        weights = await policies.files(head.weights, here / WEIGHTS)
        parent = Checkpoint(weights, await policies.files(head.state, here / STATE) if head.state else None)
    number = (head.number if head else 0) + 1
    into = directory / f"{policy}@{number}"
    trained: list[JsonValue] = [[weighted.source, weighted.advantage] for weighted in chosen]
    batch = await policies.blobs.put(json.dumps(trained).encode(), "application/json")
    step = await trainer.step(chosen, seed=seed, parent=parent, into=into)
    metrics = {**step.metrics, "imitated_episodes": float(taught.episodes)}
    return await policies.add(
        fence,
        policy,
        number,
        weights=into / WEIGHTS,
        state=into / STATE if await asyncio.to_thread((into / STATE).exists) else None,
        parent=head.name if head else None,
        batch=batch,
        metrics=metrics,
    )
