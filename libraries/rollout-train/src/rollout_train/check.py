"""`rollout env check ENVIRONMENT`: whether an environment holds together, before anything trains on it.

Without a model, it checks that the environment's rows build (keys and titles unique, `counts_for` naming rows it
has) and its description and version say something; that one seed draws one start, and its eval data is the same
each time it is asked for; that training draws no eval start (and how often a draw was one, and drawn again); and that
one episode plays to its end on the local runner with a scripted model, with a reward in the range the description
gives and a result that says what it says it does.

With a profile (`--profile P --groups N`), it plays N groups on the profile's channel, served by its base model with
nothing trained, as a run of its own, and says how each went. A group whose episodes all scored the same teaches a
group-relative update nothing; a run all of whose groups are so takes no step at all. That is said plainly.
"""

import asyncio
import json
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import JsonValue

from rollout.curriculum import curriculum_of
from rollout.environment import Description, Environment, Row, held_out, start_key, train_start
from rollout.harness.blobs import Blobs
from rollout.harness.imports import ToolBinding, ToolSet
from rollout.harness.runner import (
    DirectModel,
    ModelBinding,
    RunBinding,
    RunSpecification,
    RunStatus,
    instantiate,
    with_row,
)
from rollout.harness.sandboxes import Pool, PoolBinding
from rollout.local import LocalRunner
from rollout.testing import ScriptedModelEndpoint
from rollout_train.ledger import Ledger
from rollout_train.record import GROUPS, RESULTS, STARTS, Result, described, scope, start_header, table
from rollout_train.registry import Taken, valid
from rollout_train.rollouts.episodes import Episode, assemble
from rollout_train.rollouts.scheduler import Plan, episodes_of, plan

__all__ = ["CHECK", "Finding", "checked", "played", "scripted"]

CHECK = "check"
"""The `kind` a check's run says it is in its start."""
NOTHING_TAUGHT = "every episode scored the same: a group-relative update learns nothing from it"
REPLIES = 1000
"""Replies the scripted model has: an episode that asks for more fails."""
SEEDS = (0, 1, 2)
"""The seeds each row's start is drawn with twice, to see that one seed draws one start."""
DRAWS = 50
"""Training starts drawn of each row, to see that none is an eval start."""


@dataclass(frozen=True)
class Finding:
    check: str
    passed: bool
    said: str
    flagged: bool = False
    """Passed, with something to look at: a group that teaches nothing, say."""

    def __str__(self) -> str:
        return f"{'FAIL' if not self.passed else 'FLAG' if self.flagged else 'ok':<5} {self.check}: {self.said}"


def checked(environment: Environment) -> list[Finding]:
    """What can be checked without playing: its rows, its description, its starts (`SEEDS`), and its eval data
    against training (`DRAWS` training starts of each row)."""
    found = [_rows(environment), _description(environment)]
    if not found[0].passed:
        return found
    return [*found, _starts(environment, SEEDS), _held_out(environment, DRAWS)]


def _rows(environment: Environment) -> Finding:
    try:
        rows = list(environment.rows())
    except Exception as error:  # (whatever the environment raises is what the check reports)
        return Finding("rows", False, f"rows() raised {type(error).__name__}: {error}")
    keys, titles = [row.key for row in rows], [row.title for row in rows]
    problems = [] if rows else ["it has no rows"]
    problems += [f"two rows are {key!r}" for key in sorted({key for key in keys if keys.count(key) > 1})]
    problems += [
        f"two rows are called {title!r}" for title in sorted({each for each in titles if titles.count(each) > 1})
    ]
    problems += [
        f"{row.key} counts for {key}, which it does not have"
        for row in rows
        for key in row.counts_for
        if key not in keys
    ]
    problems += [f"{row.key}'s parameters are not JSON" for row in rows if not _json(row.parameters)]
    if problems:
        return Finding("rows", False, "; ".join(problems))
    return Finding("rows", True, f"{len(rows)} rows, {rows[0].key} to {rows[-1].key}: keys and titles unique")


def _description(environment: Environment) -> Finding:
    version, description = getattr(environment, "version", None), getattr(environment, "description", None)
    problems = [] if isinstance(version, str) and version else ["it says no version"]
    if not isinstance(description, Description):
        return Finding("description", False, "; ".join([*problems, "it has no description (a Description)"]))
    low, high = description.rewards
    if low is not None and high is not None and low >= high:
        problems.append(f"its rewards fall in [{low}, {high}], which is no range")
    if problems:
        return Finding("description", False, "; ".join(problems))
    says = " and ".join(name for name in ("solved", "saturated") if getattr(description, name)) or "neither"
    took = f"duration in {description.duration}" if description.duration else "no duration"
    shown = f"; observations shown as {description.observations}" if description.observations else ""
    said = f"version {version}; rewards in {_range(description)}; results say {says}; {took}{shown}"
    return Finding("description", True, said)


def _starts(environment: Environment, seeds: Sequence[int]) -> Finding:
    problems: list[str] = []
    for row in environment.rows():
        for seed in seeds:
            try:
                first, again = environment.start(row, random.Random(seed)), environment.start(row, random.Random(seed))
            except Exception as error:
                problems.append(f"{row.key} with seed {seed}: start() raised {type(error).__name__}: {error}")
                continue
            if not _json(first):
                problems.append(f"{row.key} with seed {seed}: the start is not JSON")
            elif start_key(first) != start_key(again):
                problems.append(f"{row.key} with seed {seed} drew two different starts")
    try:
        once, twice = _eval_keys(environment), _eval_keys(environment)
    except Exception as error:
        return Finding("starts", False, "; ".join([*problems, f"evals() raised {type(error).__name__}: {error}"]))
    if once != twice:
        problems.append("its eval data differs each time it is asked for")
    if problems:
        return Finding(
            "starts", False, "; ".join(problems[:5]) + (f" (and {len(problems) - 5} more)" if len(problems) > 5 else "")
        )
    return Finding(
        "starts",
        True,
        f"one seed draws one start, for each row and seeds {', '.join(map(str, seeds))}; its eval data is fixed",
    )


def _held_out(environment: Environment, draws: int) -> Finding:
    evals = environment.evals()
    if not evals:
        return Finding(
            "train and eval", False, "it has no eval data: say what it is measured on (`drawn` derives some)"
        )
    problems: list[str] = []
    for name, starts in evals.items():
        try:
            valid(name)
        except Taken as error:
            problems.append(f"eval data {error}")
        if not starts:
            problems.append(f"the eval data {name!r} has no starts")
    held, rows = held_out(environment), list(environment.rows())
    trained = {row.key for row in rows}
    elsewhere = sum(start.task not in trained for starts in evals.values() for start in starts)
    hits = 0
    for row in rows:
        for draw in range(draws):
            hits += start_key(environment.start(row, random.Random(f"check-{row.key}-{draw}"))) in held
            try:
                drawn = train_start(environment, row, random.Random(f"check-{row.key}-{draw}"), held)
            except ValueError as error:
                problems.append(str(error))
                break
            if start_key(drawn) in held:
                problems.append(f"training drew an eval start of {row.key}")
                break
    if problems:
        return Finding("train and eval", False, "; ".join(problems))
    count = sum(len(starts) for starts in evals.values())
    names = ", ".join(f"{name} ({len(starts)})" for name, starts in evals.items())
    apart = f", {elsewhere} of rows it does not train on" if elsewhere else ""
    were = "was an eval start" if hits == 1 else "were eval starts"
    drew = f"{hits} of {draws * len(rows)} starts drawn for training {were} (drawn again)"
    return Finding("train and eval", True, f"{count} eval starts in {names}{apart}; {drew}")


async def scripted(
    environment: Environment,
    *,
    row: str | None = None,
    reply: str = "hello",
    tool_sets: Mapping[str, ToolSet] | None = None,
    tools: Mapping[str, ToolBinding] | None = None,
    pools: Mapping[str, Pool] | None = None,
    pool_urls: Mapping[str, str] | None = None,
    within: float = 600.0,
) -> Finding:
    """One episode of `row` (by key; else the first row) on the local runner, every model slot answered by a scripted
    model that says `reply` each turn; each import served by `tool_sets` (in this process, by name) or `tools` (a URL),
    and each kind of sandbox its program declares by `pools` (in this process, by kind) or `pool_urls`.
    Fails when it does not end within `within` seconds, ends in failure, or its result is not what the description
    says."""
    rows = list(environment.rows())
    chosen = next((each for each in rows if each.key == row), None) if row else rows[0]
    if chosen is None:
        return Finding("episode", False, f"it has no row {row}")
    start = train_start(environment, chosen, random.Random("check-episode"), held_out(environment))
    reference = with_row(environment.program, start)
    program = instantiate(reference)
    tool_sets, tools = dict(tool_sets or {}), dict(tools or {})
    if missing := [name for name in program.imports() if name not in tool_sets and name not in tools]:
        return Finding(
            "episode", False, f"its program imports {', '.join(missing)}: name a tool set for each (--tools)"
        )
    pools, pool_urls = dict(pools or {}), dict(pool_urls or {})
    kinds = sorted({spec.kind for spec in program.sandboxes().values()})
    if missing := [kind for kind in kinds if kind not in pools and kind not in pool_urls]:
        return Finding(
            "episode", False, f"its program declares sandboxes of {', '.join(missing)}: name a pool for each (--pools)"
        )
    endpoint = ScriptedModelEndpoint([reply] * REPLIES)
    scripted_model = ModelBinding(direct=DirectModel(provider="scripted", model="script"))
    binding = RunBinding(
        models=dict.fromkeys(program.model_slots(), scripted_model),
        imports={name: tools.get(name) or ToolBinding(local=name) for name in program.imports()},
        pools={
            kind: PoolBinding(url=pool_urls[kind]) if kind in pool_urls else PoolBinding(local=kind) for kind in kinds
        },
    )
    runner = LocalRunner(providers={"scripted": lambda model: endpoint}, tool_sets=tool_sets, pools=pools)
    handle = await runner.start(RunSpecification(program=reference, binding=binding), labels={"check": chosen.key})
    try:
        outcome = await asyncio.wait_for(handle.result(), within)
    except TimeoutError:
        await runner.cancel(handle.run_id, reason="the check's time ran out")
        return Finding("episode", False, f"{chosen.key} had not ended after {within:.0f} seconds")
    if outcome.status is not RunStatus.COMPLETED:
        return Finding("episode", False, f"{chosen.key} {outcome.status.value}: {outcome.detail}")
    episode = assemble(handle.recorded_events(), {}, run=CHECK, group=1, number=1)
    problems = _said(environment.description, episode)
    turns, result = len(endpoint.requests), json.dumps(dict(episode.info))
    shown = f"{chosen.key}, {turns} model turn{'s' * (turns != 1)}: reward {episode.reward:g}, result {result}"
    return Finding("episode", not problems, "; ".join([shown, *problems]))


def _said(description: Description, episode: Episode) -> list[str]:
    """What an episode's reward and result say that its environment's description says they do not."""
    low, high = description.rewards
    problems: list[str] = []
    if (low is not None and episode.reward < low) or (high is not None and episode.reward > high):
        problems.append(f"its reward {episode.reward:g} is outside {_range(description)}")
    for name in ("solved", "saturated"):
        says = getattr(description, name)
        if says and not isinstance(episode.info.get(name), bool):
            problems.append(f"its result says no {name}, which its description says it reports")
        if not says and name in episode.info:
            problems.append(f"its result says {name}, which its description says it does not report")
    if description.duration and episode.duration is None:
        problems.append(f"its result says no duration, which its description counts in {description.duration}")
    if not description.duration and "duration" in episode.info:
        problems.append("its result says a duration, which its description says it does not report")
    return problems


async def played(
    environment: Environment,
    ledger: Ledger,
    blobs: Blobs,
    *,
    run: str,
    binding: RunBinding,
    groups: int,
    episodes: int,
    seed: int = 0,
    started: Mapping[str, JsonValue] | None = None,
) -> list[Finding]:
    """`groups` groups of `episodes` episodes each, asked for in the ledger as the run `run` (runners play them), of
    rows the environment's curriculum would choose first; a finding for each group, and one for them all."""
    fence = await ledger.take(scope(run))
    await plan(ledger, run, Plan(environment.program, binding), fence)
    here = start_header(kind=CHECK)
    await ledger.append(
        table(run, STARTS), str(fence.number), {**here, **described(environment), **(started or {})}, fence
    )
    curriculum, eval_starts = curriculum_of(environment), held_out(environment)
    first = max(map(int, await ledger.read(table(run, GROUPS))), default=0) + 1
    chosen: dict[int, Row] = {}
    for number in range(first, first + groups):
        rng = random.Random(f"{seed}-{number}")
        row = curriculum.sample([each.key for each in chosen.values()], rng)
        group: JsonValue = {
            "task": row.key,
            "title": row.title,
            "parameters": train_start(environment, row, rng, eval_starts),
            "episodes": episodes,
            "decided": round(time.time(), 1),
        }
        await ledger.append(table(run, GROUPS), str(number), group, fence)  # runners play it from here
        chosen[number] = row
    every = await asyncio.gather(*(episodes_of(ledger, blobs, run, number, episodes) for number in chosen))
    found: list[Finding] = []
    same = 0
    for (number, row), group_episodes in zip(chosen.items(), every, strict=True):
        good = [episode for episode in group_episodes if episode.trainable]
        rewards = [episode.reward for episode in good]
        taught = len(rewards) > 1 and max(rewards) > min(rewards)
        same += not taught
        line = Result.of(
            group_episodes, group=number, task=row.key, title=row.title, skipped=None if taught else NOTHING_TAUGHT
        )
        await ledger.append(table(run, RESULTS), str(number), line.to_json(), fence)
        problems = [each for episode in good for each in _said(environment.description, episode)]
        failed = f"; {line.failed} failed ({', '.join(sorted(set(line.failures)))})" if line.failed else ""
        said = f"{row.key}: rewards {' '.join(f'{each:g}' for each in rewards) or 'none'}{failed}"
        said += "" if taught else f": {NOTHING_TAUGHT}"
        found.append(Finding(f"group {number}", not problems, "; ".join([said, *sorted(set(problems))]), not taught))
    if same == len(chosen):
        verdict = f"in every group of {len(chosen)}, {NOTHING_TAUGHT.split(': ')[0]}: training would take no step"
    else:
        verdict = f"{len(chosen) - same} of {len(chosen)} groups have something to teach"
        verdict += f"; in {same}, {NOTHING_TAUGHT.split(': ')[0]}" if same else ""
    return [*found, Finding("groups", same < len(chosen), verdict)]


def _range(description: Description) -> str:
    low, high = description.rewards
    return f"[{'-inf' if low is None else f'{low:g}'}, {'inf' if high is None else f'{high:g}'}]"


def _json(value: object) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _eval_keys(environment: Environment) -> dict[str, list[tuple[str, str]]]:
    return {
        name: [(start.task, start_key(start.parameters)) for start in starts]
        for name, starts in environment.evals().items()
    }
