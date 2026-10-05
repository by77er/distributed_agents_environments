"""Pausing a run and resuming it, from anywhere that reaches its ledger: the monitor's page, `rollout pause` and
`rollout resume`.

A run is paused in place: its desired settings say so (`rollout_train.settings.PAUSED`), and its loop and the runners
that play it take that within seconds, while its process stays, beating and holding its engines. Resumed while its
process beats, it goes on at once. An eval a training run's schedule asked for is played by that run's process: it
pauses with the run, and can be paused and resumed on its own in place.

A run whose process is gone (stopped, failed, lost) is resumed by a launch (`rollout_train.launches`) that names it
(`resumes`) and its directory: a launcher alive that offers its profile starts `rollout train` (or `rollout eval`)
again in the run's own directory, which names the run, and the run goes on from the ledger. The launch asks what the
run's last launch asked (for a run started by hand, what its newest start says: its profile, environment, seed and
groups a step); a training run is asked for the groups it had left: those its newest start was to play, less those
played since. A run is not launched again while its process is there (it beats, and its newest start has not said how
it ended), once it finished, or while a launch of it is going.
"""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from rollout_train.evals import started_version, subject_table, suite_of
from rollout_train.launcher import LAUNCHER, launches_kind, offers_environments
from rollout_train.launches import EVAL, OPEN, RUN, Asked, Launch, launches_of
from rollout_train.ledger import Ledger
from rollout_train.presence import Beat, alive, presence_of
from rollout_train.record import ENDS, FINISHED, GROUPS, RESULTS, STARTS, newest_record, table
from rollout_train.registry import registry_of
from rollout_train.settings import GROUPS_PER_STEP, PAUSED, Desired, desired_settings_of

IN_PLACE, LAUNCHED = "in place", "launched"
"""How a run was resumed: its process, beating, goes on; or a launcher was asked to start it again."""


@dataclass(frozen=True)
class Resumed:
    run: str
    how: str
    """`IN_PLACE` or `LAUNCHED`."""
    launch: Launch | None = None
    """The launch that starts it again, if it was launched."""


async def pause(ledger: Ledger, run: str) -> Desired:
    """Pause a run (by id). Raises `KeyError` where there is no such run, or nowhere to keep its desired settings."""
    store = desired_settings_of(ledger)
    if store is None:
        raise KeyError("this ledger keeps no settings")
    if not await ledger.read(table(run, STARTS)):
        raise KeyError(f"there is no run {run}")
    return await store.want(run, {PAUSED: True})


async def resume(ledger: Ledger, run: str) -> Resumed:
    """Resume a run (by id): in place, if its process beats; else by a launch that starts it again in its directory.
    Raises `KeyError` where there is no such run, nowhere to keep what is wanted of it, or no launcher alive offers its
    profile and environments; `ValueError` for a run that cannot be resumed: running and not paused, finished, being
    launched already, a part of an eval (its eval is resumed), an eval its training run plays, or one whose start does
    not say its directory and profile."""
    store = desired_settings_of(ledger)
    if store is None:
        raise KeyError("this ledger keeps no settings")
    starts: Any = await ledger.read(table(run, STARTS))
    if not starts:
        raise KeyError(f"there is no run {run}")
    newest = newest_record(starts)
    if newest.get("part_of"):
        raise ValueError(f"{run} plays a part of the eval {newest['part_of']}: resume that eval")
    wanted = await store.desired(run)
    was = wanted is not None and wanted.settings.get(PAUSED) is True
    if was:
        await store.want(run, {PAUSED: False})
    presence = presence_of(ledger)
    beats = await presence.beats() if presence is not None else []
    by = str(newest["by"]) if newest.get("by") else None
    if await going(ledger, beats, by or run):
        if was:
            return Resumed(run, IN_PLACE)
        raise ValueError(f"{run} is running, and not paused")
    if by is not None:
        raise ValueError(f"the run that asked for the eval {run} plays it ({by}): resume that run")
    if await ending(ledger, run) == FINISHED or await played_out(ledger, run, newest):
        raise ValueError(f"{run} finished")
    return Resumed(run, LAUNCHED, await relaunch(ledger, run, newest, beats))


async def going(ledger: Ledger, beats: Sequence[Beat], run: str) -> bool:
    """Whether a run's process is there: a runner that says it is the run's beat within `STALE` seconds, and its
    newest start has not said how it ended (a run stopped a moment ago is not going, though it beat)."""
    return any(beat.about.get("run") == run and alive(beat) for beat in beats) and await ending(ledger, run) is None


async def ending(ledger: Ledger, run: str) -> str | None:
    """How a run's newest start said it ended (`rollout_train.record.ENDINGS`), if it said."""
    starts, ends = await ledger.read(table(run, STARTS)), await ledger.read(table(run, ENDS))
    said = ends.get(max(starts, key=int)) if starts else None
    return str(cast(dict[str, Any], said).get("how")) if isinstance(said, dict) else None


async def played_out(ledger: Ledger, run: str, newest: Mapping[str, Any]) -> bool:
    """Whether a run is an eval that played every start (in each of its parts)."""
    if newest.get("kind") != EVAL:
        return False
    parts = [str(each) for each in cast(list[Any], newest.get("parts") or [])] or [run]
    for each in parts:
        groups, results = await ledger.read(table(each, GROUPS)), await ledger.read(table(each, RESULTS))
        if not groups or not set(groups) <= set(results):
            return False
    return True


async def relaunch(ledger: Ledger, run: str, newest: Mapping[str, Any], beats: Sequence[Beat]) -> Launch:
    """Ask a launcher to start a run again in its own directory, with what its last launch asked (or its newest start
    says), and the groups it has left."""
    launches = launches_of(ledger)
    if launches is None:
        raise KeyError("this ledger keeps no launches")
    directory, profile = newest.get("directory"), newest.get("profile")
    if not directory or not profile:
        raise ValueError(f"{run}'s start does not say the directory and profile it was started with")
    every = await launches.all()  # (newest first)
    its = [each for each in every if each.directory == directory or each.asked.resumes == run]
    if going := [each for each in its if each.state in OPEN]:
        raise ValueError(f"{run} is being launched already ({going[0].id})")
    offered = [cast(dict[str, Any], beat.about) for beat in beats if beat.about.get("kind") == LAUNCHER and alive(beat)]
    before = its[0] if its else None
    name = before.asked.profile if before is not None and _offers(offered, before.asked.profile) else None
    name = name or _profile_of(offered, str(profile))
    if name is None:
        raise KeyError(f"no launcher alive offers {run}'s profile ({profile})")
    asked = await _asked(ledger, run, newest, before.asked if before is not None else None)
    asked = replace(asked, profile=name, resumes=run, directory=str(directory))
    able = [
        each for each in offered
        if any(
            found.get("profile") == name and launches_kind(found, asked.kind)
            and offers_environments(each, found, asked.plays())
            for found in _profiles(each)
        )
    ]  # fmt: skip
    if not able:
        raise KeyError(f"no launcher alive offers the profile {name!r} and the environments {', '.join(asked.plays())}")
    return await launches.ask(asked)


async def _asked(ledger: Ledger, run: str, newest: Mapping[str, Any], before: Asked | None) -> Asked:
    """What a run's launch to start it again asks: what its last launch asked, or what its newest start says; for a
    training run, the groups it has left."""
    said: Mapping[str, Any] = newest.get("settings") or {}
    fixed: Mapping[str, Any] = said.get("fixed") or {}
    changeable: Mapping[str, Any] = said.get("changeable") or {}
    environment = str(newest.get("environment") or (before.environment if before else ""))
    if newest.get("kind") == EVAL:
        if before is not None:
            return before
        subject: Any = (await ledger.read(subject_table(str(newest.get("suite")), run, "subject"))).get("subject") or {}
        version = started_version(newest, subject)
        if version is None:
            raise ValueError(f"{run}'s start and subject say no version of its suite")
        suite = await suite_of(ledger, version)
        return Asked(
            "", environment, await _name(ledger, run), start=newest.get("checkpoint"), kind=EVAL, suite=version,
            episodes=subject.get("episodes"), environments=suite.environments[1:] if suite is not None else (),
        )  # fmt: skip
    results: Any = await ledger.read(table(run, RESULTS))
    began = float(newest.get("started") or 0.0)
    times = [
        float(cast(dict[str, Any], each).get("time") or 0.0) for each in results.values() if isinstance(each, dict)
    ]
    earlier = sum(1 for each in times if each < began)
    playing = int(fixed.get("groups") or (before.groups if before else 100))
    left = max(0, earlier + playing - len(results))
    if before is not None:
        return replace(before, groups=left)
    return Asked(
        "", environment, await _name(ledger, run), groups=left,
        groups_per_step=int(changeable.get(GROUPS_PER_STEP) or 4), seed=int(fixed.get("seed") or 0), kind=RUN,
    )  # fmt: skip


async def _name(ledger: Ledger, run: str) -> str:
    """What a run is called, as the registry says (its id, where it says nothing)."""
    registry = registry_of(ledger)
    entry = next((each for each in await registry.runs() if each.id == run), None) if registry else None
    return entry.name if entry is not None else run


def _offers(offered: Sequence[Mapping[str, Any]], profile: str) -> bool:
    return any(each.get("profile") == profile for launcher in offered for each in _profiles(launcher))


def _profiles(launcher: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The profiles a launcher's beat says it offers."""
    return cast(list[dict[str, Any]], launcher.get("profiles") or [])


def _profile_of(offered: Sequence[Mapping[str, Any]], path: str) -> str | None:
    """The name a launcher alive offers a profile under, by its path: the same path, else one whose path (relative
    to its launcher's directory) ends the path, else one of the same file name."""
    wanted = Path(os.path.normpath(path))
    profiles = [each for launcher in offered for each in _profiles(launcher)]
    paths = [(str(each.get("profile")), Path(os.path.normpath(str(each.get("path") or "")))) for each in profiles]
    for name, there in paths:
        if there == wanted:
            return name
    for name, there in paths:
        if not there.is_absolute() and wanted.parts[-len(there.parts) :] == there.parts:
            return name
    return next((name for name, there in paths if there.name == wanted.name), None)
