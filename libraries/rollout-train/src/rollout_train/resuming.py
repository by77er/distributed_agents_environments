"""Pausing a run and resuming it, from anywhere that reaches its ledger: the monitor's page, `rollout pause` and
`rollout resume`.

A run is paused in place: its desired settings say so (`rollout_train.settings.PAUSED`), and its loop and the runners
that play it take that within seconds, while its driver stays, beating and holding its engines. Resumed while its
driver beats, it goes on at once. An eval a training run's schedule asked for is played by that run's driver: it
pauses with the run, and can be paused and resumed on its own in place.

A run whose driver is gone (stopped, failed, lost) is resumed by a launch of the same run (`resumes`), submitted with
the settings its newest start records (`rollout_train.submitting.submit`): its job's driver goes on from the ledger. A
training run is asked for the groups it had left: those its newest start was to play, less those played since. A run
is not launched again while its driver is there (it beats, and its newest start has not said how it ended), once it
finished, or while a launch of it is going.

A run started before runs recorded their providers (its start says no channel's provider) resumes only with a preset
(`preset`) whose settings match what its start recorded, key by key: every setting both say, its trainer's
implementation (`trainer.kind`) and each channel's engine (`channels.NAME.engine`). A difference is refused with the
settings that differ. The launch then carries the preset's settings with the run's own beside them (its environment,
its seed, its changeable settings), and the run's new start records them in full.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic import JsonValue

from rollout_train.evals import started_version, subject_table
from rollout_train.launches import EVAL, IMITATE, OPEN, TRAIN, Launch, launches_of
from rollout_train.ledger import Ledger
from rollout_train.presence import Beat, alive, presence_of
from rollout_train.presets import presets_of
from rollout_train.providers import INFERENCE_KINDS
from rollout_train.record import ENDS, FINISHED, GROUPS, RESULTS, STARTS, newest_record, table
from rollout_train.registry import registry_of
from rollout_train.run_settings import RunSettings, is_trainers, key_of
from rollout_train.settings import PAUSED, Desired, desired_settings_of

if TYPE_CHECKING:
    from rollout_train.cluster import Cluster
    from rollout_train.submitting import Backend

IN_PLACE, LAUNCHED = "in place", "launched"
"""How a run was resumed: its driver, beating, goes on; or its job was submitted again."""
PROFILE_ONLY = ("model", "weights", "trainer.kind", "trainer.colocated")
"""Settings a start recorded before runs recorded their providers that are no run settings."""


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


async def resume(
    ledger: Ledger,
    run: str,
    *,
    cluster: "Cluster | None" = None,
    preset: str | None = None,
    backend: "Backend | None" = None,
) -> Resumed:
    """Resume a run (by id): in place, if its driver beats; else by a launch of its recorded settings, submitted on
    `cluster` (a run whose start records no providers needs a matching `preset`). Raises `KeyError` where there is no
    such run, nowhere to keep what is wanted of it, or no cluster config to submit on; `ValueError` for a run that
    cannot be resumed: running and not paused, finished, being launched already, a part of an eval (its eval is
    resumed), an eval its training run plays, one whose settings do not say its providers and no preset that matches
    them; `rollout_train.launching.Refused` for settings the cluster refuses."""
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
    return Resumed(run, LAUNCHED, await relaunch(ledger, run, newest, cluster=cluster, preset=preset, backend=backend))


async def going(ledger: Ledger, beats: Sequence[Beat], run: str) -> bool:
    """Whether a run's driver is there: a runner that says it is the run's beat within `STALE` seconds, and its newest
    start has not said how it ended (a run stopped a moment ago is not going, though it beat)."""
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


def recorded_of(newest: Mapping[str, Any]) -> dict[str, JsonValue]:
    """The run settings a start records (`run_settings`, fixed and changeable as one mapping), less its name."""
    said = newest.get("run_settings")
    if not isinstance(said, dict):
        return {}
    said = cast(dict[str, Any], said)
    fixed = cast(dict[str, JsonValue], said.get("fixed") or {})
    values = {**fixed, **cast(dict[str, JsonValue], said.get("changeable") or {})}
    values.pop("name", None)
    return values


def provided(values: Mapping[str, JsonValue]) -> bool:
    """Whether recorded settings say a provider for some channel: those of a run that recorded its providers."""
    return any(
        key.startswith("channels.") and key.endswith((".provider", ".providers")) and value
        for key, value in values.items()
    )


def kind_of(newest: Mapping[str, Any], values: Mapping[str, JsonValue]) -> str:
    """The kind of run a start was: as its settings say, else its own `kind` (`run` and `imitation` by their names)."""
    said = values.get("kind")
    if isinstance(said, str):
        return said
    kind = str(newest.get("kind") or TRAIN)
    return {"run": TRAIN, "imitation": IMITATE}.get(kind, kind)


def differences(
    newest: Mapping[str, Any], preset: Mapping[str, JsonValue], cluster: "Cluster"
) -> list[tuple[str, JsonValue, JsonValue]]:
    """Where a preset's settings differ from what a start recorded before runs recorded their providers: each setting
    both say (`key`, the start's value, the preset's), the trainer's implementation against the start's `trainer.kind`,
    and each channel's engine against `channels.NAME.engine`."""
    profiled = cast(dict[str, Any], newest.get("settings") or {})
    started: dict[str, JsonValue] = {**recorded_of(newest), **cast(dict[str, JsonValue], profiled.get("fixed") or {})}
    found: list[tuple[str, JsonValue, JsonValue]] = []
    for key in sorted(set(started) & set(preset)):
        if started[key] != preset[key]:
            found.append((key, started[key], preset[key]))
    trainer = cluster.trainers.get(str(preset.get("trainer.provider")))
    if "trainer.kind" in started and trainer is not None and trainer.implementation != started["trainer.kind"]:
        found.append(("trainer.kind", started["trainer.kind"], trainer.implementation))
    for key, value in started.items():
        if not (key.startswith("channels.") and key.endswith(".engine")):
            continue
        provider = cluster.inference.get(str(preset.get(key.removesuffix(".engine") + ".provider")))
        if provider is None:
            continue
        engine = str(provider.settings.get("engine") or INFERENCE_KINDS[provider.kind].implementation or "")
        if engine != value:
            found.append((key, value, f"{provider.name} ({engine})"))
    return found


async def relaunch(
    ledger: Ledger,
    run: str,
    newest: Mapping[str, Any],
    *,
    cluster: "Cluster | None" = None,
    preset: str | None = None,
    backend: "Backend | None" = None,
) -> Launch:
    """Submit a run again, with the settings its newest start records (over a matching preset's, for a start that
    records no providers), and the groups it has left."""
    from rollout_train.launching import Refused
    from rollout_train.launching import checked as findings_of
    from rollout_train.submitting import submit

    launches = launches_of(ledger)
    if launches is None:
        raise KeyError("this ledger keeps no launches")
    if cluster is None:
        raise KeyError(f"{run} is launched again on the cluster config: give it one (--cluster)")
    every = await launches.all()  # (newest first)
    if going := [each for each in every if each.run == run and each.state in OPEN]:
        raise ValueError(f"{run} is being launched already ({going[0].id})")
    values = recorded_of(newest)
    kind = kind_of(newest, values)
    chosen = None
    if not provided(values):
        if preset is None:
            raise ValueError(
                f"{run}'s start says no channel's provider: resume it with a preset whose settings match its own "
                "(--preset NAME, or the monitor's resume with a preset)"
            )
        kept = presets_of(ledger)
        chosen = await kept.get(preset) if kept is not None else None
        if chosen is None:
            raise KeyError(f"there is no preset {preset!r}")
        differ = differences(newest, chosen.settings, cluster)
        if differ:
            said = "; ".join(f"{key}: the run has {before!r}, the preset {after!r}" for key, before, after in differ)
            raise ValueError(f"the preset {chosen.id} does not match {run}'s settings: {said}")
        values = _over_preset(newest, chosen.settings, values)
    elif preset is not None:
        raise ValueError(f"{run}'s start records its settings: it resumes with them, not with a preset")
    values["kind"] = kind
    if kind == EVAL:
        values |= await _evaluated(ledger, run, newest, values)
    elif kind == TRAIN:
        values["groups"] = await _left(ledger, run, newest, values)
    registry = registry_of(ledger)
    entry = next((each for each in await registry.runs() if each.id == run), None) if registry else None
    settings = RunSettings({**values, "name": entry.name if entry else run})
    findings = await findings_of(settings, cluster, ledger, own=run)
    if any(each.refuses for each in findings):
        raise Refused(findings)
    started = cast(dict[str, Any], newest.get("run_settings") or {})
    asked = chosen.id if chosen is not None else started.get("preset")
    return await submit(settings, cluster, ledger, preset=str(asked) if asked else None, resumes=run, backend=backend)


def _over_preset(
    newest: Mapping[str, Any], preset: Mapping[str, JsonValue], values: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    """A preset's settings with the run's own beside them: its environment, its seed and groups a step, and its
    changeable settings as its start recorded them (those that are run settings)."""
    profiled = cast(dict[str, Any], newest.get("settings") or {})
    own: dict[str, JsonValue] = {
        **cast(dict[str, JsonValue], profiled.get("fixed") or {}),
        **cast(dict[str, JsonValue], profiled.get("changeable") or {}),
        **values,
    }
    carried = {
        key: value for key, value in own.items()
        if key not in PROFILE_ONLY and not key.endswith((".engine", ".engines", ".reshard"))
        and (key_of(key) is not None or is_trainers(key)) and key not in preset
    }  # fmt: skip
    environment = newest.get("environment")
    return {**preset, **carried, **({"environment": str(environment)} if environment else {})}


async def _left(ledger: Ledger, run: str, newest: Mapping[str, Any], values: Mapping[str, JsonValue]) -> int:
    """The groups a training run has left: those its newest start was to play, less those played since."""
    results: Any = await ledger.read(table(run, RESULTS))
    began = float(newest.get("started") or 0.0)
    times = [
        float(cast(dict[str, Any], each).get("time") or 0.0) for each in results.values() if isinstance(each, dict)
    ]
    earlier = sum(1 for each in times if each < began)
    profiled = cast(dict[str, Any], newest.get("settings") or {})
    playing = values.get("groups") or cast(dict[str, Any], profiled.get("fixed") or {}).get("groups") or 100
    return max(0, earlier + int(cast(int, playing)) - len(results))


async def _evaluated(
    ledger: Ledger, run: str, newest: Mapping[str, Any], values: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    """What an eval launched again plays: the version of its suite its start and subject say, and its episodes."""
    if isinstance(values.get("eval.suite"), str):
        return {}
    subject: Any = (await ledger.read(subject_table(str(newest.get("suite")), run, "subject"))).get("subject") or {}
    version = started_version(newest, subject)
    if version is None:
        raise ValueError(f"{run}'s start and subject say no version of its suite")
    said: dict[str, JsonValue] = {"eval.suite": version}
    if newest.get("checkpoint"):
        said["start"] = str(newest["checkpoint"])
    if isinstance(subject.get("episodes"), int):
        said["eval.episodes"] = subject["episodes"]
    return said
