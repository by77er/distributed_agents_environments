"""`rollout`: train on an environment under a deployment profile, and watch.

rollout train PROFILE ENVIRONMENT   the training loop: PROFILE is a TOML file (`rollout_train.profile`), ENVIRONMENT
                                    names an environment as `module:name`
rollout eval PROFILE SUITE          play a suite with a checkpoint (or the base model), training nothing
rollout suite make|edit|list        evaluation suites, kept in versions (`rollout_train.evals`)
rollout report DIRECTORY ENVIRONMENT
                                    chart a run's progress and summarise it; post both to a Discord webhook
rollout env check ENVIRONMENT       whether an environment holds together; with --profile, groups played by a model
rollout imitate PROFILE             a supervised step on a dataset (--dataset), or on the run's solved episodes
                                    without their guidance
rollout dataset make RULE           make a dataset: examples chosen from runs' episodes (`rollout_train.datasets`)
rollout checkpoints                 every checkpoint, newest first: where it came from
rollout bookmark NAME CHECKPOINT    name a checkpoint, or move a bookmark there (--delete takes it away)
rollout rename WHO NAME             call a run something else (its id stays)
rollout merge CHECKPOINT            fold a LoRA checkpoint into its base: a full checkpoint of its own
rollout launcher                    start the runs and evals asked for that this machine can run
rollout monitor WHERE               the web page over a ledger and every run in it (WHERE: a run's directory, a ledger)
rollout ledger copy FROM TO         copy a ledger (a run's, files, or a database) into a database: SQLite or Postgres
rollout tools FACTORY               serve an environment's tool set over HTTP: FACTORY is `module:name`
rollout pool FACTORY                serve a pool of an environment's sandboxes over HTTP: FACTORY makes their provider
rollout engines PROFILE --run RUN   keep a profile's engines (vLLM servers) serving what the run says
rollout runner PROFILE              play runs' episodes, and nothing else
rollout pause RUN, rollout resume RUN
                                    pause a run, and resume it (in place, or launched again in its directory)
rollout gateway PROFILE             serve a replica of the gateway: it samples PROFILE's channels and records every turn
rollout cluster check               read the cluster config (`rollout_train.cluster`) and say what does not resolve here
rollout preset list|show|save|delete
                                    presets: named, versioned run settings beside the ledger (`rollout_train.presets`)

A command that starts a run (`train`, `eval`, `imitate`, `env check --profile`) takes its run settings
(`rollout_train.run_settings`) in layers over what its profile gives: `--preset NAME[@N]` (beside the profile's
ledger), `--settings FILE`, `--set KEY=VALUE`, then its own flags. Its start records them (`run_settings`).

A command over a ledger (`rename`, `bookmark`, `pause`, `resume`, `checkpoints`, `suite`, `dataset`, `merge`,
`preset`) takes it as `--ledger WHERE`, or as the cluster config's with `--cluster [PATH or NAME]` (alone:
`ROLLOUT_CLUSTER`, else `~/.config/rollout/cluster.toml`).

`rollout COMMAND --help` lists each command's options.
"""

import argparse
import asyncio
import functools
import json
import os
import signal
import sys
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from rollout.names import named

if TYPE_CHECKING:
    from pydantic import JsonValue

    from rollout_train.cluster import Cluster
    from rollout_train.ledger import Ledger
    from rollout_train.profile import Profile
    from rollout_train.registry import Registry
    from rollout_train.run_settings import RunSettings
    from rollout_train.stores import Stores


async def until_signalled(work: Coroutine[Any, Any, None]) -> int:
    """Run `work`, and cancel it on an interrupt, a termination or a hang-up, so that it stops what it started on
    the way out; returns the exit status. (A process started in the background of a script inherits "ignore" for
    interrupts, and Python then installs no handler of its own.)"""
    task = asyncio.ensure_future(work)
    received: list[int] = []

    def stop(number: int) -> None:
        received.append(number)
        task.cancel()

    loop = asyncio.get_running_loop()
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        loop.add_signal_handler(number, stop, number)
    try:
        await task
    except asyncio.CancelledError:
        if not received:
            raise
        return 128 + received[0]
    return 0


def _user_errors[**P](work: Callable[P, Coroutine[Any, Any, None]]) -> Callable[P, Coroutine[Any, Any, None]]:
    """A command whose `KeyError` or `ValueError` (a `Taken` too) is the user's to mend: it exits saying it."""

    @functools.wraps(work)
    async def said(*arguments: P.args, **options: P.kwargs) -> None:
        try:
            await work(*arguments, **options)
        except (KeyError, ValueError) as error:
            raise SystemExit(error.args[0] if error.args else str(error)) from None

    return said


async def _train(
    profile: Path,
    directory: Path | None,
    environment: str,
    groups: int | None,
    groups_per_step: int | None,
    seed: int | None,
    monitor: str | None = None,
    name: str | None = None,
    sets: list[str] | None = None,
    preset: str | None = None,
    file: Path | None = None,
    chosen: dict[str, str | None] | None = None,
) -> None:
    import dataclasses
    from collections.abc import Mapping

    from pydantic import JsonValue

    from rollout.environment import binding_for
    from rollout_train import train
    from rollout_train.evals import Schedule, environments_of, suite_for
    from rollout_train.profile import Profile
    from rollout_train.record import ending
    from rollout_train.settings import changeable, desired_settings_of, fixed

    flags = {"environment": environment, "name": name, "groups": groups, "groups_per_step": groups_per_step}
    layers = await _layered(
        profile, directory, "train", preset, file, sets or [], flags | {"seed": seed}, **chosen or {}
    )
    groups, groups_per_step, seed = (_whole(layers.settings, each) for each in ("groups", "groups_per_step", "seed"))
    called = cast(str | None, layers.settings["name"])
    described = dataclasses.replace(Profile.load(profile, directory=directory, settings=layers.profile), name=called)
    if described.trainer is None:
        raise SystemExit(f"{profile} describes no trainer")
    channel = described.trainer.channel
    offered, published = await _environment(described, environment)
    slots = _slots(layers, offered, described, channel)
    where, profiled = await asyncio.to_thread(described.directory.absolute), await asyncio.to_thread(profile.absolute)
    started: dict[str, Any] = {"directory": str(where), "profile": str(profiled), "address": monitor}  # (its `starts`)
    started["environment"] = environment
    if published is not None:
        started["published"] = published
    async with described.open() as platform:
        assert platform.trainer is not None
        started["blobs"] = platform.blobs_at  # (where the monitor reads the run's finished episodes)
        binding = binding_for(offered, channel, platform.tool_bindings, platform.pool_bindings, slots)
        wanting = desired_settings_of(platform.ledger)
        pinned = await _pinned(layers, platform.ledger, platform.registry)

        def bound(played: Any) -> Any:
            """How an entry's environment's episodes are played: each slot from the channel the run binds it to (the
            trained channel unless said), each import and pool where the profile serves it."""
            return binding_for(played, channel, platform.tool_bindings, platform.pool_bindings, slots)

        async def scheduled(name: str, every: int, episodes: int | None) -> Schedule | None:
            """The evals of a suite, by its name (the version it points to now) or a version's id: the ledger's, or the
            environment's eval data of that name, frozen on first use (`suite_for`); none for a suite neither has, or
            one whose environments do not all load here."""
            try:
                suite = await suite_for(platform.ledger, name, environment, offered)
                played = environments_of(suite, {environment: offered})
            except (KeyError, ValueError):
                return None
            return Schedule(suite, platform.eval_run, every, episodes, played, bound)

        async def desired() -> Mapping[str, JsonValue]:
            found = await wanting.desired(platform.run.id) if wanting is not None else None
            return found.settings if found is not None else {}

        schedule: Schedule | None = None
        if (asked := described.evals) is not None:  # (a suite not made yet is the environment's eval data of the name)
            try:
                suite = await suite_for(platform.ledger, asked.suite, environment, offered)
                played = environments_of(suite, {environment: offered})
            except (KeyError, ValueError) as error:
                raise SystemExit(error.args[0]) from None
            schedule = Schedule(suite, platform.eval_run, asked.every, asked.episodes, played, bound, asked.suite)
        started["settings"] = {
            "fixed": fixed(described, platform.trainer, groups=groups, seed=seed),
            "changeable": changeable(
                platform.trainer,
                groups_per_step=groups_per_step,
                max_lag=described.channels[channel].max_lag,
                evals=described.evals,
            ),
        }
        started["run_settings"] = _recorded(layers, described, pinned)  # (beside what the profile gave)
        async with ending(platform.ledger, platform.run.id):
            await train(
            offered, platform.trainer, platform.checkpoints, start=platform.origin, channel=channel,
            base=described.channels[channel].model,
            directory=described.directory / "checkpoints", publish=platform.publish, groups=groups,
            groups_per_step=groups_per_step, max_lag=described.channels[channel].max_lag, seed=seed,
            episodes_at_once=described.episodes_at_once, binding=binding,
            run=platform.run.id, started=started, hooks=[platform.feed], kept=platform.bookmarked, made=platform.made,
            reshard=platform.reshard if platform.layout else None, evals=schedule, desired=desired,
            scheduled=scheduled,
        )  # fmt: skip


async def _environment(described: "Profile", environment: str) -> tuple[Any, "dict[str, JsonValue] | None"]:
    """An environment a run plays, imported here, and what its start records of the published version it is (none
    for a built-in one): a published one by its version beside the profile's ledger (`rollout_train.published`). Exits
    saying why it does not load."""
    from rollout_train.hosting import ledger_of
    from rollout_train.published import is_published, loaded, provenance

    if not is_published(environment):
        return named(environment), None
    ledger = ledger_of(described)
    try:
        found, version = await loaded(environment, ledger)
    except KeyError as error:
        raise SystemExit(error.args[0]) from None
    finally:
        if (closing := getattr(ledger, "close", None)) is not None:
            closing()
    assert version is not None
    return found, provenance(version)


async def _evaluate(
    profile: Path,
    suite_name: str,
    reference: str | None,
    episodes: int | None,
    directory: Path | None,
    monitor: str | None = None,
    name: str | None = None,
    sets: list[str] | None = None,
    environment: str | None = None,
    preset: str | None = None,
    file: Path | None = None,
    chosen: dict[str, str | None] | None = None,
) -> None:
    import dataclasses
    import shutil

    from rollout.environment import binding_for
    from rollout_train.evals import Suite, environments_of, evaluate, suite_for, suite_of
    from rollout_train.hosting import ledger_of
    from rollout_train.profile import Profile
    from rollout_train.published import is_published, loaded, provenance
    from rollout_train.record import ending
    from rollout_train.registry import resolved

    flags = {"name": name, "start": reference, "eval.suite": suite_name, "eval.episodes": episodes}
    layers = await _layered(profile, directory, "eval", preset, file, sets or [], flags, **chosen or {})
    reference = cast(str | None, layers.given.get("start"))  # (none: the base model, whatever the profile starts from)
    episodes = cast(int | None, layers.settings["eval.episodes"])
    called = cast(str | None, layers.settings["name"])
    described = dataclasses.replace(Profile.load(profile, directory=directory, settings=layers.profile), name=called)
    if described.trainer is not None:  # (so the engines hold what the checkpoint is served over: full weights, say)
        trainer = dataclasses.replace(described.trainer, start=reference, bookmark=None)
        described = dataclasses.replace(described, trainer=trainer)
    channel = described.trainer.channel if described.trainer else next(iter(described.channels))
    slots = {key.removeprefix("slots."): str(value) for key, value in layers.given.items() if key.startswith("slots.")}
    ledger = ledger_of(described)
    published: dict[str, Any] = {}
    try:  # (the suite, and its environments, before the engines)
        found: Suite | None
        if environment is not None:  # (its eval data of that name is frozen as the suite, if it is not yet)
            found = await suite_for(ledger, suite_name, environment, (await loaded(environment, ledger))[0])
        elif (found := await suite_of(ledger, suite_name)) is None:
            raise KeyError(f"there is no suite {suite_name!r}: name its environment (--environment), or make one")
        suite: Suite = found
        imported = {each: await loaded(each, ledger) for each in suite.environments if is_published(each)}
        published = {each: provenance(version) for each, (_, version) in imported.items() if version is not None}
        played = environments_of(suite, {each: environment for each, (environment, _) in imported.items()})
    except (KeyError, ValueError) as error:
        raise SystemExit(error.args[0]) from None
    finally:
        if (closing := getattr(ledger, "close", None)) is not None:
            closing()
    where, profiled = await asyncio.to_thread(described.directory.absolute), await asyncio.to_thread(profile.absolute)
    started: dict[str, Any] = {"directory": str(where), "profile": str(profiled), "address": monitor}
    try:
        async with described.open(training=False) as platform:  # (no trainer: nothing is trained)
            try:
                subject = await resolved(platform.ledger, platform.registry, reference) if reference else None
                pinned = await _pinned(layers, platform.ledger, platform.registry)
            except (KeyError, ValueError) as error:
                raise SystemExit(error.args[0]) from None
            started["run_settings"] = _recorded(layers, described, {"start": reference, **pinned})
            started |= {"blobs": platform.blobs_at, "environment": suite.environments[0]}
            if suite.environments[0] in published:
                started["published"] = published[suite.environments[0]]

            def bound(environment: Any) -> Any:
                return binding_for(environment, channel, platform.tool_bindings, platform.pool_bindings, slots)

            async def part(number: int) -> str:
                return await platform.eval_run(part=number)

            async with ending(platform.ledger, platform.run.id):
                said = await evaluate(
                    platform.checkpoints, run=platform.run.id, suite=suite, subject=subject,
                    base=described.channels[channel].model, channel=channel,
                    directory=described.directory / "checkpoints", publish=platform.publish, environments=played,
                    binding=bound, parts=part, episodes=episodes, started=started,
                    reshard=platform.reshard if platform.layout else None, hooks=[platform.feed],
                )  # fmt: skip
    finally:  # (the files fetched to serve the checkpoint are needed only while it plays; a full one is a whole model)
        for fetched in ("bases", "checkpoints", "resharding"):
            await asyncio.to_thread(shutil.rmtree, described.directory / fetched, ignore_errors=True)
    for each in said["entries"]:
        solved = f"solved {each['solved']} of {each['played']}" if each["solved"] is not None else str(each["played"])
        print(f"{suite.id} {each['environment']}: {solved} episodes (mean reward {each['reward']})")


async def _pool(factory: str, directory: Path, where: str | None, name: str | None, host: str, port: int) -> None:
    import socket

    import uvicorn

    from rollout.harness.remote import serve_pool
    from rollout.harness.sandboxes import MemoryLeases, SandboxPool
    from rollout_train.presence import presence_of
    from rollout_train.sandboxes import admits, keep, leases_of

    provider = named(factory)(directory)
    ledger = _ledger_at(where) if where else None
    leases = (leases_of(ledger) if ledger is not None else None) or MemoryLeases()
    admitted = admits(ledger, presence_of(ledger)) if ledger is not None else None
    pool = SandboxPool(provider, name=name or f"{provider.kind}@{socket.gethostname()}", leases=leases, admits=admitted)
    keeping = (
        asyncio.ensure_future(keep(pool, ledger, presence_of(ledger), beat_as=f"pools/{pool.name}"))
        if ledger is not None
        else None
    )
    server = uvicorn.Server(uvicorn.Config(serve_pool(pool), host=host, port=port, log_level="warning"))
    try:
        await server.serve()
    finally:
        if keeping is not None:
            keeping.cancel()
            await asyncio.gather(keeping, return_exceptions=True)
        await pool.close()


async def _gateway(
    profile: Path,
    directory: Path | None,
    listen: str | None,
    certificate: Path | None,
    private_key: Path | None,
    proxied: str,
) -> None:
    import contextlib

    import uvicorn

    from rollout_train.gateway import create_app, deployed
    from rollout_train.gateway.beats import about, name_of
    from rollout_train.hosting import ledger_of
    from rollout_train.presence import beating, presence_of
    from rollout_train.profile import GatewaySpec, Profile

    described = Profile.load(profile, directory=directory)
    listening = listen or (described.gateway or GatewaySpec()).listen
    host, _, port = listening.rpartition(":")
    async with contextlib.AsyncExitStack() as stack:
        gateway = await deployed(described, stack)
        app = create_app(gateway)
        presence = presence_of(ledger_of(described))
        if presence is not None:  # (the monitor shows the replicas alive)
            said = lambda: about(gateway, listening, described.directory)  # noqa: E731
            beats = asyncio.ensure_future(beating(presence, name_of(listening), said))
            stack.callback(beats.cancel)
        config = uvicorn.Config(
            app,
            host=host,
            port=int(port),
            log_level="warning",
            proxy_headers=True,
            forwarded_allow_ips=proxied,
            ssl_certfile=str(certificate) if certificate else None,
            ssl_keyfile=str(private_key) if private_key else None,
        )
        await uvicorn.Server(config).serve()


@_user_errors
async def _suite(
    command: str,
    where: "str | Stores",
    name: str | None,
    environment: str | None,
    rows: str | None,
    seeds: str,
    episodes: int | None = None,
    thinking_tokens: int | None = None,
    answer_tokens: int | None = None,
    data: str | None = None,
    drop: str | None = None,
) -> None:
    from rollout_train.evals import SuiteEntry, edit_suite, make_suite, suite_entry, suite_of, suites_in, versions_of

    ledger = _ledger_at(where)
    if command in ("make", "edit"):
        assert name is not None
        keys = [each.strip() for each in rows.split(",") if each.strip()] if rows else None
        numbers = [int(each) for each in seeds.split(",") if each.strip()]
        current = await suite_of(ledger, name)
        if command == "edit" and current is None:
            raise KeyError(f"there is no suite {name!r}")
        entries: list[SuiteEntry] = list(current.entries) if command == "edit" and current else []
        if drop is not None:
            entries = [each for each in entries if each.environment != drop]
        if environment is None and command == "edit" and current is not None and len(current.entries) == 1:
            environment = current.environments[0]  # (the one environment it has)
        if environment is None and drop is None:
            raise SystemExit("say the entry's environment (--environment)")
        if environment is not None:
            kept = current.entry(environment) if command == "edit" and current else None
            counts: dict[str, Any] = {  # (an edit keeps what it does not change)
                "episodes": episodes or (kept.episodes if kept else 1),
                "thinking_tokens": thinking_tokens or (kept.thinking_tokens if kept else None),
                "answer_tokens": answer_tokens or (kept.answer_tokens if kept else None),
            }
            chosen: dict[str, Any] = {"rows": keys, "seeds": numbers} if numbers or keys else {}
            if not chosen and data is None and kept is not None:
                chosen = {"starts": kept.starts}  # (its starts as they were)
            chosen = chosen or {"eval_data": data or name}
            made_entry = suite_entry(environment, named(environment), **chosen, **counts)
            place = next((place for place, each in enumerate(entries) if each.environment == environment), None)
            if place is None:
                entries.append(made_entry)
            else:
                entries[place] = made_entry
        made = await (make_suite if command == "make" else edit_suite)(ledger, name, entries)
        held = ", held out of training" if made.held_out else ""
        print(f"the suite {made.id}: {len(made.starts)} starts of {', '.join(made.environments)}{held}")
        return
    listed = await suites_in(ledger)
    for each in listed:
        found = await suite_of(ledger, each)
        assert found is not None
        count = len(await versions_of(ledger, each))
        more = f"  ({count} versions)" if count > 1 else ""
        print(f"{found.id:<24} {len(found.starts):>4} starts  {', '.join(found.environments)}{more}")
    if environment is not None:  # (its eval data, frozen as a suite the first time it is played)
        for each, starts in named(environment).evals().items():
            if each not in listed:
                print(f"{each:<24} {len(starts):>4} starts  {environment}  (not played yet)")


async def _check(
    environment: str,
    row: str | None,
    reply: str,
    tools: list[str],
    profile: Path | None,
    groups: int | None,
    episodes: int | None,
    directory: Path | None,
    name: str | None,
    seed: int | None,
    sets: list[str] | None = None,
    pools: list[str] | None = None,
    preset: str | None = None,
    file: Path | None = None,
    chosen: dict[str, str | None] | None = None,
) -> int:
    import dataclasses

    from rollout.environment import binding_for
    from rollout.harness.imports import ToolBinding
    from rollout.harness.sandboxes import Pool, SandboxPool
    from rollout_train.algorithm import Grpo
    from rollout_train.check import checked, played, scripted
    from rollout_train.profile import Profile
    from rollout_train.record import ending

    offered = named(environment)
    found = checked(offered)
    for each in found:
        print(each, flush=True)
    if not found[0].passed:  # (with no rows, there is nothing to play)
        return 1
    scratch = directory or Path.home() / ".cache" / "rollout" / "checks" / (name or environment.replace(":", "-"))
    given = dict(_setting(each) for each in tools)  # (NAME=module:factory, or NAME=URL)
    kinds: dict[str, Any] = dict(_setting(each) for each in pools or [])  # (KIND=module:factory, or KIND=URL)
    if profile is not None:
        given = {**Profile.load(profile).tools, **given}
        kinds = {**Profile.load(profile).pools, **kinds}
    urls = {key: ToolBinding(url=str(where)) for key, where in given.items() if str(where).startswith("http")}
    await asyncio.to_thread(scratch.mkdir, parents=True, exist_ok=True)
    local = {key: named(str(where))(scratch) for key, where in given.items() if key not in urls}
    pool_urls = {kind: str(where) for kind, where in kinds.items() if str(where).startswith("http")}
    local_pools: dict[str, Pool] = {}
    for kind, where in kinds.items():
        if kind not in pool_urls:
            options = {"kind": where} if isinstance(where, str) else dict(where)
            local_pools[kind] = SandboxPool(named(str(options.pop("kind")))(scratch, **options))
    try:
        episode = await scripted(
            offered, row=row, reply=reply, tool_sets=local, tools=urls, pools=local_pools, pool_urls=pool_urls
        )
    finally:
        for each in [*local.values(), *local_pools.values()]:  # (as a profile closes its tool sets and pools)
            close = getattr(each, "close", None)
            if close is not None and asyncio.iscoroutine(closing := close()):
                await closing
    print(episode, flush=True)
    found.append(episode)
    flags = {"environment": environment, "name": name, "groups": groups, "seed": seed}
    layers = (
        await _layered(profile, scratch, "check", preset, file, sets or [], flags, **chosen or {}) if profile else None
    )
    groups = int(cast(int, layers.given.get("groups", 4))) if layers else 0  # (4 groups unless said otherwise)
    if profile is not None and layers is not None and groups > 0:
        loaded = Profile.load(profile, directory=scratch, settings=layers.profile)
        called = cast(str | None, layers.settings["name"]) or scratch.name
        described = dataclasses.replace(loaded, trainer=None, name=called)  # (the base model, untrained)
        channel = loaded.trainer.channel if loaded.trainer else next(iter(loaded.channels))
        slots = _slots(layers, offered, loaded, channel)
        seed = _whole(layers.settings, "seed")
        async with described.open() as platform:
            binding = binding_for(offered, channel, platform.tool_bindings, platform.pool_bindings, slots)
            started: dict[str, Any] = {"environment": environment, "profile": str(profile), "blobs": platform.blobs_at}
            started["directory"] = str(await asyncio.to_thread(scratch.absolute))
            started["run_settings"] = _recorded(layers, described, {"groups": groups})
            async with ending(platform.ledger, platform.run.id):
                groups_found = await played(
                    offered, platform.ledger, platform.blobs, run=platform.run.id, binding=binding, groups=groups,
                    episodes=episodes or Grpo().group_size, seed=seed, started=started,
                )  # fmt: skip
            for each in groups_found:
                print(each, flush=True)
                found.append(each)
    return 0 if all(each.passed for each in found) else 1


async def _imitate(
    profile: Path,
    directory: Path | None,
    kinds: list[str],
    limit: int | None,
    seed: int | None,
    dataset: str | None = None,
    start_at: str | None = None,
    name: str | None = None,
    resume_optimizer: bool = False,
    learning_rate: float | None = None,
    warmup: int | None = None,
    passes: int | None = None,
    sets: list[str] | None = None,
    preset: str | None = None,
    file: Path | None = None,
) -> None:
    import socket
    import time

    from rollout_train.checkpoints import Checkpoints
    from rollout_train.datasets import dataset_of, resolved_dataset
    from rollout_train.datasets import examples as dataset_examples
    from rollout_train.hosting import blobs_of, ledger_of
    from rollout_train.imitation import IMITATION, RATES, WARMUP, examples, imitate, passes_for
    from rollout_train.layout import BLOBS
    from rollout_train.profile import Profile
    from rollout_train.record import PROCESS, STARTS, ending, scope, table
    from rollout_train.registry import registry_of, resolved, run_of
    from rollout_train.stores import location

    flags: dict[str, Any] = {
        "name": name, "start": start_at, "seed": seed, "trainer.learning_rate": learning_rate,
        "imitation.dataset": dataset, "imitation.limit": limit, "imitation.warmup": warmup, "imitation.passes": passes,
        "imitation.resume_optimizer": resume_optimizer or None,
    }  # fmt: skip
    layers = await _layered(profile, directory, "imitate", preset, file, sets or [], flags)
    given, said = layers.given, layers.settings
    name, dataset = cast(str | None, said["name"]), cast(str | None, said["imitation.dataset"])
    limit, seed = cast(int | None, said["imitation.limit"]), _whole(said, "seed")
    learning_rate, warmup = given.get("trainer.learning_rate"), given.get("imitation.warmup")  # (not the profile's)
    passes, resume_optimizer = given.get("imitation.passes"), bool(said["imitation.resume_optimizer"])
    described = Profile.load(profile, directory=directory, settings=layers.profile)
    if described.trainer is None:
        raise SystemExit(f"{profile} describes no trainer")
    spec = described.channels[described.trainer.channel]
    renderer = named(spec.renderer)(spec.model)
    blobs, ledger = blobs_of(described), ledger_of(described)
    checkpoints, registry = Checkpoints(ledger, blobs), registry_of(ledger)
    run = await run_of(described.directory, ledger, registry, name)
    try:
        reference = cast(str | None, said["start"])  # (the profile's, unless said otherwise)
        start = await resolved(ledger, registry, reference) if reference else None
        made = await dataset_of(ledger, await resolved_dataset(ledger, registry, dataset)) if dataset else None
    except KeyError as error:
        raise SystemExit(error.args[0]) from None
    if made is not None:
        taught = await dataset_examples(ledger, made, renderer)
    else:
        taught = await examples(ledger, run.id, blobs, renderer, kinds=kinds)
    if not taught.segments:
        raise SystemExit("no solved episode of the run carried that guidance" if made is None else "no examples")
    print(f"{len(taught.segments)} segments of {taught.episodes} episodes ({taught.left_out} left out), "
          f"{taught.supervision}", flush=True)  # fmt: skip
    head = await checkpoints.head(run.id) or (await checkpoints.checkpoint(start) if start else None)
    model = spec.model  # (an adapter from full weights is trained over them)
    under = await checkpoints.under(head) if head is not None else None
    kind = named(described.trainer.kind)
    if under is not None and under.weights is not None and getattr(kind, "weights", "lora") == "lora":
        model = str(await checkpoints.files(under.weights, described.directory / "bases" / under.id))
    settings = {**described.trainer.settings, "objective": "likelihood"}
    chosen = taught.segments if limit is None else taught.segments[:limit]  # (as many as the step takes)
    weights = str(getattr(kind, "weights", "lora"))
    settings["learning_rate"] = learning_rate if learning_rate is not None else RATES.get(weights, 1e-6)
    settings["warmup_updates"] = WARMUP if warmup is None else warmup
    settings["passes"] = passes or passes_for(chosen, int(settings.get("tokens_per_step", 4096)))
    print(f"{settings['passes']} passes at {settings['learning_rate']:g}, warmed up over "
          f"{settings['warmup_updates']} updates", flush=True)  # fmt: skip
    trainer = kind(model, **settings)
    fence = await ledger.take(scope(run.id))  # (the run is stopped: imitation writes as it)
    where, profiled = await asyncio.to_thread(described.directory.absolute), await asyncio.to_thread(profile.absolute)
    started: Any = {
        "kind": IMITATION, "from": head.id if head else None, "dataset": made.id if made else None,
        "supervision": taught.supervision, "host": socket.gethostname(), "process": PROCESS,
        "started": round(time.time(), 1), "directory": str(where), "profile": str(profiled),
        "blobs": location(described.blobs, described.directory / BLOBS),
        "run_settings": _recorded(layers, described, {
            "trainer.objective": "likelihood", "trainer.learning_rate": settings["learning_rate"],
            "imitation.warmup": settings["warmup_updates"], "imitation.passes": settings["passes"],
        }),
    }  # fmt: skip
    await ledger.append(table(run.id, STARTS), str(fence.number), started, fence)
    try:
        async with ending(ledger, run.id):
            checkpoint = await imitate(
                checkpoints, trainer, taught, fence=fence, run=run.id, start=start, base=spec.model,
                directory=described.directory / "checkpoints",
                limit=limit, seed=seed, resume_optimizer=resume_optimizer,
            )  # fmt: skip
    except ValueError as error:
        raise SystemExit(str(error)) from None
    parents = ", ".join(checkpoint.parents) or "the base model"
    print(f"made {checkpoint.id} (from {parents}): "
          f"{json.dumps({key: round(value, 4) for key, value in checkpoint.metrics.items()})}")  # fmt: skip


@_user_errors
async def _dataset(
    command: str,
    where: "str | Stores",
    rule: str | None = None,
    runs: list[str] | None = None,
    turns: list[str] | None = None,
    cut: list[str] | None = None,
    per_task: int | None = None,
    name: str | None = None,
    into: Path | None = None,
) -> None:
    from rollout_train.checkpoints import short
    from rollout_train.datasets import ALL, datasets_in, make_dataset, where_blobs_are
    from rollout_train.registry import run_id
    from rollout_train.stores import FILES, opened

    ledger, registry = _registry_at(where)
    entries = await registry.runs()
    if command == "list":
        every = await datasets_in(ledger)
        shown, called = (
            short(each.id for each in every),
            {each.dataset: each.name for each in await registry.datasets()},
        )
        names = {entry.id: entry.name for entry in entries}
        for each in sorted(every, key=lambda each: each.made, reverse=True):
            label = f"  [{called[each.id]}]" if each.id in called else ""
            sizes = f"{each.counts.get('turns', 0):>6} turns of {each.counts.get('episodes', 0):>4} episodes"
            sources = ", ".join(names.get(run, run) for run in each.runs)
            print(f"{shown[each.id]:<8} {each.rule:<16} {sizes}  turns: {', '.join(each.turns)}  from {sources}{label}")
        return
    assert rule is not None and runs
    if name and any(each.name == name for each in await registry.datasets()):  # (before anything is made)
        raise SystemExit(f"another dataset is called {name!r}")
    ids = [await run_id(registry, who) for who in runs]
    if into is not None:
        kept = await asyncio.to_thread(lambda: into.expanduser().absolute())
        at: dict[str, Any] = {"kind": FILES, "directory": str(kept)}
    else:  # beside the first run's episodes
        from rollout_train.rollouts.episodes import Record
        from rollout_train.rollouts.scheduler import EPISODES

        first: Any = next(iter((await ledger.read(f"runs/{ids[0]}/{EPISODES}")).values()), None)
        at = await where_blobs_are(ledger, ids[0], Record.from_json(first).trajectories if first else None)
    made = await make_dataset(ledger, rule, ids, into=opened(at), at=at, turns=turns or [ALL], cut=cut or [],
                              per_task=per_task)  # fmt: skip
    if name:
        await registry.name_dataset(name, made.id)
    counts = ", ".join(f"{value} {key.replace('_', ' ')}" for key, value in made.counts.items())
    left = ", ".join(f"{value} {why}" for why, value in made.left_out.items()) or "none"
    print(f"made the dataset {made.id}{f' ({name})' if name else ''}: {counts}; left out: {left}")


def _setting(given: str) -> tuple[str, Any]:
    """`KEY=VALUE` (a tool set's or a pool's, say): the value read as TOML (`3e-5`, `true`, `[1, 2]`, `"text"`), or
    else as the text it is."""
    import tomllib

    key, _, value = given.partition("=")
    if not key or not _:
        raise SystemExit(f"{given!r}: it should be KEY=VALUE")
    try:
        return key.strip(), tomllib.loads(f"value = {value}")["value"]
    except tomllib.TOMLDecodeError:
        return key.strip(), value


_ALIASES = {"trainer.start": "start", "trainer.bookmark": "bookmark"}
"""Profile keys that are run settings by another name (as a launch passes them)."""
_COMMANDS = ("kind", "name", "environment", "groups", "groups_per_step", "seed", "eval.", "check.", "imitation.")
"""Run settings a command over a profile applies itself, not through the profile."""
_BUDGETS = ("thinking_tokens", "answer_tokens")
_BINDINGS = ("slots.", "self_judging")
"""Run settings that bind a program's slots to the profile's channels (`_slots`)."""
_MODES = ("mode", "follows", "lag", "checkpoint")
"""A channel's run settings that say whose records it serves (`rollout_train.serving.source_of`)."""


@dataclass(frozen=True)
class _Layered:
    """A run's settings over a profile: the keys the profile is loaded with (`profile`), the run's settings as it runs
    them (`settings`: the profile's, then a preset's, a file's and the flags'), what was given over the profile
    (`given`), and the preset they came from (`NAME@N`)."""

    profile: dict[str, Any]
    settings: "RunSettings"
    given: dict[str, Any]
    preset: str | None


async def _layered(
    path: Path,
    directory: Path | None,
    kind: str,
    preset: str | None,
    file: Path | None,
    sets: list[str],
    flags: dict[str, Any],
    *,
    model: str | None = None,
    renderer: str | None = None,
    channel: str | None = None,
) -> _Layered:
    """A run's settings in layers over what its profile gives (`_given_by`): a preset (`NAME` or `NAME@N`, kept beside
    the profile's ledger), a settings file, `--set` flags, then the command's own flags (those not `None`), `--model`
    and `--renderer` among them (of `--channel`, by default the trained one). The slots' bindings (`slots.SLOT`,
    `self_judging`) and the channels' modes (`channels.NAME.mode`, `.follows`, `.lag`, `.checkpoint`) are run settings
    only, recorded in the run's start. A run setting the profile has no place for (a provider, a spend limit) is
    refused; a key that is no run setting is the profile's own, as `--set` takes it."""
    from rollout_train.hosting import ledger_of
    from rollout_train.presets import presets_of
    from rollout_train.profile import Profile
    from rollout_train.run_settings import from_file, from_flags, is_trainers, key_of, layered, shortcuts

    plain = Profile.load(path, directory=directory)
    channel = channel or (plain.trainer.channel if plain.trainer else next(iter(plain.channels), "policy"))
    flags = {**flags, **shortcuts(model=model, renderer=renderer, channel=channel)}
    chosen = None
    if preset is not None:
        kept = presets_of(ledger_of(plain))
        chosen = await kept.get(preset) if kept is not None else None
        if chosen is None:
            raise SystemExit(f"there is no preset {preset!r} beside the profile's ledger")
    try:
        said = layered(chosen.settings if chosen else None, from_file(file) if file else None, from_flags(sets))
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
    given: dict[str, Any] = {}
    profile: dict[str, Any] = {}
    for key, value in [*said.values.items(), *((key, value) for key, value in flags.items() if value is not None)]:
        key = _ALIASES.get(key, key)
        if key_of(key) is None and not is_trainers(key):
            profile[key] = value  # (the profile's own)
            continue
        if (key == "evals.suite" and value == "") or (key.rpartition(".")[2] in _BUDGETS and value == "none"):
            value = None  # (no evals, no budget: as a launch says them)
        given[key] = value
    trained = plain.trainer.channel if plain.trainer else None
    refused: list[str] = []
    for key, value in given.items():
        last = key.rpartition(".")[2]
        if key.startswith(_COMMANDS) or key.startswith(_BINDINGS):
            continue
        if key.startswith("channels.") and key.count(".") == 2 and last in _MODES:
            continue  # (whose records a channel serves: read from the run's start by whatever serves it)
        if key in ("start", "bookmark"):
            profile[f"trainer.{key}"] = value
        elif key == "max_lag" and trained is not None:
            profile[f"channels.{trained}.max_lag"] = value
        elif key == "evals.suite":
            profile[key] = "" if value is None else value
        elif key in ("episodes_at_once", "trainer.channel") or key.startswith("evals.") or is_trainers(key):
            profile[key] = value
        elif key.startswith("channels.") and key.count(".") == 2 and last in ("model", "renderer", *_BUDGETS):
            profile[key] = "none" if value is None and last in _BUDGETS else value
        else:
            refused.append(key)
    if refused:
        raise SystemExit(f"a run over a profile cannot take {', '.join(refused)}: those need the cluster config")
    settings = layered(_given_by(plain, kind), given, {"kind": kind})
    return _Layered(profile, settings, given, chosen.id if chosen else None)


def _whole(settings: "RunSettings", key: str) -> int:
    return int(cast(int, settings[key]))


def _given_by(profile: "Profile", kind: str) -> dict[str, Any]:
    """The run settings a profile gives a run of `kind`: its trainer's, its channels', its evals'."""

    said: dict[str, Any] = {"episodes_at_once": profile.episodes_at_once}
    if (trainer := profile.trainer) is not None:
        said |= {"trainer.channel": trainer.channel, "start": trainer.start, "bookmark": trainer.bookmark}
        said |= {f"trainer.{key}": value for key, value in trainer.settings.items()}
        said["max_lag"] = profile.channels[trainer.channel].max_lag
    for name, channel in profile.channels.items():
        said |= {f"channels.{name}.model": channel.model, f"channels.{name}.renderer": channel.renderer}
        said |= {f"channels.{name}.{budget}": getattr(channel, budget) for budget in _BUDGETS}
    if (evals := profile.evals) is not None:
        said |= {"evals.suite": evals.suite, "evals.every": evals.every, "evals.episodes": evals.episodes}
    return {key: json.loads(json.dumps(value)) for key, value in said.items() if _takes(kind, key)}


def _takes(kind: str, key: str) -> bool:
    """Whether a run of `kind` takes the run setting `key` (a trainer's own: a run that trains)."""
    from rollout_train.run_settings import is_trainers, key_of

    found = key_of(key)
    return kind in found.kinds if found is not None else is_trainers(key) and kind in ("train", "imitate")


def _recorded(layers: _Layered, profile: "Profile", ran: dict[str, Any] | None = None) -> dict[str, Any]:
    """What a run's start records of its run settings (`rollout_train.run_settings.recorded`): what ran (with `ran`,
    what the command decided beyond its layers), the settings its trainer declares, and the preset."""
    from rollout_train.providers import TRAINER_KINDS, settings_of
    from rollout_train.run_settings import layered, recorded

    trainer = profile.trainer.kind if profile.trainer else None
    kinds = [each for each in TRAINER_KINDS.values() if each.implementation == trainer]
    try:
        specs = settings_of(kinds[0]) if kinds else ()
    except ImportError:  # (a trainer whose package is not installed here: its settings as given)
        specs = ()
    return recorded(layered(layers.settings.values, ran), specs, layers.preset)


def _slots(layers: _Layered, environment: Any, profile: "Profile", trained: str) -> dict[str, str]:
    """The channels a run over a profile binds its program's slots to by name (`slots.SLOT`; every other slot samples
    `trained`), refusing a binding `rollout_train.slots` refuses, a channel the profile does not describe, and a mode
    on the trained channel or one that follows a channel the profile lacks."""
    from rollout.environment import first_program
    from rollout.harness import instantiate
    from rollout_train.run_settings import layered
    from rollout_train.slots import Declared, problems

    settings = layered(layers.settings.values, {"trainer.channel": trained})
    declared = Declared.of(instantiate(first_program(environment)).model_slots())
    refused = [f"{key}: {reason}" for key, reason in problems(settings, declared)]
    named = {
        key.removeprefix("slots."): str(value) for key, value in settings.values.items() if key.startswith("slots.")
    }
    refused += [
        f"slots.{slot}: channel {channel} is not one of the profile's ({', '.join(profile.channels)})"
        for slot, channel in named.items()
        if channel not in profile.channels
    ]
    for channel in profile.channels:
        mode, follows = settings.get(f"channels.{channel}.mode"), settings.get(f"channels.{channel}.follows")
        if channel == trained and mode is not None:
            refused.append(f"channels.{channel}.mode: {channel} is the trained channel: it serves what the run trains")
        elif mode == "follows" and follows not in profile.channels:
            refused.append(f"channels.{channel}.follows: channel {channel} follows another channel of the profile")
    if refused:
        raise SystemExit("; ".join(refused))
    return named


async def _pinned(layers: _Layered, ledger: "Ledger", registry: "Registry | None") -> dict[str, Any]:
    """The checkpoints the run's fixed channels serve (`channels.NAME.checkpoint`), each resolved to its id, as the
    run's start records them."""
    from rollout_train.registry import resolved

    return {
        key: await resolved(ledger, registry, value)
        for key, value in layers.settings.values.items()
        if key.startswith("channels.") and key.endswith(".checkpoint") and isinstance(value, str)
    }


def _chosen(arguments: argparse.Namespace) -> dict[str, str | None]:
    """What `--model`, `--renderer` and `--channel` said."""
    return {"model": arguments.model, "renderer": arguments.renderer, "channel": arguments.channel}


def _layers_of(command: argparse.ArgumentParser, *, channels: bool = True) -> None:
    """A command that starts a run takes its settings in layers: `--preset`, then `--settings`, then `--set`, then
    (with `channels`) `--model` and `--renderer` of `--channel`."""
    if channels:
        command.add_argument("--model", help="the channel's model: channels.CHANNEL.model")
        command.add_argument("--renderer", help="the channel's renderer, module:name: channels.CHANNEL.renderer")
        command.add_argument("--channel", help="the channel --model and --renderer are of (by default the trained one)")
    command.add_argument("--preset", metavar="NAME[@N]", help="start from a preset's settings (`rollout preset`)")
    command.add_argument("--settings", type=Path, metavar="FILE", help="run settings: TOML or JSON, dotted keys")
    command.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="a run setting (trainer.learning_rate=3e-5) or a profile's key, the value read as JSON, then TOML, then "
        "as text (repeatable; over the preset and the file)",
    )  # fmt: skip


def _ledger_at(where: "str | Stores") -> "Ledger":
    """A ledger by where it is: a cluster's stores, a database's URL, a run's directory (as its `ledger.json` says),
    or a directory of files."""
    from rollout_train.ledger import LOCATION, FileLedger, of_run

    if not isinstance(where, str):
        return where.ledger
    if "://" in where:
        from rollout_train.database import DatabaseLedger

        return DatabaseLedger(where)
    path = Path(where).expanduser()
    return of_run(path) if (path / LOCATION).exists() or (path / "ledger").is_dir() else FileLedger(path)


async def _copy_ledger(source: str, target: str, point: bool) -> None:
    from rollout_train.database import DatabaseLedger, copy
    from rollout_train.ledger import LOCATION

    into = _ledger_at(target)
    if not isinstance(into, DatabaseLedger):
        raise SystemExit(f"{target} is not a database (a URL: sqlite:///… or postgresql://…)")
    count = await copy(_ledger_at(source), into)
    print(f"{count} records copied from {source} into {into.url}")
    if point:  # the run's directory now says its ledger is the copy (the profile should say so too)
        location = {"kind": "rollout_train.database:DatabaseLedger", "url": target}

        def pointed() -> None:
            (Path(source).expanduser() / LOCATION).write_text(json.dumps(location))

        await asyncio.to_thread(pointed)


def _registry_at(where: "str | Stores") -> "tuple[Ledger, Registry]":
    from rollout_train.registry import registry_of

    ledger = _ledger_at(where)
    registry = registry_of(ledger)
    if registry is None:
        raise SystemExit(f"the ledger at {where} has no registry beside it")
    return ledger, registry


async def _launcher(
    where: str,
    profiles: Path,
    environments: list[str],
    runs: Path,
    at_once: int,
    ray: str | None,
    gpus: float,
    name: str | None = None,
    cluster: str | None = None,
) -> None:
    from rollout_train.launcher import Launcher, name_of
    from rollout_train.launches import launches_of
    from rollout_train.presence import presence_of

    ledger = _ledger_at(where)
    launches, presence = launches_of(ledger), presence_of(ledger)
    if launches is None or presence is None:
        raise SystemExit(f"the ledger at {where} keeps no launches or heartbeats beside it")
    from rollout_train.published import environment_versions_of

    def absolute(path: Path) -> Path:  # (a run started in a published environment's source names its paths in full)
        return path.expanduser().absolute()

    profiles, runs = await asyncio.to_thread(absolute, profiles), await asyncio.to_thread(absolute, runs)
    found = Launcher(
        name_of(name), launches, presence, profiles, environments, runs, at_once=at_once, ray=ray, gpus=gpus,
        cluster=_cluster_of(cluster) if cluster is not None else None, versions=environment_versions_of(ledger),
    )  # fmt: skip
    await found.serve()


def _as_job(ray: str, given: list[str]) -> None:
    """Submit this launcher (the command as given, without `--as-job`) as a Ray job, unless one already runs here."""
    import shlex
    import socket

    from rollout_train.ray_cluster import prepare

    prepare()
    from ray.job_submission import JobSubmissionClient

    client, host = JobSubmissionClient(ray), socket.gethostname()
    for job in client.list_jobs():
        said: Any = job.metadata or {}
        if said.get("kind") == "launcher" and said.get("host") == host and not job.status.is_terminal():
            raise SystemExit(f"a launcher already runs here as Ray job {job.submission_id}: `ray job stop` it first")
    command = [sys.executable, "-m", "rollout_train.cli", *(each for each in given if each != "--as-job")]
    entrypoint = f"cd {shlex.quote(os.getcwd())} && exec {shlex.join(command)}"
    job = client.submit_job(entrypoint=entrypoint, metadata={"kind": "launcher", "host": host}, entrypoint_num_cpus=0)
    print(f"the launcher runs as Ray job {job}: `ray job logs {job} --follow` shows its output")


@_user_errors
async def _merge(who: str, where: "str | Stores", base: str | None, merger: str, bookmark: str | None) -> None:
    from rollout.harness.blobs import Blobs, FileBlobStore
    from rollout_train.checkpoints import Checkpoints
    from rollout_train.datasets import where_blobs_are
    from rollout_train.layout import BLOBS
    from rollout_train.merging import SCOPE, merge
    from rollout_train.registry import resolved
    from rollout_train.stores import opened

    ledger, registry = _registry_at(where)
    lora = await resolved(ledger, registry, who)
    if lora is None:
        raise SystemExit("the base model has no adapter to merge")
    if isinstance(where, str):
        beside: Blobs = FileBlobStore(await asyncio.to_thread(lambda: Path(where).expanduser() / BLOBS))
    else:
        beside = where.blobs  # (the cluster's)
    made = await Checkpoints(ledger, beside).checkpoint(lora)  # (its record only)
    try:
        blobs = opened(await where_blobs_are(ledger, made.run)) if made.run else beside
    except ValueError:  # (a run that does not say where its blobs are keeps them beside its ledger)
        blobs = beside
    fence = await ledger.take(SCOPE)
    scratch = Path.home() / ".cache" / "rollout" / "merging"  # (on disk: a merged model may be gigabytes)
    merged = await merge(Checkpoints(ledger, blobs), fence, lora, base=base, merger=merger, scratch=scratch)
    if bookmark:
        await registry.bookmark(bookmark, merged.id)
    print(f"merged {lora} into its base: {merged.id} (full weights, depth {merged.depth}, base {merged.base})")


@_user_errors
async def _preset(
    command: str,
    where: "str | Stores",
    reference: str | None = None,
    run: str | None = None,
    file: Path | None = None,
    sets: list[str] | None = None,
    note: str = "",
) -> None:
    """`rollout preset`: list the presets, show one (`NAME` or `NAME@N`), save a version, or delete one."""
    import time

    from rollout_train.presets import presets_of
    from rollout_train.record import STARTS, newest_record, table
    from rollout_train.registry import registry_of, run_id
    from rollout_train.run_settings import from_file, from_flags, layered

    ledger = _ledger_at(where)
    presets = presets_of(ledger)
    if presets is None:
        raise SystemExit("this ledger keeps no presets beside it")
    if command == "list":
        for each in await presets.all():
            saved = time.strftime("%Y-%m-%d %H:%M", time.localtime(each.saved))
            noted = f"  {each.note}" if each.note else ""
            print(f"{each.id:<32} {len(each.settings):>3} settings  saved {saved}{noted}")
        return
    assert reference is not None
    if command == "show":
        found = await presets.get(reference)
        if found is None:
            raise KeyError(f"there is no preset {reference!r}")
        versions = ", ".join(str(each.version) for each in await presets.versions(found.name))
        print(f"{found.id} (versions {versions}){f': {found.note}' if found.note else ''}")
        for key, value in sorted(found.settings.items()):
            print(f"{key} = {json.dumps(value)}")
        return
    if command == "delete":
        gone = await presets.delete(reference)
        print(f"the preset {gone.name} is deleted (its versions stay readable by number)")
        return
    copied: dict[str, Any] = {}
    if run is not None:  # (the settings a run's newest start records, less its name)
        started = newest_record(await ledger.read(table(await run_id(registry_of(ledger), run), STARTS)))
        kept = started.get("run_settings")
        if not isinstance(kept, dict):
            raise SystemExit(f"the run {run} records no run settings in its start")
        copied = {**cast(dict[str, Any], kept).get("fixed", {}), **cast(dict[str, Any], kept).get("changeable", {})}
        copied.pop("name", None)
    said = layered(copied, from_file(file) if file else None, from_flags(sets or []))
    saved = await presets.save(reference, said.values, note)
    print(f"saved {saved.id}: {len(saved.settings)} settings")


@_user_errors
async def _rename(who: str, name: str, where: "str | Stores") -> None:
    _, registry = _registry_at(where)
    entry = await registry.rename(who, name)
    print(f"the run {entry.id} is called {entry.name}")


@_user_errors
async def _pause_or_resume(command: str, who: str, where: "str | Stores") -> None:
    from rollout_train.registry import registry_of, run_id
    from rollout_train.resuming import IN_PLACE, pause, resume

    ledger = _ledger_at(where)
    run = await run_id(registry_of(ledger), who)
    if command == "pause":
        await pause(ledger, run)
        print(f"{who} is paused: what is playing plays out, and nothing new starts")
        return
    resumed = await resume(ledger, run)
    if resumed.how == IN_PLACE:
        print(f"{who} goes on")
    else:
        assert resumed.launch is not None
        print(f"{who} is asked to start again in {resumed.launch.asked.directory} ({resumed.launch.id})")


@_user_errors
async def _bookmark(name: str, reference: str | None, delete: bool, where: "str | Stores") -> None:
    from rollout_train.registry import resolved

    ledger, registry = _registry_at(where)
    if delete:
        await registry.unbookmark(name)
        print(f"no bookmark {name} any more")
        return
    checkpoint = await resolved(ledger, registry, reference or "")
    if checkpoint is None:
        raise SystemExit("a bookmark names a checkpoint, not the base model")
    await registry.bookmark(name, checkpoint)
    print(f"{name} is {checkpoint}")


async def _checkpoints(where: "str | Stores") -> None:
    from rollout_train.checkpoints import checkpoints_in, short
    from rollout_train.registry import names

    ledger, registry = _registry_at(where)
    every, called = await checkpoints_in(ledger), await names(registry)
    shown = short(checkpoint.id for checkpoint in every)
    marks: dict[str, list[str]] = {}
    for mark, checkpoint in called["bookmarks"].items():
        marks.setdefault(checkpoint, []).append(mark)
    for checkpoint in sorted(every, key=lambda each: (each.made, each.depth), reverse=True):
        origin = called["runs"].get(checkpoint.run, checkpoint.run) if checkpoint.run else "made outside a run"
        at = f":{checkpoint.step}" if checkpoint.step is not None else ""
        parents = ", ".join(shown.get(parent, parent) for parent in checkpoint.parents) or "the base model"
        kept = "" if checkpoint.weights is not None else "  (released)"
        bookmarked = f"  [{', '.join(marks[checkpoint.id])}]" if checkpoint.id in marks else ""
        print(f"{shown[checkpoint.id]:<8} depth {checkpoint.depth:<4} {origin}{at}  from {parents}{bookmarked}{kept}")


def _over_a_ledger(command: argparse.ArgumentParser) -> None:
    """A command over a ledger takes it as `--ledger`, or the cluster config's as `--cluster` (`_ledger_of`)."""
    where = "a run's directory, a ledger's directory, or a database's URL (by default this directory)"
    given = command.add_mutually_exclusive_group()
    given.add_argument("--ledger", help=where)
    given.add_argument("--cluster", **_cluster_option("the cluster config whose ledger it is"))


def _cluster_option(what: str) -> dict[str, Any]:
    """`--cluster [PATH or NAME]`: alone, the cluster config is found as `rollout_train.cluster.find` finds it."""
    return {
        "nargs": "?", "const": "", "metavar": "PATH or NAME",
        "help": f"{what}: a path, or a name under ~/.config/rollout/clusters (alone: ROLLOUT_CLUSTER, else "
        "~/.config/rollout/cluster.toml)",
    }  # fmt: skip


def _cluster_of(given: str | None) -> "Cluster":
    """The cluster config `--cluster` says (`""`: found as `rollout_train.cluster.find` finds it), read and checked;
    exits saying what is wrong."""
    from rollout_train.cluster import ClusterError, find, load

    try:
        return load(find(given or None))
    except ClusterError as error:
        raise SystemExit(str(error)) from None


def _ledger_of(arguments: argparse.Namespace) -> "str | Stores":
    """Where a command over a ledger finds it: the cluster config's stores (`--cluster`), else `--ledger`, else this
    directory."""
    if arguments.cluster is None:
        return arguments.ledger or "."
    from rollout_train.cluster import ClusterError
    from rollout_train.stores import Stores

    try:
        return Stores.open(_cluster_of(arguments.cluster))
    except ClusterError as error:
        raise SystemExit(str(error)) from None


def _check_cluster(given: str | None) -> int:
    """Say what the cluster config holds and, on this node, what of it does not resolve; 1 if something does not."""
    from rollout_train.cluster import ClusterError, find, inspect, load

    try:
        path = find(given or None)
        cluster = load(path)
    except ClusterError as error:
        print(error)
        return 1
    print(f"{path}: the cluster {cluster.name} (Ray namespace {cluster.namespace})")
    for kind, names in (("inference providers", cluster.inference), ("trainers", cluster.trainers),
                        ("sandbox pools", cluster.sandboxes), ("environments", cluster.environments)):  # fmt: skip
        print(f"  {kind}: {', '.join(names) or 'none'}")
    problems = inspect(cluster)
    for problem in problems:
        print(f"  {problem}")
    print(f"  {len(problems)} not resolved on this node" if problems else "  everything it names resolves on this node")
    return 1 if problems else 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="rollout", description="Train on an environment under a deployment profile.")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="run the training loop")
    training.add_argument("profile", type=Path)
    training.add_argument("environment")
    training.add_argument("--directory", type=Path, help="the run's directory (instead of the profile's)")
    training.add_argument("--groups", type=int, help="groups it plays (100)")
    training.add_argument("--groups-per-step", type=int, help="groups a step waits for (4)")
    training.add_argument("--seed", type=int, help="(0)")
    training.add_argument("--monitor", help="where the monitor on this machine serves, as other machines reach it")
    training.add_argument("--name", help="what a new run is called (by default its directory's name)")
    _layers_of(training)
    reporting = commands.add_parser("report", help="chart a run's progress, and post it to a Discord webhook")
    reporting.add_argument("directory", type=Path)
    reporting.add_argument("environment")
    reporting.add_argument("--watch", action="store_true", help="report again after every group, until interrupted")
    reporting.add_argument("--webhook", help="a Discord webhook (default: the environment's DISCORD_WEBHOOK_URL)")
    imitating = commands.add_parser(
        "imitate", help="a supervised step on a dataset, or on the run's solved episodes without their guidance"
    )
    imitating.add_argument("profile", type=Path)
    imitating.add_argument("--directory", type=Path, help="the run's directory (instead of the profile's)")
    imitating.add_argument("--dataset", help="a dataset to train on, by its name or id (`rollout dataset make`)")
    imitating.add_argument(
        "--start", help="the checkpoint to train from, if the run made none (instead of the profile's)"
    )
    imitating.add_argument("--name", help="what a new run is called (by default its directory's name)")
    imitating.add_argument("--without", nargs="+", default=["way"], help="the kinds of guidance to take out")
    imitating.add_argument("--limit", type=int, help="at most this many segments, drawn at random")
    imitating.add_argument("--seed", type=int, help="(0)")
    imitating.add_argument("--learning-rate", type=float, help="the step's rate (by default 1e-6 full, 1e-4 LoRA)")
    imitating.add_argument("--warmup", type=int, help="updates a fresh optimizer warms up over (4)")
    imitating.add_argument("--passes", type=int, help="passes over the examples (by default enough for 8 updates)")
    imitating.add_argument(
        "--resume-optimizer", action="store_true",
        help="go on from the trainer state of the checkpoint it trains from (by default the optimizer starts afresh)",
    )  # fmt: skip
    _layers_of(imitating, channels=False)
    presets = commands.add_parser("preset", help="list, show, save or delete presets: named, versioned run settings")
    preset_commands = presets.add_subparsers(dest="preset_command", required=True)
    _over_a_ledger(preset_commands.add_parser("list", help="every preset's newest version"))
    preset_showing = preset_commands.add_parser("show", help="a preset's settings: its newest version, or NAME@N")
    preset_showing.add_argument("preset", metavar="NAME[@N]")
    _over_a_ledger(preset_showing)
    preset_saving = preset_commands.add_parser(
        "save", help="save settings as a preset's next version: a run's, a file's, flags' (each over the last)"
    )
    preset_saving.add_argument("preset", metavar="NAME")
    preset_saving.add_argument("--from-run", metavar="RUN", help="the settings a run's start records, by name or id")
    preset_saving.add_argument("--settings", type=Path, metavar="FILE", help="run settings: TOML or JSON, dotted keys")
    preset_saving.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="a run setting")
    preset_saving.add_argument("--note", default="", help="what this version is for, or what it changed")
    _over_a_ledger(preset_saving)
    preset_deleting = preset_commands.add_parser("delete", help="delete a preset (its versions stay readable)")
    preset_deleting.add_argument("preset", metavar="NAME")
    _over_a_ledger(preset_deleting)
    datasets = commands.add_parser("dataset", help="make or list datasets: examples chosen from runs' episodes")
    dataset_commands = datasets.add_subparsers(dest="dataset_command", required=True)
    dataset_making = dataset_commands.add_parser("make", help="make a dataset by an episode rule and turn filters")
    dataset_making.add_argument("rule", choices=["solved-all", "best-of-group", "capped-per-task"])
    dataset_making.add_argument("--run", action="append", required=True, help="a run, by name or id (repeatable)")
    dataset_making.add_argument(
        "--turns", action="append", help="a turn filter: all (the default) or module:name (repeatable: every one keeps)"
    )
    dataset_making.add_argument(
        "--without", nargs="*", default=["way"], help="the kinds of guidance cut from the examples (way; none: empty)"
    )
    dataset_making.add_argument("--per-task", type=int, help="with capped-per-task: episodes of each task (3)")
    dataset_making.add_argument("--name", help="a name for the dataset")
    dataset_making.add_argument(
        "--blobs", type=Path, help="where to keep its manifest (by default beside the episodes)"
    )
    _over_a_ledger(dataset_making)
    dataset_listing = dataset_commands.add_parser("list", help="every dataset, newest first")
    _over_a_ledger(dataset_listing)
    monitoring = commands.add_parser("monitor", help="serve the monitor's page over a ledger and every run in it")
    monitoring.add_argument("where", help="a run's directory, a ledger's directory, or a database's URL")
    monitoring.add_argument("--host", default="127.0.0.1")
    monitoring.add_argument("--port", type=int, default=8765)
    monitoring.add_argument(
        "--cluster", **_cluster_option("import environments from git with this cluster config's Ray and blob store")
    )
    ledgers = commands.add_parser("ledger", help="work with ledgers")
    ledger_commands = ledgers.add_subparsers(dest="ledger_command", required=True)
    copying = ledger_commands.add_parser("copy", help="copy a ledger into a database (SQLite or Postgres)")
    copying.add_argument("source", help="a run's directory, a directory of files, or a database's URL")
    copying.add_argument("target", help="a database's URL: sqlite:///path or postgresql://…")
    copying.add_argument("--point", action="store_true", help="make the source run's directory name the copy")
    renaming = commands.add_parser("rename", help="call a run something else (its id stays)")
    renaming.add_argument("who", help="the run, by its name or its id")
    renaming.add_argument("name", help="what it is called from now on")
    _over_a_ledger(renaming)
    pausing = commands.add_parser("pause", help="pause a run: what is playing plays out, and nothing new starts")
    pausing.add_argument("who", help="the run, by its name or its id")
    _over_a_ledger(pausing)
    resuming = commands.add_parser(
        "resume", help="resume a run: a paused one goes on; a stopped, failed or lost one is launched again"
    )
    resuming.add_argument("who", help="the run, by its name or its id")
    _over_a_ledger(resuming)
    marking = commands.add_parser("bookmark", help="name a checkpoint, move a bookmark, or take one away")
    marking.add_argument("name")
    marking.add_argument("checkpoint", nargs="?", help="a bookmark, RUN:STEP, RUN, or a checkpoint's id or its start")
    marking.add_argument("--delete", action="store_true", help="take the bookmark away (the checkpoint stays)")
    _over_a_ledger(marking)
    launching = commands.add_parser("launcher", help="start the training runs asked for that this machine can run")
    launching.add_argument("--ledger", required=True, help="the database's URL (or a ledger's directory)")
    launching.add_argument("--profiles", type=Path, required=True, help="a directory of profiles it offers")
    launching.add_argument("--environment", action="append", default=[], help="an environment it offers (repeatable)")
    launching.add_argument("--runs", type=Path, required=True, help="where it makes each run's directory")
    launching.add_argument("--at-once", type=int, default=1, help="runs it plays at once (1: one GPU)")
    launching.add_argument("--ray", help="a Ray cluster's job server (http://127.0.0.1:8265): each run is a Ray job")
    launching.add_argument("--gpus", type=float, default=1.0, help="accelerators each run's Ray job asks for (1)")
    launching.add_argument("--as-job", action="store_true", help="submit the launcher itself as a Ray job (with --ray)")
    launching.add_argument("--name", help="what it beats as besides its host, where a machine has several launchers")
    offering = _cluster_option("the cluster config whose inference providers' models it offers evals")
    launching.add_argument("--cluster", **offering)
    merging = commands.add_parser("merge", help="fold a LoRA checkpoint into its base: a full checkpoint of its own")
    merging.add_argument("checkpoint", help="the LoRA checkpoint: a bookmark, RUN:STEP, RUN, or an id or its start")
    merging.add_argument("--base", help="the model to merge into (by default the one it was trained over)")
    merging.add_argument("--merger", default="rollout_lora.merge:merge", help="what folds the adapter in (module:name)")
    merging.add_argument("--bookmark", help="a bookmark to name the merged checkpoint")
    _over_a_ledger(merging)
    evaluating = commands.add_parser(
        "eval", help="play a suite with a checkpoint (or the base model), training nothing"
    )
    evaluating.add_argument("profile", type=Path)
    evaluating.add_argument("suite")
    evaluating.add_argument(
        "--checkpoint", help="a bookmark, RUN:STEP, RUN, or a checkpoint's id (none: the base model)"
    )
    evaluating.add_argument("--episodes", type=int, help="episodes of each start (by default the suite's)")
    evaluating.add_argument(
        "--environment", help="module:name: a suite not made yet is its eval data of that name, frozen now"
    )
    evaluating.add_argument("--directory", type=Path, help="the eval's directory (instead of the profile's)")
    evaluating.add_argument("--name", help="what the eval is called (by default its directory's name)")
    evaluating.add_argument("--monitor", help="where the monitor on this machine serves, as other machines reach it")
    _layers_of(evaluating)
    suites = commands.add_parser("suite", help="make, edit or list evaluation suites")
    suite_commands = suites.add_subparsers(dest="suite_command", required=True)
    making = suite_commands.add_parser("make", help="make a suite: its eval data, or a start of each row for each seed")
    editing = suite_commands.add_parser("edit", help="make a suite's next version, and point its name to it")
    for each in (making, editing):
        each.add_argument("name")
        each.add_argument(
            "--environment", required=each is making,
            help="module:name: the entry made, or changed (an edit of a suite of one entry: that one), or added",
        )  # fmt: skip
        each.add_argument("--rows", help="row keys, comma-separated (by default every row)")
        each.add_argument(
            "--seeds", default="",
            help="seeds, comma-separated: each row started once with each (none: its eval data, see --data)",
        )  # fmt: skip
        each.add_argument("--data", help="the environment's eval data its starts are (by default the suite's name)")
        each.add_argument("--episodes", type=int, help="episodes of each start an eval plays (1; an edit keeps them)")
        each.add_argument("--thinking-tokens", type=int, help="the entry's episodes' tokens of thinking per turn")
        each.add_argument("--answer-tokens", type=int, help="the entry's episodes' tokens of answer after thinking")
        _over_a_ledger(each)
    editing.add_argument("--drop", metavar="ENVIRONMENT", help="module:name: the entry left out of the next version")
    suite_listing = suite_commands.add_parser("list", help="every suite")
    _over_a_ledger(suite_listing)
    suite_listing.add_argument("--environment", help="module:name: its eval data not played yet too")
    environments = commands.add_parser("env", help="work with environments")
    environment_commands = environments.add_subparsers(dest="env_command", required=True)
    checking = environment_commands.add_parser("check", help="whether an environment holds together")
    checking.add_argument("environment", help="module:name")
    checking.add_argument("--row", help="the row the scripted episode plays (by default the first)")
    checking.add_argument("--reply", default="hello", help="what the scripted model says each turn")
    checking.add_argument(
        "--tools", action="append", default=[], metavar="NAME=WHERE",
        help="a tool set its program imports: module:factory, or a URL (repeatable; a profile's are used too)",
    )  # fmt: skip
    checking.add_argument(
        "--pools", action="append", default=[], metavar="KIND=WHERE",
        help="a pool of the sandboxes its program declares: module:factory of their provider, or a URL (repeatable; a "
        "profile's are used too)",
    )  # fmt: skip
    checking.add_argument("--profile", type=Path, help="play groups on this profile's channel, with its base model")
    checking.add_argument("--groups", type=int, help="groups played with --profile (4)")
    checking.add_argument("--episodes", type=int, help="episodes of each group (by default the algorithm's group size)")
    checking.add_argument("--directory", type=Path, help="the check's directory (~/.cache/rollout/checks/NAME)")
    checking.add_argument("--name", help="what the check's run is called")
    checking.add_argument("--seed", type=int, help="(0)")
    _layers_of(checking)
    listing = commands.add_parser("checkpoints", help="every checkpoint, newest first: where it came from")
    _over_a_ledger(listing)
    hosting = commands.add_parser("engines", help="keep a profile's engines serving what a run says, and nothing else")
    hosting.add_argument("profile", type=Path)
    hosting.add_argument("--run", required=True, help="the run whose channels they serve, by its name or its id")
    hosting.add_argument("--directory", type=Path, help="where fetched checkpoints are kept (instead of the profile's)")
    hosting.add_argument("--name", help="what it beats as (by default this machine's name)")
    playing = commands.add_parser("runner", help="play runs' episodes, and nothing else")
    playing.add_argument("profile", type=Path)
    playing.add_argument(
        "--run", action="append", default=[], help="a run it plays, by name or id (repeatable; none: all it reaches)"
    )
    playing.add_argument("--directory", type=Path, help="the runner's directory (instead of the profile's)")
    serving = commands.add_parser("tools", help="serve a tool set over HTTP")
    serving.add_argument("factory")
    serving.add_argument("--directory", type=Path, default=Path("."))
    serving.add_argument("--host", default="127.0.0.1")
    serving.add_argument("--port", type=int, default=8700)
    pooling = commands.add_parser("pool", help="serve a pool of sandboxes over HTTP")
    pooling.add_argument("factory", help="`module:name` of what makes the sandboxes' provider, called with --directory")
    pooling.add_argument("--directory", type=Path, default=Path("."))
    pooling.add_argument(
        "--ledger",
        help="keep the leases beside this ledger, ending with their claims: a run's directory, a ledger's "
        "directory, or a database's URL (without it, they are kept in the process, and end only when released)",
    )
    pooling.add_argument("--name", help="what the pool is called among those sharing the ledger (KIND@HOST)")
    pooling.add_argument("--host", default="127.0.0.1")
    pooling.add_argument("--port", type=int, default=8710)
    gateway = commands.add_parser("gateway", help="serve a replica of the gateway, which records every turn")
    gateway.add_argument("profile", type=Path, help="a profile: its channels, ledger, blobs and [gateway] table")
    gateway.add_argument("--directory", type=Path, help="in place of the profile's own")
    gateway.add_argument("--listen", help="host:port, in place of the profile's [gateway] listen")
    gateway.add_argument(
        "--certificate", type=Path, help="serve TLS with this certificate (else a proxy terminates it)"
    )
    gateway.add_argument("--private-key", type=Path, help="the certificate's private key")
    gateway.add_argument(
        "--proxied", default="127.0.0.1", help="addresses of proxies whose X-Forwarded-* headers are trusted ('*': any)"
    )
    clusters = commands.add_parser("cluster", help="work with the cluster config")
    cluster_commands = clusters.add_subparsers(dest="cluster_command", required=True)
    cluster_checking = cluster_commands.add_parser(
        "check", help="read the cluster config, and say which of its secrets and projects do not resolve on this node"
    )
    cluster_checking.add_argument("--cluster", **_cluster_option("the cluster config"))
    arguments = parser.parse_args()
    if arguments.command == "cluster":
        sys.exit(_check_cluster(arguments.cluster))
    if arguments.command == "train":
        work = _train(
            arguments.profile,
            arguments.directory,
            arguments.environment,
            arguments.groups,
            arguments.groups_per_step,
            arguments.seed,
            arguments.monitor,
            arguments.name,
            arguments.set,
            arguments.preset,
            arguments.settings,
            _chosen(arguments),
        )
        sys.exit(asyncio.run(until_signalled(work)))
    if arguments.command == "merge":
        asyncio.run(
            _merge(arguments.checkpoint, _ledger_of(arguments), arguments.base, arguments.merger, arguments.bookmark)
        )
        return
    if arguments.command == "eval":
        work = _evaluate(
            arguments.profile, arguments.suite, arguments.checkpoint, arguments.episodes, arguments.directory,
            arguments.monitor, arguments.name, arguments.set, arguments.environment, arguments.preset,
            arguments.settings, _chosen(arguments),
        )  # fmt: skip
        sys.exit(asyncio.run(until_signalled(work)))
    if arguments.command == "suite":
        if arguments.suite_command == "list":
            asyncio.run(_suite("list", _ledger_of(arguments), None, arguments.environment, None, ""))
            return
        asyncio.run(_suite(
            arguments.suite_command, _ledger_of(arguments), arguments.name, arguments.environment, arguments.rows,
            arguments.seeds, arguments.episodes, arguments.thinking_tokens, arguments.answer_tokens, arguments.data,
            getattr(arguments, "drop", None),
        ))  # fmt: skip
        return
    if arguments.command == "env":
        sys.exit(asyncio.run(_check(
            arguments.environment, arguments.row, arguments.reply, arguments.tools, arguments.profile,
            arguments.groups, arguments.episodes, arguments.directory, arguments.name, arguments.seed,
            arguments.set, arguments.pools, arguments.preset, arguments.settings, _chosen(arguments),
        )))  # fmt: skip
    if arguments.command == "preset":
        asyncio.run(_preset(
            arguments.preset_command, _ledger_of(arguments), getattr(arguments, "preset", None),
            getattr(arguments, "from_run", None), getattr(arguments, "settings", None), getattr(arguments, "set", None),
            getattr(arguments, "note", ""),
        ))  # fmt: skip
        return
    if arguments.command == "rename":
        asyncio.run(_rename(arguments.who, arguments.name, _ledger_of(arguments)))
        return
    if arguments.command in ("pause", "resume"):
        asyncio.run(_pause_or_resume(arguments.command, arguments.who, _ledger_of(arguments)))
        return
    if arguments.command == "bookmark":
        if arguments.checkpoint is None and not arguments.delete:
            parser.error("bookmark: name a checkpoint, or --delete")
        asyncio.run(_bookmark(arguments.name, arguments.checkpoint, arguments.delete, _ledger_of(arguments)))
        return
    if arguments.command == "launcher":
        if arguments.as_job:
            if not arguments.ray:
                parser.error("launcher --as-job: say the Ray cluster with --ray")
            _as_job(arguments.ray, sys.argv[1:])
            return
        work = _launcher(
            arguments.ledger, arguments.profiles, arguments.environment, arguments.runs, arguments.at_once,
            arguments.ray, arguments.gpus, arguments.name, arguments.cluster,
        )  # fmt: skip
        sys.exit(asyncio.run(until_signalled(work)))
    if arguments.command in ("engines", "runner"):
        from rollout_train.hosting import host_engines, run_episodes
        from rollout_train.profile import Profile

        described = Profile.load(arguments.profile, directory=arguments.directory)
        if arguments.command == "runner":
            sys.exit(asyncio.run(until_signalled(run_episodes(described, arguments.run))))
        work = host_engines(described, arguments.run, name=arguments.name)
        sys.exit(asyncio.run(until_signalled(work)))
    if arguments.command == "checkpoints":
        asyncio.run(_checkpoints(_ledger_of(arguments)))
        return
    if arguments.command == "gateway":
        work = _gateway(arguments.profile, arguments.directory, arguments.listen, arguments.certificate,
                        arguments.private_key, arguments.proxied)  # fmt: skip
        sys.exit(asyncio.run(until_signalled(work)))
    if arguments.command == "pool":
        work = _pool(arguments.factory, arguments.directory, arguments.ledger, arguments.name, arguments.host,
                     arguments.port)  # fmt: skip
        sys.exit(asyncio.run(until_signalled(work)))
    if arguments.command == "ledger":
        asyncio.run(_copy_ledger(arguments.source, arguments.target, arguments.point))
        return
    if arguments.command == "imitate":
        work = _imitate(
            arguments.profile, arguments.directory, arguments.without, arguments.limit, arguments.seed,
            arguments.dataset, arguments.start, arguments.name, arguments.resume_optimizer,
            arguments.learning_rate, arguments.warmup, arguments.passes, arguments.set, arguments.preset,
            arguments.settings,
        )  # fmt: skip
        sys.exit(asyncio.run(until_signalled(work)))
    if arguments.command == "dataset":
        if arguments.dataset_command == "list":
            asyncio.run(_dataset("list", _ledger_of(arguments)))
            return
        asyncio.run(_dataset(
            "make", _ledger_of(arguments), arguments.rule, arguments.run, arguments.turns, arguments.without,
            arguments.per_task, arguments.name, arguments.blobs,
        ))  # fmt: skip
        return
    if arguments.command == "report":
        from rollout_train.report import report

        webhook = arguments.webhook or os.environ.get("DISCORD_WEBHOOK_URL")
        asyncio.run(report(arguments.directory, named(arguments.environment).rows(), webhook, watch=arguments.watch))
    if arguments.command in ("monitor", "tools"):
        import uvicorn

        if arguments.command == "monitor":
            from rollout_train.monitor.app import create_app
            from rollout_train.publishing import Importer

            importer = Importer.of(_cluster_of(arguments.cluster)) if arguments.cluster is not None else None
            app = create_app(arguments.where, importer=importer)
        else:
            from rollout.harness.remote import serve

            app = serve(named(arguments.factory)(arguments.directory))
        uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="warning")


if __name__ == "__main__":
    main()
