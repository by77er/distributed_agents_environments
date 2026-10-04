"""`rollout`: train on an environment under a deployment profile, and watch.

rollout train PROFILE ENVIRONMENT   the training loop: PROFILE is a TOML file (`rollout_train.profile`), ENVIRONMENT
                                    names an environment as `module:name`
rollout report RUN ENVIRONMENT      chart a run's progress and summarise it; post both to a Discord webhook
rollout env check ENVIRONMENT       whether an environment holds together; with --profile, groups played by a model
rollout imitate PROFILE             a supervised step on a dataset (--dataset), or on the run's solved episodes
                                    without their guidance
rollout dataset make RULE           make a dataset: examples chosen from runs' episodes (`rollout_train.datasets`)
rollout monitor WHERE               the web page over a ledger and every run in it (WHERE: a run's directory, a ledger)
rollout ledger copy FROM TO         copy a ledger (a run's, files, or a database) into a database: SQLite or Postgres
rollout tools FACTORY               serve an environment's tool set over HTTP: FACTORY is `module:name`
rollout pool FACTORY                serve a pool of an environment's sandboxes over HTTP: FACTORY makes their provider
rollout engines PROFILE --run RUN   keep a profile's engines (vLLM servers) serving what the run says
rollout runner PROFILE              play runs' episodes, and nothing else
rollout pause RUN, rollout resume RUN
                                    pause a run, and resume it (in place, or launched again in its directory)
rollout gateway PROFILE             serve a replica of the gateway: it samples PROFILE's channels and records every turn

`rollout COMMAND --help` lists each command's options.
"""

import argparse
import asyncio
import json
import os
import signal
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rollout.names import named

if TYPE_CHECKING:
    from rollout_train.ledger import Ledger
    from rollout_train.registry import Registry


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


async def _train(
    profile: Path,
    directory: Path | None,
    environment: str,
    groups: int,
    groups_per_step: int,
    seed: int,
    monitor: str | None = None,
    name: str | None = None,
    settings: dict[str, Any] | None = None,
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

    described = dataclasses.replace(Profile.load(profile, directory=directory, settings=settings), name=name)
    if described.trainer is None:
        raise SystemExit(f"{profile} describes no trainer")
    channel, offered = described.trainer.channel, named(environment)
    where, profiled = await asyncio.to_thread(described.directory.absolute), await asyncio.to_thread(profile.absolute)
    started: dict[str, Any] = {"directory": str(where), "profile": str(profiled), "address": monitor}  # (its `starts`)
    started["environment"] = environment
    async with described.open() as platform:
        assert platform.trainer is not None
        started["blobs"] = platform.blobs_at  # (where the monitor reads the run's finished episodes)
        binding = binding_for(offered, channel, platform.tool_bindings, platform.pool_bindings)
        wanting = desired_settings_of(platform.ledger)

        def bound(played: Any) -> Any:
            """How an entry's environment's episodes are played: every slot from the trained channel, each import and
            pool where the profile serves it."""
            return binding_for(played, channel, platform.tool_bindings, platform.pool_bindings)

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


async def _evaluate(
    profile: Path,
    suite_name: str,
    reference: str | None,
    episodes: int | None,
    directory: Path | None,
    monitor: str | None = None,
    name: str | None = None,
    settings: dict[str, Any] | None = None,
    environment: str | None = None,
) -> None:
    import dataclasses
    import shutil

    from rollout.environment import binding_for
    from rollout_train.evals import Suite, environments_of, evaluate, suite_for, suite_of
    from rollout_train.layout import LEDGER
    from rollout_train.ledger import opened
    from rollout_train.profile import Profile
    from rollout_train.record import ending
    from rollout_train.registry import resolved

    described = dataclasses.replace(Profile.load(profile, directory=directory, settings=settings), name=name)
    if described.trainer is not None:  # (so the engines hold what the checkpoint is served over: full weights, say)
        trainer = dataclasses.replace(described.trainer, start=reference, bookmark=None)
        described = dataclasses.replace(described, trainer=trainer)
    channel = described.trainer.channel if described.trainer else next(iter(described.channels))
    ledger = opened(dict(described.ledger) or {"directory": str(described.directory / LEDGER)})
    try:  # (the suite, and its environments, before the engines)
        found: Suite | None
        if environment is not None:  # (its eval data of that name is frozen as the suite, if it is not yet)
            found = await suite_for(ledger, suite_name, environment, named(environment))
        elif (found := await suite_of(ledger, suite_name)) is None:
            raise KeyError(f"there is no suite {suite_name!r}: name its environment (--environment), or make one")
        suite: Suite = found
        played = environments_of(suite)
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
            except (KeyError, ValueError) as error:
                raise SystemExit(error.args[0]) from None
            started |= {"blobs": platform.blobs_at, "environment": suite.environments[0]}

            def bound(environment: Any) -> Any:
                return binding_for(environment, channel, platform.tool_bindings, platform.pool_bindings)

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


async def _suite(
    command: str,
    where: str,
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
        try:
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
        except (KeyError, ValueError) as error:
            raise SystemExit(error.args[0]) from None
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
    groups: int,
    episodes: int | None,
    directory: Path | None,
    name: str | None,
    seed: int,
    settings: dict[str, Any] | None = None,
    pools: list[str] | None = None,
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
    if profile is not None and groups > 0:
        loaded = Profile.load(profile, directory=scratch, settings=settings)
        described = dataclasses.replace(loaded, trainer=None, name=name or scratch.name)  # (the base model, untrained)
        channel = loaded.trainer.channel if loaded.trainer else next(iter(loaded.channels))
        async with described.open() as platform:
            binding = binding_for(offered, channel, platform.tool_bindings, platform.pool_bindings)
            started: dict[str, Any] = {"environment": environment, "profile": str(profile), "blobs": platform.blobs_at}
            started["directory"] = str(await asyncio.to_thread(scratch.absolute))
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
    seed: int,
    dataset: str | None = None,
    start_at: str | None = None,
    name: str | None = None,
    resume_optimizer: bool = False,
    learning_rate: float | None = None,
    warmup: int | None = None,
    passes: int | None = None,
) -> None:
    import socket
    import time

    from rollout.harness.blobs import FileBlobStore
    from rollout_train.checkpoints import Checkpoints
    from rollout_train.datasets import dataset_of, resolved_dataset
    from rollout_train.datasets import examples as dataset_examples
    from rollout_train.imitation import IMITATION, RATES, WARMUP, examples, imitate, passes_for
    from rollout_train.layout import BLOBS, LEDGER
    from rollout_train.ledger import opened
    from rollout_train.profile import Profile
    from rollout_train.record import PROCESS, STARTS, ending, scope, table
    from rollout_train.registry import registry_of, resolved, run_of
    from rollout_train.stores import location

    described = Profile.load(profile, directory=directory)
    if described.trainer is None:
        raise SystemExit(f"{profile} describes no trainer")
    spec = described.channels[described.trainer.channel]
    renderer = named(spec.renderer)(spec.model)
    store = dict(described.blobs)
    blobs = named(store.pop("kind"))(**store) if store else FileBlobStore(described.directory / BLOBS)
    ledger = opened(dict(described.ledger) or {"directory": str(described.directory / LEDGER)})
    checkpoints, registry = Checkpoints(ledger, blobs), registry_of(ledger)
    run = await run_of(described.directory, ledger, registry, name)
    try:
        reference = start_at or described.trainer.start
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
    print(f"{len(taught.segments)} segments of {taught.episodes} episodes ({taught.left_out} left out)", flush=True)
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
        "host": socket.gethostname(), "process": PROCESS, "started": round(time.time(), 1), "directory": str(where),
        "profile": str(profiled), "blobs": location(described.blobs, described.directory / BLOBS),
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


async def _dataset(
    command: str,
    where: str,
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
    from rollout_train.record import runs_in
    from rollout_train.registry import Taken, found
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
    known = await runs_in(ledger)
    ids: list[str] = []
    for who in runs:
        entry = found(entries, who)
        if entry is None and who not in known:
            raise SystemExit(f"there is no run {who!r}")
        ids.append(entry.id if entry is not None else who)
    if into is not None:
        kept = await asyncio.to_thread(lambda: into.expanduser().absolute())
        at: dict[str, Any] = {"kind": FILES, "directory": str(kept)}
    else:  # beside the first run's episodes
        from rollout_train.rollouts.episodes import Record
        from rollout_train.rollouts.scheduler import EPISODES

        first: Any = next(iter((await ledger.read(f"runs/{ids[0]}/{EPISODES}")).values()), None)
        at = await where_blobs_are(ledger, ids[0], Record.from_json(first).trajectories if first else None)
    try:
        made = await make_dataset(ledger, rule, ids, into=opened(at), at=at, turns=turns or [ALL], cut=cut or [],
                          per_task=per_task)  # fmt: skip
        if name:
            await registry.name_dataset(name, made.id)
    except (ValueError, Taken) as error:
        raise SystemExit(str(error)) from None
    counts = ", ".join(f"{value} {key.replace('_', ' ')}" for key, value in made.counts.items())
    left = ", ".join(f"{value} {why}" for why, value in made.left_out.items()) or "none"
    print(f"made the dataset {made.id}{f' ({name})' if name else ''}: {counts}; left out: {left}")


def _setting(given: str) -> tuple[str, Any]:
    """`KEY=VALUE` as a profile setting: the value read as TOML (`3e-5`, `true`, `[1, 2]`, `"text"`), or else as
    the text it is."""
    import tomllib

    key, _, value = given.partition("=")
    if not key or not _:
        raise SystemExit(f"--set {given!r}: it should be KEY=VALUE")
    try:
        return key.strip(), tomllib.loads(f"value = {value}")["value"]
    except tomllib.TOMLDecodeError:
        return key.strip(), value


def _ledger_at(where: str) -> "Ledger":
    """A ledger by where it is: a database's URL, a run's directory (as its `ledger.json` says), or a directory of
    files."""
    from rollout_train.ledger import LOCATION, FileLedger, of_run

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


def _registry_at(where: str) -> "tuple[Ledger, Registry]":
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
) -> None:
    from rollout_train.launcher import Launcher, name_of
    from rollout_train.launches import launches_of
    from rollout_train.presence import presence_of

    ledger = _ledger_at(where)
    launches, presence = launches_of(ledger), presence_of(ledger)
    if launches is None or presence is None:
        raise SystemExit(f"the ledger at {where} keeps no launches or heartbeats beside it")
    profiles, runs = await asyncio.to_thread(profiles.expanduser), await asyncio.to_thread(runs.expanduser)
    found = Launcher(
        name_of(name=name), launches, presence, profiles, environments, runs, at_once=at_once, ray=ray, gpus=gpus
    )
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


async def _merge(who: str, where: str, base: str | None, merger: str, bookmark: str | None) -> None:
    from rollout.harness.blobs import FileBlobStore
    from rollout_train.checkpoints import Checkpoints
    from rollout_train.layout import BLOBS
    from rollout_train.merging import SCOPE, merge
    from rollout_train.record import STARTS, table
    from rollout_train.registry import resolved
    from rollout_train.stores import opened

    ledger, registry = _registry_at(where)
    try:
        lora = await resolved(ledger, registry, who)
    except KeyError as error:
        raise SystemExit(error.args[0]) from None
    if lora is None:
        raise SystemExit("the base model has no adapter to merge")
    made = await Checkpoints(ledger, FileBlobStore(Path(where) / BLOBS)).checkpoint(lora)  # (its record only)
    starts: Any = await ledger.read(table(made.run, STARTS)) if made.run else {}
    kept: Any = starts[max(starts, key=int)].get("blobs") if starts else None  # (where the run keeps its blobs)
    here = await asyncio.to_thread(Path(where).expanduser)
    blobs = opened(kept) if kept else FileBlobStore(here / BLOBS)
    fence = await ledger.take(SCOPE)
    scratch = Path.home() / ".cache" / "rollout" / "merging"  # (on disk: a merged model may be gigabytes)
    merged = await merge(Checkpoints(ledger, blobs), fence, lora, base=base, merger=merger, scratch=scratch)
    if bookmark:
        await registry.bookmark(bookmark, merged.id)
    print(f"merged {lora} into its base: {merged.id} (full weights, depth {merged.depth}, base {merged.base})")


async def _rename(who: str, name: str, where: str) -> None:
    from rollout_train.registry import Taken

    _, registry = _registry_at(where)
    try:
        entry = await registry.rename(who, name)
    except (KeyError, Taken) as error:
        raise SystemExit(error.args[0]) from None
    print(f"the run {entry.id} is called {entry.name}")


async def _pause_or_resume(command: str, who: str, where: str) -> None:
    from rollout_train.record import runs_in
    from rollout_train.registry import found, registry_of
    from rollout_train.resuming import IN_PLACE, pause, resume

    ledger = _ledger_at(where)
    registry = registry_of(ledger)
    entry = found(await registry.runs(), who) if registry is not None else None
    run = entry.id if entry is not None else who
    if entry is None and who not in await runs_in(ledger):
        raise SystemExit(f"there is no run {who!r}")
    try:
        if command == "pause":
            await pause(ledger, run)
            print(f"{who} is paused: what is playing plays out, and nothing new starts")
            return
        resumed = await resume(ledger, run)
    except (KeyError, ValueError) as error:
        raise SystemExit(error.args[0]) from None
    if resumed.how == IN_PLACE:
        print(f"{who} goes on")
    else:
        assert resumed.launch is not None
        print(f"{who} is asked to start again in {resumed.launch.asked.directory} ({resumed.launch.id})")


async def _bookmark(name: str, reference: str | None, delete: bool, where: str) -> None:
    from rollout_train.registry import Taken, resolved

    ledger, registry = _registry_at(where)
    try:
        if delete:
            await registry.unbookmark(name)
            print(f"no bookmark {name} any more")
            return
        checkpoint = await resolved(ledger, registry, reference or "")
        if checkpoint is None:
            raise SystemExit("a bookmark names a checkpoint, not the base model")
        await registry.bookmark(name, checkpoint)
    except (KeyError, Taken) as error:
        raise SystemExit(error.args[0]) from None
    print(f"{name} is {checkpoint}")


async def _checkpoints(where: str) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(prog="rollout", description="Train on an environment under a deployment profile.")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="run the training loop")
    training.add_argument("profile", type=Path)
    training.add_argument("environment")
    training.add_argument("--directory", type=Path, help="the run's directory (instead of the profile's)")
    training.add_argument("--groups", type=int, default=100)
    training.add_argument("--groups-per-step", type=int, default=4, help="groups a step waits for (4)")
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--monitor", help="where the monitor on this machine serves, as other machines reach it")
    training.add_argument("--name", help="what a new run is called (by default its directory's name)")
    training.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="change a profile setting, by dotted key: --set trainer.learning_rate=3e-5 (a TOML value; repeatable)",
    )  # fmt: skip
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
    imitating.add_argument("--seed", type=int, default=0)
    imitating.add_argument("--learning-rate", type=float, help="the step's rate (by default 1e-6 full, 1e-4 LoRA)")
    imitating.add_argument("--warmup", type=int, help="updates a fresh optimizer warms up over (4)")
    imitating.add_argument("--passes", type=int, help="passes over the examples (by default enough for 8 updates)")
    imitating.add_argument(
        "--resume-optimizer", action="store_true",
        help="go on from the trainer state of the checkpoint it trains from (by default the optimizer starts afresh)",
    )  # fmt: skip
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
    dataset_making.add_argument(
        "--ledger", default=".", help="a run's directory, a ledger's directory, or a database's URL"
    )
    dataset_listing = dataset_commands.add_parser("list", help="every dataset, newest first")
    dataset_listing.add_argument(
        "--ledger", default=".", help="a run's directory, a ledger's directory, or a database's URL"
    )
    monitoring = commands.add_parser("monitor", help="serve the monitor's page over a ledger and every run in it")
    monitoring.add_argument("where", help="a run's directory, a ledger's directory, or a database's URL")
    monitoring.add_argument("--host", default="127.0.0.1")
    monitoring.add_argument("--port", type=int, default=8765)
    ledgers = commands.add_parser("ledger", help="work with ledgers")
    ledger_commands = ledgers.add_subparsers(dest="ledger_command", required=True)
    copying = ledger_commands.add_parser("copy", help="copy a ledger into a database (SQLite or Postgres)")
    copying.add_argument("source", help="a run's directory, a directory of files, or a database's URL")
    copying.add_argument("target", help="a database's URL: sqlite:///path or postgresql://…")
    copying.add_argument("--point", action="store_true", help="make the source run's directory name the copy")
    where = "a run's directory, a ledger's directory, or a database's URL (by default this directory)"
    renaming = commands.add_parser("rename", help="call a run something else (its id stays)")
    renaming.add_argument("who", help="the run, by its name or its id")
    renaming.add_argument("name", help="what it is called from now on")
    renaming.add_argument("--ledger", default=".", help=where)
    pausing = commands.add_parser("pause", help="pause a run: what is playing plays out, and nothing new starts")
    pausing.add_argument("who", help="the run, by its name or its id")
    pausing.add_argument("--ledger", default=".", help=where)
    resuming = commands.add_parser(
        "resume", help="resume a run: a paused one goes on; a stopped, failed or lost one is launched again"
    )
    resuming.add_argument("who", help="the run, by its name or its id")
    resuming.add_argument("--ledger", default=".", help=where)
    marking = commands.add_parser("bookmark", help="name a checkpoint, move a bookmark, or take one away")
    marking.add_argument("name")
    marking.add_argument("checkpoint", nargs="?", help="a bookmark, RUN:STEP, RUN, or a checkpoint's id or its start")
    marking.add_argument("--delete", action="store_true", help="take the bookmark away (the checkpoint stays)")
    marking.add_argument("--ledger", default=".", help=where)
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
    merging = commands.add_parser("merge", help="fold a LoRA checkpoint into its base: a full checkpoint of its own")
    merging.add_argument("checkpoint", help="the LoRA checkpoint: a bookmark, RUN:STEP, RUN, or an id or its start")
    merging.add_argument("--base", help="the model to merge into (by default the one it was trained over)")
    merging.add_argument("--merger", default="rollout_lora.merge:merge", help="what folds the adapter in (module:name)")
    merging.add_argument("--bookmark", help="a bookmark to name the merged checkpoint")
    merging.add_argument("--ledger", default=".", help=where)
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
    evaluating.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="change a profile setting")
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
        each.add_argument("--ledger", default=".", help=where)
    editing.add_argument("--drop", metavar="ENVIRONMENT", help="module:name: the entry left out of the next version")
    suite_listing = suite_commands.add_parser("list", help="every suite")
    suite_listing.add_argument("--ledger", default=".", help=where)
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
    checking.add_argument("--groups", type=int, default=4, help="groups played with --profile (4)")
    checking.add_argument("--episodes", type=int, help="episodes of each group (by default the algorithm's group size)")
    checking.add_argument("--directory", type=Path, help="the check's directory (~/.cache/rollout/checks/NAME)")
    checking.add_argument("--name", help="what the check's run is called")
    checking.add_argument("--seed", type=int, default=0)
    checking.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="change a profile setting")
    listing = commands.add_parser("checkpoints", help="every checkpoint, newest first: where it came from")
    listing.add_argument("--ledger", default=".", help=where)
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
    arguments = parser.parse_args()
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
            dict(_setting(each) for each in arguments.set),
        )
        sys.exit(asyncio.run(until_signalled(work)))
    if arguments.command == "merge":
        asyncio.run(
            _merge(arguments.checkpoint, arguments.ledger, arguments.base, arguments.merger, arguments.bookmark)
        )
        return
    if arguments.command == "eval":
        work = _evaluate(
            arguments.profile, arguments.suite, arguments.checkpoint, arguments.episodes, arguments.directory,
            arguments.monitor, arguments.name, dict(_setting(each) for each in arguments.set), arguments.environment,
        )  # fmt: skip
        sys.exit(asyncio.run(until_signalled(work)))
    if arguments.command == "suite":
        if arguments.suite_command == "list":
            asyncio.run(_suite("list", arguments.ledger, None, arguments.environment, None, ""))
            return
        asyncio.run(_suite(
            arguments.suite_command, arguments.ledger, arguments.name, arguments.environment, arguments.rows,
            arguments.seeds, arguments.episodes, arguments.thinking_tokens, arguments.answer_tokens, arguments.data,
            getattr(arguments, "drop", None),
        ))  # fmt: skip
        return
    if arguments.command == "env":
        sys.exit(asyncio.run(_check(
            arguments.environment, arguments.row, arguments.reply, arguments.tools, arguments.profile,
            arguments.groups, arguments.episodes, arguments.directory, arguments.name, arguments.seed,
            dict(_setting(each) for each in arguments.set), arguments.pools,
        )))  # fmt: skip
    if arguments.command == "rename":
        asyncio.run(_rename(arguments.who, arguments.name, arguments.ledger))
        return
    if arguments.command in ("pause", "resume"):
        asyncio.run(_pause_or_resume(arguments.command, arguments.who, arguments.ledger))
        return
    if arguments.command == "bookmark":
        if arguments.checkpoint is None and not arguments.delete:
            parser.error("bookmark: name a checkpoint, or --delete")
        asyncio.run(_bookmark(arguments.name, arguments.checkpoint, arguments.delete, arguments.ledger))
        return
    if arguments.command == "launcher":
        if arguments.as_job:
            if not arguments.ray:
                parser.error("launcher --as-job: say the Ray cluster with --ray")
            _as_job(arguments.ray, sys.argv[1:])
            return
        work = _launcher(
            arguments.ledger, arguments.profiles, arguments.environment, arguments.runs, arguments.at_once,
            arguments.ray, arguments.gpus, arguments.name,
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
        asyncio.run(_checkpoints(arguments.ledger))
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
            arguments.learning_rate, arguments.warmup, arguments.passes,
        )  # fmt: skip
        sys.exit(asyncio.run(until_signalled(work)))
    if arguments.command == "dataset":
        if arguments.dataset_command == "list":
            asyncio.run(_dataset("list", arguments.ledger))
            return
        asyncio.run(_dataset(
            "make", arguments.ledger, arguments.rule, arguments.run, arguments.turns, arguments.without,
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

            app = create_app(arguments.where)
        else:
            from rollout.harness.remote import serve

            app = serve(named(arguments.factory)(arguments.directory))
        uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="warning")


if __name__ == "__main__":
    main()
