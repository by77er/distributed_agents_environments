"""`rollout`: ask for runs on a cluster, train, evaluate, and watch.

rollout train ENVIRONMENT           a training run: its settings in layers (below), its job submitted and followed
rollout eval SUITE                  play a suite with a checkpoint (--checkpoint) or a base model, training nothing
rollout imitate --dataset DATASET   a supervised step on a dataset (`rollout dataset make`)
rollout env check ENVIRONMENT       whether an environment holds together; with a model's settings, groups it plays
rollout resume RUN, rollout pause RUN
                                    resume a run (in place, or its job submitted again with its recorded settings), or
                                    pause it
rollout suite make|edit|list        evaluation suites, kept in versions (`rollout_train.evals`)
rollout report DIRECTORY ENVIRONMENT
                                    chart a run's progress and summarise it; post both to a Discord webhook
rollout dataset make RULE           make a dataset: examples chosen from runs' episodes (`rollout_train.datasets`)
rollout checkpoints                 every checkpoint, newest first: where it came from
rollout bookmark NAME CHECKPOINT    name a checkpoint, or move a bookmark there (--delete takes it away)
rollout rename WHO NAME             call a run something else (its id stays)
rollout merge CHECKPOINT            fold a LoRA checkpoint into its base: a full checkpoint of its own
rollout monitor [WHERE]             the web page over a ledger and every run in it; with --cluster, it asks for runs
rollout ledger copy FROM TO         copy a ledger (a run's, files, or a database) into a database: SQLite or Postgres
rollout tools FACTORY               serve an environment's tool set over HTTP: FACTORY is `module:name`
rollout pool FACTORY                serve a pool of an environment's sandboxes over HTTP: FACTORY makes their provider
rollout gateway                     serve a replica of the gateway: every channel a run's start names on the cluster's
                                    providers, and every turn recorded
rollout cluster check               read the cluster config (`rollout_train.cluster`) and say what does not resolve here
rollout preset list|show|save|load|delete
                                    presets: named, versioned run settings beside the ledger (`rollout_train.presets`)

A command that asks for a run (`train`, `eval`, `imitate`, `env check`) takes its run settings
(`rollout_train.run_settings`) in layers: `--preset NAME[@N]`, then `--settings FILE`, then `--set KEY=VALUE`, then
its own flags (`--model`, `--provider`, `--renderer` of `--channel`, `--trainer` among them). It checks them against the
cluster config (`--cluster`, as `rollout_train.cluster.find` finds it) and submits the run's job
(`rollout_train.submitting.submit`), following it until it ends; `--check` says what would be refused and stops,
`--detach` returns once the job is submitted, and `--here` runs the job in this process, on the cluster's Ray.

A command over a ledger (`rename`, `bookmark`, `pause`, `resume`, `checkpoints`, `suite`, `dataset`, `merge`,
`preset`) takes it as `--ledger WHERE`, or as the cluster config's with `--cluster [PATH or NAME]` (alone:
`ROLLOUT_CLUSTER`, else `~/.config/rollout/cluster.toml`).

`rollout COMMAND --help` lists each command's options.
"""

import argparse
import asyncio
import contextlib
import functools
import json
import os
import signal
import sys
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from rollout.names import named

if TYPE_CHECKING:
    from pydantic import JsonValue

    from rollout_train.cluster import Cluster
    from rollout_train.launches import Launch
    from rollout_train.ledger import Ledger
    from rollout_train.registry import Registry
    from rollout_train.run_settings import RunSettings
    from rollout_train.stores import Stores


async def until_signalled(work: Coroutine[Any, Any, object]) -> int:
    """Run `work`, and cancel it on an interrupt, a termination or a hang-up, so that it stops what it started on
    the way out; returns the exit status: what `work` returns, where that is a number, else 0. (A process started in
    the background of a script inherits "ignore" for interrupts, and Python then installs no handler of its own.)"""
    task = asyncio.ensure_future(work)
    received: list[int] = []

    def stop(number: int) -> None:
        received.append(number)
        task.cancel()

    loop = asyncio.get_running_loop()
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        loop.add_signal_handler(number, stop, number)
    try:
        result = await task
    except asyncio.CancelledError:
        if not received:
            raise
        return 128 + received[0]
    return result if isinstance(result, int) and not isinstance(result, bool) else 0


def _user_errors[**P](work: Callable[P, Coroutine[Any, Any, None]]) -> Callable[P, Coroutine[Any, Any, None]]:
    """A command whose `KeyError` or `ValueError` (a `Taken` too) is the user's to mend: it exits saying it."""

    @functools.wraps(work)
    async def said(*arguments: P.args, **options: P.kwargs) -> None:
        try:
            await work(*arguments, **options)
        except (KeyError, ValueError) as error:
            raise SystemExit(error.args[0] if error.args else str(error)) from None

    return said


FOLLOWING = 2.0
"""Seconds between a followed launch's looks at its job."""
DONE = {"ended": 0, "failed": 1, "stopped": 130}
"""A followed launch's exit status, by how it finished."""


async def _asked(
    arguments: argparse.Namespace,
    kind: str,
    flags: "dict[str, JsonValue]",
    *,
    named_as: Callable[["RunSettings"], str],
    under: "dict[str, JsonValue] | None" = None,
) -> int:
    """Ask for a run of `kind` with its settings in layers (`under`, then `--preset`, `--settings`, `--set`, the
    shortcuts and `flags`), checked against the cluster config; then, as the arguments say, stop there (`--check`),
    run its job here (`--here`), or submit it and follow it (unless `--detach`). Returns the exit status."""
    from rollout_train.launching import checked, free_name, ray_free, settled
    from rollout_train.run_settings import from_file, from_flags, layered, shortcuts
    from rollout_train.stores import Stores
    from rollout_train.submitting import ask, submit

    cluster = _cluster_of(arguments.cluster)
    stores = Stores.open(cluster)
    channel = getattr(arguments, "channel", None) or "policy"
    said = shortcuts(
        model=getattr(arguments, "model", None), provider=getattr(arguments, "provider", None),
        renderer=getattr(arguments, "renderer", None), trainer=getattr(arguments, "trainer", None), channel=channel,
    )  # fmt: skip
    try:
        given = layered(
            from_file(arguments.settings) if arguments.settings else None, from_flags(arguments.set), said,
            {key: value for key, value in flags.items() if value is not None},
        )  # fmt: skip
        settings, preset = await settled(kind, arguments.name, {**(under or {}), **given.values},
                                         preset=arguments.preset, presets=stores.presets)  # fmt: skip
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error.args[0]) if error.args else str(error)) from None
    if kind == "eval" and settings["environment"] is None and isinstance(settings["eval.suite"], str):
        from rollout_train.evals import suite_of

        suite = await suite_of(stores.ledger, str(settings["eval.suite"]))
        if suite is None:
            raise SystemExit(f"there is no suite {settings['eval.suite']!r}: make it with `rollout suite make`")
        settings = layered(settings.values, {"environment": suite.environments[0]})
    if not settings["name"]:
        taken = {each.name for each in await stores.registry.runs()}
        settings = layered(settings.values, {"name": free_name(named_as(settings), taken)})
    free = await asyncio.to_thread(ray_free) if arguments.here else None
    findings = await checked(settings, cluster, stores.ledger, free=free)
    for each in findings:
        print(f"{'refused' if each.refuses else 'note'}: {each.key}: {each.reason}", flush=True)
    if any(each.refuses for each in findings):
        return 2
    if arguments.check:
        print(f"{settings['name']}: a {kind} run that may be asked for, with {len(settings.values)} settings given")
        return 0
    if arguments.here:
        import ray

        from rollout_train.jobs import driven
        from rollout_train.ray_cluster import connect

        if not ray.is_initialized():
            await asyncio.to_thread(connect, cluster.ray.address)
        launch = await ask(settings, stores.ledger, preset=preset)
        print(f"{settings['name']} ({launch.run}): launch {launch.id}, run here", flush=True)
        return await driven(launch.id, cluster, stores)
    launch = await submit(settings, cluster, stores.ledger, preset=preset)
    print(f"{settings['name']} ({launch.run}): launch {launch.id}, job {launch.job} ({launch.state})", flush=True)
    if launch.state == "failed":
        print(launch.detail, flush=True)
        return 1
    if arguments.detach:
        return 0
    return await _followed(launch, cluster, stores)


async def _followed(launch: "Launch", cluster: "Cluster", stores: "Stores") -> int:
    """Say how a launch goes until it finishes; an interrupt asks it to stop. Returns its exit status."""
    from rollout_train.launches import launch_of, launches_of
    from rollout_train.submitting import followed, stopped

    launches = launches_of(stores.ledger)
    assert launches is not None
    said = (launch.state, launch.detail)
    try:
        while launch.state not in DONE:
            await asyncio.sleep(FOLLOWING)
            launch = await followed(await launch_of(launches, launch.id), launches, cluster)
            if (launch.state, launch.detail) != said:
                said = (launch.state, launch.detail)
                detail = f": {launch.detail}" if launch.detail and launch.detail != launch.state else ""
                print(f"{launch.state}{detail}", flush=True)
    except asyncio.CancelledError:
        with contextlib.suppress(KeyError):
            await asyncio.shield(stopped(await launch_of(launches, launch.id), launches, cluster))
        raise
    return DONE[launch.state]


def _named_after(environment: "JsonValue") -> str:
    """A run's name after its environment: the name of a published one; a built-in one's attribute (`words`), or its
    module's first part where the attribute says nothing (`environment`)."""
    said = str(environment or "run")
    if "@" in said:
        return said.partition("@")[0]
    module, _, attribute = said.partition(":")
    return attribute if attribute and attribute not in ("environment", "environments") else module.partition(".")[0]


async def _train(arguments: argparse.Namespace) -> int:
    flags: dict[str, JsonValue] = {
        "environment": arguments.environment, "groups": arguments.groups,
        "groups_per_step": arguments.groups_per_step, "seed": arguments.seed,
    }  # fmt: skip
    return await _asked(arguments, "train", flags, named_as=lambda said: _named_after(said["environment"]))


async def _evaluate(arguments: argparse.Namespace) -> int:
    """An eval of a suite: by a checkpoint (`--checkpoint`, its channel's model, renderer and provider those of the run
    that made it, unless the settings say others), or by a base model the settings name."""
    flags: dict[str, JsonValue] = {
        "eval.suite": arguments.suite, "start": arguments.checkpoint, "eval.episodes": arguments.episodes,
    }  # fmt: skip
    under = await _made_by(arguments) if arguments.checkpoint else None

    def named_as(said: "RunSettings") -> str:
        subject = said["start"] or said.get(f"channels.{arguments.channel or 'policy'}.model") or "the base model"
        return f"{arguments.suite} on {str(subject).split('/')[-1]}"

    return await _asked(arguments, "eval", flags, named_as=named_as, under=under)


async def _made_by(arguments: argparse.Namespace) -> "dict[str, JsonValue]":
    """The channel settings of the run that made the checkpoint an eval plays (its trained channel's model, renderer,
    providers and budgets), as the eval's channel's."""
    from rollout_train.record import recorded_settings
    from rollout_train.registry import resolved
    from rollout_train.stores import Stores

    stores = Stores.open(_cluster_of(arguments.cluster))
    try:
        id = await resolved(stores.ledger, stores.registry, arguments.checkpoint)
    except KeyError as error:
        raise SystemExit(error.args[0]) from None
    if id is None:
        return {}
    made = await stores.checkpoints.checkpoint(id)
    recorded = await recorded_settings(stores.ledger, made.run) if made.run else None
    if not recorded:
        return {}
    trained = str(recorded.get("trainer.channel") or "policy")
    channel = arguments.channel or "policy"
    keys = ("model", "renderer", "provider", "providers", "thinking_tokens", "answer_tokens")
    return {
        f"channels.{channel}.{key}": recorded[f"channels.{trained}.{key}"]
        for key in keys if recorded.get(f"channels.{trained}.{key}") is not None
    }  # fmt: skip


async def _imitate(arguments: argparse.Namespace) -> int:
    flags: dict[str, JsonValue] = {
        "start": arguments.start, "seed": arguments.seed, "trainer.learning_rate": arguments.learning_rate,
        "imitation.dataset": arguments.dataset, "imitation.limit": arguments.limit,
        "imitation.warmup": arguments.warmup, "imitation.passes": arguments.passes,
        "imitation.resume_optimizer": arguments.resume_optimizer or None,
    }  # fmt: skip
    return await _asked(arguments, "imitate", flags, named_as=lambda said: f"imitate {said['imitation.dataset']}")


async def _pool(arguments: argparse.Namespace) -> None:
    """Serve a pool over HTTP: of the provider `factory` names, made with `--directory`; or, with `--kind`, the
    cluster config's `[sandboxes.KIND]` (its provider, `size` and settings, made with `--directory` or the cluster's
    `[scratch]/sandboxes/KIND`), its leases beside the cluster's ledger."""
    import socket

    import uvicorn

    from rollout.harness.remote import serve_pool
    from rollout.harness.sandboxes import MemoryLeases, SandboxPool
    from rollout_train.presence import presence_of
    from rollout_train.sandboxes import admits, keep, leases_of
    from rollout_train.stores import Stores

    where: str | Stores | None = arguments.ledger
    name: str | None = arguments.name
    if arguments.kind is not None:
        cluster = _cluster_of(arguments.cluster)
        section = cluster.sandboxes.get(arguments.kind)
        if section is None or section.provider is None:
            raise SystemExit(f"the cluster config names no provider of {arguments.kind} sandboxes ([sandboxes])")
        scratch = Path(cluster.scratch).expanduser()  # noqa: ASYNC240 (before it serves)
        directory = arguments.directory or scratch / "sandboxes" / arguments.kind
        directory.mkdir(parents=True, exist_ok=True)
        provider = named(section.provider)(directory, size=section.size, **dict(section.settings))
        where = where or Stores.open(cluster)
        name = name or arguments.kind
    elif arguments.factory is not None:
        provider = named(arguments.factory)(arguments.directory or Path("."))
    else:
        raise SystemExit("say what makes the sandboxes: a factory (module:name), or --kind with --cluster")
    ledger = _ledger_at(where) if where else None
    leases = (leases_of(ledger) if ledger is not None else None) or MemoryLeases()
    admitted = admits(ledger, presence_of(ledger)) if ledger is not None else None
    pool = SandboxPool(provider, name=name or f"{provider.kind}@{socket.gethostname()}", leases=leases, admits=admitted)
    keeping = (
        asyncio.ensure_future(keep(pool, ledger, presence_of(ledger), beat_as=f"pools/{pool.name}"))
        if ledger is not None
        else None
    )
    config = uvicorn.Config(serve_pool(pool), host=arguments.host, port=arguments.port, log_level="warning")
    server = uvicorn.Server(config)
    try:
        await server.serve()
    finally:
        if keeping is not None:
            keeping.cancel()
            await asyncio.gather(keeping, return_exceptions=True)
        await pool.close()


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
    if environment is not None:  # (its eval data, which `rollout suite make NAME --environment …` makes a suite of)
        for each, starts in named(environment).evals().items():
            if each not in listed:
                print(f"{each:<24} {len(starts):>4} starts  {environment}  (eval data, not a suite)")


async def _check(arguments: argparse.Namespace) -> int:
    """Whether an environment holds together, here: its checks, and an episode answered by a scripted model with the
    tool sets and pools given; then, where the settings name a model's channel (a preset, `--provider`), groups played
    by it as a check run on the cluster."""
    from rollout.harness.imports import ToolBinding
    from rollout.harness.sandboxes import Pool, SandboxPool
    from rollout_train.check import checked, scripted

    environment: str = arguments.environment
    offered = named(environment)
    found = checked(offered)
    for each in found:
        print(each, flush=True)
    if not found[0].passed:  # (with no rows, there is nothing to play)
        return 1
    scratch = Path.home() / ".cache" / "rollout" / "checks" / (arguments.name or environment.replace(":", "-"))
    given = dict(_setting(each) for each in arguments.tools)  # (NAME=module:factory, or NAME=URL)
    kinds: dict[str, Any] = dict(_setting(each) for each in arguments.pools)  # (KIND=module:factory, or KIND=URL)
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
            offered, row=arguments.row, reply=arguments.reply, tool_sets=local, tools=urls, pools=local_pools,
            pool_urls=pool_urls,
        )  # fmt: skip
    finally:
        for each in [*local.values(), *local_pools.values()]:
            close = getattr(each, "close", None)
            if close is not None and asyncio.iscoroutine(closing := close()):
                await closing
    print(episode, flush=True)
    found.append(episode)
    passed = all(each.passed for each in found)
    if not (arguments.preset or arguments.provider or arguments.settings or arguments.set):
        return 0 if passed else 1
    flags: dict[str, JsonValue] = {
        "environment": environment, "groups": arguments.groups or 4, "check.episodes": arguments.episodes,
        "seed": arguments.seed,
    }  # fmt: skip
    played = await _asked(arguments, "check", flags, named_as=lambda said: f"check {_named_after(environment)}")
    return played or (0 if passed else 1)


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
    if point:  # the run's directory now says its ledger is the copy (the cluster config should say so too)
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
    directory: Path | None = None,
) -> None:
    """`rollout preset`: list the presets, show one (`NAME` or `NAME@N`), save a version, load a directory of them
    (each `NAME.toml` saved as preset `NAME`, where its newest version holds other settings), or delete one."""
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
    if command == "load":
        assert directory is not None
        for path in sorted(await asyncio.to_thread(lambda: list(directory.glob("*.toml")))):
            said = from_file(path)
            newest = await presets.get(path.stem)
            if newest is not None and dict(newest.settings) == said:
                print(f"{newest.id}: as {path.name} says")
                continue
            saved = await presets.save(path.stem, said, f"loaded from {path.name}")
            print(f"saved {saved.id}: {len(saved.settings)} settings from {path.name}")
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
async def _pause_or_resume(command: str, who: str, where: "str | Stores", arguments: argparse.Namespace) -> None:
    from rollout_train.launching import Refused
    from rollout_train.registry import registry_of, run_id
    from rollout_train.resuming import IN_PLACE, pause, resume

    ledger = _ledger_at(where)
    run = await run_id(registry_of(ledger), who)
    if command == "pause":
        await pause(ledger, run)
        print(f"{who} is paused: what is playing plays out, and nothing new starts")
        return
    cluster = _cluster_of(arguments.cluster) if arguments.cluster is not None else None
    try:
        resumed = await resume(ledger, run, cluster=cluster, preset=arguments.preset)
    except Refused as refused:
        raise SystemExit("; ".join(f"{each.key}: {each.reason}" for each in refused.refusals)) from None
    if resumed.how == IN_PLACE:
        print(f"{who} goes on")
    else:
        assert resumed.launch is not None
        print(f"{who} is launched again: launch {resumed.launch.id}, job {resumed.launch.job} "
              f"({resumed.launch.state})")  # fmt: skip


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


async def _pods(command: str, cluster: "Cluster") -> int:
    """`rollout pods list` and `rollout pods reap`."""
    from rollout_train.pods.leases import pod_leases_of
    from rollout_train.pods.leasing import reap
    from rollout_train.stores import ledger_of

    ledger = ledger_of(cluster)
    if command == "reap":
        for each in await reap(cluster, ledger):
            print(each, flush=True)
        return 0
    store = pod_leases_of(ledger)
    if store is None:
        raise SystemExit("this ledger keeps no pods' leases")
    now = await store.now()
    for lease in await store.all():
        since = lease.released if lease.state == "idle" else lease.held
        age = f"{(now - since) / 60:.0f} min" if since is not None else "-"
        print(f"{lease.pod}\t{lease.provider}\t{lease.state}\t{lease.run or '-'}\t{lease.gpu}\t${lease.price:.2f}/h"
              f"\t{age}", flush=True)  # fmt: skip
    return 0


def _cluster_of(given: str | None) -> "Cluster":
    """The cluster config `--cluster` says (none or `""`: the one this process was handed, else the one
    `rollout_train.cluster.find` finds), read and checked; exits saying what is wrong."""
    from rollout_train.cluster import ClusterError, located

    try:
        return located(given or None)
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


def _monitor_token(cluster: "Cluster | None", host: str, port: int) -> str:
    """The monitor's token: the one the cluster config names (`[monitor] token_env` or `token_file`), else
    `ROLLOUT_MONITOR_TOKEN`; where neither is set, one made up now. Says where the page is, and with a token made up
    here, its sign-in link."""
    from rollout_train.monitor.access import TOKEN_ENV, new_token
    from rollout_train.providers import Secret

    named = cluster.monitor.token if cluster is not None else None
    token = (named or Secret(env=TOKEN_ENV)).resolve()
    if named is not None and token is None:
        raise SystemExit(f"monitor: its token ({named}, the cluster config's [monitor] token) is not set here")
    shown = "localhost" if host in ("0.0.0.0", "::") else f"[{host}]" if ":" in host else host
    page = f"http://{shown}:{port}"
    if token is None:
        token = new_token()
        print(f"the monitor's page: {page}/login?token={token}", file=sys.stderr, flush=True)
    else:
        print(f"the monitor's page: {page}/login?token=TOKEN, its token {named or Secret(env=TOKEN_ENV)}",
              file=sys.stderr, flush=True)  # fmt: skip
    return token


def _check_cluster(given: str | None, role: str = "run") -> int:
    """Say what the cluster config holds and, on this node, what of it that `role` reads does not resolve; 1 if
    something does not."""
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
    problems = inspect(cluster, role=role)
    if cluster.kubernetes is not None:
        import httpx

        from rollout_train.submitting import KubernetesApi, pod_security

        if (KubernetesApi.ACCOUNT / "token").exists():
            try:
                problems += asyncio.run(pod_security(cluster.kubernetes))
            except (RuntimeError, httpx.HTTPError) as error:
                print(f"  the namespace's Pod Security labels were not read: {error}")
        else:
            print("  the namespace's Pod Security labels were not read: no Kubernetes account here (run this in a pod)")
    for problem in problems:
        print(f"  {problem}")
    reads = "everything it names" if role == "run" else f"everything a {role} reads"
    print(f"  {len(problems)} not resolved on this node" if problems else f"  {reads} resolves on this node")
    return 1 if problems else 0


async def _gateway(
    cluster_given: str | None, listen: str | None, certificate: Path | None, private_key: Path | None, proxied: str
) -> None:
    """A replica of the cluster's gateway (`rollout_train.gateway.ChannelDirectory.of`): every channel a run's start
    names on the cluster's providers whose servers it reaches, sampled and recorded, with the keys `[gateway]` names."""
    import contextlib

    import uvicorn

    from rollout_train.gateway import ChannelDirectory, Gateway, Keyring, TurnStore, create_app
    from rollout_train.gateway.beats import about, name_of
    from rollout_train.presence import beating, presence_of
    from rollout_train.stores import Stores

    cluster = _cluster_of(cluster_given)
    stores = Stores.open(cluster)
    listening = listen or cluster.gateway.listen
    host, _, port = listening.rpartition(":")
    keys = cluster.gateway.keys
    if keys is not None and keys.file is not None:
        keyring = Keyring.load(Path(keys.file).expanduser())  # noqa: ASYNC240 (before it serves)
    elif keys is not None and keys.env is not None:
        keyring = Keyring.from_environment({"ROLLOUT_GATEWAY_KEYS": os.environ.get(keys.env, "")})
    else:
        keyring = Keyring.from_environment()
    async with contextlib.AsyncExitStack() as stack:
        directory = ChannelDirectory.of(cluster, stores.ledger)
        stack.callback(directory.close)
        gateway = Gateway(TurnStore(stores.ledger, stores.blobs), keyring, directory=directory)
        presence = presence_of(stores.ledger)
        if presence is not None:  # (the monitor shows the replicas alive)
            scratch = Path(cluster.scratch).expanduser()  # noqa: ASYNC240 (a path, read nothing)
            said_of = lambda: about(gateway, listening, scratch)  # noqa: E731
            beats = asyncio.ensure_future(beating(presence, name_of(listening), said_of))
            stack.callback(beats.cancel)
        config = uvicorn.Config(
            create_app(gateway), host=host, port=int(port), log_level="warning", proxy_headers=True,
            forwarded_allow_ips=proxied, ssl_certfile=str(certificate) if certificate else None,
            ssl_keyfile=str(private_key) if private_key else None,
        )  # fmt: skip
        await uvicorn.Server(config).serve()


def _asks(command: argparse.ArgumentParser, *, channels: bool = True, trains: bool = True) -> None:
    """A command that asks for a run takes its settings in layers: `--preset`, then `--settings`, then `--set`, then
    its flags (with `channels`, `--model`, `--provider` and `--renderer` of `--channel`; with `trains`, `--trainer`);
    the cluster config; and what to do once they are checked."""
    command.add_argument("--name", help="what the run is called (by default a free name after what it plays)")
    if channels:
        command.add_argument("--model", help="the channel's model: channels.CHANNEL.model")
        command.add_argument("--provider", help="the inference provider that samples it: channels.CHANNEL.provider")
        command.add_argument("--renderer", help="the channel's renderer, module:name: channels.CHANNEL.renderer")
        command.add_argument("--channel", help="the channel the flags above are of (policy)")
    if trains:
        command.add_argument("--trainer", help="the trainer, a [trainers.NAME] of the cluster: trainer.provider")
    command.add_argument("--preset", metavar="NAME[@N]", help="start from a preset's settings (`rollout preset`)")
    command.add_argument("--settings", type=Path, metavar="FILE", help="run settings: TOML or JSON, dotted keys")
    command.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="a run setting (trainer.learning_rate=3e-5), the value read as JSON, then TOML, then as text (repeatable; "
        "over the preset and the file)",
    )  # fmt: skip
    command.add_argument("--cluster", **_cluster_option("the cluster config the run is asked for on"))
    doing = command.add_mutually_exclusive_group()
    doing.add_argument("--check", action="store_true", help="say what would be refused, and ask for nothing")
    doing.add_argument("--detach", action="store_true", help="return once the run's job is submitted")
    doing.add_argument("--here", action="store_true", help="run the job in this process, on the cluster's Ray")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rollout", description="Ask for runs on a cluster, train, and watch.")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="a training run, submitted and followed")
    training.add_argument("environment", help="module:name, or a published one as NAME@VERSION")
    training.add_argument("--groups", type=int, help="groups it plays (100)")
    training.add_argument("--groups-per-step", type=int, help="groups a step waits for (4)")
    training.add_argument("--seed", type=int, help="(0)")
    _asks(training)
    reporting = commands.add_parser("report", help="chart a run's progress, and post it to a Discord webhook")
    reporting.add_argument("directory", type=Path)
    reporting.add_argument("environment")
    reporting.add_argument("--watch", action="store_true", help="report again after every group, until interrupted")
    reporting.add_argument("--webhook", help="a Discord webhook (default: the environment's DISCORD_WEBHOOK_URL)")
    imitating = commands.add_parser("imitate", help="a supervised step on a dataset")
    imitating.add_argument("--dataset", help="the dataset to train on, by its name or id (`rollout dataset make`)")
    imitating.add_argument("--start", help="the checkpoint to train from, if the run made none")
    imitating.add_argument("--limit", type=int, help="at most this many segments, drawn at random")
    imitating.add_argument("--seed", type=int, help="(0)")
    imitating.add_argument("--learning-rate", type=float, help="the step's rate (by default 1e-6 full, 1e-4 LoRA)")
    imitating.add_argument("--warmup", type=int, help="updates a fresh optimizer warms up over (4)")
    imitating.add_argument("--passes", type=int, help="passes over the examples (by default enough for 8 updates)")
    imitating.add_argument(
        "--resume-optimizer", action="store_true",
        help="go on from the trainer state of the checkpoint it trains from (by default the optimizer starts afresh)",
    )  # fmt: skip
    _asks(imitating, channels=False)
    presets = commands.add_parser("preset", help="list, show, save, load or delete presets: named, versioned settings")
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
    preset_loading = preset_commands.add_parser(
        "load", help="save each NAME.toml of a directory as preset NAME, where its newest version says otherwise"
    )
    preset_loading.add_argument("directory", type=Path)
    _over_a_ledger(preset_loading)
    preset_deleting = preset_commands.add_parser("delete", help="delete a preset (its versions stay readable)")
    preset_deleting.add_argument("preset", metavar="NAME")
    _over_a_ledger(preset_deleting)
    datasets = commands.add_parser("dataset", help="make or list datasets: examples chosen from runs' episodes")
    dataset_commands = datasets.add_subparsers(dest="dataset_command", required=True)
    dataset_making = dataset_commands.add_parser("make", help="make a dataset by an episode rule and turn filters")
    dataset_making.add_argument(
        "rule", choices=["solved-all", "best-of-group", "capped-per-task", "best-and-worst", "above-and-below"],
        help="an episode rule (examples to imitate), or a preference rule: best-and-worst (pairs), above-and-below "
        "(labelled examples)",
    )  # fmt: skip
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
    monitoring.add_argument(
        "where", nargs="?", help="a run's directory, a ledger's directory, or a database's URL (by default the cluster "
        "config's ledger)",
    )  # fmt: skip
    monitoring.add_argument("--host", default="127.0.0.1")
    monitoring.add_argument("--port", type=int, default=8765)
    monitoring.add_argument(
        "--allow-host", action="append", default=[], metavar="NAME",
        help="another name the page is served under, beside localhost, 127.0.0.1, [::1] and this machine's "
        "(repeatable; *.DOMAIN for every name in a domain)",
    )  # fmt: skip
    monitoring.add_argument(
        "--cluster", **_cluster_option("ask for runs on this cluster config, and import environments from git with it")
    )
    ledgers = commands.add_parser("ledger", help="work with ledgers")
    ledger_commands = ledgers.add_subparsers(dest="ledger_command", required=True)
    copying = ledger_commands.add_parser("copy", help="copy a ledger into a database (SQLite or Postgres)")
    copying.add_argument("source", help="a run's directory, a directory of files, or a database's URL")
    copying.add_argument("target", help="a database's URL: sqlite:///path or postgresql://…")
    copying.add_argument("--point", action="store_true", help="make the source run's directory name the copy")
    ledger_serving = ledger_commands.add_parser(
        "serve", help="serve the cluster config's ledger over HTTP, to whoever holds a token (the ledger service)"
    )
    ledger_serving.add_argument("--cluster", **_cluster_option("the cluster config whose ledger and token it serves"))
    ledger_serving.add_argument("--listen", default="127.0.0.1:8840", help="host:port (127.0.0.1:8840)")
    renaming = commands.add_parser("rename", help="call a run something else (its id stays)")
    renaming.add_argument("who", help="the run, by its name or its id")
    renaming.add_argument("name", help="what it is called from now on")
    _over_a_ledger(renaming)
    pausing = commands.add_parser("pause", help="pause a run: what is playing plays out, and nothing new starts")
    pausing.add_argument("who", help="the run, by its name or its id")
    _over_a_ledger(pausing)
    resuming = commands.add_parser(
        "resume", help="resume a run: a paused one goes on; a stopped, failed or lost one is submitted again"
    )
    resuming.add_argument("who", help="the run, by its name or its id")
    resuming.add_argument(
        "--preset", metavar="NAME[@N]", help="for a run whose start records no providers: a preset that matches it"
    )
    _over_a_ledger(resuming)
    marking = commands.add_parser("bookmark", help="name a checkpoint, move a bookmark, or take one away")
    marking.add_argument("name")
    marking.add_argument("checkpoint", nargs="?", help="a bookmark, RUN:STEP, RUN, or a checkpoint's id or its start")
    marking.add_argument("--delete", action="store_true", help="take the bookmark away (the checkpoint stays)")
    _over_a_ledger(marking)
    merging = commands.add_parser("merge", help="fold a LoRA checkpoint into its base: a full checkpoint of its own")
    merging.add_argument("checkpoint", help="the LoRA checkpoint: a bookmark, RUN:STEP, RUN, or an id or its start")
    merging.add_argument("--base", help="the model to merge into (by default the one it was trained over)")
    merging.add_argument("--merger", default="rollout_lora.merge:merge", help="what folds the adapter in (module:name)")
    merging.add_argument("--bookmark", help="a bookmark to name the merged checkpoint")
    _over_a_ledger(merging)
    evaluating = commands.add_parser("eval", help="play a suite with a checkpoint (or a base model), training nothing")
    evaluating.add_argument("suite", help="the suite, by name (its newest version) or NAME@N")
    evaluating.add_argument(
        "--checkpoint", help="a bookmark, RUN:STEP, RUN, or a checkpoint's id (none: the base model the settings name)"
    )
    evaluating.add_argument("--episodes", type=int, help="episodes of each start (by default the suite's)")
    _asks(evaluating, trains=False)
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
    suite_listing.add_argument("--environment", help="module:name: its eval data not made into a suite yet too")
    environments = commands.add_parser("env", help="work with environments")
    environment_commands = environments.add_subparsers(dest="env_command", required=True)
    checking = environment_commands.add_parser("check", help="whether an environment holds together")
    checking.add_argument("environment", help="module:name")
    checking.add_argument("--row", help="the row the scripted episode plays (by default the first)")
    checking.add_argument("--reply", default="hello", help="what the scripted model says each turn")
    checking.add_argument(
        "--tools", action="append", default=[], metavar="NAME=WHERE",
        help="a tool set its program imports: module:factory, or a URL (repeatable)",
    )  # fmt: skip
    checking.add_argument(
        "--pools", action="append", default=[], metavar="KIND=WHERE",
        help="a pool of the sandboxes its program declares: module:factory of their provider, or a URL (repeatable)",
    )  # fmt: skip
    checking.add_argument("--groups", type=int, help="groups played by the model the settings name (4)")
    checking.add_argument("--episodes", type=int, help="episodes of each group (by default the algorithm's group size)")
    checking.add_argument("--seed", type=int, help="(0)")
    _asks(checking, trains=False)
    listing = commands.add_parser("checkpoints", help="every checkpoint, newest first: where it came from")
    _over_a_ledger(listing)
    serving = commands.add_parser("tools", help="serve a tool set over HTTP")
    serving.add_argument("factory")
    serving.add_argument("--directory", type=Path, default=Path("."))
    serving.add_argument("--host", default="127.0.0.1")
    serving.add_argument("--port", type=int, default=8700)
    pooling = commands.add_parser("pool", help="serve a pool of sandboxes over HTTP")
    pooling.add_argument(
        "factory", nargs="?", help="`module:name` of what makes the sandboxes' provider, called with --directory"
    )
    pooling.add_argument(
        "--kind", help="serve the cluster config's [sandboxes.KIND] (its provider, size and settings) in place of a "
        "factory, its leases beside the cluster's ledger",
    )  # fmt: skip
    pooling.add_argument("--cluster", **_cluster_option("the cluster config whose [sandboxes.KIND] it serves"))
    pooling.add_argument(
        "--directory", type=Path,
        help="where the provider keeps its state (by default this directory, or with --kind [scratch]/sandboxes/KIND)",
    )  # fmt: skip
    pooling.add_argument(
        "--ledger",
        help="keep the leases beside this ledger, ending with their claims: a run's directory, a ledger's "
        "directory, or a database's URL (without it, they are kept in the process, and end only when released)",
    )
    pooling.add_argument("--name", help="what the pool is called among those sharing the ledger (KIND@HOST, or KIND)")
    pooling.add_argument("--host", default="127.0.0.1")
    pooling.add_argument("--port", type=int, default=8710)
    gateway = commands.add_parser("gateway", help="serve a replica of the gateway, which records every turn")
    gateway.add_argument("--cluster", **_cluster_option("the cluster config: its stores, keys and providers"))
    gateway.add_argument("--listen", help="host:port, in place of the cluster config's [gateway] listen")
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
        "check",
        help="read the cluster config, and say which of its secrets and projects do not resolve on this node "
        "(and, in a pod, whether its namespace is labelled for Pod Security)",
    )
    cluster_checking.add_argument("--cluster", **_cluster_option("the cluster config"))
    cluster_checking.add_argument(
        "--role", default="run", choices=["run", "gateway", "monitor", "ledger", "pool", "reaper"],
        help="check only the secrets this role reads (run: every secret, the default)",
    )  # fmt: skip
    pki = commands.add_parser("pki", help="the certificates the platform holds, from the cluster's step-ca")
    pki_commands = pki.add_subparsers(dest="pki_command", required=True)
    pki_publishing = pki_commands.add_parser(
        "publish", help="write step-ca's root and provisioner key, and a new gateway certificate, as Secrets"
    )
    pki_publishing.add_argument("--ca-directory", type=Path, default=Path("/home/step"), help="step-ca's state")
    pki_publishing.add_argument("--url", required=True, help="where step-ca answers (https://step-ca.NAMESPACE:9000)")
    pki_publishing.add_argument("--namespace", required=True, help="where the Secrets are written")
    pki_publishing.add_argument("--secret", default="step-ca", help="the root and the provisioner's key (step-ca)")
    pki_publishing.add_argument("--tls-secret", default="gateway-tls", help="the gateway's certificate (gateway-tls)")
    pki_publishing.add_argument("--provisioner", help="the JWK provisioner, by name (its first)")
    pods = commands.add_parser("pods", help="the GPU pods runs rent on RunPod: their leases, and the reaper")
    pod_commands = pods.add_subparsers(dest="pod_command", required=True)
    pod_listing = pod_commands.add_parser("list", help="every pod's lease: who holds it, since when, at what price")
    pod_listing.add_argument("--cluster", **_cluster_option("the cluster config"))
    pod_reaping = pod_commands.add_parser(
        "reap", help="delete the pods no run holds: idle past their idle stop, stale leases', and those no lease names"
    )
    pod_reaping.add_argument("--cluster", **_cluster_option("the cluster config"))
    arguments = parser.parse_args()
    if arguments.command == "cluster":
        sys.exit(_check_cluster(arguments.cluster, arguments.role))
    if arguments.command == "pods":
        sys.exit(asyncio.run(_pods(arguments.pod_command, _cluster_of(arguments.cluster))))
    if arguments.command == "pki":
        from rollout_train.pki import publish

        password = os.environ.get("STEP_CA_PASSWORD", "")
        if not password:
            raise SystemExit("STEP_CA_PASSWORD is not set: it opens the provisioner's key")
        print(asyncio.run(publish(arguments.ca_directory, password, arguments.url, arguments.namespace,
                                  secret=arguments.secret, tls_secret=arguments.tls_secret,
                                  provisioner=arguments.provisioner)), flush=True)  # fmt: skip
        return
    asking = {"train": _train, "eval": _evaluate, "imitate": _imitate}
    if arguments.command in asking:
        sys.exit(asyncio.run(until_signalled(asking[arguments.command](arguments))))
    if arguments.command == "merge":
        asyncio.run(
            _merge(arguments.checkpoint, _ledger_of(arguments), arguments.base, arguments.merger, arguments.bookmark)
        )
        return
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
        sys.exit(asyncio.run(until_signalled(_check(arguments))))
    if arguments.command == "preset":
        asyncio.run(_preset(
            arguments.preset_command, _ledger_of(arguments), getattr(arguments, "preset", None),
            getattr(arguments, "from_run", None), getattr(arguments, "settings", None), getattr(arguments, "set", None),
            getattr(arguments, "note", ""), getattr(arguments, "directory", None),
        ))  # fmt: skip
        return
    if arguments.command == "rename":
        asyncio.run(_rename(arguments.who, arguments.name, _ledger_of(arguments)))
        return
    if arguments.command in ("pause", "resume"):
        if arguments.command == "pause":
            arguments.preset = None
        asyncio.run(_pause_or_resume(arguments.command, arguments.who, _ledger_of(arguments), arguments))
        return
    if arguments.command == "bookmark":
        if arguments.checkpoint is None and not arguments.delete:
            parser.error("bookmark: name a checkpoint, or --delete")
        asyncio.run(_bookmark(arguments.name, arguments.checkpoint, arguments.delete, _ledger_of(arguments)))
        return
    if arguments.command == "checkpoints":
        asyncio.run(_checkpoints(_ledger_of(arguments)))
        return
    if arguments.command == "gateway":
        work = _gateway(arguments.cluster, arguments.listen, arguments.certificate, arguments.private_key,
                        arguments.proxied)  # fmt: skip
        sys.exit(asyncio.run(until_signalled(work)))
    if arguments.command == "pool":
        sys.exit(asyncio.run(until_signalled(_pool(arguments))))
    if arguments.command == "ledger" and arguments.ledger_command == "serve":
        import uvicorn

        from rollout_train.ledger_service.service import for_cluster

        host, _, port = arguments.listen.rpartition(":")
        uvicorn.run(for_cluster(_cluster_of(arguments.cluster)), host=host or "127.0.0.1", port=int(port),
                    log_level="warning")  # fmt: skip
        return
    if arguments.command == "ledger":
        asyncio.run(_copy_ledger(arguments.source, arguments.target, arguments.point))
        return
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

            cluster = _cluster_of(arguments.cluster) if arguments.cluster is not None else None
            where = arguments.where
            if where is None:
                if cluster is None:
                    parser.error("monitor: say WHERE, or --cluster for the cluster config's ledger")
                from rollout_train.stores import ledger_url

                where = ledger_url(cluster)
            importer = Importer.of(cluster) if cluster is not None else None
            token = _monitor_token(cluster, arguments.host, arguments.port)
            app = create_app(where, importer=importer, cluster=cluster, token=token, hosts=arguments.allow_host)
        else:
            from rollout.harness.remote import serve

            app = serve(named(arguments.factory)(arguments.directory))
        uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="warning")


if __name__ == "__main__":
    main()
