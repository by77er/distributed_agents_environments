"""`rollout`: train on an environment under a deployment profile, and watch.

rollout train PROFILE ENVIRONMENT   the training loop: PROFILE is a TOML file (`rollout_train.profile`), ENVIRONMENT
                                    names an environment as `module:name`
rollout report RUN ENVIRONMENT      chart a run's progress and summarise it; post both to a Discord webhook
rollout imitate PROFILE             a supervised step on the run's solved episodes, without their guidance
rollout monitor WHERE               the web page over a ledger and every run in it (WHERE: a run's directory, a ledger)
rollout ledger copy FROM TO         copy a ledger (a run's, files, or a database) into a database: SQLite or Postgres
rollout tools FACTORY               serve an environment's tool set over HTTP: FACTORY is `module:name`

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

    from rollout.environment import binding_for
    from rollout_train import train
    from rollout_train.evals import Schedule, suite_of
    from rollout_train.profile import Profile

    described = dataclasses.replace(Profile.load(profile, directory=directory, settings=settings), name=name)
    if described.trainer is None:
        raise SystemExit(f"{profile} describes no trainer")
    channel, rows = described.trainer.channel, named(environment)
    where, profiled = await asyncio.to_thread(described.directory.absolute), await asyncio.to_thread(profile.absolute)
    started: dict[str, Any] = {"directory": str(where), "profile": str(profiled), "address": monitor}  # (its `starts`)
    async with described.open() as platform:
        assert platform.trainer is not None
        started["blobs"] = platform.blobs_at  # (where the monitor reads the run's finished episodes)
        binding = binding_for(rows, channel, platform.tool_bindings)
        schedule: Schedule | None = None
        if (asked := described.evals) is not None:
            suite = await suite_of(platform.ledger, asked.suite)
            if suite is None:
                raise SystemExit(f"there is no suite {asked.suite!r}: make one with `rollout suite make`")
            played = named(suite.environment)
            schedule = Schedule(
                suite, played, platform.eval_run, asked.every, asked.episodes,
                binding_for(played, channel, platform.tool_bindings),
            )  # fmt: skip
        await train(
            rows, platform.trainer, platform.checkpoints, start=platform.origin, channel=channel,
            base=described.channels[channel].model,
            directory=described.directory / "checkpoints", publish=platform.publish, groups=groups,
            groups_per_step=groups_per_step, seed=seed, episodes_at_once=described.episodes_at_once, binding=binding,
            run=platform.run.id, started=started, hooks=[platform.feed], kept=platform.bookmarked, made=platform.made,
            reshard=platform.reshard if platform.layout else None, evals=schedule,
        )  # fmt: skip


async def _evaluate(
    profile: Path,
    suite_name: str,
    reference: str | None,
    episodes: int,
    directory: Path | None,
    monitor: str | None = None,
    name: str | None = None,
    settings: dict[str, Any] | None = None,
) -> None:
    import dataclasses
    import shutil

    from rollout.environment import binding_for
    from rollout_train.evals import evaluate, suite_of
    from rollout_train.profile import Profile
    from rollout_train.registry import resolved

    described = dataclasses.replace(Profile.load(profile, directory=directory, settings=settings), name=name)
    if described.trainer is not None:  # (so the engines hold what the checkpoint is served over: full weights, say)
        trainer = dataclasses.replace(described.trainer, start=reference, bookmark=None)
        described = dataclasses.replace(described, trainer=trainer)
    channel = described.trainer.channel if described.trainer else next(iter(described.channels))
    where, profiled = await asyncio.to_thread(described.directory.absolute), await asyncio.to_thread(profile.absolute)
    started: dict[str, Any] = {"directory": str(where), "profile": str(profiled), "address": monitor}
    try:
        async with described.open(training=False) as platform:  # (no trainer: nothing is trained)
            suite = await suite_of(platform.ledger, suite_name)
            if suite is None:
                raise SystemExit(f"there is no suite {suite_name!r}: make one with `rollout suite make`")
            try:
                subject = await resolved(platform.ledger, platform.registry, reference) if reference else None
            except KeyError as error:
                raise SystemExit(error.args[0]) from None
            rows = named(suite.environment)
            started["blobs"] = platform.blobs_at
            said = await evaluate(
                rows, platform.checkpoints, run=platform.run.id, suite=suite, subject=subject,
                base=described.channels[channel].model, channel=channel,
                directory=described.directory / "checkpoints", publish=platform.publish, episodes=episodes,
                binding=binding_for(rows, channel, platform.tool_bindings), started=started,
                reshard=platform.reshard if platform.layout else None, hooks=[platform.feed],
            )  # fmt: skip
    finally:  # (the files fetched to serve the checkpoint are needed only while it plays; a full one is a whole model)
        for fetched in ("bases", "checkpoints", "resharding"):
            await asyncio.to_thread(shutil.rmtree, described.directory / fetched, ignore_errors=True)
    print(f"{suite_name}: solved {said['solved']} of {said['played']} episodes (mean reward {said['reward']})")


async def _suite(
    command: str, where: str, name: str | None, environment: str | None, rows: str | None, seeds: str
) -> None:
    from rollout_train.evals import make_suite, suite_of, suites_in

    ledger = _ledger_at(where)
    if command == "make":
        assert name is not None and environment is not None
        keys = [each.strip() for each in rows.split(",") if each.strip()] if rows else None
        try:
            numbers = [int(each) for each in seeds.split(",") if each.strip()]
            made = await make_suite(ledger, name, environment, named(environment), rows=keys, seeds=numbers)
        except ValueError as error:
            raise SystemExit(str(error)) from None
        print(f"the suite {made.name}: {len(made.starts)} starts of {environment}")
        return
    for each in await suites_in(ledger):
        found = await suite_of(ledger, each)
        assert found is not None
        print(f"{each:<24} {len(found.starts):>4} starts  {found.environment}")


async def _imitate(profile: Path, directory: Path | None, kinds: list[str], limit: int | None, seed: int) -> None:
    from rollout.harness.blobs import FileBlobStore
    from rollout_train.checkpoints import Checkpoints
    from rollout_train.imitation import examples, imitate
    from rollout_train.layout import BLOBS, LEDGER
    from rollout_train.ledger import opened
    from rollout_train.profile import Profile
    from rollout_train.record import scope
    from rollout_train.registry import registry_of, resolved, run_of

    described = Profile.load(profile, directory=directory)
    if described.trainer is None:
        raise SystemExit(f"{profile} describes no trainer")
    spec = described.channels[described.trainer.channel]
    renderer = named(spec.renderer)(spec.model)
    store = dict(described.blobs)
    blobs = named(store.pop("kind"))(**store) if store else FileBlobStore(described.directory / BLOBS)
    ledger = opened(dict(described.ledger) or {"directory": str(described.directory / LEDGER)})
    checkpoints, registry = Checkpoints(ledger, blobs), registry_of(ledger)
    run = await run_of(described.directory, ledger, registry)
    start = await resolved(ledger, registry, described.trainer.start) if described.trainer.start else None
    taught = await examples(ledger, run.id, blobs, renderer, kinds=kinds)
    if not taught.segments:
        raise SystemExit("no solved episode of the run carried that guidance")
    print(f"{len(taught.segments)} segments of {taught.episodes} episodes ({taught.left_out} left out)", flush=True)
    settings = {**described.trainer.settings, "objective": "likelihood"}
    trainer = named(described.trainer.kind)(spec.model, **settings)
    fence = await ledger.take(scope(run.id))  # (the run is stopped: imitation writes as it)
    checkpoint = await imitate(
        checkpoints, trainer, taught, fence=fence, run=run.id, start=start, base=spec.model,
        directory=described.directory / "checkpoints",
        limit=limit, seed=seed,
    )  # fmt: skip
    print(f"made {checkpoint.id}: {json.dumps({key: round(value, 4) for key, value in checkpoint.metrics.items()})}")


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
    where: str, profiles: Path, environments: list[str], runs: Path, at_once: int, ray: str | None, gpus: float
) -> None:
    from rollout_train.launcher import Launcher, name_of
    from rollout_train.launches import launches_of
    from rollout_train.presence import presence_of

    ledger = _ledger_at(where)
    launches, presence = launches_of(ledger), presence_of(ledger)
    if launches is None or presence is None:
        raise SystemExit(f"the ledger at {where} keeps no launches or heartbeats beside it")
    profiles, runs = await asyncio.to_thread(profiles.expanduser), await asyncio.to_thread(runs.expanduser)
    found = Launcher(name_of(), launches, presence, profiles, environments, runs, at_once=at_once, ray=ray, gpus=gpus)
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
    imitating = commands.add_parser("imitate", help="a supervised step on solved episodes, without their guidance")
    imitating.add_argument("profile", type=Path)
    imitating.add_argument("--directory", type=Path, help="the run's directory (instead of the profile's)")
    imitating.add_argument("--without", nargs="+", default=["way"], help="the kinds of guidance to take out")
    imitating.add_argument("--limit", type=int, help="at most this many segments, drawn at random")
    imitating.add_argument("--seed", type=int, default=0)
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
    evaluating.add_argument("--episodes", type=int, default=1, help="episodes of each start (1)")
    evaluating.add_argument("--directory", type=Path, help="the eval's directory (instead of the profile's)")
    evaluating.add_argument("--name", help="what the eval is called (by default its directory's name)")
    evaluating.add_argument("--monitor", help="where the monitor on this machine serves, as other machines reach it")
    evaluating.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="change a profile setting")
    suites = commands.add_parser("suite", help="make or list evaluation suites")
    suite_commands = suites.add_subparsers(dest="suite_command", required=True)
    making = suite_commands.add_parser("make", help="make a suite: a start of each row for each seed, frozen")
    making.add_argument("name")
    making.add_argument("--environment", required=True, help="module:name")
    making.add_argument("--rows", help="row keys, comma-separated (by default every row)")
    making.add_argument("--seeds", required=True, help="seeds, comma-separated: each row is started once with each")
    making.add_argument("--ledger", default=".", help=where)
    suite_listing = suite_commands.add_parser("list", help="every suite")
    suite_listing.add_argument("--ledger", default=".", help=where)
    listing = commands.add_parser("checkpoints", help="every checkpoint, newest first: where it came from")
    listing.add_argument("--ledger", default=".", help=where)
    serving = commands.add_parser("tools", help="serve a tool set over HTTP")
    serving.add_argument("factory")
    serving.add_argument("--directory", type=Path, default=Path("."))
    serving.add_argument("--host", default="127.0.0.1")
    serving.add_argument("--port", type=int, default=8700)
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
            arguments.monitor, arguments.name, dict(_setting(each) for each in arguments.set),
        )  # fmt: skip
        sys.exit(asyncio.run(until_signalled(work)))
    if arguments.command == "suite":
        made = arguments.suite_command == "make"
        asyncio.run(_suite(
            arguments.suite_command, arguments.ledger, arguments.name if made else None,
            arguments.environment if made else None, arguments.rows if made else None, arguments.seeds if made else "",
        ))  # fmt: skip
        return
    if arguments.command == "rename":
        asyncio.run(_rename(arguments.who, arguments.name, arguments.ledger))
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
            arguments.ray, arguments.gpus,
        )  # fmt: skip
        sys.exit(asyncio.run(until_signalled(work)))
    if arguments.command == "checkpoints":
        asyncio.run(_checkpoints(arguments.ledger))
        return
    if arguments.command == "ledger":
        asyncio.run(_copy_ledger(arguments.source, arguments.target, arguments.point))
        return
    if arguments.command == "imitate":
        work = _imitate(arguments.profile, arguments.directory, arguments.without, arguments.limit, arguments.seed)
        sys.exit(asyncio.run(until_signalled(work)))
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
