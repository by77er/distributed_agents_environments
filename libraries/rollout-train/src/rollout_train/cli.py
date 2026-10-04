"""`rollout`: train on a catalog under a deployment profile, and watch.

rollout train PROFILE CATALOG    the training loop: PROFILE is a TOML file (`rollout_train.profile`), CATALOG names an
                                 environment's catalog as `module:name`
rollout report RUN CATALOG       chart a run's progress and summarise it; post both to a Discord webhook
rollout imitate PROFILE          a supervised step on the run's solved episodes, without their guidance
rollout monitor WHERE            the web page over a ledger and every run in it (WHERE: a run's directory, a ledger)
rollout ledger copy FROM TO      copy a ledger (a run's, files, or a database) into a database: SQLite or Postgres
rollout tools FACTORY            serve an environment's tool set over HTTP: FACTORY is `module:name`

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
    catalog: str,
    groups: int,
    groups_per_step: int,
    seed: int,
    monitor: str | None = None,
    name: str | None = None,
    settings: dict[str, Any] | None = None,
) -> None:
    import dataclasses

    from rollout.catalog import binding_for
    from rollout_train import train
    from rollout_train.profile import Profile

    described = dataclasses.replace(Profile.load(profile, directory=directory, settings=settings), name=name)
    if described.trainer is None:
        raise SystemExit(f"{profile} describes no trainer")
    channel, rows = described.trainer.channel, named(catalog)
    where, profiled = await asyncio.to_thread(described.directory.absolute), await asyncio.to_thread(profile.absolute)
    started: dict[str, Any] = {"directory": str(where), "profile": str(profiled), "address": monitor}  # (its `starts`)
    async with described.open() as platform:
        assert platform.trainer is not None
        started["blobs"] = platform.blobs_at  # (where the monitor reads the run's finished episodes)
        binding = binding_for(rows, channel, platform.tool_bindings)
        await train(
            rows, platform.trainer, platform.versions, start=platform.origin, channel=channel,
            base=described.channels[channel].model,
            directory=described.directory / "versions", publish=platform.publish, groups=groups,
            groups_per_step=groups_per_step, seed=seed, episodes_at_once=described.episodes_at_once, binding=binding,
            run=platform.run.id, started=started, hooks=[platform.feed], kept=platform.bookmarked, made=platform.made,
            reshard=platform.reshard if platform.layout else None,
        )  # fmt: skip


async def _imitate(profile: Path, directory: Path | None, kinds: list[str], limit: int | None, seed: int) -> None:
    from rollout.harness.blobs import FileBlobStore
    from rollout_train.imitation import examples, imitate
    from rollout_train.layout import BLOBS, LEDGER
    from rollout_train.ledger import opened
    from rollout_train.profile import Profile
    from rollout_train.record import scope
    from rollout_train.registry import registry_of, resolved, run_of
    from rollout_train.versions import Versions

    described = Profile.load(profile, directory=directory)
    if described.trainer is None:
        raise SystemExit(f"{profile} describes no trainer")
    spec = described.channels[described.trainer.channel]
    renderer = named(spec.renderer)(spec.model)
    store = dict(described.blobs)
    blobs = named(store.pop("kind"))(**store) if store else FileBlobStore(described.directory / BLOBS)
    ledger = opened(dict(described.ledger) or {"directory": str(described.directory / LEDGER)})
    versions, registry = Versions(ledger, blobs), registry_of(ledger)
    run = await run_of(described.directory, ledger, registry)
    start = await resolved(ledger, registry, described.trainer.start) if described.trainer.start else None
    taught = await examples(ledger, run.id, blobs, renderer, kinds=kinds)
    if not taught.segments:
        raise SystemExit("no solved episode of the run carried that guidance")
    print(f"{len(taught.segments)} segments of {taught.episodes} episodes ({taught.left_out} left out)", flush=True)
    settings = {**described.trainer.settings, "objective": "likelihood"}
    trainer = named(described.trainer.kind)(spec.model, **settings)
    fence = await ledger.take(scope(run.id))  # (the run is stopped: imitation writes as it)
    version = await imitate(
        versions, trainer, taught, fence=fence, run=run.id, start=start, base=spec.model,
        directory=described.directory / "versions",
        limit=limit, seed=seed,
    )  # fmt: skip
    print(f"made {version.id}: {json.dumps({key: round(value, 4) for key, value in version.metrics.items()})}")


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
    where: str, profiles: Path, catalogs: list[str], runs: Path, at_once: int, ray: str | None, gpus: float
) -> None:
    from rollout_train.launcher import Launcher, name_of
    from rollout_train.launches import launches_of
    from rollout_train.presence import presence_of

    ledger = _ledger_at(where)
    launches, presence = launches_of(ledger), presence_of(ledger)
    if launches is None or presence is None:
        raise SystemExit(f"the ledger at {where} keeps no launches or heartbeats beside it")
    profiles, runs = await asyncio.to_thread(profiles.expanduser), await asyncio.to_thread(runs.expanduser)
    found = Launcher(name_of(), launches, presence, profiles, catalogs, runs, at_once=at_once, ray=ray, gpus=gpus)
    await found.serve()


def _as_job(ray: str, given: list[str]) -> None:
    """Submit this launcher (the command as given, without `--as-job`) as a Ray job, unless one already runs here."""
    import shlex
    import socket

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
        version = await resolved(ledger, registry, reference or "")
        if version is None:
            raise SystemExit("a bookmark names a version, not the base model")
        await registry.bookmark(name, version)
    except (KeyError, Taken) as error:
        raise SystemExit(error.args[0]) from None
    print(f"{name} is {version}")


async def _versions(where: str) -> None:
    from rollout_train.registry import names
    from rollout_train.versions import short, versions_in

    ledger, registry = _registry_at(where)
    every, called = await versions_in(ledger), await names(registry)
    shown = short(version.id for version in every)
    marks: dict[str, list[str]] = {}
    for mark, version in called["bookmarks"].items():
        marks.setdefault(version, []).append(mark)
    for version in sorted(every, key=lambda each: (each.made, each.depth), reverse=True):
        origin = called["runs"].get(version.run, version.run) if version.run else "made outside a run"
        at = f":{version.step}" if version.step is not None else ""
        parents = ", ".join(shown.get(parent, parent) for parent in version.parents) or "the base model"
        kept = "" if version.weights is not None else "  (released)"
        bookmarked = f"  [{', '.join(marks[version.id])}]" if version.id in marks else ""
        print(f"{shown[version.id]:<8} depth {version.depth:<4} {origin}{at}  from {parents}{bookmarked}{kept}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rollout", description="Train on a catalog under a deployment profile.")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="run the training loop")
    training.add_argument("profile", type=Path)
    training.add_argument("catalog")
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
    reporting.add_argument("catalog")
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
    marking = commands.add_parser("bookmark", help="name a version, move a bookmark, or take one away")
    marking.add_argument("name")
    marking.add_argument("version", nargs="?", help="a bookmark, RUN:STEP, RUN, or a version's id or its start")
    marking.add_argument("--delete", action="store_true", help="take the bookmark away (the version stays)")
    marking.add_argument("--ledger", default=".", help=where)
    launching = commands.add_parser("launcher", help="start the training runs asked for that this machine can run")
    launching.add_argument("--ledger", required=True, help="the database's URL (or a ledger's directory)")
    launching.add_argument("--profiles", type=Path, required=True, help="a directory of profiles it offers")
    launching.add_argument("--catalog", action="append", default=[], help="a catalog it offers (repeatable)")
    launching.add_argument("--runs", type=Path, required=True, help="where it makes each run's directory")
    launching.add_argument("--at-once", type=int, default=1, help="runs it plays at once (1: one GPU)")
    launching.add_argument("--ray", help="a Ray cluster's job server (http://127.0.0.1:8265): each run is a Ray job")
    launching.add_argument("--gpus", type=float, default=1.0, help="accelerators each run's Ray job asks for (1)")
    launching.add_argument("--as-job", action="store_true", help="submit the launcher itself as a Ray job (with --ray)")
    listing = commands.add_parser("versions", help="every version, newest first: where it came from")
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
            arguments.catalog,
            arguments.groups,
            arguments.groups_per_step,
            arguments.seed,
            arguments.monitor,
            arguments.name,
            dict(_setting(each) for each in arguments.set),
        )
        sys.exit(asyncio.run(until_signalled(work)))
    if arguments.command == "rename":
        asyncio.run(_rename(arguments.who, arguments.name, arguments.ledger))
        return
    if arguments.command == "bookmark":
        if arguments.version is None and not arguments.delete:
            parser.error("bookmark: name a version, or --delete")
        asyncio.run(_bookmark(arguments.name, arguments.version, arguments.delete, arguments.ledger))
        return
    if arguments.command == "launcher":
        if arguments.as_job:
            if not arguments.ray:
                parser.error("launcher --as-job: say the Ray cluster with --ray")
            _as_job(arguments.ray, sys.argv[1:])
            return
        work = _launcher(
            arguments.ledger, arguments.profiles, arguments.catalog, arguments.runs, arguments.at_once,
            arguments.ray, arguments.gpus,
        )  # fmt: skip
        sys.exit(asyncio.run(until_signalled(work)))
    if arguments.command == "versions":
        asyncio.run(_versions(arguments.ledger))
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
        asyncio.run(report(arguments.directory, named(arguments.catalog).rows(), webhook, watch=arguments.watch))
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
