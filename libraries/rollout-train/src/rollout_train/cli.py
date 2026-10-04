"""`rollout`: train on a catalog under a deployment profile, and watch.

rollout train PROFILE CATALOG    the training loop: PROFILE is a TOML file (`rollout_train.profile`), CATALOG names an
                                 environment's catalog as `module:name`
rollout report RUN CATALOG       chart a run's progress and summarise it; post both to a Discord webhook
rollout imitate PROFILE          a supervised step on the solved episodes of the run's log, without their guidance
rollout monitor RUN              the web page over a run's directory: where it stands, and every episode
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
    profile: Path, directory: Path | None, catalog: str, groups: int, groups_per_step: int, seed: int
) -> None:
    from rollout.catalog import binding_for
    from rollout_train import train
    from rollout_train.profile import Profile

    described = Profile.load(profile, directory=directory)
    if described.trainer is None:
        raise SystemExit(f"{profile} describes no trainer")
    channel, rows = described.trainer.channel, named(catalog)
    async with described.open() as platform:
        assert platform.trainer is not None
        binding = binding_for(rows, channel, platform.tool_bindings)
        await train(
            platform.jobs, rows, platform.trainer, platform.policies, policy=platform.policy, channel=channel,
            directory=described.directory / "versions", groups=groups, groups_per_step=groups_per_step, seed=seed,
            episodes_at_once=described.episodes_at_once, binding=binding, run=described.directory.name,
        )  # fmt: skip


async def _imitate(profile: Path, directory: Path | None, kinds: list[str], limit: int | None, seed: int) -> None:
    from rollout.harness.blobs import FileBlobStore
    from rollout_train.imitation import examples, imitate
    from rollout_train.layout import BLOBS, JOBS, LEDGER
    from rollout_train.ledger import opened
    from rollout_train.policies import Policies
    from rollout_train.profile import Profile

    described = Profile.load(profile, directory=directory)
    if described.trainer is None:
        raise SystemExit(f"{profile} describes no trainer")
    spec = described.channels[described.trainer.channel]
    renderer = named(spec.renderer)(spec.model)
    store = dict(described.blobs)
    blobs = named(store.pop("kind"))(**store) if store else FileBlobStore(described.directory / BLOBS)
    policies = Policies(opened(dict(described.ledger) or {"directory": str(described.directory / LEDGER)}), blobs)
    policy = described.trainer.policy or described.directory.name
    taught = None
    for log in sorted((described.directory / JOBS).iterdir()):
        found = await examples(log, blobs, renderer, kinds=kinds)
        if taught is None:
            taught = found
        else:
            taught.segments += found.segments
            taught.episodes += found.episodes
            taught.left_out += found.left_out
    if taught is None or not taught.segments:
        raise SystemExit("no solved episode in the run's log carried that guidance")
    print(f"{len(taught.segments)} segments of {taught.episodes} episodes ({taught.left_out} left out)", flush=True)
    settings = {**described.trainer.settings, "objective": "likelihood"}
    trainer = named(described.trainer.kind)(spec.model, **settings)
    writer = await policies.writer(policy)
    version = await imitate(
        policies, trainer, taught, fence=writer, policy=policy, directory=described.directory / "versions",
        limit=limit, seed=seed,
    )  # fmt: skip
    print(f"made {version.name}: {json.dumps({key: round(value, 4) for key, value in version.metrics.items()})}")


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
    monitoring = commands.add_parser("monitor", help="serve the monitor's page over a run's directory")
    monitoring.add_argument("directory", type=Path)
    monitoring.add_argument("--host", default="127.0.0.1")
    monitoring.add_argument("--port", type=int, default=8765)
    ledgers = commands.add_parser("ledger", help="work with ledgers")
    ledger_commands = ledgers.add_subparsers(dest="ledger_command", required=True)
    copying = ledger_commands.add_parser("copy", help="copy a ledger into a database (SQLite or Postgres)")
    copying.add_argument("source", help="a run's directory, a directory of files, or a database's URL")
    copying.add_argument("target", help="a database's URL: sqlite:///path or postgresql://…")
    copying.add_argument("--point", action="store_true", help="make the source run's directory name the copy")
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
        )
        sys.exit(asyncio.run(until_signalled(work)))
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

            app = create_app(arguments.directory)
        else:
            from rollout.harness.remote import serve

            app = serve(named(arguments.factory)(arguments.directory))
        uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="warning")


if __name__ == "__main__":
    main()
