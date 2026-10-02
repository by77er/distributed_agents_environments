"""`rollout`: train on a catalog under a deployment profile, and watch.

rollout train PROFILE CATALOG    the training loop: PROFILE is a TOML file (`rollout.profile`), CATALOG names an
                                 environment's catalog as `module:name`
rollout report RUN CATALOG       chart a run's progress and summarise it; post both to a Discord webhook
rollout monitor FEED             the web page over a run's feed (RUN/feed)
rollout tools FACTORY            serve an environment's tool set over HTTP: FACTORY is `module:name`

`rollout COMMAND --help` lists each command's options.
"""

import argparse
import asyncio
import os
import signal
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from rollout.names import named


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


async def _train(profile: Path, directory: Path | None, catalog: str, groups: int, seed: int) -> None:
    from rollout.profile import Profile
    from rollout.rollouts import binding_for
    from rollout.training import train

    described = Profile.load(profile, directory=directory)
    if described.trainer is None:
        raise SystemExit(f"{profile} describes no trainer")
    channel, rows = described.trainer.channel, named(catalog)
    async with described.open() as platform:
        assert platform.trainer is not None
        binding = binding_for(rows, channel, platform.tool_bindings)
        await train(
            platform.jobs, rows, platform.trainer, platform.store, channel=channel, groups=groups, seed=seed,
            binding=binding,
        )  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(prog="rollout", description="Train on a catalog under a deployment profile.")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="run the training loop")
    training.add_argument("profile", type=Path)
    training.add_argument("catalog")
    training.add_argument("--directory", type=Path, help="the run's directory (instead of the profile's)")
    training.add_argument("--groups", type=int, default=100)
    training.add_argument("--seed", type=int, default=0)
    reporting = commands.add_parser("report", help="chart a run's progress, and post it to a Discord webhook")
    reporting.add_argument("directory", type=Path)
    reporting.add_argument("catalog")
    reporting.add_argument("--watch", action="store_true", help="report again after every group, until interrupted")
    reporting.add_argument("--webhook", help="a Discord webhook (default: the environment's DISCORD_WEBHOOK_URL)")
    monitoring = commands.add_parser("monitor", help="serve the monitor's page over a feed directory")
    monitoring.add_argument("directory", type=Path)
    monitoring.add_argument("--host", default="127.0.0.1")
    monitoring.add_argument("--port", type=int, default=8765)
    serving = commands.add_parser("tools", help="serve a tool set over HTTP")
    serving.add_argument("factory")
    serving.add_argument("--directory", type=Path, default=Path("."))
    serving.add_argument("--host", default="127.0.0.1")
    serving.add_argument("--port", type=int, default=8700)
    arguments = parser.parse_args()
    if arguments.command == "train":
        work = _train(arguments.profile, arguments.directory, arguments.catalog, arguments.groups, arguments.seed)
        sys.exit(asyncio.run(until_signalled(work)))
    if arguments.command == "report":
        from rollout.training.report import report

        webhook = arguments.webhook or os.environ.get("DISCORD_WEBHOOK_URL")
        asyncio.run(report(arguments.directory, named(arguments.catalog).rows(), webhook, watch=arguments.watch))
    if arguments.command in ("monitor", "tools"):
        import uvicorn

        if arguments.command == "monitor":
            from rollout.monitor.app import create_app

            app = create_app(arguments.directory)
        else:
            from rollout.core.harness.remote import serve

            app = serve(named(arguments.factory)(arguments.directory))
        uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="warning")


if __name__ == "__main__":
    main()
