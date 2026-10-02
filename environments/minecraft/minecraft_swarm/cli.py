"""`minecraft-swarm`: run the Minecraft swarm environment.

minecraft-swarm server [--seed N]     start a temporary server and keep it up (Ctrl-C stops and deletes it)
minecraft-swarm train RUN             train the swarm on the curriculum
minecraft-swarm report RUN [--watch]  chart a run's progress and summarise it; post both to a Discord webhook
"""

import argparse
import asyncio
import contextlib
import json
import os
from pathlib import Path

from minecraft_swarm.paper import Installation, PaperServer


def main() -> None:
    parser = argparse.ArgumentParser(prog="minecraft-swarm", description="The Minecraft swarm environment.")
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("server", help="start a temporary server and keep it up")
    server.add_argument("--seed", type=int, default=12345)
    server.add_argument("--keep", action="store_true", help="keep the server's directory when it stops")
    training = commands.add_parser("train", help="train the swarm")
    training.add_argument("directory", type=Path)
    training.add_argument("--iterations", type=int, default=100)
    training.add_argument("--group-size", type=int, default=4)
    training.add_argument("--max-minutes", type=float, help="cap each task's budget of game time")
    training.add_argument("--max-turns", type=int, help="cap each episode's turns (for smoke tests)")
    training.add_argument(
        "--stragglers",
        type=int,
        default=1,
        help="start the next group when at most this many episodes of earlier groups are still running",
    )
    training.add_argument("--update-turns", type=int, default=384, help="turns trained on per update (sampled)")
    training.add_argument("--thinking-budget", type=int, default=1024, help="tokens of thinking per turn")
    training.add_argument("--learning-rate", type=float, default=2e-5)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--tasks", help="comma-separated task ids to train on (default: the whole curriculum)")
    training.add_argument("--exercise-updates", action="store_true", help=argparse.SUPPRESS)
    reporting = commands.add_parser("report", help="chart a run's progress, and post it to a Discord webhook")
    reporting.add_argument("directory", type=Path)
    reporting.add_argument("--watch", action="store_true", help="report again after every iteration until the run ends")
    reporting.add_argument("--webhook", help="a Discord webhook (default: the environment's DISCORD_WEBHOOK_URL)")
    arguments = parser.parse_args()
    if arguments.command == "report":
        from minecraft_swarm.report import report

        webhook = arguments.webhook or os.environ.get("DISCORD_WEBHOOK_URL")
        asyncio.run(report(arguments.directory, webhook, watch=arguments.watch))
    if arguments.command == "train":
        from minecraft_swarm.train import TrainingSettings, train

        settings = TrainingSettings(
            directory=arguments.directory,
            iterations=arguments.iterations,
            group_size=arguments.group_size,
            max_minutes=arguments.max_minutes,
            max_turns=arguments.max_turns,
            stragglers=arguments.stragglers,
            update_turns=arguments.update_turns,
            thinking_budget=arguments.thinking_budget,
            learning_rate=arguments.learning_rate,
            seed=arguments.seed,
            tasks=arguments.tasks.split(",") if arguments.tasks else None,
            exercise_updates=arguments.exercise_updates,
        )
        asyncio.run(train(settings))
    if arguments.command == "server":
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(_server(arguments.seed, keep=arguments.keep))


async def _server(seed: int, *, keep: bool) -> None:
    server = PaperServer(Installation(), seed=seed)
    await server.start()
    address = {"port": server.port, "control": server.control_url, "directory": str(server.directory)}
    print(json.dumps(address), flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        await server.stop(keep=keep)


if __name__ == "__main__":
    main()
