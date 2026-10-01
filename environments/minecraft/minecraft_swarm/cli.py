"""`minecraft-swarm`: run the Minecraft swarm environment.

minecraft-swarm server [--seed N]     start a temporary server and keep it up (Ctrl-C stops and deletes it)
"""

import argparse
import asyncio
import contextlib
import json
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
    training.add_argument("--max-turns", type=int, default=16)
    training.add_argument("--thinking-budget", type=int, default=384)
    training.add_argument("--learning-rate", type=float, default=2e-5)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--tasks", help="comma-separated task ids to train on (default: the whole curriculum)")
    arguments = parser.parse_args()
    if arguments.command == "train":
        from minecraft_swarm.train import TrainingSettings, train

        settings = TrainingSettings(
            directory=arguments.directory,
            iterations=arguments.iterations,
            group_size=arguments.group_size,
            max_turns=arguments.max_turns,
            thinking_budget=arguments.thinking_budget,
            learning_rate=arguments.learning_rate,
            seed=arguments.seed,
            tasks=arguments.tasks.split(",") if arguments.tasks else None,
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
