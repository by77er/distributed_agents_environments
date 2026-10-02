"""`minecraft-swarm server [--seed N]`: a temporary server to look at (join with any client; Ctrl-C stops and
deletes it). Training is `rollout train PROFILE minecraft_swarm.catalog:catalog`."""

import argparse
import asyncio
import contextlib
import json

from minecraft_swarm.paper import Installation, PaperServer


def main() -> None:
    parser = argparse.ArgumentParser(prog="minecraft-swarm", description="The Minecraft swarm environment.")
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("server", help="start a temporary server and keep it up")
    server.add_argument("--seed", type=int, default=12345)
    server.add_argument("--keep", action="store_true", help="keep the server's directory when it stops")
    arguments = parser.parse_args()
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
