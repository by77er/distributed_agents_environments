"""Machines in one process, for the tests of what runs on several: engines that say which weights sampled them, an
engine host (its engines served over HTTP, a follower beating beside a shared ledger), and ports to serve on."""

import asyncio
import contextlib
import socket
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path
from typing import Any, cast

from rollout_train.checkpoints import Checkpoints
from rollout_train.following import Follower
from rollout_train.inference import Channel, Generation, Limits, serve_engines
from rollout_train.presence import Presence
from rollout_train.recorder import Renderer
from rollout_train.recorder.renderers import Tokenizer
from rollout_train.testing import Characters, PlainRenderer, ScriptedEngine

PORTS = range(8820, 8830)
"""What the servers of these tests listen on, on 127.0.0.1."""


def free_port() -> int:
    for port in PORTS:
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no port of {PORTS} is free")


class Saying(ScriptedEngine):
    """An engine whose reply is the adapter it was asked to sample from (`base` for none), so that a recorded turn says
    which weights sampled it."""

    def __init__(self) -> None:
        super().__init__(cast(Tokenizer, Characters()))

    async def generate(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Sequence[int],
        adapter: str | None,
    ) -> Generation:
        self.script = [(f"{adapter or 'base'}\n", "stop")]
        return await super().generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_token_ids=stop_token_ids,
            adapter=adapter,
        )


def saying_channel(replicas: int = 1, name: str = "policy") -> Channel:
    return Channel(name, [Saying() for _ in range(replicas)], cast(Renderer, PlainRenderer()), Limits())


@contextlib.asynccontextmanager
async def served(app: Any) -> AsyncGenerator[tuple[str, Any]]:
    """`app` served on 127.0.0.1: its base URL, and the server (whose `should_exit` stops it)."""
    import uvicorn

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:  # (the server says when it is up)
        await asyncio.sleep(0.01)
        if task.done():
            task.result()
    try:
        yield f"http://127.0.0.1:{port}", server
    finally:
        server.should_exit = True
        await task


@contextlib.asynccontextmanager
async def engine_host(
    name: str,
    checkpoints: Checkpoints,
    run: str,
    channels: dict[str, Channel],
    directory: Path,
    presence: Presence,
    *,
    following: bool = True,
) -> AsyncGenerator[tuple[Follower, Any]]:
    """An engine host: `channels` served over HTTP and followed (unless not `following`: it serves what it has),
    beating as `name`. Yields the follower and the server (whose `should_exit` makes the host stop answering)."""
    async with served(serve_engines(channels)) as (address, server):
        follower = Follower(
            name, checkpoints, run, channels, directory, address=address, presence=presence, every=0.05, beating=0.2
        )
        task = asyncio.create_task(follower.serve() if following else _beating(follower))
        try:
            await follower.beat()
            yield follower, server
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def _beating(follower: Follower) -> None:
    """Beat, following nothing: a host whose follower is stuck."""
    while True:
        await asyncio.sleep(0.2)
        await follower.beat()
