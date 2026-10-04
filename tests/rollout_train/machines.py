"""Machines in one process, for the tests of what runs on several: engines that say which weights sampled them, an
engine host (its engines served over HTTP, a follower beating beside a shared ledger), and ports to serve on."""

import asyncio
import contextlib
import random
import socket
from collections.abc import AsyncGenerator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import httpx
from pydantic import JsonValue

from rollout.contracts import Message
from rollout.environment import Description, Row, Start
from rollout.harness import End, Observation, ProgramReference, RunContext, Task, agent_program
from rollout_train.checkpoints import Checkpoints
from rollout_train.following import Follower
from rollout_train.inference import Channel, Generation, Limits, serve_engines
from rollout_train.inference.remote import REPLICA
from rollout_train.presence import Presence
from rollout_train.recorder import Renderer
from rollout_train.recorder.renderers import Tokenizer
from rollout_train.testing import Characters, PlainRenderer, ScriptedEngine

PORTS = range(8820, 8830)
"""What the servers of these tests listen on, on 127.0.0.1."""


def free_port() -> int:
    for port in PORTS:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # (as the server does: closed ones may linger)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no port of {PORTS} is free")


class Saying(ScriptedEngine):
    """An engine whose reply is the adapter it was asked to sample from (`base` for none), so that a recorded turn says
    which weights sampled it; with `words`, then one of them, each in turn."""

    def __init__(self, words: Sequence[str] = ()) -> None:
        super().__init__(cast(Tokenizer, Characters()))
        self.words = list(words)
        self.said = 0

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
        word = f" {self.words[self.said % len(self.words)]}" if self.words else ""
        self.said += 1
        self.script = [(f"{adapter or 'base'}{word}\n", "stop")]
        return await super().generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_token_ids=stop_token_ids,
            adapter=adapter,
        )


class SaysYes(Task):
    """One turn: a point for a reply that ends in yes."""

    async def start(self, run: RunContext) -> Observation:
        return Observation("Say yes.")

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        said = reply.text.strip().endswith("yes")
        await run.emit("result", {"solved": said, "saturated": said, "duration": 1})
        return End(reward=1.0 if said else 0.0)


class Yes:
    """An environment of one row, played by `SaysYes`."""

    program: ProgramReference = agent_program(SaysYes)
    version = "1"
    description = Description(saturated=True, duration="turns")

    def rows(self) -> Sequence[Row]:
        return [Row("yes", "say yes", {})]

    def start(self, row: Row, rng: random.Random) -> JsonValue:
        return {"seed": rng.randrange(1000)}

    def evals(self) -> Mapping[str, Sequence[Start]]:
        return {}


yes = Yes()


def saying_engine(model: str, **options: Any) -> Saying:
    """What a profile names for an engine that says which weights sampled it, then yes or no in turn."""
    return Saying(words=("yes", "no"))


def saying_channel(replicas: int = 1, name: str = "policy") -> Channel:
    return Channel(name, [Saying() for _ in range(replicas)], cast(Renderer, PlainRenderer()), Limits())


@contextlib.asynccontextmanager
async def served(app: Any) -> AsyncGenerator[tuple[str, Any]]:
    """`app` served on 127.0.0.1: its base URL, and the server (whose `should_exit` stops it)."""
    import uvicorn

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
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
    token: str | None = None,
) -> AsyncGenerator[tuple[Follower, Any]]:
    """An engine host: `channels` served over HTTP (asking for `token`, if given) and followed (unless not
    `following`: it serves what it has), beating as `name`, its replicas' ids beginning with it. Yields the follower and
    the server (whose `should_exit` makes the host stop answering)."""
    async with served(serve_engines(channels, name, token=token)) as (address, server):
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


def passing_on(hosts: Mapping[str, str], *, caching: bool = False) -> Any:
    """A reverse proxy that knows nothing of the protocol but the header naming a request's replica: it sends each
    request, token and all, to the host of the server the replica's id begins with (`hosts`, by server). `caching`: it
    answers a sample from the first answer it passed on for that replica, as a proxy that caches would."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import Route

    kept: dict[str, tuple[int, bytes]] = {}

    async def forward(request: Request) -> Response:
        replica = request.headers[REPLICA]
        path = request.url.path
        if caching and path in kept:
            status, body = kept[path]
            return Response(body, status, media_type="application/json")
        headers = {key: value for key, value in request.headers.items() if key not in ("host", "content-length")}
        async with httpx.AsyncClient() as client:
            answer = await client.request(
                request.method, hosts[replica.split(".")[0]] + path, content=await request.body(), headers=headers
            )
        if path.endswith("/generate") and answer.status_code == 200:
            kept.setdefault(path, (answer.status_code, answer.content))
        return Response(answer.content, answer.status_code, media_type="application/json")

    return Starlette(routes=[Route("/{path:path}", forward, methods=["GET", "POST", "DELETE"])])


async def _beating(follower: Follower) -> None:
    """Beat, following nothing: a host whose follower is stuck."""
    while True:
        await asyncio.sleep(0.2)
        await follower.beat()
