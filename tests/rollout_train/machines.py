"""Machines in one process, for the tests of what runs on several: engines that say which weights sampled them, a fake
of vLLM's OpenAI-compatible server over one, an engine host (that server, and a follower that loads into it and beats
beside a shared ledger), a proxy that passes requests on, and ports to serve on."""

import asyncio
import contextlib
import itertools
import random
import socket
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
from pydantic import JsonValue

from rollout.contracts import Message
from rollout.environment import Description, Row, Start
from rollout.harness import End, Observation, ProgramReference, RunContext, Task, agent_program
from rollout_train.checkpoints import Checkpoints
from rollout_train.following import Follower
from rollout_train.inference import Channel, Connection, Generation, RemoteEngine
from rollout_train.presence import Presence
from rollout_train.recorder import Renderer
from rollout_train.recorder.renderers import Tokenizer
from rollout_train.testing import Characters, PlainRenderer, ScriptedEngine

PORTS = range(8820, 8830)
"""What the servers of these tests listen on, on 127.0.0.1."""
MODEL = "a-checkpoint"
"""The base model the fake servers serve, under its own name."""


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
        thinking_budget: int | None = None,
        top: int = 0,
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
            top=top,
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


def fake_vllm(
    engine: ScriptedEngine, *, model: str = MODEL, token: str | None = None, bodies: list[dict[str, Any]] | None = None
) -> Any:
    """A fake of vLLM's OpenAI-compatible server over `engine`, serving `model` under its name and the adapters loaded
    under theirs (as `VLLM_ALLOW_RUNTIME_LORA_UPDATING` allows): `/v1/models`, `/v1/completions` of token ids with
    `return_token_ids` and logprobs, the most likely tokens at each (`logprobs` above 0, by id with
    `return_tokens_as_token_ids`), and prompt logprobs (`prompt_logprobs`), answered in vLLM's shapes (404 for a model
    it does not have), `/v1/load_lora_adapter` and `/v1/unload_lora_adapter`. With `token`, a request without it as a
    bearer token is refused (401). Each completion's body is kept in `bodies`, if given."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, PlainTextResponse, Response
    from starlette.routing import Route

    loaded: dict[str, str] = {}

    def refused(request: Request) -> Response | None:
        if token is not None and request.headers.get("authorization") != f"Bearer {token}":
            return JSONResponse({"error": {"message": "Unauthorized"}}, status_code=401)
        return None

    async def models(request: Request) -> Response:
        if (refusal := refused(request)) is not None:
            return refusal
        cards = [{"id": model, "object": "model", "max_model_len": engine.max_model_len}]
        cards += [{"id": name, "object": "model", "parent": model, "root": path} for name, path in loaded.items()]
        return JSONResponse({"object": "list", "data": cards})

    async def completions(request: Request) -> Response:
        if (refusal := refused(request)) is not None:
            return refusal
        body = await request.json()
        if bodies is not None:
            bodies.append(body)
        asked = body["model"]
        if asked != model and asked not in loaded:
            error = {"message": f"The model `{asked}` does not exist.", "type": "NotFoundError", "code": 404}
            return JSONResponse({"error": error}, status_code=404)
        if body.get("prompt_logprobs") is not None:
            return JSONResponse(await scored(body, None if asked == model else asked))
        top = int(body.get("logprobs") or 0)
        generation = await engine.generate(
            body["prompt"],
            max_tokens=body["max_tokens"],
            temperature=body["temperature"],
            top_p=body["top_p"],
            stop_token_ids=body["stop_token_ids"],
            adapter=None if asked == model else asked,
            top=top,
        )
        tops: list[Any] = [None] * len(generation.tokens)
        if top:  # (the sampled token, and the most likely ones beside it)
            tops = [
                {f"token_id:{each}": value for each, value in zip(ids, values, strict=True)}
                | {f"token_id:{token}": logprob}
                for ids, values, token, logprob in zip(
                    generation.top_tokens, generation.top_logprobs, generation.tokens, generation.logprobs, strict=True
                )
            ]
        logprobs = {
            "tokens": [f"token_id:{each}" for each in generation.tokens],
            "token_logprobs": generation.logprobs,
            "top_logprobs": tops,
            "text_offset": [0] * len(generation.tokens),
        }
        choice = {
            "index": 0,
            "text": "".join(map(chr, generation.tokens)),
            "token_ids": generation.tokens,
            "logprobs": logprobs,
            "finish_reason": generation.finish_reason,
        }
        usage = {"prompt_tokens": len(body["prompt"]), "completion_tokens": len(generation.tokens)}
        return JSONResponse({"object": "text_completion", "model": asked, "choices": [choice], "usage": usage})

    async def scored(body: dict[str, Any], adapter: str | None) -> dict[str, Any]:
        """A completion with prompt logprobs, as vLLM answers it: a logprob of each prompt token but the first, by its
        id, beside the most likely tokens there, ranked; and one token generated."""
        prompt: list[int] = body["prompt"]
        scores = await engine.score(prompt, start=1, top=body["prompt_logprobs"], adapter=adapter)
        listed: list[Any] = [None]
        for at, logprob in enumerate(scores.logprobs, start=1):
            entry = {str(prompt[at]): {"logprob": logprob, "rank": 1, "decoded_token": chr(prompt[at])}}
            ranked = (
                zip(scores.top_tokens[at - 1], scores.top_logprobs[at - 1], strict=True) if scores.top_tokens else ()
            )
            for rank, (each, value) in enumerate(ranked, start=1):
                entry[str(each)] = {"logprob": value, "rank": rank, "decoded_token": chr(each)}
            listed.append(entry)
        choice = {"index": 0, "text": "", "logprobs": None, "finish_reason": "length", "prompt_logprobs": listed}
        usage = {"prompt_tokens": len(prompt), "completion_tokens": 1}
        return {"object": "text_completion", "model": body["model"], "choices": [choice], "usage": usage}

    async def load(request: Request) -> Response:
        if (refusal := refused(request)) is not None:
            return refusal
        body = await request.json()
        if body["lora_name"] in loaded:  # (as vLLM refuses a name it holds already)
            message = f"The lora adapter '{body['lora_name']}' has already been loaded."
            refusal = {"message": message, "type": "BadRequestError", "code": 400}
            return JSONResponse({"error": refusal}, status_code=400)
        await engine.load_adapter(body["lora_name"], body["lora_path"])
        loaded[body["lora_name"]] = body["lora_path"]
        return PlainTextResponse(f"Success: LoRA adapter '{body['lora_name']}' added successfully.")

    async def unload(request: Request) -> Response:
        if (refusal := refused(request)) is not None:
            return refusal
        body = await request.json()
        await engine.remove_adapter(body["lora_name"])
        loaded.pop(body["lora_name"], None)
        return PlainTextResponse(f"Success: LoRA adapter '{body['lora_name']}' removed successfully.")

    return Starlette(
        routes=[
            Route("/v1/models", models),
            Route("/v1/completions", completions, methods=["POST"]),
            Route("/v1/load_lora_adapter", load, methods=["POST"]),
            Route("/v1/unload_lora_adapter", unload, methods=["POST"]),
        ]
    )


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


@dataclass
class Host:
    """An engine host, as a test sees it: its vLLM server's address, the server, the engine behind it, the channel its
    follower publishes to, and the follower."""

    address: str
    server: Any
    engine: Saying
    channel: Channel
    follower: Follower


@contextlib.asynccontextmanager
async def engine_host(
    name: str,
    checkpoints: Checkpoints,
    run: str,
    directory: Path,
    presence: Presence,
    *,
    following: bool = True,
    token: str | None = None,
    connection: Connection | None = None,
) -> AsyncGenerator[Host]:
    """An engine host: a fake vLLM server (asking for `token`, if given), and a follower that loads what `run` says its
    channel `policy` should serve into it (unless not `following`: it serves what it has), beating as `name`. The
    server's `should_exit` makes it stop answering."""
    engine = Saying(words=("yes", "no"))
    async with served(fake_vllm(engine, token=token)) as (address, server):
        channel = Channel(
            "policy", [RemoteEngine(MODEL, address=address, connection=connection)], cast(Renderer, PlainRenderer())
        )
        follower = Follower(name, checkpoints, run, {"policy": channel}, directory, presence=presence, every=0.05,
                            beating=0.2)  # fmt: skip
        task = asyncio.create_task(follower.serve() if following else _beating(follower))
        try:
            await follower.beat()
            yield Host(address, server, engine, channel, follower)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def passing_on(upstreams: Sequence[str], *, caching: bool = False) -> Any:
    """A reverse proxy that knows nothing of the API: it passes each request on, token and all, to the next of
    `upstreams` in turn (a router, as far as a runner can tell). `caching`: it answers every completion after the first
    from the first answer, as a proxy that caches would."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import Route

    turns = itertools.cycle(upstreams)
    kept: list[tuple[int, bytes]] = []

    async def forward(request: Request) -> Response:
        path = request.url.path
        if caching and path == "/v1/completions" and kept:
            status, body = kept[0]
            return Response(body, status, media_type="application/json")
        headers = {key: value for key, value in request.headers.items() if key not in ("host", "content-length")}
        async with httpx.AsyncClient() as client:
            answer = await client.request(
                request.method, next(turns) + path, content=await request.body(), headers=headers
            )
        if path == "/v1/completions" and answer.status_code == 200:
            kept.append((answer.status_code, answer.content))
        kind = answer.headers.get("content-type", "application/json")
        return Response(answer.content, answer.status_code, media_type=kind)

    return Starlette(routes=[Route("/{path:path}", forward, methods=["GET", "POST"])])


async def _beating(follower: Follower) -> None:
    """Beat, following nothing: a host whose follower is stuck."""
    while True:
        await asyncio.sleep(0.2)
        await follower.beat()
