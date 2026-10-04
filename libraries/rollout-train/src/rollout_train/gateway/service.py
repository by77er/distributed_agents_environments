"""The gateway: a stateless service between programs and harnesses, and the endpoints that sample a policy.

A request carries a signed key (`keys`), which says whose turn it is; the gateway renders the request with the
channel's renderer, samples it with the thinking budget (`rollout_train.recorder.sampling`) from the checkpoint the
channel serves the session (a routed channel asks its servers for the one its run says it should serve, by name:
`rollout_train.inference.remote`), records the turn (`turns`), and replies in the request's own API. It keeps no
session: any replica answers any request, and a replica can die at any moment.

- **A retried request is not sampled twice.** A request under an effect id (its `Idempotency-Key`) that was recorded is
  answered with the recorded reply.
- **A reply is sent only once its turn is recorded.** A replica that dies before recording leaves nothing; the client's
  retry is sampled again.
- **A turn samples one checkpoint**, the one chosen when it began, through both phases of its thinking budget. An
  endpoint that drops it in between has the turn sampled again from the start.

    POST {base}/v1/chat/completions      OpenAI's Chat Completions
    POST {base}/v1/responses             OpenAI's Responses
    POST {base}/v1/messages              Anthropic's Messages
    POST {base}/v1/messages/count_tokens a Messages request's prompt, counted with the channel's renderer
    POST {base}/v1/samples               a `SampleRequest`, answered with a `SampleResult` (for programs in a runner)
    GET  {base}/v1/models                the channels, as models (each this process hosts with its `contract`)
    GET  {base}/healthz                  alive
    GET  {base}/readyz                   ready: the ledger and the blob store answer

A reply says which checkpoint served it and at what depth (`X-Rollout-Checkpoint`, `X-Rollout-Depth`), the request id
it was recorded under (`X-Rollout-Request-Id`), and whether it was recorded before (`X-Rollout-Replayed`). A request may
say how it follows from earlier ones (`X-Rollout-Links`: a JSON list of `{"type", "source"}`, each source a request id).
"""

import asyncio
import itertools
import json
import logging
import time
import uuid
from array import array
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from rollout.contracts import CapabilityContract, ModelEndpointError, SampleRequest
from rollout.harness.hooks import ModelSample, RunHooks
from rollout_train.gateway.keys import Grant, KeyRefused, Keyring
from rollout_train.gateway.turns import Link, Reply, TurnRecord, TurnStore
from rollout_train.inference import Channel, Generation, Limits, Routes
from rollout_train.inference.channel import Sampler, Unserved
from rollout_train.inference.remote import NoReplica
from rollout_train.ledger import Fenced
from rollout_train.recorder.compat import SERVED_UNDER, chat, key, messages, refused, replied, requested, responses
from rollout_train.recorder.compat.wire import Failure, Format, Prompt
from rollout_train.recorder.sampling import sample_turn
from rollout_train.recorder.segments import TOKEN_LEVEL
from rollout_train.serving import BASE, parts

LINKS = "x-rollout-links"
"""The header a request declares its links to earlier requests in."""
ATTEMPTS = 3
"""Times a turn is sampled before it fails, when the weights it began with stop being served (`Unserved`)."""


logger = logging.getLogger(__name__)


class Refused(Exception):
    """A request the gateway will not sample: why, as a `Failure`."""

    def __init__(self, failure: Failure, message: str) -> None:
        super().__init__(message)
        self.failure = failure


@dataclass
class Gateway:
    """What a replica serves: where it records, the keys it takes, and the channels it samples: those whose engines this
    process publishes to (`channels`, by name; `models` names each one's base model), and those whose engines serve
    elsewhere (`routes`), each run's sampled from what that run says it serves. `hooks` are told of each sample a
    harness asks for in one of the three APIs and the gateway records (a runner's own samples reach its hooks through
    its endpoints). Each turn records what its sampler samples with: the sampler's `sampled_with` where it says, else
    `TOKEN_LEVEL` (every engine and server a channel samples from is token-exact, with sampled-token logprobs)."""

    store: TurnStore
    keyring: Keyring
    channels: Mapping[str, Channel] = field(default_factory=dict[str, Channel])
    routes: Routes | None = None
    models: Mapping[str, str] = field(default_factory=dict[str, str])
    hooks: Sequence[RunHooks] = ()

    def granted(self, key: str) -> Grant:
        """The grant a key carries; `Refused` if it is not one this gateway takes."""
        try:
            grant = self.keyring.verify(key)
        except KeyRefused as error:
            raise Refused(Failure.KEY, str(error)) from None
        self.sampler(grant)
        return grant

    def sampler(self, grant: Grant) -> Sampler:
        """What a grant's turns sample from (`sampler_of` its run and channel)."""
        return self.sampler_of(grant.run, grant.channel)

    def sampler_of(self, run: str, channel: str) -> Sampler:
        """What a run's channel samples from: the channel of this process it names, else the run's routed channel of
        that name. A channel named within its run (`RUN/NAME`) is that run's."""
        named, name = parts(channel)
        run = named or run
        if name in self.channels:
            return self.channels[name]
        if self.routes is not None and self.routes.routed(name):
            return self.routes.channel(run, name)
        raise Refused(Failure.KEY, f"this gateway serves no channel {channel!r}")

    async def reaches(self, run: str, channel: str) -> bool:
        """Whether a run's channel can be sampled now: one of this process, or a routed one whose servers have a
        checkpoint close enough to what the run says it should serve."""
        _, name = parts(channel)
        if name in self.channels:
            return True
        return self.routes is not None and self.routes.routed(name) and await self.routes.reaches(run, name)

    @property
    def names(self) -> list[str]:
        """The channels it samples, by name."""
        routed: Mapping[str, object] = self.routes.routes if self.routes is not None else {}
        return [*self.channels, *(name for name in routed if name not in self.channels)]

    def describe(self, grant: Grant) -> CapabilityContract:
        return contract_of(self.sampler(grant), limits_of(self.sampler(grant).limits, grant.thinking, grant.answer))

    async def sample(self, grant: Grant, request: SampleRequest, links: Sequence[Link] = ()) -> Reply:
        """One reply, recorded before it is returned: the one recorded under the request's effect id, if there is one.
        `links` are those its harness declared besides the request's own. Raises `Refused`, or the endpoint's
        `ModelEndpointError` (`ContextOverflow` when the context is too long)."""
        if request.session_id != grant.session_id:
            raise Refused(Failure.KEY, f"this key is for session {grant.session_id}, not {request.session_id}")
        try:
            links = [*links, *(Link(link.type, link.source) for link in request.links)]
        except ValueError as error:
            raise Refused(Failure.REQUEST, str(error)) from None
        index = await self.store.index(grant.run, grant.run_id)
        recorded = await self.store.reply(grant.run, grant.run_id, request.effect_id, index)
        if recorded is None:
            turn = await self._sampled(grant, request, links)
            try:
                recorded = await self.store.record(turn, grant.fence, index)
            except Fenced:
                raise Refused(
                    Failure.KEY, "this key's attempt was taken over: its turns are no longer recorded"
                ) from None
        if recorded.slot != grant.slot:
            raise Refused(Failure.REQUEST, f"request id {request.effect_id} was used by another slot of the run")
        return recorded

    async def _sampled(self, grant: Grant, request: SampleRequest, links: Sequence[Link]) -> TurnRecord:
        """A turn sampled from the weights its session samples from when it begins, and sampled again from the start
        when they stop being served before it ends (`Unserved`), up to `ATTEMPTS` times."""
        sampler = self.sampler(grant)
        failure: Exception | None = None
        for attempt in range(1, ATTEMPTS + 1):
            try:
                return await self._turn(grant, sampler, request, links, attempt)
            except Unserved as error:
                failure = error
            except NoReplica as error:
                raise ModelEndpointError(str(error)) from None
        raise ModelEndpointError(f"the turn was not served in {ATTEMPTS} attempts: {failure}")

    async def _turn(
        self, grant: Grant, sampler: Sampler, request: SampleRequest, links: Sequence[Link], attempt: int
    ) -> TurnRecord:
        started, began = time.time(), time.monotonic()
        adapter, version = await sampler.weights(request.session_id)  # (both phases sample these weights)
        phases = itertools.count(1)

        async def generate(context: Sequence[int], room: int, stop: Sequence[int]) -> Generation:
            generation = await sampler.generate(
                context,
                max_tokens=room,
                temperature=grant.temperature,
                top_p=grant.top_p,
                stop_token_ids=stop,
                adapter=adapter,
                session=request.session_id,
                version=version,
                request=f"{request.effect_id}/{attempt}/{next(phases)}",
            )
            if generation.model is not None and adapter is not None and generation.model != adapter:
                raise Unserved(f"{generation.model} answered for {adapter}")  # (its tokens would be misstamped)
            return generation

        limits = limits_of(sampler.limits, grant.thinking, grant.answer)
        turn = await sample_turn(request, sampler.renderer, limits, sampler.context_limit, generate)
        held = getattr(sampler, "held", None) or getattr(sampler, "model", None)
        return TurnRecord(
            effect_id=request.effect_id,
            run=grant.run,
            run_id=grant.run_id,
            slot=grant.slot,
            channel=sampler.name,
            checkpoint=adapter or held or self.models.get(sampler.name, BASE),
            depth=version,
            prompt=array("i", turn.prompt),
            completion=turn.completion,
            mask=turn.mask,
            logprobs=turn.logprobs,
            result=turn.result,
            episode=grant.episode,
            attempt=grant.attempt,
            links=tuple(links),
            timings={
                "started": round(started, 3),
                "attempt": attempt,
                "phases": [round(seconds, 4) for seconds in turn.phases],
                "seconds": round(time.monotonic() - began, 4),
            },
            sampled_with=tuple(getattr(sampler, "sampled_with", TOKEN_LEVEL)),
        )

    def observe(self, grant: Grant, request: SampleRequest, reply: Reply, seconds: float) -> None:
        """Tell the hooks of a sample a harness asked for, newly recorded."""
        if reply.replayed or not self.hooks:
            return
        sample = ModelSample(grant.run_id, grant.slot, request, reply.result, seconds)
        for hook in self.hooks:
            try:
                hook.on_sample(sample)
            except Exception:
                logger.exception("a hook failed on a sample of %s in run %s", grant.slot, grant.run_id)

    async def count(self, grant: Grant, prompt: Prompt) -> int:
        """How many tokens a prompt renders to with the channel's renderer: what a turn's prompt would hold."""
        renderer = self.sampler(grant).renderer
        return len(await asyncio.to_thread(renderer.render, prompt.messages, prompt.tools))

    async def ready(self) -> dict[str, str]:
        """What is not ready, by part (empty: ready): the ledger and the blob store must answer."""
        problems: dict[str, str] = {}
        try:
            await asyncio.wait_for(self.store.ledger.read("gateway/ready"), 5.0)
        except Exception as error:
            problems["ledger"] = f"{type(error).__name__}: {error}"
        try:
            reference = await asyncio.wait_for(self.store.blobs.put(b"ready", "text/plain"), 5.0)
            await asyncio.wait_for(self.store.blobs.read(reference), 5.0)
        except Exception as error:
            problems["blobs"] = f"{type(error).__name__}: {error}"
        return problems


def create_app(gateway: Gateway) -> Starlette:
    """Serve `gateway` over HTTP (behind a proxy that terminates TLS, or with uvicorn's own certificates)."""

    def hosted(name: str) -> dict[str, Any]:
        """For a channel whose engines are in this process, what it guarantees a session (`contract`): what a runner
        that records through this gateway tells its programs of the channel."""
        channel = gateway.channels.get(name)
        return {"contract": contract_of(channel).model_dump(mode="json")} if channel is not None else {}

    async def models(request: Request) -> Response:
        names = gateway.names
        listed = [
            {"id": name, "object": "model", "type": "model", "display_name": name, "created": 0, "owned_by": "rollout"}
            | hosted(name)
            for name in names
        ]
        page = {"has_more": False, "first_id": names[0] if names else None, "last_id": names[-1] if names else None}
        return JSONResponse({"object": "list", "data": listed} | page)

    def answering(format: Format) -> Callable[[Request], Awaitable[Response]]:
        async def answer(request: Request) -> Response:
            try:
                grant = gateway.granted(key(request))
                allowed = gateway.describe(grant).max_output_tokens
                fallback = f"{grant.session_id}:harness:{uuid.uuid4().hex}"
                try:
                    body, sample = await requested(request, format, grant.session_id, allowed, fallback)
                    links = linked(request)
                except (KeyError, TypeError, ValueError) as error:
                    raise Refused(Failure.REQUEST, f"the request could not be read: {error}") from None
                began = time.monotonic()
                reply = await gateway.sample(grant, sample, links)
                gateway.observe(grant, sample, reply, time.monotonic() - began)
            except Refused as error:
                return format.error(error.failure, str(error))
            except ModelEndpointError as error:
                return refused(format, error)
            return replied(format, reply.result, sample.effect_id, body, said(reply))

        return answer

    async def count_tokens(request: Request) -> Response:
        try:
            grant = gateway.granted(key(request))
            try:
                body = await request.json()
                if not isinstance(body, dict):
                    raise TypeError("the body is not a JSON object")
                prompt = messages.prompt(cast(dict[str, Any], body))
            except (KeyError, TypeError, ValueError) as error:
                raise Refused(Failure.REQUEST, f"the request could not be read: {error}") from None
            counted = await gateway.count(grant, prompt)
        except Refused as error:
            return messages.FORMAT.error(error.failure, str(error))
        return JSONResponse({"input_tokens": counted})

    async def samples(request: Request) -> Response:
        try:
            grant = gateway.granted(key(request))
            try:
                sample = SampleRequest.model_validate(await request.json())
                links = linked(request)
            except (ValidationError, ValueError, TypeError, KeyError) as error:
                raise Refused(Failure.REQUEST, f"the request could not be read: {error}") from None
            reply = await gateway.sample(grant, sample, links)
        except Refused as error:
            return _native_error(error.failure.value, str(error))
        except ModelEndpointError as error:
            return _native_error(type(error).__name__, str(error), getattr(error, "context_limit", None))
        return JSONResponse(reply.result.model_dump(mode="json"), headers=said(reply))

    async def healthy(request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def ready(request: Request) -> Response:
        problems = await gateway.ready()
        return JSONResponse({"status": "not ready" if problems else "ready", **problems}, 503 if problems else 200)

    return Starlette(
        routes=[
            Route("/healthz", healthy),
            Route("/readyz", ready),
            Route(f"{SERVED_UNDER}/models", models),
            Route(f"{SERVED_UNDER}/chat/completions", answering(chat.FORMAT), methods=["POST"]),
            Route(f"{SERVED_UNDER}/responses", answering(responses.FORMAT), methods=["POST"]),
            Route(f"{SERVED_UNDER}/messages", answering(messages.FORMAT), methods=["POST"]),
            Route(f"{SERVED_UNDER}/messages/count_tokens", count_tokens, methods=["POST"]),
            Route(f"{SERVED_UNDER}/samples", samples, methods=["POST"]),
        ]
    )


def limits_of(limits: Limits, thinking: int | None, answer: int | None) -> Limits:
    """A channel's limits, with the thinking and answer room a binding gives in place of its own (none: the
    channel's)."""
    return replace(
        limits,
        thinking=limits.thinking if thinking is None else thinking,
        answer=limits.answer if answer is None else answer,
    )


def contract_of(sampler: Sampler, limits: Limits | None = None) -> CapabilityContract:
    """What a channel guarantees a session: its context, and room for thinking and an answer (by `limits`, else the
    channel's own). Where either budget is unset, a reply may take all the context its prompt leaves: the most output
    is the context limit."""
    limits = limits or sampler.limits
    context = sampler.context_limit
    room = context if limits.thinking is None or limits.answer is None else limits.thinking + limits.answer
    return CapabilityContract(context_limit=context, max_output_tokens=room)


def linked(request: Request) -> list[Link]:
    """The links a request declares (`X-Rollout-Links`). Raises `ValueError` when they cannot be read."""
    given = request.headers.get(LINKS)
    if not given:
        return []
    entries: Any = json.loads(given)
    if not isinstance(entries, list):
        raise ValueError(f"{LINKS} is a JSON list of {{type, source}}")
    links: list[Link] = []
    for entry in cast(list[Any], entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{LINKS} is a JSON list of {{type, source}}")
        said = cast(dict[str, Any], entry)
        links.append(Link(str(said["type"]), str(said["source"])))
    return links


def said(reply: Reply) -> dict[str, str]:
    """What a reply's headers say of it."""
    return {
        "x-rollout-checkpoint": reply.checkpoint,
        "x-rollout-depth": str(reply.depth),
        "x-rollout-request-id": reply.effect_id,
        "x-rollout-replayed": "true" if reply.replayed else "false",
    }


NATIVE_STATUS = {Failure.KEY.value: 401, Failure.REQUEST.value: 400, "ContextOverflow": 400}


def _native_error(kind: str, message: str, context_limit: int | None = None) -> Response:
    """An error of the native path: `{"error": {"type", "message"}}`, with the context limit of a `ContextOverflow`."""
    error: dict[str, Any] = {"type": kind, "message": message}
    if context_limit is not None:
        error["context_limit"] = context_limit
    return JSONResponse({"error": error}, status_code=NATIVE_STATUS.get(kind, 503))
