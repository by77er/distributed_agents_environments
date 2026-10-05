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
    POST {base}/v1/scores                a `ScoreRequest`: the logprobs the channel gives the tokens it is handed
    GET  {base}/v1/models                the channels, as models (each this process hosts with its `contract`); with
                                         `?run=RUN`, the channels that run's start names (`RUN/NAME`), with theirs
    GET  {base}/healthz                  alive
    GET  {base}/readyz                   ready: the ledger and the blob store answer

A score request is recorded as a turn of its own use (`score`): the tokens it was handed are its prompt, it samples
nothing, and it is never trained on. Its tokens count as tokens in, as a sample's prompt does.

A channel on a hosted API (`rollout_train.inference.api.ApiChannel`) is sampled by message: the request goes to its
provider's endpoint as it is, with the binding's sampling and budgets, and its turn records the reply (text, tool calls,
usage) with no tokens (`sampled_with` empty, never trained on) and what it cost (`spend`). What a run spends so is
counted (`spending`), and a run whose cap is reached samples no more on hosted APIs. Such a channel scores no tokens and
counts none.

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

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from rollout.contracts import (
    CapabilityContract,
    ContextOverflow,
    FinishReason,
    Message,
    ModelEndpointError,
    Role,
    SampleRequest,
    SampleResult,
    Usage,
)
from rollout.harness.hooks import ModelSample, RunHooks
from rollout_train.gateway.directory import ChannelDirectory
from rollout_train.gateway.keys import Grant, KeyRefused, Keyring
from rollout_train.gateway.spending import Spending
from rollout_train.gateway.turns import SCORE, Link, Reply, TurnRecord, TurnStore
from rollout_train.inference import Channel, Generation, Limits, Routes, Scores
from rollout_train.inference.api import ApiChannel
from rollout_train.inference.channel import Sampler, Unserved, scored_range
from rollout_train.inference.remote import NoReplica
from rollout_train.ledger import Fenced
from rollout_train.recorder.compat import SERVED_UNDER, chat, key, messages, refused, replied, requested, responses
from rollout_train.recorder.compat.wire import Failure, Format, Prompt
from rollout_train.recorder.sampling import sample_turn
from rollout_train.recorder.segments import TOKEN_LEVEL
from rollout_train.serving import BASE, parts, qualified

LINKS = "x-rollout-links"
"""The header a request declares its links to earlier requests in."""
ATTEMPTS = 3
"""Times a turn is sampled before it fails, when the weights it began with stop being served (`Unserved`)."""
BACKOFF = 1.0
"""Seconds before a turn's second attempt, doubled before each after it."""


logger = logging.getLogger(__name__)


class ScoreRequest(BaseModel):
    """A request to score tokens: the logprobs the channel gives the tokens at positions `start` to `end` of `tokens`
    (`end` absent: to the end), each given those before it, with the `top` most likely tokens at each."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    effect_id: str = Field(min_length=1)
    """Its request id: a request under one that was recorded is answered with what was recorded."""
    session_id: str
    """The key's session."""
    tokens: list[int] = Field(min_length=2)
    start: int = Field(ge=1)
    end: int | None = None
    top: int = Field(default=0, ge=0)


class Refused(Exception):
    """A request the gateway will not sample: why, as a `Failure`."""

    def __init__(self, failure: Failure, message: str) -> None:
        super().__init__(message)
        self.failure = failure


@dataclass
class Gateway:
    """What a replica serves: where it records, the keys it takes, and the channels it samples: every channel a run's
    start names with a provider `directory` knows (`rollout_train.gateway.directory`), those whose engines this process
    publishes to (`channels`, by name; `models` names each one's base model), and those whose engines serve elsewhere
    (`routes`), each run's sampled from what that run says it serves. `hooks` are told of each sample a
    harness asks for in one of the three APIs and the gateway records (a runner's own samples reach its hooks through
    its endpoints). Each turn records what its sampler samples with: the sampler's `sampled_with` where it says, else
    `TOKEN_LEVEL` (every engine and server a channel samples from is token-exact, with sampled-token logprobs). The
    channels on hosted APIs it samples by name are `hosted`; what runs spend on them is counted in `spending`."""

    store: TurnStore
    keyring: Keyring
    channels: Mapping[str, Channel] = field(default_factory=dict[str, Channel])
    routes: Routes | None = None
    models: Mapping[str, str] = field(default_factory=dict[str, str])
    hooks: Sequence[RunHooks] = ()
    directory: ChannelDirectory | None = None
    hosted: Mapping[str, ApiChannel] = field(default_factory=dict[str, ApiChannel])
    spending: Spending | None = None

    def __post_init__(self) -> None:
        if self.spending is None:
            self.spending = Spending(self.store.ledger)

    async def granted(self, key: str) -> Grant:
        """The grant a key carries, its run's channels loaded from its start (where there is a directory); `Refused`
        if it is not one this gateway takes."""
        try:
            grant = self.keyring.verify(key)
        except KeyRefused as error:
            raise Refused(Failure.KEY, str(error)) from None
        await self.load(grant.run, grant.channel)
        self.sampler(grant)
        return grant

    async def load(self, run: str, channel: str) -> None:
        """Build a run's channels from its start, if there is a directory and they are not built yet (a channel named
        within its run, `RUN/NAME`, is that run's)."""
        if self.directory is not None:
            await self.directory.load(parts(channel)[0] or run)

    def sampler(self, grant: Grant) -> "Sampler | ApiChannel":
        """What a grant's turns sample from (`sampler_of` its run and channel)."""
        return self.sampler_of(grant.run, grant.channel)

    def sampler_of(self, run: str, channel: str) -> "Sampler | ApiChannel":
        """What a run's channel samples from: the channel its start names, built by the directory (once the run is
        loaded: `load`); else the channel of this process it names (its engines here, or a hosted API); else the run's
        routed channel of that name. A channel named within its run (`RUN/NAME`) is that run's."""
        named, name = parts(channel)
        run = named or run
        if self.directory is not None and (built := self.directory.channel(run, name)) is not None:
            return built
        if name in self.channels:
            return self.channels[name]
        if name in self.hosted:
            return self.hosted[name]
        if self.routes is not None and self.routes.routed(name):
            return self.routes.channel(run, name)
        raise Refused(Failure.KEY, f"this gateway serves no channel {channel!r}")

    async def reaches(self, run: str, channel: str) -> bool:
        """Whether a run's channel can be sampled now: one its start names or a routed one, whose servers have a
        checkpoint close enough to what the run says it should serve; or one of this process."""
        await self.load(run, channel)
        named, name = parts(channel)
        if self.directory is not None and (built := self.directory.channel(named or run, name)) is not None:
            return await built.reaches()
        if name in self.channels or name in self.hosted:
            return True
        return self.routes is not None and self.routes.routed(name) and await self.routes.reaches(run, name)

    @property
    def names(self) -> list[str]:
        """The channels it samples, by name."""
        routed: Mapping[str, object] = self.routes.routes if self.routes is not None else {}
        local = [*self.channels, *(name for name in self.hosted if name not in self.channels)]
        return [*local, *(name for name in routed if name not in local)]

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
            spending = self.spending
            assert spending is not None
            if turn.spend is not None:
                await spending.prepare(grant.run)  # (what earlier starts spent, read before this turn is in the ledger)
            try:
                recorded = await self.store.record(turn, grant.fence, index)
            except Fenced:
                raise Refused(
                    Failure.KEY, "this key's attempt was taken over: its turns are no longer recorded"
                ) from None
            if turn.spend is not None and not recorded.replayed:
                await spending.counted(grant.run, turn.spend)
        if recorded.slot != grant.slot:
            raise Refused(Failure.REQUEST, f"request id {request.effect_id} was used by another slot of the run")
        return recorded

    async def _sampled(self, grant: Grant, request: SampleRequest, links: Sequence[Link]) -> TurnRecord:
        """A turn sampled from the weights its session samples from when it begins, and sampled again from the start
        when they stop being served before it ends (`Unserved`), up to `ATTEMPTS` times."""
        sampler = self.sampler(grant)
        if isinstance(sampler, ApiChannel):
            return await self._hosted_turn(grant, sampler, request, links)
        failure: Exception | None = None
        for attempt in range(1, ATTEMPTS + 1):
            if attempt > 1:
                await asyncio.sleep(BACKOFF * 2 ** (attempt - 2))  # (a server that did not answer, a moment to again)
            try:
                return await self._turn(grant, sampler, request, links, attempt)
            except Unserved as error:
                failure = error
            except NoReplica as error:
                raise ModelEndpointError(str(error)) from None
        raise ModelEndpointError(f"the turn was not served in {ATTEMPTS} attempts: {failure}")

    async def _hosted_turn(
        self, grant: Grant, channel: ApiChannel, request: SampleRequest, links: Sequence[Link]
    ) -> TurnRecord:
        """A turn a hosted API sampled: the request as it is, with the grant's sampling and budgets; recorded with no
        tokens, never trained on, with what it cost. A run whose spending cap is reached samples none."""
        assert self.spending is not None
        if (why := await self.spending.over(grant.run)) is not None:
            raise ModelEndpointError(f"no more is sampled on {channel.provider.name}: {why}")
        started, began = time.time(), time.monotonic()
        result = await channel.sample(
            request, temperature=grant.temperature, top_p=grant.top_p, thinking=grant.thinking, answer=grant.answer
        )
        return TurnRecord(
            effect_id=request.effect_id,
            run=grant.run,
            run_id=grant.run_id,
            slot=grant.slot,
            channel=channel.name,
            checkpoint=channel.held,
            depth=0,
            prompt=array("i"),
            completion=[],
            mask=[],
            logprobs=[],
            result=result,
            episode=grant.episode,
            attempt=grant.attempt,
            links=tuple(links),
            timings={"started": round(started, 3), "attempt": 1, "seconds": round(time.monotonic() - began, 4)},
            sampled_with=channel.sampled_with,
            trained=False,
            spend=channel.dollars(result.usage),
        )

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
            trained=grant.trained,
        )

    async def score(self, grant: Grant, request: ScoreRequest) -> Reply:
        """The scores the grant's channel gives the request's tokens (`Reply.scores`), recorded as a turn of its own
        use before they are returned: those recorded under the request's effect id, if there are any. Raises `Refused`
        (`ValueError` from the channel, for a range or a `top` it does not take, is a request refused), or the
        endpoint's `ModelEndpointError` (`ContextOverflow` for a sequence too long to score)."""
        if request.session_id != grant.session_id:
            raise Refused(Failure.KEY, f"this key is for session {grant.session_id}, not {request.session_id}")
        try:
            scored_range(len(request.tokens), request.start, request.end)
        except ValueError as error:
            raise Refused(Failure.REQUEST, str(error)) from None
        index = await self.store.index(grant.run, grant.run_id)
        recorded = await self.store.reply(grant.run, grant.run_id, request.effect_id, index)
        if recorded is None:
            turn = await self._scored(grant, request)
            try:
                recorded = await self.store.record(turn, grant.fence, index)
            except Fenced:
                raise Refused(
                    Failure.KEY, "this key's attempt was taken over: its turns are no longer recorded"
                ) from None
        if recorded.slot != grant.slot or recorded.scores is None:
            raise Refused(Failure.REQUEST, f"request id {request.effect_id} was used by another request of the run")
        return recorded

    async def _scored(self, grant: Grant, request: ScoreRequest) -> TurnRecord:
        """A scoring turn, scored by the weights the session samples from when it begins, and scored again when they
        stop being served before it ends (`Unserved`), up to `ATTEMPTS` times."""
        sampler = self.sampler(grant)
        if isinstance(sampler, ApiChannel):
            raise Refused(Failure.REQUEST, f"channel {sampler.name} is a hosted API's, which scores no tokens")
        failure: Exception | None = None
        for attempt in range(1, ATTEMPTS + 1):
            try:
                return await self._score_turn(grant, sampler, request, attempt)
            except Unserved as error:
                failure = error
            except NoReplica as error:
                raise ModelEndpointError(str(error)) from None
            except ValueError as error:
                raise Refused(Failure.REQUEST, str(error).splitlines()[-1]) from None
        raise ModelEndpointError(f"the scores were not served in {ATTEMPTS} attempts: {failure}")

    async def _score_turn(self, grant: Grant, sampler: Sampler, request: ScoreRequest, attempt: int) -> TurnRecord:
        started, began = time.time(), time.monotonic()
        end = len(request.tokens) if request.end is None else request.end
        adapter, version = await sampler.weights(request.session_id)
        limit = sampler.context_limit
        if limit and end >= limit:  # (vLLM generates a token after the sequence, and drops it)
            raise ContextOverflow(limit - 1)
        scores: Scores = await sampler.score(
            request.tokens, start=request.start, end=request.end, top=request.top, adapter=adapter,
            session=request.session_id, version=version, request=f"{request.effect_id}/{attempt}",
        )  # fmt: skip
        if scores.model is not None and adapter is not None and scores.model != adapter:
            raise Unserved(f"{scores.model} answered for {adapter}")
        held = getattr(sampler, "held", None) or getattr(sampler, "model", None)
        usage = Usage(context_used=end, context_limit=limit or end, input_tokens=end, output_tokens=0)
        return TurnRecord(
            effect_id=request.effect_id,
            run=grant.run,
            run_id=grant.run_id,
            slot=grant.slot,
            channel=sampler.name,
            checkpoint=adapter or held or self.models.get(sampler.name, BASE),
            depth=version,
            prompt=array("i", request.tokens[:end]),
            completion=[],
            mask=[],
            logprobs=[],
            result=SampleResult(message=Message(role=Role.ASSISTANT), finish_reason=FinishReason.STOP, usage=usage),
            episode=grant.episode,
            attempt=grant.attempt,
            timings={
                "started": round(started, 3),
                "attempt": attempt,
                "seconds": round(time.monotonic() - began, 4),
            },
            sampled_with=tuple(getattr(sampler, "sampled_with", TOKEN_LEVEL)),
            use=SCORE,
            scores=scores,
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
        """How many tokens a prompt renders to with the channel's renderer: what a turn's prompt would hold. A hosted
        API's channel has no renderer, and counts none (`Refused`)."""
        sampler = self.sampler(grant)
        if isinstance(sampler, ApiChannel):
            raise Refused(Failure.REQUEST, f"channel {sampler.name} is a hosted API's, which renders messages itself: "
                          "it counts no tokens here")  # fmt: skip
        renderer = sampler.renderer
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
        """For a channel whose engines are in this process (or on a hosted API), what it guarantees a session
        (`contract`): what a runner that records through this gateway tells its programs of the channel."""
        channel = gateway.channels.get(name) or gateway.hosted.get(name)
        return {"contract": contract_of(channel).model_dump(mode="json")} if channel is not None else {}

    async def models(request: Request) -> Response:
        run = request.query_params.get("run")
        if run is not None:  # (a run's channels, as its start names them, each with what it guarantees)
            built = await gateway.directory.load(run) if gateway.directory is not None else {}
            await asyncio.gather(*(channel.refresh() for channel in built.values()), return_exceptions=True)
            names = [qualified(run, name) for name in built]
            contracts = [{"contract": contract_of(channel).model_dump(mode="json")} for channel in built.values()]
        else:
            names, contracts = gateway.names, [hosted(name) for name in gateway.names]
        listed = [
            {"id": name, "object": "model", "type": "model", "display_name": name, "created": 0, "owned_by": "rollout"}
            | contract
            for name, contract in zip(names, contracts, strict=True)
        ]
        page = {"has_more": False, "first_id": names[0] if names else None, "last_id": names[-1] if names else None}
        return JSONResponse({"object": "list", "data": listed} | page)

    def answering(format: Format) -> Callable[[Request], Awaitable[Response]]:
        async def answer(request: Request) -> Response:
            try:
                grant = await gateway.granted(key(request))
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
            grant = await gateway.granted(key(request))
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
            grant = await gateway.granted(key(request))
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

    async def scores(request: Request) -> Response:
        try:
            grant = await gateway.granted(key(request))
            try:
                asked = ScoreRequest.model_validate(await request.json())
            except (ValidationError, ValueError, TypeError) as error:
                raise Refused(Failure.REQUEST, f"the request could not be read: {error}") from None
            reply = await gateway.score(grant, asked)
        except Refused as error:
            return _native_error(error.failure.value, str(error))
        except ModelEndpointError as error:
            return _native_error(type(error).__name__, str(error), getattr(error, "context_limit", None))
        assert reply.scores is not None
        said_scores = {
            "start": reply.scores.start,
            "logprobs": reply.scores.logprobs,
            "top_tokens": reply.scores.top_tokens,
            "top_logprobs": reply.scores.top_logprobs,
        }
        return JSONResponse(said_scores, headers=said(reply))

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
            Route(f"{SERVED_UNDER}/scores", scores, methods=["POST"]),
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


def contract_of(sampler: "Sampler | ApiChannel", limits: Limits | None = None) -> CapabilityContract:
    """What a channel guarantees a session: its context, and room for thinking and an answer (by `limits`, else the
    channel's own). Where either budget is unset, a reply may take all the context its prompt leaves (the most output
    is the context limit), or on a hosted API the most its model writes."""
    limits = limits or sampler.limits
    context = sampler.context_limit
    most = sampler.max_output_tokens if isinstance(sampler, ApiChannel) else context
    room = most if limits.thinking is None or limits.answer is None else limits.thinking + limits.answer
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
