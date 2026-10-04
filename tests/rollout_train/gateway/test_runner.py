"""A runner's programs sampling through the gateway: the episode they leave is the one the in-process recorder leaves
(the same trajectories, in the same blob), a sample whose answer was lost is retried without being recorded twice, and
the turns of an attempt that was taken over are neither recorded further nor read into the episode."""

from pathlib import Path
from typing import Any, cast

import httpx

from rollout.contracts import CapabilityContract, Message, RunEvent
from rollout.harness import (
    End,
    ModelBinding,
    Observation,
    RecordedModel,
    RunBinding,
    RunContext,
    RunSpecification,
    Task,
    agent_program,
)
from rollout.local import LocalRunner
from rollout_train.gateway import Attempt, GatewayEndpoints, TurnStore, create_app
from rollout_train.recorder import Recorder
from rollout_train.rollouts.episodes import Outcome, assemble, loaded, stored
from tests.rollout_train.gateway.support import EchoEngine, echo_channel, gateway_over, keyring, stores

POLICY = RecordedModel(channel="policy")
BINDING = RunBinding(models={"policy": ModelBinding(recorded=POLICY)})


class Walk(Task):
    """Three steps, then a reward."""

    def __init__(self, configuration: object = None) -> None:
        self.steps = 0

    async def start(self, run: RunContext) -> Observation:
        return Observation("Walk three steps.")

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        self.steps += 1
        return End(reward=1.0) if self.steps == 3 else Observation(f"Step {self.steps} done.")


SPECIFICATION = RunSpecification(program=agent_program(Walk), binding=BINDING)


def contracts() -> dict[str, CapabilityContract]:
    channel = echo_channel()
    limits = channel.limits
    return {
        "policy": CapabilityContract(
            context_limit=channel.context_limit, max_output_tokens=limits.thinking + limits.answer
        )
    }


class Losing(httpx.AsyncBaseTransport):
    """Delivers every request, and loses the answers to the first `losing` of them (a replica that died after
    recording, before replying)."""

    def __init__(self, app: Any, losing: int) -> None:
        self.inner = httpx.ASGITransport(app=app)
        self.losing = losing

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self.inner.handle_async_request(request)
        if self.losing:
            self.losing -= 1
            raise httpx.RemoteProtocolError("the connection closed before the answer", request=request)
        return response


async def played(runner: LocalRunner, run_id: str) -> list[RunEvent]:
    handle = await runner.start(SPECIFICATION, run_id=run_id)
    await handle.result()
    return handle.recorded_events()


async def test_an_episode_played_through_the_gateway_is_the_episode_the_in_process_recorder_leaves(
    tmp_path: Path,
) -> None:
    ledger, blobs = stores(tmp_path)
    channel = echo_channel()
    app = create_app(gateway_over(channel, ledger, blobs))
    http = httpx.AsyncClient(transport=Losing(app, losing=2))
    endpoints = GatewayEndpoints("http://gateway", keyring(), TurnStore(ledger, blobs), contracts(), http=http)
    endpoints.admit("r_same", Attempt("train", await ledger.take("runs/train/episodes/1/1"), "1/1", 1))
    events = await played(LocalRunner(recorder=endpoints), "r_same")
    through_gateway = assemble(events, await endpoints.sessions("train", "r_same"), run="train", group=1, number=1)
    assert (
        len(cast(EchoEngine, channel.engines[0]).prompts) == 3
    )  # (three turns: the two answers lost were not sampled again)

    recorder = Recorder({"policy": echo_channel()})
    events = await played(LocalRunner(recorder=recorder), "r_same")
    in_process = assemble(events, recorder.sessions("r_same"), run="train", group=1, number=1)

    assert through_gateway.outcome is Outcome.COMPLETED and through_gateway.reward == 1.0
    assert through_gateway.trajectories == in_process.trajectories
    kept, expected = await stored(through_gateway, [], blobs), await stored(in_process, [], blobs)
    assert kept.trajectories == expected.trajectories and kept.sampled == expected.sampled
    assert (await loaded(kept, blobs)).trajectories == in_process.trajectories


async def test_the_turns_of_an_attempt_taken_over_are_not_recorded_further_nor_read_into_the_episode(
    tmp_path: Path,
) -> None:
    ledger, blobs = stores(tmp_path)
    app = create_app(gateway_over(echo_channel(), ledger, blobs))
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
    endpoints = GatewayEndpoints("http://gateway", keyring(), TurnStore(ledger, blobs), contracts(), http=http)
    first = await ledger.take("runs/train/episodes/1/1")
    endpoints.admit("r_old", Attempt("train", first, "1/1", 1))
    endpoint = endpoints.endpoint(POLICY)
    from rollout_train.testing import sample_request

    await endpoint.sample(sample_request([Message.user("Walk.")], "r_old:0:0", session_id="r_old/policy"))
    second = await ledger.take("runs/train/episodes/1/1")  # another runner claims attempt 2
    endpoints.admit("r_new", Attempt("train", second, "1/1", 2))
    events = await played(LocalRunner(recorder=endpoints), "r_new")
    try:
        await endpoint.sample(sample_request([Message.user("Walk on.")], "r_old:0:1", session_id="r_old/policy"))
        raise AssertionError("a stale attempt's sample was recorded")
    except Exception as error:
        assert "taken over" in str(error)
    episode = assemble(events, await endpoints.sessions("train", "r_new"), run="train", group=1, number=1)
    spans = [span for segment in episode.trajectories["policy"].segments for span in segment.spans]
    assert spans and all(span.effect_id.startswith("r_new:") for span in spans)
    assert [turn.effect_id for turn in await endpoints.store.turns("train", "r_old")] == ["r_old:0:0"]
