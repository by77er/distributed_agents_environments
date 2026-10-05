"""A runner's programs sampling through the gateway: the episode they leave through replicas at a URL is the one a
gateway in the runner's own process leaves (the same trajectories, in the same blob), a sample whose answer was lost is
retried without being recorded twice, the turns of an attempt that was taken over are neither recorded further nor
read into the episode, and the links a program declares are kept with its turns."""

from pathlib import Path
from typing import Any, cast

import httpx

from rollout.contracts import CapabilityContract, Message, RunEvent
from rollout.harness import (
    End,
    Memory,
    ModelBinding,
    Observation,
    RecordedModel,
    RunBinding,
    RunContext,
    RunSpecification,
    Task,
    agent_program,
)
from rollout.harness.model import EndpointModel
from rollout.local import LocalRunner
from rollout.local.context import LocalRunContext
from rollout_train.gateway import Attempt, GatewayEndpoints, TurnStore, create_app, unaccepted
from rollout_train.rollouts.episodes import Outcome, assemble, loaded, stored
from rollout_train.testing import admitted, gateway_endpoints, keyring
from tests.rollout_train.gateway.support import EchoEngine, echo_channel, gateway_over, stores

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
    return {"policy": CapabilityContract(context_limit=channel.context_limit, max_output_tokens=channel.context_limit)}


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


async def test_an_episode_played_through_the_gateway_elsewhere_is_the_episode_one_in_process_leaves(
    tmp_path: Path,
) -> None:
    ledger, blobs = stores(tmp_path)
    channel = echo_channel()
    app = create_app(gateway_over(channel, ledger, blobs))
    http = httpx.AsyncClient(transport=Losing(app, losing=2))
    endpoints = GatewayEndpoints("http://gateway", keyring(), TurnStore(ledger, blobs), contracts(), http=http)
    endpoints.admit("r_same", Attempt("train", await ledger.take("runs/train/episodes/1/1"), "1/1", 1))
    events = await played(LocalRunner(gateway=endpoints), "r_same")
    through_gateway = assemble(events, await endpoints.sessions("train", "r_same"), run="train", group=1, number=1)
    assert (
        len(cast(EchoEngine, channel.engines[0]).prompts) == 3
    )  # (three turns: the two answers lost were not sampled again)

    here_ledger, here_blobs = stores(tmp_path / "here")
    here = gateway_endpoints(echo_channel(), ledger=here_ledger, blobs=here_blobs)
    here.admit("r_same", Attempt("train", await here_ledger.take("runs/train/episodes/1/1"), "1/1", 1))
    events = await played(LocalRunner(gateway=here), "r_same")
    in_process = assemble(events, await here.sessions("train", "r_same"), run="train", group=1, number=1)

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
    events = await played(LocalRunner(gateway=endpoints), "r_new")
    try:
        await endpoint.sample(sample_request([Message.user("Walk on.")], "r_old:0:1", session_id="r_old/policy"))
        raise AssertionError("a stale attempt's sample was recorded")
    except Exception as error:
        assert "taken over" in str(error)
    episode = assemble(events, await endpoints.sessions("train", "r_new"), run="train", group=1, number=1)
    spans = [span for segment in episode.trajectories["policy"].segments for span in segment.spans]
    assert spans and all(span.effect_id.startswith("r_new:") for span in spans)
    assert [turn.effect_id for turn in await endpoints.store.turns("train", "r_old")] == ["r_old:0:0"]


async def test_the_links_a_program_declares_are_kept_with_its_turns(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    endpoints = gateway_endpoints(echo_channel(), ledger=ledger, blobs=blobs)
    await admitted(endpoints, "r_mem")
    endpoint = endpoints.endpoint(POLICY)
    policy = EndpointModel(endpoint, "r_mem/policy", LocalRunContext("r_mem", {"policy": endpoint}))
    memory = Memory()
    for step in range(3):
        reply = await memory.sample(policy, current=[Message.user(f"Step {step}.")])
        memory.remember(Message.user(f"Step {step}."), reply)
    await memory.compact(policy, keep=1)  # (as a team compacts: its summary is a sample of its own)
    await memory.sample(policy, current=[Message.user("Go on.")])
    turns = await endpoints.store.turns("train", "r_mem")
    links = [[(link.type, link.source) for link in turn.links] for turn in turns]
    assert links == [[], [], [], [("compaction_attempt", turns[2].effect_id)], [("compaction", turns[3].effect_id)]]
    assert unaccepted(turns) == set()  # (it went on from its summary: what it sampled for it is trained on)
