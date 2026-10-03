"""A whole swarm episode on a live server, with a scripted policy instead of a model: the lockstep loop, the tool set,
and the shared reward. Needs Java and Node."""

import re
import shutil
from collections.abc import Mapping

import httpx
import pytest

from minecraft_swarm.episode import SwarmEpisode
from minecraft_swarm.tasks import TEAM
from minecraft_swarm.worlds import MinecraftTools, MinecraftWorlds
from rollout.contracts import (
    CapabilityContract,
    FinishReason,
    RunEventType,
    SampleRequest,
    SampleResult,
    ToolCall,
    Usage,
)
from rollout.harness import (
    DirectModel,
    ModelBinding,
    ProgramReference,
    RunBinding,
    RunSpecification,
    RunStatus,
    ToolBinding,
    ToolSet,
    register,
)
from rollout.harness.remote import RemoteToolSet, serve
from rollout.local import LocalRunner
from rollout.testing import payload, tool_call_reply

DROPPED = re.compile(r"(\d+) diamond at \((-?\d+), (-?\d+), (-?\d+)\)")


class WalkToDiamonds:
    """Walks to the nearest diamonds it sees; waits otherwise. Stateless: decided by the last observation."""

    def describe(self, session_id: str) -> CapabilityContract:
        return CapabilityContract(context_limit=100_000, max_output_tokens=1_000)

    async def sample(self, request: SampleRequest) -> SampleResult:
        last = request.context.append[-1].text
        found = DROPPED.search(last)
        if found:
            _, x, y, z = found.groups()
            call = ToolCall(call_id="c", name="move_to", arguments={"x": int(x), "y": int(y), "z": int(z)})
        else:
            call = ToolCall(call_id="c", name="wait", arguments={})
        return SampleResult(
            message=tool_call_reply(call),
            finish_reason=FinishReason.TOOL_USE,
            usage=Usage(context_used=1, context_limit=100_000),
        )

    async def cancel(self, effect_id: str) -> None:
        pass


def binding() -> RunBinding:
    return RunBinding(
        models={name: ModelBinding(direct=DirectModel(provider="scripted", model="walk")) for name in TEAM},
        imports={"minecraft": ToolBinding(local="minecraft")},
    )


@pytest.mark.skipif(shutil.which("java") is None or shutil.which("node") is None, reason="Java and Node are needed")
@pytest.mark.parametrize("through", ["in process", "over HTTP"])
async def test_a_scripted_swarm_picks_up_diamonds_and_shares_the_reward(through: str) -> None:
    worlds = MinecraftWorlds()
    tools: ToolSet = MinecraftTools(worlds)
    if through == "over HTTP":  # the worlds served as a tool set, as from a machine of their own
        transport = httpx.ASGITransport(app=serve(tools))
        client = httpx.AsyncClient(transport=transport, base_url="http://worlds", timeout=300)
        tools = RemoteToolSet("http://worlds", client=client, specifications=tools.specifications())
    runner = LocalRunner(providers={"scripted": lambda model: WalkToDiamonds()}, tool_sets={"minecraft": tools})
    names = ["ada", "ben", "cy", "dee"]
    parameters: Mapping[str, object] = {"task": "t001", "world_seed": 12345, "layout_seed": 3, "turns": 4}
    parameters = {**parameters, "names": names}
    specification = RunSpecification(
        program=ProgramReference(program=register(SwarmEpisode), parameters=dict(parameters)),  # type: ignore[arg-type]
        binding=binding(),
    )
    try:
        handle = await runner.start(specification)
        outcome = await handle.result()
    finally:
        await worlds.close()
    assert outcome.status is RunStatus.COMPLETED, outcome
    events = handle.recorded_events()
    rewards = {str(payload(e)["slot"]): payload(e)["value"] for e in events if e.type is RunEventType.REWARD_ASSIGNED}
    assert set(rewards) == set(TEAM) and len(set(rewards.values())) == 1  # the swarm shares one reward
    (result,) = [payload(e)["payload"] for e in events if e.type is RunEventType.OUTPUT_EMITTED]
    assert isinstance(result, dict)
    diamonds = result["team_diamonds"]
    assert isinstance(diamonds, int) and diamonds > 0 and rewards[TEAM[0]] == diamonds, result
    assert result["objective"] == "diamonds" and result["solved"] is True
    turns = result["turns"]
    assert isinstance(turns, int) and 1 <= turns <= 4  # it ends early once every diamond is held
    assert (turns < 4) == (diamonds == result["available_diamonds"])
    assert isinstance(result["duration"], float) and 0 < result["duration"] <= 3
    assert result["saturated"] == (diamonds == result["available_diamonds"])
    world_operations = [
        e for e in events if e.type is RunEventType.EFFECT_REQUESTED and payload(e).get("kind") == "tool.call"
    ]
    assert len(world_operations) > 4 * 4, "every world operation is a recorded effect"
