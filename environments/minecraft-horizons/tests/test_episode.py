"""The horizon episode on a made-up world and a scripted model (no server): the clock in every observation, a game
that lasts its whole budget, and the reward and result it reports."""

import json
import math
from collections.abc import Mapping, Sequence

from pydantic import JsonValue

from minecraft_horizons.episode import HorizonEpisode
from minecraft_horizons.tasks import TASKS
from minecraft_horizons.worlds import KIND, OPERATIONS, HorizonWorlds
from minecraft_team.episode import TICKS_PER_MINUTE
from minecraft_team.tasks import TEAM
from rollout.contracts import (
    CapabilityContract,
    FinishReason,
    RunEvent,
    RunEventType,
    SampleRequest,
    SampleResult,
    Text,
    ToolCall,
    ToolResult,
    ToolSpecification,
    Usage,
)
from rollout.harness import (
    DirectModel,
    ModelBinding,
    PoolBinding,
    ProgramReference,
    Reach,
    RunBinding,
    RunSpecification,
    RunStatus,
    SandboxPool,
    SandboxSpec,
    register,
)
from rollout.local import LocalRunner
from rollout.testing import payload, tool_call_reply

CREW = ["ada", "ben"]
TASK = "iron-fresh-5m"
WINDOW = 400
"""Game ticks of each made-up window: 20 seconds, so five minutes take fifteen windows."""


class MadeUpWorld:
    """Worlds without a server: every window runs `WINDOW` ticks; the score is twelve iron, of which two were the
    team's at the start, so ten count."""

    kind = KIND
    size = 4
    deduplicates = False

    def __init__(self, done_after: int | None = None) -> None:
        self.windows = 0
        self.done_after = done_after
        """Windows after which a speedrun's goal is reached (none: never)."""
        self.made: dict[str, SandboxSpec] = {}

    def operations(self) -> Sequence[ToolSpecification]:
        return HorizonWorlds().operations()

    async def create(self, handle: str, spec: SandboxSpec, environment: Mapping[str, str]) -> Reach:
        self.made[handle] = spec
        return Reach()

    async def delete(self, handle: str) -> None:
        self.made.pop(handle, None)

    async def held(self) -> Sequence[str]:
        return list(self.made)

    async def call(
        self, handle: str, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        assert name in OPERATIONS, name
        value: JsonValue
        match name:
            case "observe":
                value = observation(str(arguments["agent"]))
            case "act":
                value = {"started": True}
            case "brief":
                value = {"sites": [{"x": 10, "y": 64, "z": 20}]}
            case "window":
                self.windows += 1
                value = {"ticks": WINDOW, "done": self.done_after is not None and self.windows >= self.done_after}
            case _:
                value = {"reward": math.log1p(10.0), "amount": 10.0, "objective": "iron", "unit": "iron"}
        return ToolResult(content=[Text(text=json.dumps(value))], structured=value)


def observation(agent: str) -> dict[str, JsonValue]:
    return {
        "self": {
            "name": agent,
            "position": {"x": 0, "y": 64, "z": 0},
            "dimension": "overworld",
            "health": 20,
            "food": 20,
            "holding": None,
            "wearing": {},
            "inventory": {},
        },
        "world": {"time": {"phase": "day"}, "sky": True, "biome": "plains"},
        "map": {"center": {"x": 0, "y": 64, "z": 0}, "radius": 0, "palette": [], "layers": []},
        "notable": [],
        "items": [],
        "teammates": [],
        "mobs": [],
        "animals": [],
        "messages": [],
        "last_action": None,
        "died": False,
    }


class Waiting:
    """Waits every turn, and keeps every request it gets."""

    def __init__(self) -> None:
        self.requests: list[SampleRequest] = []

    def describe(self, session_id: str) -> CapabilityContract:
        return CapabilityContract(context_limit=1_000_000, max_output_tokens=1_000)

    async def cancel(self, effect_id: str) -> None:
        pass

    async def sample(self, request: SampleRequest) -> SampleResult:
        self.requests.append(request)
        call = ToolCall(call_id=f"c{len(self.requests)}", name="wait", arguments={})
        usage = Usage(context_used=10, context_limit=1_000_000, input_tokens=10)
        return SampleResult(message=tool_call_reply(call), finish_reason=FinishReason.TOOL_USE, usage=usage)


async def play(model: Waiting, world: MadeUpWorld, task: str) -> list[RunEvent]:
    runner = LocalRunner(providers={"scripted": lambda _: model}, pools={"worlds": SandboxPool(world)})
    binding = RunBinding(
        models={name: ModelBinding(direct=DirectModel(provider="scripted", model="m")) for name in TEAM},
        pools={KIND: PoolBinding(local="worlds")},
    )
    parameters: dict[str, JsonValue] = {"task": task, "world_seed": 1, "layout_seed": 2, "names": list[JsonValue](CREW)}
    handle = await runner.start(
        RunSpecification(
            program=ProgramReference(program=register(HorizonEpisode), parameters=parameters), binding=binding
        )
    )
    outcome = await handle.result()
    assert outcome.status is RunStatus.COMPLETED, outcome
    return [event async for event in handle.events()]


async def test_the_game_lasts_its_budget_every_observation_shows_the_clock_and_the_amount_is_the_reward() -> None:
    model, world = Waiting(), MadeUpWorld()
    events = await play(model, world, TASK)
    (result,) = [payload(event)["payload"] for event in events if event.type is RunEventType.OUTPUT_EMITTED]
    assert isinstance(result, dict)
    windows = math.ceil(5 * TICKS_PER_MINUTE / WINDOW)
    assert world.windows == windows and result["turns"] == windows and result["ended"] == "game time"
    assert result["amount"] == 10.0 and result["solved"] is True and result["budget_minutes"] == 5
    assert result["amount_per_minute"] == 10.0 / result["game_minutes"]  # type: ignore[operator]
    rewards = [payload(event) for event in events if event.type is RunEventType.REWARD_ASSIGNED]
    assert sorted(str(each["slot"]) for each in rewards) == ["agent-1", "agent-2"]  # (each playing agent, the same)
    assert all(each["value"] == math.log1p(10.0) for each in rewards)
    acting = [request for request in model.requests if request.session_id.endswith("/agent-1")]
    assert len(acting) == windows
    first, last = acting[0].context.append[-1].text, acting[-1].context.append[-1].text
    assert first.startswith(f"Time left: 5.0 of 5 minutes of game time; {TASKS[TASK].turns} of {TASKS[TASK].turns}")
    left = 5 - (windows - 1) * WINDOW / TICKS_PER_MINUTE
    assert last.startswith(f"Time left: {left:.1f} of 5 minutes of game time")
    assert "You are ada" in first


async def test_a_speedrun_ends_when_its_goal_is_reached() -> None:
    model, world = Waiting(), MadeUpWorld(done_after=3)
    events = await play(model, world, "nether-portal-kit-10m")
    (result,) = [payload(event)["payload"] for event in events if event.type is RunEventType.OUTPUT_EMITTED]
    assert isinstance(result, dict)
    assert world.windows == 3 and result["turns"] == 3  # (of the thirty a ten-minute budget of such windows holds)
    assert result["ended"] == "goal reached" and result["solved"] is True


async def test_a_building_task_tells_agents_the_sites_its_world_laid_out() -> None:
    model, world = Waiting(), MadeUpWorld()
    await play(model, world, "huts-fresh-5m")
    system = model.requests[0].context.append[0].text  # (built once the world said where its sites are)
    assert "1 at (10, 64, 20)" in system and "71 blocks a hut" in system
