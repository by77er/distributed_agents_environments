"""The swarm episode on a made-up world and a scripted model (no server): what each agent's context holds as the
game goes on, when the team compacts, how an episode ends and what it reports. Under the local runner and the
durable one."""

import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from minecraft_swarm.catalog import Swarm
from minecraft_swarm.episode import TICKS_PER_MINUTE, SwarmEpisode, action, answer
from minecraft_swarm.prompts import COMPACT, NO_CALL, ONE_CALL
from minecraft_swarm.tasks import TEAM, catalog
from minecraft_swarm.worlds import DROP_TICKS, OPERATIONS, WINDOW_TICKS, MinecraftTools, MinecraftWorlds
from rollout.contracts import (
    CapabilityContract,
    FinishReason,
    Message,
    RunEventType,
    SampleRequest,
    SampleResult,
    Text,
    ToolCall,
    ToolResult,
    ToolResultBlock,
    ToolSpecification,
    Usage,
)
from rollout.harness import (
    DirectModel,
    Memory,
    ModelBinding,
    ProgramReference,
    RunBinding,
    RunSpecification,
    RunStatus,
    ToolBinding,
    register,
)
from rollout.local import LocalRunner
from rollout.testing import payload, tool_call_reply

LIMIT, OUTPUT = 5_000, 1_400
"""The scripted model's context limit and the room it may use to answer. It counts 100 tokens a message."""
TURNS = 19
SHORT_WINDOW = 110
"""Game ticks of a made-up window, unless a test says otherwise: `TURNS` of them fit in the first task's game time."""
FULL_WINDOW = WINDOW_TICKS + DROP_TICKS
"""Game ticks of a window that no action ends early."""


class MadeUpWorld:
    """The `minecraft` tool set without a server: each agent stands one block further east every turn."""

    deduplicates = False

    def __init__(self, ticks: int = SHORT_WINDOW) -> None:
        self.observed: dict[str, int] = dict.fromkeys(TEAM, 0)
        self.ticks = ticks
        """Game time each window takes."""

    def specifications(self) -> Sequence[ToolSpecification]:
        return MinecraftTools(MinecraftWorlds()).specifications()

    async def call(
        self, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        assert name in OPERATIONS, name  # an episode asks only for what the tool set offers
        value: JsonValue
        match name:
            case "begin":
                value = {"episode": "e-1"}
            case "observe":
                agent = str(arguments["agent"])
                self.observed[agent] += 1
                value = observation(agent, self.observed[agent])
            case "act":
                value = {"started": True}
            case "window":
                value = {"ticks": self.ticks, "done": False}
            case "score":
                value = {"reward": 3.0, "solved": True, "team_diamonds": 3}
            case _:
                value = {"ended": True}
        return ToolResult(content=[Text(text=json.dumps(value))], structured=value)


def observation(agent: str, turn: int) -> dict[str, JsonValue]:
    cells: list[JsonValue] = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    return {
        "self": {
            "name": agent,
            "position": {"x": turn, "y": 64, "z": 0},
            "dimension": "overworld",
            "health": 20,
            "food": 20,
            "holding": None,
            "wearing": {},
            "inventory": {},
        },
        "world": {"time": {"phase": "day"}, "sky": True, "biome": "plains"},
        "map": {
            "center": {"x": turn, "y": 64, "z": 0},
            "radius": 1,
            "palette": [{"name": "stone", "solid": True}, {"name": "air", "solid": False}],
            "layers": [{"dy": 0, "y": 64, "cells": cells}],
        },
        "notable": [{"block": "chest", "count": 1, "x": turn + 2, "y": 64, "z": 0, "distance": 2.0, "also": []}],
        "items": [],
        "teammates": [],
        "mobs": [],
        "animals": [],
        "messages": [],
        "last_action": {"action": {"name": "wait"}, "ok": True, "waited": True} if turn > 1 else None,
        "died": False,
    }


class Remembering:
    """Waits every turn; asked what to remember, answers with a numbered summary. Keeps every request it gets. It
    reports 100 tokens of input per message; the prompts of the agent named `wordy` take 600 more than the others'."""

    def __init__(self, wordy: str | None = None) -> None:
        self.requests: list[SampleRequest] = []
        self.summaries: dict[str, int] = {}
        self.wordy = wordy

    def describe(self, session_id: str) -> CapabilityContract:
        return CapabilityContract(context_limit=LIMIT, max_output_tokens=OUTPUT)

    async def cancel(self, effect_id: str) -> None:
        pass

    async def sample(self, request: SampleRequest) -> SampleResult:
        self.requests.append(request)
        agent = request.session_id.rsplit("/", 1)[-1]
        tokens = 100 * len(request.context.append) + (600 if agent == self.wordy else 0)
        usage = Usage(context_used=tokens, context_limit=LIMIT, input_tokens=tokens)
        if request.context.append[-1].text == COMPACT:
            self.summaries[agent] = self.summaries.get(agent, 0) + 1
            summary = Message.assistant(f"SUMMARY {self.summaries[agent]} for {agent}")
            return SampleResult(message=summary, finish_reason=FinishReason.STOP, usage=usage)
        call = ToolCall(call_id=f"c{len(self.requests)}", name="wait", arguments={})
        return SampleResult(message=tool_call_reply(call), finish_reason=FinishReason.TOOL_USE, usage=usage)


def specification(turns: int | None = TURNS) -> RunSpecification:
    binding = RunBinding(
        models={name: ModelBinding(direct=DirectModel(provider="scripted", model="m")) for name in TEAM},
        imports={"minecraft": ToolBinding(local="minecraft")},
    )
    parameters: dict[str, JsonValue] = {"task": "t001", "turns": turns}
    return RunSpecification(
        program=ProgramReference(program=register(SwarmEpisode), parameters=parameters), binding=binding
    )


def texts(request: SampleRequest) -> list[str]:
    return [message.text for message in request.context.append]


def of(model: Remembering, agent: str) -> tuple[list[SampleRequest], list[SampleRequest]]:
    """An agent's acting requests and its compaction requests, in order."""
    mine = [request for request in model.requests if request.session_id.endswith(f"/{agent}")]
    compactions = [request for request in mine if texts(request)[-1] == COMPACT]
    return [request for request in mine if texts(request)[-1] != COMPACT], compactions


@pytest.mark.parametrize("runner_kind", ["local", "durable"])
async def test_an_agent_sees_the_map_once_and_remembers_its_turns_in_brief_and_older_ones_as_a_summary(
    runner_kind: str, tmp_path: Path
) -> None:
    model = Remembering()
    world = MadeUpWorld()
    runner: Any
    if runner_kind == "durable":
        pytest.importorskip("dbos")
        from rollout_durable import DurableRunner

        runner = DurableRunner(
            tmp_path / "state", providers={"scripted": lambda _: model}, tool_sets={"minecraft": world}
        )
        await runner.launch()
    else:
        runner = LocalRunner(providers={"scripted": lambda _: model}, tool_sets={"minecraft": world})
    try:
        handle = await runner.start(specification())
        outcome = await handle.result()
        events = [event async for event in handle.events()]
    finally:
        if runner_kind == "durable":
            await runner.close()
    assert outcome.status is RunStatus.COMPLETED, outcome
    (result,) = [payload(event)["payload"] for event in events if event.type is RunEventType.OUTPUT_EMITTED]
    assert isinstance(result, dict) and result["turns"] == TURNS and result["compactions"] != 0
    # How it went, in the game's terms: nothing here ended the game early, and its time is game minutes.
    assert result["solved"] is True and result["saturated"] is False and result["ended"] == "turns"
    assert result["duration"] == pytest.approx(TURNS * SHORT_WINDOW / TICKS_PER_MINUTE)

    acting, compactions = of(model, "ada")
    assert len(acting) == TURNS and len(compactions) == result["compactions"]
    assert max(100 * len(request.context.append) for request in acting) <= LIMIT - OUTPUT  # always room to reply

    # Only the current observation carries a map; remembered turns keep what was in sight.
    late = texts(acting[8])
    assert sum("Map of what you have seen" in text for text in late) == 1
    assert "Map of what you have seen" in late[-1] and "You are ada, at (9, 64, 0)" in late[-1]
    assert "You are ada, at (1, 64, 0)" in late[1] and "Notable in sight: chest at (3, 64, 0)" in late[1]

    # A compaction shows the older turns once more, with no tools; from then on its summary stands for them.
    first = compactions[0]
    assert not first.tools and "You are ada, at (1, 64, 0)" in texts(first)[1]
    after = next(request for request in acting if "SUMMARY 1 for ada" in texts(request)[1])
    assert "your own summary" in texts(after)[1] and not any("at (1, 64, 0)" in text for text in texts(after))


async def test_the_team_compacts_in_the_same_turn() -> None:
    model = Remembering(wordy="cy")  # cy's memory fills two turns before the others' would
    runner = LocalRunner(providers={"scripted": lambda _: model}, tool_sets={"minecraft": MadeUpWorld()})
    handle = await runner.start(specification())
    assert (await handle.result()).status is RunStatus.COMPLETED
    when: dict[str, list[int]] = {name: [] for name in TEAM}  # the turn each compaction came before
    turn = dict.fromkeys(TEAM, 0)
    for request in model.requests:
        agent = request.session_id.rsplit("/", 1)[-1]
        if texts(request)[-1] == COMPACT:
            when[agent].append(turn[agent] + 1)
        else:
            turn[agent] += 1
    assert when["cy"] and all(turns == when["cy"] for turns in when.values()), when


async def test_an_episode_ends_when_its_turns_are_spent_however_little_game_time_they_took() -> None:
    async def play(ticks: int) -> dict[str, Any]:
        runner = LocalRunner(
            providers={"scripted": lambda _: Remembering()}, tool_sets={"minecraft": MadeUpWorld(ticks)}
        )
        handle = await runner.start(specification(turns=None))
        assert (await handle.result()).status is RunStatus.COMPLETED
        (result,) = [payload(e)["payload"] for e in handle.recorded_events() if e.type is RunEventType.OUTPUT_EMITTED]
        assert isinstance(result, dict)
        return result

    first = catalog()[0]  # t001: three minutes of game time, and so 36 turns
    game_ticks = first.minutes * TICKS_PER_MINUTE
    quick = await play(ticks=20)  # actions that end at once: a second of game time a turn
    assert quick["turns"] == first.turns == 36 and quick["ended"] == "turns"
    assert quick["duration"] == pytest.approx(first.turns * 20 / TICKS_PER_MINUTE)
    slow = await play(ticks=FULL_WINDOW)  # full windows: the game time runs out first
    assert slow["turns"] == math.ceil(game_ticks / FULL_WINDOW) < first.turns and slow["ended"] == "game time"


def test_every_reply_is_answered_whatever_it_called() -> None:
    seen = Message.user("You are ada.")
    done = {"last_action": {"action": {"name": "mine"}, "ok": True, "mined": "stone"}}
    silent = Memory(turns=[[seen, Message.assistant("I wonder.")]])
    answer(silent, done)
    assert silent.turns[0][-1].text == NO_CALL

    first = ToolCall(call_id="c1", name="mine", arguments={"x": 1, "y": 2, "z": 3})
    second = ToolCall(call_id="c2", name="move", arguments={"direction": "north"})
    eager = Memory(turns=[[seen, tool_call_reply(first).model_copy(update={"content": [first, second]})]])
    answer(eager, done)
    results = [block for block in eager.turns[0][-1].content if isinstance(block, ToolResultBlock)]
    said: list[Any] = [block.result.content[0] for block in results]
    assert [part.text for part in said] == ["mined: stone", ONE_CALL]
    # An argument called `name` does not rename the action.
    assert action(ToolCall(call_id="c1", name="craft", arguments={"name": "oak_planks", "count": 4})) == {
        "name": "craft",
        "count": 4,
    }


def test_the_catalog_offers_every_task_and_draws_one_start_for_a_whole_group() -> None:
    swarm = Swarm()
    rows = swarm.rows()
    assert len(rows) == 100 and rows[0].key == "t001" and rows[0].parameters == {"task": "t001"}
    assert [row.key for row in Swarm(only=("t003", "t007")).rows()] == ["t003", "t007"]
    start = swarm.start(rows[6], random.Random(5))
    assert isinstance(start, dict) and set(start) == {"task", "world_seed", "layout_seed"} and start["task"] == "t007"
    starts: list[Any] = [swarm.start(rows[0], random.Random(seed)) for seed in range(200)]
    worlds = {start["world_seed"] for start in starts}
    assert len(worlds) == 12  # a dozen worlds, each generated once
    assert swarm.program.program.endswith("SwarmEpisode")
