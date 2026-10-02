"""An agent's memory over a long episode: recent turns without their maps, and of everything older a summary the
agent writes itself when its oldest turns are compacted. On a made-up world and a scripted model (no server), under
the local runner and the durable one."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from minecraft_swarm.episode import COMPACT_AT, KEEP_TURNS, SwarmEpisode
from minecraft_swarm.prompts import COMPACT, TEAM
from minecraft_swarm.worlds import MinecraftTools, MinecraftWorlds
from pydantic import JsonValue

from rollout.core.contracts import (
    CapabilityContract,
    FinishReason,
    Message,
    RunEventType,
    SampleRequest,
    SampleResult,
    Text,
    ToolCall,
    ToolResult,
    ToolSpecification,
    Usage,
)
from rollout.core.harness import (
    DirectModel,
    ModelBinding,
    ProgramReference,
    RunBinding,
    RunSpecification,
    RunStatus,
    ToolBinding,
    register,
)
from rollout.core.local import LocalRunner
from rollout.core.testing import payload, tool_call_reply

DROPPED = COMPACT_AT - KEEP_TURNS
"""Turns a compaction replaces with the summary."""
TURNS = COMPACT_AT + DROPPED + 2
"""Long enough for two compactions: before turn `COMPACT_AT + 1`, and `DROPPED` turns later."""


class MadeUpWorld:
    """The `minecraft` tool set without a server: each agent stands one block further east every turn."""

    def __init__(self) -> None:
        self.observed: dict[str, int] = dict.fromkeys(TEAM, 0)

    def specifications(self) -> Sequence[ToolSpecification]:
        return MinecraftTools(MinecraftWorlds()).specifications()

    async def call(
        self, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
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
                value = {"ticks": 110, "done": False}
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
        "world": {"time": {"ticks": 1000, "phase": "day"}, "light": 15, "sky": True, "biome": "plains"},
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
    """Waits every turn; asked what to remember, answers with a numbered summary. Keeps every request it gets."""

    def __init__(self) -> None:
        self.requests: list[SampleRequest] = []
        self.summaries: dict[str, int] = {}

    def describe(self, session_id: str) -> CapabilityContract:
        return CapabilityContract(context_limit=100_000, max_output_tokens=1_000)

    async def cancel(self, effect_id: str) -> None:
        pass

    async def sample(self, request: SampleRequest) -> SampleResult:
        self.requests.append(request)
        usage = Usage(context_used=1, context_limit=100_000)
        if request.context.append[-1].text == COMPACT:
            agent = request.session_id.rsplit("/", 1)[-1]
            self.summaries[agent] = self.summaries.get(agent, 0) + 1
            summary = Message.assistant(f"SUMMARY {self.summaries[agent]} for {agent}")
            return SampleResult(message=summary, finish_reason=FinishReason.STOP, usage=usage)
        call = ToolCall(call_id=f"c{len(self.requests)}", name="wait", arguments={})
        return SampleResult(message=tool_call_reply(call), finish_reason=FinishReason.TOOL_USE, usage=usage)


def specification() -> RunSpecification:
    binding = RunBinding(
        models={name: ModelBinding(direct=DirectModel(provider="scripted", model="m")) for name in TEAM},
        imports={"minecraft": ToolBinding(local="minecraft")},
    )
    parameters: dict[str, JsonValue] = {"task": "t001", "turns": TURNS}
    return RunSpecification(
        program=ProgramReference(program=register(SwarmEpisode), parameters=parameters), binding=binding
    )


def texts(request: SampleRequest) -> list[str]:
    return [message.text for message in request.context.append]


@pytest.mark.parametrize("runner_kind", ["local", "durable"])
async def test_old_turns_are_compacted_into_a_summary_and_the_context_stays_bounded(
    runner_kind: str, tmp_path: Path
) -> None:
    model = Remembering()
    world = MadeUpWorld()
    runner: Any
    if runner_kind == "durable":
        pytest.importorskip("dbos")
        from rollout.durable import DurableRunner

        runner = DurableRunner(
            tmp_path / "state", providers={"scripted": lambda _: model}, tool_sets={"minecraft": world}
        )
        await runner.launch()
    else:
        runner = LocalRunner(providers={"scripted": lambda _: model}, tool_sets={"minecraft": world})
    try:
        handle = await runner.start(specification())
        outcome = await handle.result()
        events = handle.recorded_events()
    finally:
        if runner_kind == "durable":
            await runner.close()
    assert outcome.status is RunStatus.COMPLETED, outcome
    (result,) = [payload(event)["payload"] for event in events if event.type is RunEventType.OUTPUT_EMITTED]
    assert isinstance(result, dict) and result["turns"] == TURNS and result["compactions"] == 2

    ada = [request for request in model.requests if request.session_id.endswith("/ada")]
    compactions = [request for request in ada if texts(request)[-1] == COMPACT]
    acting = [request for request in ada if texts(request)[-1] != COMPACT]
    assert len(acting) == TURNS and len(compactions) == 2

    # However long the episode, a context holds at most the system prompt, a summary, the recent turns (what was
    # seen, the reply, how it went) and the current observation.
    assert max(len(request.context.append) for request in acting) == 1 + 1 + 3 * (COMPACT_AT - 1) + 1
    assert [len(request.context.append) for request in acting[:3]] == [2, 5, 8]

    # Only the current observation carries a map; remembered turns keep what was in sight.
    before = texts(acting[COMPACT_AT - 1])  # the last turn before the first compaction
    assert sum("Map of what you have seen" in text for text in before) == 1
    assert "Map of what you have seen" in before[-1] and f"You are ada, at ({COMPACT_AT}, 64, 0)" in before[-1]
    assert "You are ada, at (1, 64, 0)" in before[1] and "Notable in sight: chest at (3, 64, 0)" in before[1]

    # The first compaction: the oldest turns are shown once more, and no tools are offered.
    first = compactions[0]
    assert not first.tools and len(first.context.append) == 1 + 3 * DROPPED + 1
    assert "You are ada, at (1, 64, 0)" in texts(first)[1]
    assert f"You are ada, at ({DROPPED}, 64, 0)" in texts(first)[-4]
    assert not any(f"You are ada, at ({DROPPED + 1}, 64, 0)" in text for text in texts(first))

    # From then on the summary stands for them.
    after = texts(acting[COMPACT_AT])
    assert after[1].endswith("SUMMARY 1 for ada") and "your own summary" in after[1]
    assert f"You are ada, at ({DROPPED + 1}, 64, 0)" in after[2]
    assert not any("You are ada, at (1, 64, 0)" in text for text in after)
    assert len(after) == 1 + 1 + 3 * KEEP_TURNS + 1

    # The second compaction builds on the first summary, and replaces it.
    second = texts(compactions[1])
    assert second[1].endswith("SUMMARY 1 for ada") and f"You are ada, at ({DROPPED + 1}, 64, 0)" in second[2]
    assert texts(acting[-1])[1].endswith("SUMMARY 2 for ada")
