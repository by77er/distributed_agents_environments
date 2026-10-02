"""An agent's memory over a long episode: recent turns without their maps, and of everything older a summary the
agent writes itself when its oldest turns are compacted. On a made-up world and a scripted model (no server), under
the local runner and the durable one."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from minecraft_swarm.episode import NO_CALL, ONE_CALL, SUMMARY_TOKENS, TURN_GROWTH, Memory, SwarmEpisode, action, answer
from minecraft_swarm.prompts import COMPACT, TEAM
from minecraft_swarm.worlds import MinecraftTools, MinecraftWorlds
from pydantic import JsonValue

from rollout.core.contracts import (
    CapabilityContract,
    ContextOverflow,
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

LIMIT, OUTPUT = 5_050, 1_400
"""The scripted model's context limit and the room it may use to answer. It counts 100 tokens a message, so with
`TURN_GROWTH` a context is crowded from 30 messages on: after turn 11, and again after turn 17."""
TURNS = 19


class MadeUpWorld:
    """The `minecraft` tool set without a server: each agent stands one block further east every turn."""

    def __init__(self, ticks: int = 110) -> None:
        self.observed: dict[str, int] = dict.fromkeys(TEAM, 0)
        self.ticks = ticks
        """Game time each window takes."""

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
    """Waits every turn; asked what to remember, answers with a numbered summary. Keeps every request it gets.

    It reports 100 tokens of input per message, as an endpoint reports the tokens a prompt took; or, with `overflow`,
    reports nothing and refuses any acting context of more than that many messages. The prompts of the agent named
    `wordy` take 600 tokens more than the others'."""

    def __init__(self, overflow: int | None = None, wordy: str | None = None) -> None:
        self.requests: list[SampleRequest] = []
        self.summaries: dict[str, int] = {}
        self.overflow = overflow
        self.wordy = wordy

    def describe(self, session_id: str) -> CapabilityContract:
        return CapabilityContract(context_limit=LIMIT, max_output_tokens=OUTPUT)

    async def cancel(self, effect_id: str) -> None:
        pass

    async def sample(self, request: SampleRequest) -> SampleResult:
        messages = len(request.context.append)
        compacting = request.context.append[-1].text == COMPACT
        if self.overflow is not None and not compacting and messages > self.overflow:
            raise ContextOverflow(LIMIT)
        self.requests.append(request)
        agent = request.session_id.rsplit("/", 1)[-1]
        tokens = None if self.overflow is not None else 100 * messages + (600 if agent == self.wordy else 0)
        usage = Usage(context_used=tokens or 1, context_limit=LIMIT, input_tokens=tokens)
        if compacting:
            self.summaries[agent] = self.summaries.get(agent, 0) + 1
            summary = Message.assistant(f"SUMMARY {self.summaries[agent]} for {agent}")
            return SampleResult(message=summary, finish_reason=FinishReason.STOP, usage=usage)
        call = ToolCall(call_id=f"c{len(self.requests)}", name="wait", arguments={})
        return SampleResult(message=tool_call_reply(call), finish_reason=FinishReason.TOOL_USE, usage=usage)


def specification(turns: int = TURNS) -> RunSpecification:
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
async def test_a_crowded_context_is_compacted_into_a_summary_and_stays_bounded(
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

    acting, compactions = of(model, "ada")
    assert len(acting) == TURNS and len(compactions) == 2

    # A context grows by a turn (what was seen, the reply, how it went) until it is crowded; it never passes that.
    assert [len(request.context.append) for request in acting[:3]] == [2, 5, 8]
    assert max(len(request.context.append) for request in acting) == 32
    crowded = acting[10]  # turn 11: 32 messages, 3,200 tokens: one more turn would cut into the room to answer
    assert 100 * len(crowded.context.append) + TURN_GROWTH > LIMIT - OUTPUT

    # Only the current observation carries a map; remembered turns keep what was in sight.
    before = texts(crowded)
    assert sum("Map of what you have seen" in text for text in before) == 1
    assert "Map of what you have seen" in before[-1] and "You are ada, at (11, 64, 0)" in before[-1]
    assert "You are ada, at (1, 64, 0)" in before[1] and "Notable in sight: chest at (3, 64, 0)" in before[1]

    # The compaction: the older half of the turns is shown once more, and no tools are offered.
    first = compactions[0]
    assert not first.tools and len(first.context.append) == 1 + 3 * 6 + 1
    assert "You are ada, at (1, 64, 0)" in texts(first)[1] and "You are ada, at (6, 64, 0)" in texts(first)[-4]
    assert not any("You are ada, at (7, 64, 0)" in text for text in texts(first))

    # From then on the summary stands for them.
    after = texts(acting[11])
    assert after[1].endswith("SUMMARY 1 for ada") and "your own summary" in after[1]
    assert "You are ada, at (7, 64, 0)" in after[2] and len(after) == 1 + 1 + 3 * 5 + 1
    assert not any("You are ada, at (1, 64, 0)" in text for text in after)

    # The second compaction builds on the first summary, and replaces it.
    second = texts(compactions[1])
    assert second[1].endswith("SUMMARY 1 for ada") and "You are ada, at (7, 64, 0)" in second[2]
    assert texts(acting[-1])[1].endswith("SUMMARY 2 for ada")


async def test_the_team_compacts_in_the_same_turn_and_a_compaction_has_room_for_the_summary_only() -> None:
    model = Remembering(wordy="cy")  # cy's context is crowded two turns before the others' would be
    runner = LocalRunner(providers={"scripted": lambda _: model}, tool_sets={"minecraft": MadeUpWorld()})
    handle = await runner.start(specification())
    assert (await handle.result()).status is RunStatus.COMPLETED
    when: dict[str, list[int]] = {name: [] for name in TEAM}  # the turn each compaction came before
    turn = dict.fromkeys(TEAM, 0)
    for request in model.requests:
        agent = request.session_id.rsplit("/", 1)[-1]
        if texts(request)[-1] == COMPACT:
            when[agent].append(turn[agent] + 1)
            assert request.max_output_tokens == SUMMARY_TOKENS and not request.tools
        else:
            turn[agent] += 1
    assert when["cy"] and when["cy"][0] == 10  # (the others alone would compact before turn 12)
    assert all(turns == when["cy"] for turns in when.values()), when


async def test_an_episode_ends_when_its_turns_are_spent_however_little_game_time_they_took() -> None:
    def spec() -> RunSpecification:
        binding = RunBinding(
            models={name: ModelBinding(direct=DirectModel(provider="scripted", model="m")) for name in TEAM},
            imports={"minecraft": ToolBinding(local="minecraft")},
        )
        program = ProgramReference(program=register(SwarmEpisode), parameters={"task": "t001"})
        return RunSpecification(program=program, binding=binding)

    async def play(ticks: int) -> dict[str, Any]:
        runner = LocalRunner(
            providers={"scripted": lambda _: Remembering()}, tool_sets={"minecraft": MadeUpWorld(ticks)}
        )
        handle = await runner.start(spec())
        assert (await handle.result()).status is RunStatus.COMPLETED
        (result,) = [payload(e)["payload"] for e in handle.recorded_events() if e.type is RunEventType.OUTPUT_EMITTED]
        assert isinstance(result, dict)
        return result

    # t001 has three minutes of game time, and so 36 turns.
    quick = await play(ticks=20)  # actions that end at once: a second of game time a turn
    assert quick["turns"] == 36 and quick["ended"] == "turns" and quick["game_minutes"] == 0.6
    slow = await play(ticks=110)  # full windows: the game time runs out first
    assert slow["turns"] == 33 and slow["ended"] == "game time"


async def test_a_context_that_overflows_is_compacted_and_tried_again() -> None:
    model = Remembering(overflow=20)  # says nothing of its tokens, and refuses contexts of more than 20 messages
    runner = LocalRunner(providers={"scripted": lambda _: model}, tool_sets={"minecraft": MadeUpWorld()})
    handle = await runner.start(specification(turns=12))
    outcome = await handle.result()
    assert outcome.status is RunStatus.COMPLETED, outcome  # an overflow does not end the episode
    acting, compactions = of(model, "ada")
    assert len(acting) == 12 and compactions
    assert max(len(request.context.append) for request in acting) <= 20
    assert texts(acting[-1])[1].startswith("What you remember from earlier in this game")


def test_every_reply_is_answered_whatever_it_called() -> None:
    from rollout.core.contracts import Message, Role, ToolResultBlock

    seen = Message.user("You are ada.")
    done = {"last_action": {"action": {"name": "mine"}, "ok": True, "mined": "stone"}}

    silent = Memory(turns=[[seen, Message.assistant("I wonder.")]])
    answer(silent, done)
    assert silent.turns[0][-1].role is Role.USER and silent.turns[0][-1].text == NO_CALL

    first = ToolCall(call_id="c1", name="mine", arguments={"x": 1, "y": 2, "z": 3})
    second = ToolCall(call_id="c2", name="move", arguments={"direction": "north"})
    eager = Memory(turns=[[seen, tool_call_reply(first).model_copy(update={"content": [first, second]})]])
    answer(eager, done)
    results = [block for block in eager.turns[0][-1].content if isinstance(block, ToolResultBlock)]
    assert [block.call_id for block in results] == ["c1", "c2"]  # each call has its answer
    assert results[0].result.content[0].text == "mined: stone"  # type: ignore[union-attr]
    assert results[1].result.content[0].text == ONE_CALL  # type: ignore[union-attr]


def test_an_argument_called_name_does_not_rename_the_action() -> None:
    call = ToolCall(call_id="c1", name="craft", arguments={"name": "oak_planks", "count": 4})
    assert action(call) == {"name": "craft", "count": 4}
