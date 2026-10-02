"""The swarm episode: four agents, one shared reward, the world frozen while they think.

Each turn, every agent observes, thinks and calls one action tool, all at once while the world is frozen; then the
world runs one window while the actions happen. The episode ends when its budget of game time or of turns is spent,
or earlier when nothing is left to earn. Its reward, the task's objective scored from the plugin's ground truth,
goes to every agent: the swarm is rewarded equally.

Each agent is a model slot of its own (`ada`, `ben`, `cy`, `dee`): its own context and its own recorded session.
Bound to the same recorded channel, they are one policy. The team talks through the game's chat: an observation
shows the last few messages an agent has heard or said, each with its age in turns.

What an agent remembers is what its context holds (`Memory`): the current observation in full, with the map; its
recent turns without their maps (what was in sight, what it did, how that went); and, of everything older, a summary
it wrote itself. When an agent's context is nearly full (by the tokens its last prompt actually took, which the
model endpoint reports), every agent with turns to spare is shown its oldest turns once more, with its earlier
summary, and asked what to remember (`prompts.COMPACT`); its answer replaces them. The team compacts in the same
turn because a turn takes as long as its slowest agent: four compactions in one turn cost what one does, and in four
turns four times that. A compaction is a sample like any other, on the agent's own slot, with room for the summary
and none for thinking it over: it costs no game time, it is recorded, and it is trained with the episode's
advantage, since what an agent chooses to remember is part of how it plays. An episode of any length thus keeps a
context that fits; should a prompt overflow all the same, the agent compacts and tries again.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import JsonValue

from minecraft_swarm.prompts import ACTIONS, CHAT_LINES, COMPACT, REMEMBERED, TEAM, describe, system_prompt
from minecraft_swarm.tasks import TURNS_PER_MINUTE, Task, catalog
from rollout.core.contracts import ContextOverflow, Message, Role, Text, ToolCall, ToolResult, ToolResultBlock
from rollout.core.harness import Model, ModelSlot, Program, RunContext

TURN_GROWTH = 700
"""Tokens one more turn may add to a context (a reply, its result, a larger observation): memory is compacted when
the last prompt, grown by this, would leave less than the room the model may use to think and answer."""
KEEP_CHARACTERS = 3500
"""Recent turns a compaction leaves as they are: the newest that fit this many characters (about 1,000 tokens, four
or five turns), and never more than half of them."""
MAX_SUMMARY = 2400
"""Characters of summary kept (the engine's limit on an answer is the tighter one)."""
NO_CALL = "You called no tool, so you did nothing that turn."
ONE_CALL = "Not done: only your first call of a turn counts."
SUMMARY_TOKENS = 400
"""Output a compaction may take: the summary, written straight out."""
SPARE_TURNS = 4
"""An agent compacts along with a crowded teammate if it remembers at least this many turns."""
TICKS_PER_MINUTE = 1200
SATURATED = "nothing left to earn"
"""How a result says its episode ended when the team reached everything the task has to give."""


@dataclass
class Memory:
    """What one agent remembers besides what it sees now."""

    summary: str = ""
    """Its own account of everything older than `turns`."""
    turns: list[list[Message]] = field(default_factory=list[list[Message]])
    """Its recent turns, oldest first: each is what it saw (without the map), what it replied, and how that went."""
    compactions: int = 0

    def messages(self) -> list[Message]:
        """Memory as the start of a context, after the system prompt."""
        remembered = [Message.user(REMEMBERED.format(summary=self.summary))] if self.summary else []
        return [*remembered, *(message for turn in self.turns for message in turn)]


class SwarmEpisode(Program):
    """Parameters: `task` (an id from the catalog), `world_seed`, `layout_seed`; optionally `minutes` (a shorter
    budget of game time) and `turns` (a lower cap on turns than the task's own)."""

    def __init__(self, parameters: Mapping[str, JsonValue] | None = None) -> None:
        parameters = parameters or {}
        tasks = {task.id: task for task in catalog()}
        self.task: Task = tasks[str(parameters.get("task", "t001"))]
        self.world_seed = int(cast(int, parameters.get("world_seed", 12345)))
        self.layout_seed = int(cast(int, parameters.get("layout_seed", 0)))
        self.minutes = min(self.task.minutes, float(cast(float, parameters.get("minutes", self.task.minutes))))
        turns = parameters.get("turns")
        budget = round(self.minutes * TURNS_PER_MINUTE)  # the task's budget of turns, for the game time it is given
        self.max_turns = budget if turns is None else min(budget, int(cast(int, turns)))
        self.chat: dict[str, list[tuple[int, str, str]]] = {name: [] for name in TEAM}
        """What each agent has heard and said lately: (turn, speaker, message), oldest first."""

    def model_slots(self) -> Mapping[str, ModelSlot]:
        return {name: ModelSlot() for name in TEAM}

    def imports(self) -> list[str]:
        return ["minecraft"]

    async def main(self, run: RunContext) -> None:
        begun = await self._call(
            run, "begin", {"task": self.task.id, "world_seed": self.world_seed, "layout_seed": self.layout_seed}
        )
        episode = str(begun["episode"])
        memories = {name: Memory() for name in TEAM}
        budget = self.minutes * TICKS_PER_MINUTE
        spent = 0.0
        turn = 0
        try:
            done = False
            while spent < budget and turn < self.max_turns:
                turn += 1
                observations = await run.gather(
                    *(self._call(run, "observe", {"episode": episode, "agent": name}) for name in TEAM)
                )
                for name, observation in zip(TEAM, observations, strict=True):
                    answer(memories[name], observation)
                if any(crowded(run.models[name]) for name in TEAM):
                    await run.gather(
                        *(
                            self._compact(run, name, memories[name])
                            for name in TEAM
                            if crowded(run.models[name]) or len(memories[name].turns) >= SPARE_TURNS
                        )
                    )
                actions = await run.gather(
                    *(
                        self._think(run, name, turn, observation, memories[name])
                        for name, observation in zip(TEAM, observations, strict=True)
                    )
                )
                await run.gather(
                    *(
                        self._call(run, "act", {"episode": episode, "agent": name, "action": action})
                        for name, action in zip(TEAM, actions, strict=True)
                    )
                )
                window = await self._call(run, "window", {"episode": episode})
                spent += float(cast(int, window["ticks"]))
                if window.get("done"):
                    done = True
                    break
            score = await self._call(run, "score", {"episode": episode})
        finally:
            await self._call(run, "end", {"episode": episode})
        reward = float(cast(float, score["reward"]))
        for name in TEAM:  # the swarm is rewarded equally
            run.reward(reward, slot=name)
        compactions = max(memory.compactions for memory in memories.values())
        ended = SATURATED if done else "game time" if spent >= budget else "turns"
        result = {
            "task": self.task.id,
            "turns": turn,
            "game_minutes": spent / TICKS_PER_MINUTE,
            "ended": ended,
            **score,
        }
        await run.emit("result", {**result, "compactions": compactions})

    async def _think(
        self, run: RunContext, name: str, turn: int, observation: Mapping[str, Any], memory: Memory
    ) -> dict[str, JsonValue]:
        """One agent's turn: its context, one sample, and the action it chose (`idle` if it chose none)."""
        model = run.models[name]
        heard = self.chat[name]
        # (What a teammate said reached this agent after the turn it was said in: the turn before this one.)
        heard.extend((turn - 1, str(said["from"]), str(said["message"])) for said in observation.get("messages", []))
        del heard[:-CHAT_LINES]
        chat = [(turn - at, who, message) for at, who, message in heard]
        seen = Message.user(describe(observation, chat=chat))
        while True:
            context = [Message.system(system_prompt(self.task)), *memory.messages(), seen]
            try:
                reply = await model.sample(context, tools=ACTIONS)
                break
            except ContextOverflow:  # the estimate was short: make room and ask again
                if not memory.turns:
                    raise
                await self._compact(run, name, memory, keep=0 if len(memory.turns) == 1 else None)
        memory.turns.append([Message.user(describe(observation, recalled=True)), reply])
        calls = reply.tool_calls
        if not calls:
            return {"name": "idle"}
        call: ToolCall = calls[0]  # one action per turn
        said = " ".join(str(call.arguments.get("message", "")).split())[:240]
        if call.name == "chat" and said:  # an agent sees what it said among what it heard
            heard.append((turn, name, said))
        return action(call)

    async def _compact(self, run: RunContext, name: str, memory: Memory, keep: int | None = None) -> None:
        """Replace the agent's oldest turns with what it says it needs to remember of them. `keep` turns stay as they
        are (by default the newest that fit `KEEP_CHARACTERS`, and at most half)."""
        if keep is None:
            keep, size = 0, 0
            for turn in reversed(memory.turns[len(memory.turns) - len(memory.turns) // 2 :]):
                size += sum(
                    len(message.text) + sum(len(str(call.arguments)) for call in message.tool_calls) for message in turn
                )
                if size > KEEP_CHARACTERS:
                    break
                keep += 1
        old = memory.turns[: len(memory.turns) - keep]
        if not old:
            return
        context = [
            Message.system(system_prompt(self.task)),
            *Memory(memory.summary, old).messages(),
            Message.user(COMPACT),
        ]
        model = run.models[name]
        try:
            reply = await model.sample(
                context, max_output_tokens=min(SUMMARY_TOKENS, model.capabilities.max_output_tokens)
            )
            summary = reply.text.strip()
        except ContextOverflow:  # too much even to reread: it is forgotten unsummarized
            summary = ""
        if summary:  # (a reply with no text keeps the earlier summary; the old turns go either way)
            memory.summary = summary[:MAX_SUMMARY]
        memory.turns = memory.turns[len(memory.turns) - keep :]
        memory.compactions += 1

    async def _call(self, run: RunContext, operation: str, arguments: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        result = await run.tools.call(operation, arguments)
        if result.is_error or not isinstance(result.structured, dict):
            detail = "".join(part.text for part in result.content if isinstance(part, Text))
            raise RuntimeError(f"minecraft.{operation} failed: {detail}")
        return result.structured


def action(call: ToolCall) -> dict[str, JsonValue]:
    """A tool call as the harness takes an action: its arguments, and its name (whatever an argument is called)."""
    return {**dict(call.arguments), "name": call.name}


def answer(memory: Memory, observation: Mapping[str, Any]) -> None:
    """Close the agent's last remembered turn with how its action went (the observation that follows says)."""
    if not memory.turns:
        return
    last = memory.turns[-1]
    previous = last[-1].tool_calls
    if not previous:
        last.append(Message.user(NO_CALL))
        return
    results = [describe_result(observation.get("last_action")), *[ONE_CALL] * (len(previous) - 1)]
    answered = [
        ToolResultBlock(call_id=call.call_id, result=ToolResult(content=[Text(text=result)]))
        for call, result in zip(previous, results, strict=True)
    ]
    last.append(Message(role=Role.TOOL, content=answered))


def crowded(model: Model) -> bool:
    """Whether one more turn would leave the model less than its full room to think and answer."""
    usage = model.usage
    if usage is None or usage.input_tokens is None:
        return False
    return usage.input_tokens + TURN_GROWTH > usage.context_limit - model.capabilities.max_output_tokens


def describe_result(result: Any) -> str:
    if not isinstance(result, dict):
        return "No result."
    outcome = cast(dict[str, Any], result)
    if outcome.get("ok"):
        details = {key: value for key, value in outcome.items() if key not in ("action", "ok")}
        return ", ".join(f"{key}: {item}" for key, item in details.items()) or "Done."
    if outcome.get("interrupted"):
        at = cast(dict[str, Any], outcome.get("now_at") or {})
        return f"Cut off when the world froze; you got to ({at.get('x')}, {at.get('y')}, {at.get('z')})."
    return f"Failed: {outcome.get('error', 'unknown error')}"
