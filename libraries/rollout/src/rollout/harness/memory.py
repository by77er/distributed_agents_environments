"""Memory for long episodes: an agent's recent turns as they were, and its own summary of everything older.

A long game outgrows any model's context. `Memory` keeps a context that fits, whatever the model: when the model
reports that its context is nearly full, the agent is shown its oldest turns once more, with its earlier summary, and
asked what to remember (`prompt`); its answer replaces them. That is a sample like any other, on the agent's own model
slot, with room for the summary and none to think it over. A prompt that overflows all the same is compacted and
tried again.

Each request says how it follows from the ones before (`SampleLink`): a summary's request is a `compaction_attempt` of
the latest reply's, and the request that goes on from a summary names that summary's as its `compaction`, so a recording
endpoint can tell what each sample was for.

Code that uses it says nothing about tokens or limits. It says what a turn should look like once it is no longer
the current one (`remember`: an observation without its bulky part, say), and what to ask when turns must go.

`CompactingAgent` is the same for the task loop.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from rollout.contracts import (
    ContextOverflow,
    Message,
    Role,
    SampleLink,
    Text,
    ToolResult,
    ToolResultBlock,
    ToolSpecification,
)
from rollout.harness.agent import Agent
from rollout.harness.context import Model, RunContext
from rollout.harness.history import History
from rollout.harness.model import EFFECT_ID_META

PROMPT = (
    "The turns above are about to leave your memory. Write what you need to remember from them, and from your "
    "earlier summary if there is one, to go on well: what you have learned, what has been done and agreed, and what "
    "you intend to do next. Be specific and brief, and leave out what no longer matters. Reply with the summary only."
)
REMEMBERED = "What you remember from earlier (your own summary):\n{summary}"


@dataclass
class Memory:
    prompt: str = PROMPT
    """What the agent is asked when its oldest turns are compacted into a summary."""
    remembered: str = REMEMBERED
    """How the summary is shown to it afterwards."""
    summary: str = ""
    turns: list[list[Message]] = field(default_factory=list[list[Message]])
    """Its recent turns, oldest first: each a few messages (what it saw, what it replied, how that went)."""
    compactions: int = 0
    _last: int | None = None
    _growth: int = 0
    _sampled: str | None = None
    """The latest reply's effect id."""
    _summarised: str | None = None
    """The effect id of the summary the next request goes on from."""

    def context(self, system: Message | None = None, current: Sequence[Message] = ()) -> list[Message]:
        """The system prompt, the summary, the remembered turns, and what is in front of the agent now."""
        first = [system] if system is not None else []
        summary = [Message.user(self.remembered.format(summary=self.summary))] if self.summary else []
        return [*first, *summary, *(message for turn in self.turns for message in turn), *current]

    def remember(self, *messages: Message) -> None:
        """Add a turn."""
        self.turns.append(list(messages))

    def answer(self, result: str, *, others: str = "Not done: only your first call of a turn counts.") -> None:
        """Close the latest turn with how its reply went: the result of its first tool call (further calls are
        answered with `others`), or, for a reply that called nothing, a note to the agent."""
        if not self.turns:
            return
        latest = self.turns[-1]
        calls = latest[-1].tool_calls
        if not calls:
            latest.append(Message.user(result))
            return
        results = [result, *[others] * (len(calls) - 1)]
        latest.append(
            Message(
                role=Role.TOOL,
                content=[
                    ToolResultBlock(call_id=call.call_id, result=ToolResult(content=[Text(text=text)]))
                    for call, text in zip(calls, results, strict=True)
                ],
            )
        )

    def crowded(self, model: Model) -> bool:
        """Whether one more turn might leave the model less than its full room to reply (by what its last prompt
        took, which the model reports, and by how much a turn has been seen to add). The room kept is the contract's
        most output, up to a quarter of the context: a model with no output budget may reply up to its whole context,
        which no compaction could keep free."""
        usage = model.usage
        if usage is None or usage.input_tokens is None:
            return False
        if self._last is not None:
            self._growth = max(self._growth, usage.input_tokens - self._last)
        self._last = usage.input_tokens
        margin = 2 * self._growth or usage.context_limit // 10
        room = min(model.capabilities.max_output_tokens, usage.context_limit // 4)
        return usage.input_tokens + margin > usage.context_limit - room

    async def compact(self, model: Model, system: Message | None = None, *, keep: int | None = None) -> None:
        """Replace the oldest turns with what the agent says it needs to remember of them. The newest `keep` stay as
        they are (by default the newest third)."""
        keep = len(self.turns) // 3 if keep is None else keep
        old = self.turns[: len(self.turns) - keep]
        if not old:
            return
        context = [*Memory(summary=self.summary, turns=old, remembered=self.remembered).context(system)]
        try:
            room = min(model.capabilities.max_output_tokens, model.capabilities.context_limit // 20)
            compacted = [SampleLink(type="compaction_attempt", source=self._sampled)] if self._sampled else []
            reply = await model.sample([*context, Message.user(self.prompt)], max_output_tokens=room, links=compacted)
            summary = reply.text.strip()
        except ContextOverflow:  # too much even to reread: it is forgotten unsummarized
            reply, summary = None, ""
        self._summarised = _effect_of(reply) if summary else None
        if summary:  # (a reply with no text keeps the earlier summary; the old turns go either way)
            self.summary = summary
        self.turns = self.turns[len(self.turns) - keep :]
        self.compactions += 1
        self._last = None  # (the next prompt is shorter: it says nothing of how turns grow)

    async def sample(
        self,
        model: Model,
        *,
        system: Message | None = None,
        current: Sequence[Message] = (),
        tools: Sequence[ToolSpecification] = (),
        keep: int = 0,
    ) -> Message:
        """One reply to the context. If the model refuses the context as too long, memory is compacted and the reply
        asked for again (the newest `keep` turns are never compacted)."""
        while True:
            accepted = [SampleLink(type="compaction", source=self._summarised)] if self._summarised else []
            try:
                reply = await model.sample(self.context(system, current), tools=tools, links=accepted)
            except ContextOverflow:
                if len(self.turns) <= keep:
                    raise
                spare = len(self.turns) - keep
                await self.compact(model, system, keep=keep if spare == 1 else max(keep, len(self.turns) // 3))
                continue
            self._summarised, self._sampled = None, _effect_of(reply)
            return reply


def _effect_of(reply: Message | None) -> str | None:
    """The effect id a reply carries, if it carries one."""
    effect = reply.meta.get(EFFECT_ID_META) if reply is not None else None
    return str(effect) if effect else None


class CompactingAgent(Agent):
    """The default agent, with a memory that fits: one sample of the policy slot per turn, over the recent turns and
    the agent's own summary of the older ones."""

    compact_prompt: str = PROMPT

    def __init__(self, configuration: object = None) -> None:
        super().__init__(configuration)
        self.memory = Memory(prompt=self.compact_prompt)
        self._folded = 0

    async def act(self, run: RunContext, history: History, tools: list[ToolSpecification]) -> Message:
        for turn in history.turns[self._folded :]:  # each: a reply (none for the first) and the observation after it
            reply = [turn.reply] if turn.reply is not None else []
            self.memory.remember(*reply, *(turn.observation.messages if turn.observation is not None else ()))
        self._folded = len(history.turns)
        system = Message.system(self.system_prompt) if self.system_prompt else None
        if self.memory.crowded(run.model) and len(self.memory.turns) > 1:
            await self.memory.compact(run.model, system, keep=max(1, len(self.memory.turns) // 3))
        return await self.memory.sample(run.model, system=system, tools=tools, keep=1)
