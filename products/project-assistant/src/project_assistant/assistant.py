"""The project assistant: a long-lived conversation about one code repository.

The task waits for messages, answers them with the help of imported repository and notes tools, and emits each
answer as a `reply` output. Follow-ups the assistant schedules are `WaitFor` timeouts: when one is due and no message
has arrived, the loop wakes the assistant with a follow-up observation.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from rollout.contracts import Message, ToolSpecification
from rollout.harness import Agent, History, Observation, RunContext, Task, WaitFor, tool

SYSTEM_PROMPT = """You are a project assistant for the code repository "{repository}". You talk with its developers in \
one long conversation that can span days. The current time is {now}.

Ground every answer in the repository: search, list and read files, and check git history before you answer. Cite \
files as path:line. Say so when you cannot find something rather than guessing.

You have notes that persist across conversations. Search them when earlier decisions or preferences may matter. Save a \
note when the developer asks you to remember something, or when a decision or finding will matter later.

When asked to follow up or remind later, call schedule_follow_up. When a follow-up is due you will be woken with it; \
act on it and write a message for the developer.

Answer concisely."""


@dataclass
class FollowUp:
    identifier: int
    due: datetime
    about: str


class ProjectAssistant(Task):
    imports = ["repository", "notes"]

    def __init__(self, parameters: dict[str, str] | None = None) -> None:
        self.follow_ups: list[FollowUp] = []
        self.scheduled = 0

    async def start(self, run: RunContext) -> WaitFor:
        return self._wait(run)

    async def respond(self, run: RunContext, reply: Message) -> Observation | WaitFor:
        if reply.tool_calls:
            return await self.run_tools(run, reply)
        if reply.text:
            origin = run.conversation.origin if run.conversation else None
            await run.emit("reply", reply.text, to=origin)
        return self._wait(run)

    @tool
    async def schedule_follow_up(self, run: RunContext, minutes_from_now: float, about: str) -> str:
        """Schedule a follow-up: you will be woken after this many minutes if no message arrives first; otherwise it
        stays scheduled. Describe what to do in `about`."""
        due = run.now() + timedelta(minutes=minutes_from_now)
        self.scheduled += 1
        self.follow_ups.append(FollowUp(self.scheduled, due, about))
        return f"Follow-up scheduled for {due.isoformat(timespec='minutes')}: {about}"

    def _wait(self, run: RunContext) -> WaitFor:
        """Wait for the next message, or until the next follow-up is due."""
        woken = self._woken(run)
        self.follow_ups = [follow_up for follow_up in self.follow_ups if follow_up.identifier not in woken]
        if not self.follow_ups:
            return WaitFor("message")
        next_up = min(self.follow_ups, key=lambda follow_up: follow_up.due)
        wake = Observation(
            f"(Follow-up due, scheduled for {next_up.due.isoformat(timespec='minutes')}) {next_up.about}\n"
            "Act on it now; the developer will read your reply.",
            info={"follow_up": next_up.identifier},
        )
        return WaitFor("message", timeout=max(next_up.due - run.now(), timedelta(0)), on_timeout=wake)

    @staticmethod
    def _woken(run: RunContext) -> set[int]:
        """Follow-ups that already fired: their wake observations are in the history."""
        return {
            int(turn.observation.info["follow_up"])
            for turn in run.history.turns
            if turn.observation is not None and "follow_up" in turn.observation.info
        }


class ProjectAgent(Agent):
    """The default agent, with a system prompt that names the repository and the current time."""

    def __init__(self, configuration: dict[str, str] | None = None) -> None:
        super().__init__(configuration)
        self.repository_name = (configuration or {}).get("repository_name", "the project")

    async def act(self, run: RunContext, history: History, tools: list[ToolSpecification]) -> Message:
        now = run.now().isoformat(timespec="minutes")
        system = Message.system(SYSTEM_PROMPT.format(repository=self.repository_name, now=now))
        return await run.model.sample([system, *history.messages(run.context_hints)], tools=tools)
