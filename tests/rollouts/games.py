"""Two small tasks for the tests of everything above a run: a guessing game on the task loop, and a gate that holds
a run open until the test lets it through."""

import asyncio
from typing import Any

from rollout.core.contracts import Message
from rollout.core.harness import End, Observation, RunContext, Task

GATES: dict[str, asyncio.Event] = {}
"""By name: a `Gated` run waits for its gate to be set."""


class Guess(Task):
    """Say the word in `parameters["word"]` within `parameters["tries"]` turns: one point for saying it, and the
    result says whether it was said and how many turns it took."""

    def __init__(self, parameters: Any = None) -> None:
        super().__init__(parameters)
        self.word = str(parameters["word"])
        self.tries = int(parameters.get("tries", 1))
        self.turns = 0
        if parameters.get("broken") == "at once":
            raise RuntimeError("this row cannot be made")

    async def setup(self, run: RunContext) -> None:
        if self.parameters.get("broken"):
            raise RuntimeError("this row cannot be set up")

    async def start(self, run: RunContext) -> Observation:
        return Observation("Say the word.")

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        self.turns += 1
        said = reply.text.strip() == self.word
        if said or self.turns >= self.tries:
            await run.emit("result", {"solved": said, "saturated": said, "duration": self.turns})
            return End(reward=1.0 if said else 0.0)
        return Observation("Not that. Again.")


class Gated(Guess):
    """A guess that does not begin until the test opens its gate (`parameters["gate"]`)."""

    async def start(self, run: RunContext) -> Observation:
        await GATES.setdefault(str(self.parameters["gate"]), asyncio.Event()).wait()
        return await super().start(run)
