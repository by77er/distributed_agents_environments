"""Small tasks for the tests of everything above a run: a guessing game on the task loop, and a gate that holds a run
open until the test lets it through; an environment of the guessing game (`words`), and of it beside a sandbox of
each episode's own (`boxed`, its sandboxes' provider `boxes`); and a guessing game a real model plays (`guessing`).

`guessing` is an example environment for a small model: of three words, the model names the one the start says, and
only what it says after thinking counts. Qwen3-0.6B names the right one about a third of the time, so a group's rewards
differ, and a group-relative update has something to learn from:

    PYTHONPATH=. uv run rollout env check tests.rollout_train.rollouts.games:guessing --profile PROFILE --groups 4
"""

import asyncio
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from rollout.contracts import Message
from rollout.environment import Description, Row, Start, drawn
from rollout.harness import (
    End,
    ModelSlot,
    Observation,
    ProgramReference,
    RunContext,
    SandboxSpec,
    Task,
    agent_program,
)
from rollout.testing import FakeSandboxes

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


class Words:
    """An environment of the guessing game: three rows, each a word to say."""

    program: ProgramReference = agent_program(Guess)
    version = "1"
    description = Description(saturated=True, duration="turns")

    def rows(self) -> Sequence[Row]:
        return [Row(f"say-{word}", f"say {word}", {"word": word}) for word in ("yes", "no", "maybe")]

    def start(self, row: Row, rng: random.Random) -> JsonValue:
        return {**row.parameters, "seed": rng.randrange(1000)}

    def evals(self) -> Mapping[str, Sequence[Start]]:
        return {"words-held-out": drawn(self, seeds=[1, 2])}


words = Words()

CHOICES = ("apple", "river", "candle")


class Choose(Task):
    """Name the word in `parameters["word"]` of `CHOICES`; only what is said after `</think>` counts, and naming every
    word is no guess."""

    def __init__(self, parameters: Any = None) -> None:
        super().__init__(parameters)
        self.word = str(parameters["word"])

    async def start(self, run: RunContext) -> Observation:
        return Observation(
            f"I am thinking of one of these words: {', '.join(CHOICES)}. Reply with that word only. /no_think"
        )

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        answer = reply.text.rsplit("</think>", 1)[-1].lower()  # (what it said, not what it thought)
        named = [word for word in CHOICES if word in answer]
        said = named == [self.word]
        await run.emit("result", {"solved": said, "saturated": said, "duration": 1})
        return End(reward=1.0 if said else 0.0)


class Guessing:
    """The guessing game for a real model: a row for each word of `CHOICES`. A start's seed is never read, so its eval
    starts are the same situations as its training starts: a toy, not a held-out measure."""

    program: ProgramReference = agent_program(Choose)
    version = "1"
    description = Description(saturated=True, duration="turns")

    def rows(self) -> Sequence[Row]:
        return [Row(f"guess-{word}", f"guess {word}", {"word": word}) for word in CHOICES]

    def start(self, row: Row, rng: random.Random) -> JsonValue:
        return {**row.parameters, "seed": rng.randrange(1000)}

    def evals(self) -> Mapping[str, Sequence[Start]]:
        return {"guessing-held-out": drawn(self, seeds=[1, 2, 3])}


guessing = Guessing()


class JudgedGuess(Guess):
    """A guess a judge scores: the policy answers once, and the judge, not trained, says whether the answer is the word;
    a point when its verdict ends with yes."""

    models = {"policy": ModelSlot(), "judge": ModelSlot(trained=False, judge=True)}

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        verdict = await run.models["judge"].sample([Message.user(f"Is {reply.text.strip()!r} the word {self.word}?")])
        said = verdict.text.strip().endswith("yes")
        await run.emit("result", {"solved": said, "saturated": said, "duration": 1})
        return End(reward=1.0 if said else 0.0)


class Judged(Words):
    """The guessing game, scored by a judge."""

    program: ProgramReference = agent_program(JudgedGuess)

    def evals(self) -> Mapping[str, Sequence[Start]]:
        return {"judged-held-out": drawn(self, seeds=[1, 2])}


judged = Judged()

BOXES: list[FakeSandboxes] = []
"""Every provider `boxes` made, in order."""


def boxes(directory: Path, size: int = 4) -> FakeSandboxes:
    """A provider of `fake` sandboxes (the cluster config's `[sandboxes.fake] provider`), at most `size` at once."""
    made = FakeSandboxes(size=size)
    BOXES.append(made)
    return made


class BoxedGuess(Guess):
    """A guess played beside a sandbox: it asks its box who it is, and says so in its result."""

    sandboxes = {"box": SandboxSpec(kind="fake")}

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        described: Any = (await run.sandbox("box").call("describe")).structured
        said = reply.text.strip() == self.word
        await run.emit("result", {"solved": said, "duration": 1, "handle": described["handle"],
                                  "key": run.sandbox("box").lease.key})  # fmt: skip
        return End(reward=1.0 if said else 0.0)


class Boxed(Words):
    """The guessing game, each episode with a sandbox of its own."""

    program: ProgramReference = agent_program(BoxedGuess)
    description = Description(duration="turns")


boxed = Boxed()
