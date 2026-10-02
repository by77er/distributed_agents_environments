"""What the runners in multi-runner tests execute. Any runner may execute any run, so everything here is importable
and deterministic, and every runner process records what it does to ledger files shared by the test."""

import asyncio
import json
import time
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

from rollout.contracts import (
    CapabilityContract,
    FinishReason,
    Message,
    OutcomeUnknown,
    SampleRequest,
    SampleResult,
    Usage,
)
from rollout.harness import (
    Deployment,
    DirectModel,
    End,
    EnvironmentSpecification,
    ExecutionResult,
    ModelBinding,
    ModelSlot,
    Observation,
    Program,
    ProgramReference,
    RunBinding,
    RunContext,
    RunSpecification,
    Task,
    WaitFor,
    agent_program,
    register,
)

TURNS = 5


def record(ledger: Path, entry: Mapping[str, Any]) -> None:
    with ledger.open("a") as file:
        file.write(json.dumps({**entry, "at": time.time()}) + "\n")


def read(ledger: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []


class EchoModel:
    """Replies "reply to <the last message>", after `delay` seconds; records each sample and which runner made it."""

    def __init__(self, ledger: Path, runner_id: str, delay: float) -> None:
        self.ledger = ledger
        self.runner_id = runner_id
        self.delay = delay

    def describe(self, session_id: str) -> CapabilityContract:
        return CapabilityContract(context_limit=100_000, max_output_tokens=100)

    async def sample(self, request: SampleRequest) -> SampleResult:
        record(self.ledger, {"effect_id": request.effect_id, "runner": self.runner_id})
        await asyncio.sleep(self.delay)
        return SampleResult(
            message=Message.assistant(f"reply to {request.context.append[-1].text}"),
            finish_reason=FinishReason.STOP,
            usage=Usage(context_used=1, context_limit=100_000),
        )

    async def cancel(self, effect_id: str) -> None:
        pass


class SlowEnvironments:
    """An environment service that records what it does, and takes three seconds per command."""

    def __init__(self, ledger: Path, runner_id: str) -> None:
        self.ledger = ledger
        self.runner_id = runner_id

    async def create(self, environment_id: str, specification: EnvironmentSpecification) -> None:
        record(self.ledger, {"operation": "create", "runner": self.runner_id})

    async def execute(
        self, environment_id: str, command: str, *, timeout_seconds: float, cwd: str | None, effect_id: str = ""
    ) -> ExecutionResult:
        record(self.ledger, {"operation": "execute", "runner": self.runner_id, "effect_id": effect_id})
        await asyncio.sleep(3)
        return ExecutionResult(exit_code=0, output="done\n")

    async def put(self, environment_id: str, path: str, data: bytes) -> None: ...

    async def get(self, environment_id: str, path: str) -> bytes:
        return b""

    async def destroy(self, environment_id: str) -> None:
        record(self.ledger, {"operation": "destroy", "runner": self.runner_id})


# Programs


class Chat(Task):
    """A conversation: replies to each message, then waits for the next."""

    wait_seconds = 600.0

    async def start(self, run: RunContext) -> WaitFor:
        return WaitFor("message")

    async def respond(self, run: RunContext, reply: Message) -> WaitFor:
        await run.emit("reply", reply.text)
        return WaitFor("message", timeout=timedelta(seconds=self.wait_seconds))

    async def teardown(self, run: RunContext) -> None:
        if teardowns is not None:
            record(teardowns, {"run_id": run.run_id})


class ShortWait(Chat):
    wait_seconds = 2.0


class Counting(Task):
    """Five turns with the model, one reply each."""

    async def start(self, run: RunContext) -> Observation:
        return Observation("turn 1")

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        await run.emit("reply", reply.text)
        if run.turn + 1 >= TURNS:
            return End(reward=1.0)
        return Observation(f"turn {run.turn + 2}")


class Deploy(Program):
    """Runs one slow command in an environment."""

    def model_slots(self) -> Mapping[str, ModelSlot]:
        return {}

    async def main(self, run: RunContext) -> None:
        assert run.environments is not None
        environment = await run.environments.create()
        try:
            await environment.execute("deploy --production")
            await run.emit("result", "deployed")
        except OutcomeUnknown:
            await run.emit("result", "outcome unknown")


teardowns: Path | None = None
"""Where `Chat.teardown` records the runs it tears down; set by `configure`."""


def _binding(model: str) -> RunBinding:
    return RunBinding(models={"policy": ModelBinding(direct=DirectModel(provider="echo", model=model))})


def deployments() -> list[Deployment]:
    return [
        Deployment(
            name="scenarios/chat", specification=RunSpecification(program=agent_program(Chat), binding=_binding("fast"))
        ),
        Deployment(
            name="scenarios/shortwait",
            specification=RunSpecification(program=agent_program(ShortWait), binding=_binding("fast")),
        ),
    ]


def specification(program: str) -> RunSpecification:
    if program == "counting":
        return RunSpecification(program=agent_program(Counting), binding=_binding("slow"))
    if program == "deploy":
        return RunSpecification(program=ProgramReference(program=register(Deploy)), binding=RunBinding(models={}))
    raise KeyError(program)


def configure(directory: Path, runner_id: str) -> dict[str, Any]:
    """A runner's model providers and environments: the same in every runner, recording to `directory`."""
    global teardowns
    teardowns = directory / "teardowns.jsonl"

    def echo(model: DirectModel) -> EchoModel:
        return EchoModel(directory / "models.jsonl", runner_id, delay=1.0 if model.model == "slow" else 0.0)

    return {"providers": {"echo": echo}, "environments": SlowEnvironments(directory / "environments.jsonl", runner_id)}
