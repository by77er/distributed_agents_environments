"""A child process for the attempt-marker test: a program runs a slow command in an environment and is killed."""

import asyncio
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from rollout.core.contracts import OutcomeUnknown
from rollout.core.harness import (
    EnvironmentSpecification,
    ExecutionResult,
    ModelSlot,
    Program,
    ProgramReference,
    RunBinding,
    RunContext,
    RunSpecification,
    register,
)
from rollout.durable import DurableRunner

RUN_ID = "r_guardtest"


class LedgerEnvironments:
    """A fake environment service that records every command it starts, and takes a few seconds per command."""

    def __init__(self, ledger: Path) -> None:
        self.ledger = ledger

    async def create(self, environment_id: str, specification: EnvironmentSpecification) -> None:
        self._write({"operation": "create", "environment_id": environment_id})

    async def execute(
        self, environment_id: str, command: str, *, timeout_seconds: float, cwd: str | None, effect_id: str = ""
    ) -> ExecutionResult:
        self._write({"operation": "execute", "command": command})
        await asyncio.sleep(3)
        return ExecutionResult(exit_code=0, output="done\n")

    async def put(self, environment_id: str, path: str, data: bytes) -> None: ...
    async def get(self, environment_id: str, path: str) -> bytes:
        return b""

    async def destroy(self, environment_id: str) -> None:
        self._write({"operation": "destroy", "environment_id": environment_id})

    def _write(self, entry: Mapping[str, str]) -> None:
        with self.ledger.open("a") as file:
            file.write(json.dumps(entry) + "\n")


class Deploy(Program):
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


async def main(mode: str, directory: Path) -> None:
    runner = DurableRunner(directory / "state", environments=LedgerEnvironments(directory / "ledger.jsonl"))
    await runner.launch()
    if mode == "start":
        specification = RunSpecification(
            program=ProgramReference(program=register(Deploy)), binding=RunBinding(models={})
        )
        await runner.start(specification, run_id=RUN_ID)
        await asyncio.sleep(3600)  # killed by the test
    else:
        outcome = await asyncio.wait_for(runner.run(RUN_ID).result(), 60)
        print("OUTCOME " + outcome.model_dump_json(), flush=True)
    await runner.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
