"""A child process for the crash-recovery test: `start` runs an episode until killed; `resume` recovers it.

The model is a ledger: every sample it serves is appended to a file, so the test can count calls across processes.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from rollout.contracts import CapabilityContract, FinishReason, Message, SampleRequest, SampleResult, Usage
from rollout.harness import (
    DirectModel,
    End,
    ModelBinding,
    Observation,
    RunBinding,
    RunContext,
    RunSpecification,
    Task,
    agent_program,
)
from rollout_durable import DurableRunner

TURNS = 5
RUN_ID = "r_crashtest"


class Counting(Task):
    async def start(self, run: RunContext) -> Observation:
        return Observation("turn 1")

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        await run.emit("reply", reply.text)
        if run.turn + 1 >= TURNS:  # run.turn counts completed turns; this reply is not recorded yet
            return End(reward=1.0)
        return Observation(f"turn {run.turn + 2}")


class LedgerEndpoint:
    def __init__(self, ledger: Path) -> None:
        self.ledger = ledger

    def describe(self, session_id: str) -> CapabilityContract:
        return CapabilityContract(context_limit=10_000, max_output_tokens=100)

    async def sample(self, request: SampleRequest) -> SampleResult:
        with self.ledger.open("a") as file:
            file.write(json.dumps({"effect_id": request.effect_id, "at": time.time()}) + "\n")
        await asyncio.sleep(1.0)
        last = request.context.append[-1].text
        return SampleResult(
            message=Message.assistant(f"reply to {last}"),
            finish_reason=FinishReason.STOP,
            usage=Usage(context_used=1, context_limit=10_000),
        )

    async def cancel(self, effect_id: str) -> None:
        pass


async def main(mode: str, directory: Path) -> None:
    ledger = LedgerEndpoint(directory / "ledger.jsonl")
    runner = DurableRunner(directory / "state", providers={"ledger": lambda model: ledger})
    await runner.launch()  # in `resume`, DBOS recovers the unfinished run here
    if mode == "start":
        binding = RunBinding(models={"policy": ModelBinding(direct=DirectModel(provider="ledger", model="x"))})
        await runner.start(RunSpecification(program=agent_program(Counting), binding=binding), run_id=RUN_ID)
        await asyncio.sleep(3600)  # killed by the test
    else:
        outcome = await asyncio.wait_for(runner.run(RUN_ID).result(), 60)
        print("OUTCOME " + outcome.model_dump_json(), flush=True)
    await runner.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
