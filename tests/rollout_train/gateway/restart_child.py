"""A runner's process for the restart test: an episode runner over a durable runner, recording through a gateway in
its own process, over the ledger and blob store in a directory.

`start` plays until killed: after the gateway has recorded the episode's third turn, the reply never reaches the program
(the process is stuck, as one killed between recording a turn and the program hearing of it would be), and it writes
`recorded` so that the test knows to kill it. `resume` starts again over the same state, adopts the run, and plays it to
its end. The engine notes every generation in `generations.jsonl`, so that the test counts them across processes.
"""

import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from rollout.contracts import Message, SampleRequest, SampleResult
from rollout.harness import End, Observation, RecordedModel, RunContext, Task
from rollout.harness.blobs import FileBlobStore
from rollout_train.gateway import GatewayEndpoints
from rollout_train.inference import Channel, Generation, Limits
from rollout_train.ledger import FileLedger
from rollout_train.presence import FilePresence
from rollout_train.record import table
from rollout_train.recorder import Renderer
from rollout_train.rollouts import EpisodeRunner
from rollout_train.rollouts.scheduler import EPISODES
from rollout_train.testing import PlainRenderer, recording
from tests.rollout_train.gateway.support import EchoEngine

STEPS = 5
STUCK_AFTER = 3
"""The turn whose reply the first process never hears of."""


class Walk(Task):
    """Five steps, then a reward."""

    def __init__(self, parameters: object = None) -> None:
        super().__init__(parameters)
        self.steps = 0

    async def start(self, run: RunContext) -> Observation:
        return Observation("Walk five steps.")

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        self.steps += 1
        return End(reward=1.0) if self.steps == STEPS else Observation(f"Step {self.steps} done.")


class Noting(EchoEngine):
    """The echo engine, noting each prompt it is asked to continue in a file."""

    def __init__(self, notes: Path) -> None:
        super().__init__()
        self.notes = notes

    async def generate(self, prompt: Sequence[int], **options: Any) -> Generation:
        with self.notes.open("a") as file:
            file.write(json.dumps({"prompt": list(prompt)}) + "\n")
        return await super().generate(prompt, **options)


class Stuck:
    """The runner's view of the gateway: after the gateway recorded turn `after`, its reply is never delivered."""

    def __init__(self, endpoints: GatewayEndpoints, after: int, marker: Path) -> None:
        self.endpoints = endpoints
        self.after = after
        self.marker = marker
        self.samples = 0

    def endpoint(self, binding: RecordedModel) -> Any:
        return StuckEndpoint(self, self.endpoints.endpoint(binding))


class StuckEndpoint:
    def __init__(self, stuck: Stuck, inner: Any) -> None:
        self.stuck = stuck
        self.inner = inner

    def describe(self, session_id: str) -> Any:
        return self.inner.describe(session_id)

    def address(self, session_id: str, **options: Any) -> Any:
        return self.inner.address(session_id, **options)

    async def cancel(self, effect_id: str) -> None:
        await self.inner.cancel(effect_id)

    async def sample(self, request: SampleRequest) -> SampleResult:
        result = await self.inner.sample(request)
        self.stuck.samples += 1
        if self.stuck.samples == self.stuck.after:
            self.stuck.marker.write_text(request.effect_id)
            await asyncio.Event().wait()  # (killed here)
        return result


async def main(mode: str, directory: Path) -> None:
    from rollout_durable import DurableRunner

    ledger, blobs = FileLedger(directory / "ledger"), FileBlobStore(directory / "blobs")
    channel = Channel("policy", [Noting(directory / "generations.jsonl")], cast(Renderer, PlainRenderer()), Limits())
    endpoints = recording(channel, ledger=ledger, blobs=blobs)
    served: Any = Stuck(endpoints, STUCK_AFTER, directory / "recorded") if mode == "start" else endpoints
    durable = DurableRunner(directory / "runs", recorder=served)
    runner = EpisodeRunner(
        "here", ledger, durable, endpoints, blobs, places=1, presence=FilePresence(ledger.directory), every=0.05
    )
    await runner.prepare()  # (before the durable runner recovers its runs: they find their claims adopted)
    await durable.launch()
    serving = asyncio.create_task(runner.serve())
    try:
        async with asyncio.timeout(120):
            while "1/1" not in await ledger.read(table("train", EPISODES)):  # noqa: ASYNC110 (another task writes)
                await asyncio.sleep(0.05)
        print("ENDED", flush=True)
    finally:
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
        await durable.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
