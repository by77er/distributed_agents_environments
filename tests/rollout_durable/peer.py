"""One runner of a multi-runner test: a `DurableRunner` on a shared database, driven by JSON lines on stdin.

    python peer.py DIRECTORY DATABASE_URL RUNNER_ID EVICT_AFTER TAKEOVER_AFTER

It prints {"ready": RUNNER_ID} once launched, then answers each request {"id", "op", ...} with {"id", "result"} or
{"id", "error"}. Requests run concurrently. Operations: start, send, cancel, resident.
"""

import asyncio
import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import scenarios

from rollout.contracts import Text
from rollout.harness import Address, Envelope, Priority
from rollout_durable import DurableRunner


async def handle(runner: DurableRunner, request: dict[str, Any]) -> Any:
    op = request["op"]
    if op == "start":
        handle = await runner.start(scenarios.specification(request["program"]), run_id=request["run_id"])
        return handle.run_id
    if op == "send":
        return await runner.send(
            Address(kind="conversation", value=request["address"]),
            Envelope(content=[Text(text=request["text"])]),
            priority=Priority(request.get("priority", "normal")),
            idempotency_key=request.get("key"),
        )
    if op == "cancel":
        await runner.cancel(request["run_id"], reason="cancelled by the test")
        return None
    if op == "resident":
        return sorted(runner.workflow_tasks)
    raise ValueError(f"unknown operation {op!r}")


async def answer(runner: DurableRunner, request: dict[str, Any]) -> None:
    try:
        reply = {"id": request["id"], "result": await handle(runner, request)}
    except Exception as error:
        reply = {"id": request["id"], "error": f"{type(error).__name__}: {error}"}
    print(json.dumps(reply), flush=True)


async def main(directory: Path, database: str, runner_id: str, evict_after: float, takeover_after: float) -> None:
    runner = DurableRunner(
        directory / runner_id,
        database=database,
        runner_id=runner_id,
        evict_after=timedelta(seconds=evict_after) if evict_after > 0 else None,
        eviction_interval=0.2,
        heartbeat_interval=0.5,
        takeover_after=timedelta(seconds=takeover_after),
        **scenarios.configure(directory, runner_id),
    )
    for deployment in scenarios.deployments():
        runner.deploy(deployment)
    await runner.launch()
    print(json.dumps({"ready": runner_id}), flush=True)

    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    pending: set[asyncio.Task[None]] = set()
    while line := await reader.readline():
        task = asyncio.create_task(answer(runner, json.loads(line)))
        pending.add(task)
        task.add_done_callback(pending.discard)
    await runner.close()  # stdin closed: the test is done with this runner


if __name__ == "__main__":
    directory, database, runner_id, evict_after, takeover_after = sys.argv[1:6]
    asyncio.run(main(Path(directory), database, runner_id, float(evict_after), float(takeover_after)))
