"""Runners in separate processes sharing one Postgres database, driven from a test (see peer.py)."""

import asyncio
import functools
import itertools
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any, Self

import scenarios

from rollout import testing
from rollout.contracts import RunEvent, RunEventType
from rollout_durable import RunStore
from rollout_durable.database import Database

PEER = Path(__file__).with_name("peer.py")


class Peer:
    def __init__(self, runner_id: str, process: asyncio.subprocess.Process) -> None:
        self.runner_id = runner_id
        self.process = process
        self.replies: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self.reader: asyncio.Task[None] | None = None

    async def read(self) -> None:
        assert self.process.stdout is not None
        while line := await self.process.stdout.readline():
            if not line.startswith(b"{"):
                continue
            reply = json.loads(line)
            future = self.replies.pop(reply.get("id", -1), None)
            if future is not None and not future.done():
                future.set_result(reply)
        for future in self.replies.values():
            if not future.done():
                future.set_exception(ConnectionError(f"runner {self.runner_id} exited"))


class Cluster:
    def __init__(
        self, directory: Path, database: str, size: int, *, evict_after: float = 0, takeover_after: float = 3.0
    ) -> None:
        self.directory = directory
        self.database = database
        self.size = size
        self.evict_after = evict_after
        self.takeover_after = takeover_after
        self.peers: list[Peer | None] = [None] * size
        self._ids = itertools.count()
        self._shared = Database(database)
        self.store = RunStore(self._shared)
        """The runners' store, read by the test."""

    async def __aenter__(self) -> Self:
        await asyncio.gather(*(self.start(index) for index in range(self.size)))
        return self

    async def __aexit__(self, *exception: object) -> None:
        for index in range(self.size):
            await self.stop(index)
        self.store.close()
        self._shared.close()

    async def start(self, index: int) -> None:
        """Start (or restart) runner `index`, which keeps its runner id across restarts."""
        runner_id = f"runner-{index}"
        log = (self.directory / f"{runner_id}.log").open("a")
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(PEER), str(self.directory), self.database, runner_id,
            str(self.evict_after), str(self.takeover_after),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log,
        )  # fmt: skip
        assert process.stdout is not None
        async with asyncio.timeout(60):
            while not (line := await process.stdout.readline()).startswith(b'{"ready"'):
                if not line:
                    raise RuntimeError(f"{runner_id} exited while starting; see {log.name}")
        peer = Peer(runner_id, process)
        peer.reader = asyncio.create_task(peer.read())
        self.peers[index] = peer

    def kill(self, index: int) -> None:
        """SIGKILL runner `index`: no cleanup, its heartbeat stops."""
        peer = self.peers[index]
        assert peer is not None
        os.kill(peer.process.pid, signal.SIGKILL)
        self.peers[index] = None

    async def stop(self, index: int) -> None:
        peer = self.peers[index]
        if peer is None:
            return
        self.peers[index] = None
        if peer.process.stdin is not None:
            peer.process.stdin.close()
        try:
            async with asyncio.timeout(20):
                await peer.process.wait()
        except TimeoutError:
            peer.process.kill()
            await peer.process.wait()

    async def call(self, index: int, op: str, *, wait_seconds: float = 60, **arguments: Any) -> Any:
        peer = self.peers[index]
        assert peer is not None and peer.process.stdin is not None, f"runner {index} is not running"
        request_id = next(self._ids)
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        peer.replies[request_id] = future
        peer.process.stdin.write(json.dumps({"id": request_id, "op": op, **arguments}).encode() + b"\n")
        await peer.process.stdin.drain()
        reply = await asyncio.wait_for(future, wait_seconds)
        if "error" in reply:
            raise RuntimeError(f"runner {index}: {reply['error']}")
        return reply["result"]

    # What happened, read from the shared store and ledgers

    def ledger(self, name: str) -> list[dict[str, Any]]:
        return scenarios.read(self.directory / f"{name}.jsonl")

    def events(self, run_id: str) -> list[RunEvent]:
        return self.store.events(run_id)

    def replies(self, run_id: str) -> list[str]:
        return [
            str(event.payload["payload"])  # type: ignore[index, call-overload]
            for event in self.events(run_id)
            if event.type is RunEventType.OUTPUT_EMITTED
        ]

    def received(self, run_id: str) -> list[str]:
        """The message ids the run received, in order."""
        return [
            str(event.payload["envelope"]["message_id"])  # type: ignore[index, call-overload]
            for event in self.events(run_id)
            if event.type is RunEventType.MESSAGE_RECEIVED
        ]

    def conversation_runs(self, address: str) -> list[str]:
        return self.store.conversation_runs(address)


until = functools.partial(testing.until, seconds=60, every=0.1)
"""Waiting as `rollout.testing.until` does, as long as a cluster of processes may take."""
