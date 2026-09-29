"""Fault evaluation for agent sessions: SIGKILL the server during a fan-out over the board, and check invariants.

A lead session creates two workers and posts three tasks; the workers claim and do them on their own computers and
the lead reports to the operator. Meanwhile the server is killed at seeded random moments and restarted. Model calls
in flight at a kill may be re-sampled, and a re-sample may decide differently: that is not a fault. The checks are
the system's guarantees:

- every message in the outbox was delivered, and each reached its recipient exactly once;
- no command started in an environment ran twice (commands in flight at a kill report an unknown outcome);
- model calls repeat at most once per session per kill (the calls in flight);
- every task was resolved exactly once; no post or session was duplicated;
- every run's event stream is gapless; each live session has exactly one environment;
- the operator received a summary with the correct answers.
"""

import asyncio
import os
import random
import signal
import socket
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx

from rollout.coordination import CoordinationStore
from rollout.core.contracts import RunEventType
from rollout.core.testing import read_ledger
from rollout.durable import RunStore

INSTRUCTIONS = (
    "You coordinate. First create two worker sessions named w1 and w2, instructing each to subscribe to the board "
    "channel 'jobs', claim open tasks there one at a time, do them on their own computer, and resolve each with the "
    "answer. Then post these three tasks to 'jobs': (1) Which Python version does Alpine's apk install? (2) Fetch "
    "https://example.com and report the page's <title>. (3) What is the sha256 of the text 'rollout' (no newline)? "
    "When all three are resolved, send the operator one message summarizing the answers."
)
EXPECTED = ["3.14", "Example Domain", "a4fa034cc780dbd72a36bf51ba5ee7afd509020953aae10021794638543fd997"]


class Server:
    def __init__(self, work: Path) -> None:
        self.work = work
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.process: asyncio.subprocess.Process | None = None
        self.log = (work / "server.log").open("a")

    async def start(self) -> None:
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "agent_sessions.cli", "serve", "--state", str(self.work / "state"),
            "--port", str(self.port), "--model-ledger", str(self.work / "models.jsonl"),
            "--command-ledger", str(self.work / "commands.jsonl"),
            stdout=self.log, stderr=asyncio.subprocess.STDOUT,
        )  # fmt: skip
        async with httpx.AsyncClient() as client:
            for _ in range(600):
                try:
                    if (await client.get(f"{self.url}/sessions", timeout=1)).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                if self.process.returncode is not None:
                    raise RuntimeError(f"the server exited; see {self.log.name}")
                await asyncio.sleep(0.1)
        raise RuntimeError("the server did not become ready")

    async def kill(self) -> None:
        if self.process is not None and self.process.returncode is None:
            os.kill(self.process.pid, signal.SIGKILL)
            await self.process.wait()

    async def stop(self) -> None:
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            async with asyncio.timeout(30):
                await self.process.wait()


async def run_faults(*, kills: int = 3, seed: int = 1, deadline_seconds: float = 420) -> dict[str, Any]:
    rng = random.Random(seed)
    kill_times = sorted(rng.uniform(4, 45) for _ in range(kills))
    report: dict[str, Any] = {"kills": 0, "kill_seconds": [round(t, 1) for t in kill_times], "problems": []}
    with tempfile.TemporaryDirectory(prefix="session-faults-") as directory:
        work = Path(directory)
        server = Server(work)
        await server.start()
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                await _retry(
                    lambda: client.post(f"{server.url}/sessions", json={"name": "lead", "instructions": INSTRUCTIONS})
                )
                while time.monotonic() - started < deadline_seconds:
                    if kill_times and time.monotonic() - started >= kill_times[0]:
                        kill_times.pop(0)
                        await server.kill()
                        report["kills"] += 1
                        await server.start()
                    if not kill_times and _summary(await _inbox(client, server)):
                        break
                    await asyncio.sleep(0.5)
                await asyncio.sleep(3)  # let final deliveries settle
                report["seconds"] = round(time.monotonic() - started, 1)
                inbox = await _inbox(client, server)
        finally:
            await server.stop()
        _check(report, work, inbox)
    report["passed"] = not report["problems"]
    return report


def _check(report: dict[str, Any], work: Path, inbox: list[dict[str, str]]) -> None:
    problems: list[str] = report["problems"]
    state = work / "state"
    coordination = CoordinationStore(state / "coordination.sqlite")
    runs = RunStore(state / "runs" / "runs.sqlite")
    try:
        summary = _summary(inbox)
        report["operator_messages"] = len(inbox)
        if summary is None:
            problems.append("the operator never received a summary with every answer")

        sessions = [p.name for p in coordination.participants() if p.name != "operator"]
        report["sessions"] = sessions
        if len(sessions) != len(set(sessions)) or set(sessions) != {"lead", "w1", "w2"}:
            problems.append(f"unexpected sessions: {sessions}")

        tasks = [post for post in coordination.posts(limit=100) if post.kind == "task"]
        report["tasks"] = {post.id: post.status for post in tasks}
        if len(tasks) != 3 or any(post.status != "done" for post in tasks):
            problems.append(f"tasks not all resolved exactly once: {report['tasks']}")
        if len({post.title for post in tasks}) != len(tasks):
            problems.append("a task was posted twice")

        pending = coordination.pending()
        if pending:
            problems.append(f"{len(pending)} outbox messages were never delivered")
        received: Counter[str] = Counter()
        run_ids = runs.read_all_run_ids()
        for run_id in run_ids:
            events = runs.events(run_id)
            if [event.seq for event in events] != list(range(len(events))):
                problems.append(f"the events of {run_id} are not gapless")
            for event in events:
                data = event.payload if isinstance(event.payload, dict) else {}
                envelope = data.get("envelope")
                if event.type is RunEventType.MESSAGE_RECEIVED and isinstance(envelope, dict):
                    received[str(envelope.get("message_id"))] += 1
        outbox_keys = coordination.read(
            lambda db: [row[0] for row in db.execute("SELECT key FROM outbox WHERE recipient != 'operator'")]
        )
        duplicated = [key for key in outbox_keys if received[f"outbox:{key}"] > 1]
        missing = [key for key in outbox_keys if received[f"outbox:{key}"] == 0]
        report["messages_delivered"] = len(outbox_keys)
        if duplicated:
            problems.append(f"{len(duplicated)} messages reached their recipient more than once")
        if missing:
            problems.append(f"{len(missing)} messages never reached their recipient")

        commands = Counter(entry["effect_id"] for entry in read_ledger(work / "commands.jsonl"))
        report["commands"] = sum(commands.values())
        rerun = [effect for effect, count in commands.items() if count > 1]
        if rerun:
            problems.append(f"{len(rerun)} commands ran more than once: {rerun}")
        unknown = sum(
            1
            for run_id in run_ids
            for event in runs.events(run_id)
            if event.type is RunEventType.EFFECT_COMPLETED
            and isinstance(event.payload, dict)
            and event.payload.get("status") == "outcome_unknown"
        )
        report["commands_with_unknown_outcome"] = unknown

        models = Counter(entry["effect_id"] for entry in read_ledger(work / "models.jsonl"))
        repeated = sum(count - 1 for count in models.values())
        report["model_calls"] = sum(models.values())
        report["repeated_model_calls"] = repeated
        if repeated > report["kills"] * max(len(sessions), 1):
            problems.append(f"{repeated} repeated model calls for {report['kills']} kills")

        environments = [path for path in (state / "environments").iterdir() if path.name.startswith("e_")]
        report["environments"] = len(environments)
        if len(environments) != len(sessions):
            problems.append(f"{len(environments)} environments for {len(sessions)} sessions")
    finally:
        coordination.close()
        runs.close()


def _summary(inbox: list[dict[str, str]]) -> str | None:
    for message in inbox:
        if message["from"] == "lead" and all(answer in message["text"] for answer in EXPECTED):
            return message["text"]
    return None


async def _inbox(client: httpx.AsyncClient, server: Server) -> list[dict[str, str]]:
    response = await _retry(lambda: client.get(f"{server.url}/inbox"))
    return response.json()["messages"]


async def _retry(request: Callable[[], Awaitable[httpx.Response]]) -> httpx.Response:
    for _ in range(100):
        try:
            response = await request()
            response.raise_for_status()
            return response
        except httpx.HTTPError:
            await asyncio.sleep(0.2)
    raise RuntimeError("the server did not answer")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
