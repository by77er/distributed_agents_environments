"""Durability under faults: the assistant's server is killed with SIGKILL at random moments during a conversation.

A scripted client holds one conversation with the assistant running on the `DurableRunner`. While a reply is being
produced, the client kills the server at a random moment, restarts it, and carries on; a message whose send failed is
resent with the same idempotency key. Afterwards it checks that no message was lost and nothing happened twice:

- every message got exactly one reply, and no reply was duplicated;
- no note was saved twice;
- no model call was repeated, except calls in flight when the server was killed (at most one per kill);
- the final reply recalls what the conversation asked the assistant to remember.
"""

import asyncio
import json
import os
import random
import signal
import socket
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from project_assistant.evaluation.fixture import create_fixture_repository
from project_assistant.notes import NotesStore

MESSAGES = [
    "Where is the invoice total computed?",
    "Please remember: the staging database is called kelp-staging-2.",
    "What is the session timeout, and who last changed it?",
    "Please remember: releases go out on Thursdays.",
    "Is there a refund function in the billing module?",
    "What did I ask you to remember in this conversation? List both things.",
]
CONVERSATION = "faults"


@dataclass
class FaultReport:
    kills: int = 0
    messages: int = 0
    replies: int = 0
    lost_messages: int = 0
    duplicate_replies: int = 0
    notes: int = 0
    duplicate_notes: int = 0
    model_calls: int = 0
    repeated_model_calls: int = 0
    final_reply_recalls_both: bool = False
    recovery_seconds: list[float] = field(default_factory=list[float])
    passed: bool = False
    detail: str = ""


class Server:
    """The assistant's HTTP server as a child process, restartable on the same state."""

    def __init__(self, repository: Path, work: Path, model: str, reasoning_effort: str) -> None:
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.command = [
            sys.executable, "-m", "project_assistant", "serve",
            "--repository", str(repository), "--state", str(work / "state"), "--notes", str(work / "notes.sqlite"),
            "--ledger", str(work / "ledger.jsonl"), "--port", str(self.port),
            "--model", model, "--reasoning-effort", reasoning_effort,
        ]  # fmt: skip
        self.log = (work / "server.log").open("a")
        self.process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        self.process = await asyncio.create_subprocess_exec(
            *self.command, stdout=self.log, stderr=asyncio.subprocess.STDOUT
        )
        async with httpx.AsyncClient() as client:
            for _ in range(600):
                try:
                    if (await client.get(f"{self.url}/health", timeout=1)).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                if self.process.returncode is not None:
                    raise RuntimeError(f"the server exited with {self.process.returncode}; see {self.log.name}")
                await asyncio.sleep(0.1)
        raise RuntimeError("the server did not become healthy")

    async def kill(self) -> None:
        if self.process is not None and self.process.returncode is None:
            os.kill(self.process.pid, signal.SIGKILL)
            await self.process.wait()

    async def stop(self) -> None:
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            async with asyncio.timeout(30):
                await self.process.wait()


async def run_faults(
    *, kills: int = 3, seed: int = 1, model: str = "gpt-6-astra", reasoning_effort: str = "low"
) -> FaultReport:
    rng = random.Random(seed)
    kill_turns = set(rng.sample(range(len(MESSAGES)), min(kills, len(MESSAGES))))
    report = FaultReport(messages=len(MESSAGES))
    with tempfile.TemporaryDirectory(prefix="faults-") as directory:
        work = Path(directory)
        repository = create_fixture_repository(work / "tidepool")
        server = Server(repository, work, model, reasoning_effort)
        await server.start()
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                for index, text in enumerate(MESSAGES):
                    await _send(client, server, text, f"message-{index}")
                    kill_at = time.monotonic() + rng.uniform(0.5, 6.0) if index in kill_turns else None
                    if kill_at is not None and not await _await_replies(client, server, index + 1, until=kill_at):
                        await server.kill()
                        report.kills += 1
                        restarted = time.monotonic()
                        await server.start()
                        await _send(client, server, text, f"message-{index}")  # a retry: deduplicated
                        await _await_replies(client, server, index + 1)
                        report.recovery_seconds.append(round(time.monotonic() - restarted, 1))
                    else:
                        await _await_replies(client, server, index + 1)
                transcript = await _transcript(client, server)
        finally:
            await server.stop()
        _check(report, transcript, work)
    return report


async def _send(client: httpx.AsyncClient, server: Server, text: str, key: str) -> None:
    for _ in range(50):
        try:
            response = await client.post(
                f"{server.url}/conversations/{CONVERSATION}/messages",
                json={"text": text, "idempotency_key": key, "wait": False},
            )
            response.raise_for_status()
            return
        except httpx.HTTPError:
            await asyncio.sleep(0.2)
    raise RuntimeError(f"could not send {key}")


async def _transcript(client: httpx.AsyncClient, server: Server) -> list[dict[str, str]]:
    for _ in range(50):
        try:
            response = await client.get(f"{server.url}/conversations/{CONVERSATION}/transcript")
            return response.json()["messages"]
        except httpx.HTTPError:
            await asyncio.sleep(0.2)
    raise RuntimeError("could not read the transcript")


async def _replies(client: httpx.AsyncClient, server: Server) -> list[dict[str, str]]:
    return [entry for entry in await _transcript(client, server) if entry["role"] == "assistant"]


async def _await_replies(client: httpx.AsyncClient, server: Server, count: int, until: float | None = None) -> bool:
    """Poll until the conversation has `count` replies (True), or `until` (a monotonic time) passes (False)."""
    while len(await _replies(client, server)) < count:
        if until is not None and time.monotonic() >= until:
            return False
        await asyncio.sleep(0.25)
    return True


def _check(report: FaultReport, transcript: list[dict[str, str]], work: Path) -> None:
    users = [entry["text"] for entry in transcript if entry["role"] == "user"]
    replies = [entry["text"] for entry in transcript if entry["role"] == "assistant"]
    report.replies = len(replies)
    report.lost_messages = sum(1 for text in MESSAGES if text not in users)
    report.duplicate_replies = max(0, len(replies) - len(MESSAGES)) + (len(users) - len(set(users)))
    notes = [f"{title} {body}" for _, title, body in NotesStore(work / "notes.sqlite").notes()]
    report.notes = len(notes)
    report.duplicate_notes = max(0, len(notes) - 2)
    calls = Counter(json.loads(line)["effect_id"] for line in (work / "ledger.jsonl").read_text().splitlines())
    report.model_calls = sum(calls.values())
    report.repeated_model_calls = sum(count - 1 for count in calls.values())
    final = replies[-1].lower() if replies else ""
    report.final_reply_recalls_both = "kelp-staging-2" in final and "thursday" in final
    problems = [
        f"{report.lost_messages} lost messages" if report.lost_messages else "",
        f"{report.duplicate_replies} duplicate replies" if report.duplicate_replies else "",
        f"{report.duplicate_notes} duplicate notes" if report.duplicate_notes else "",
        f"{report.repeated_model_calls} repeated model calls for {report.kills} kills"
        if report.repeated_model_calls > report.kills
        else "",
        "the final reply does not recall both notes" if not report.final_reply_recalls_both else "",
        f"{len(replies)} replies for {len(MESSAGES)} messages" if len(replies) != len(MESSAGES) else "",
    ]
    report.detail = "; ".join(problem for problem in problems if problem)
    report.passed = not report.detail


def summarize_faults(report: FaultReport) -> str:
    return json.dumps(asdict(report), indent=2)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
