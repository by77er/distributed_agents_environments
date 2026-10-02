"""Fault evaluation for agent sessions: SIGKILL servers during a fan-out over the board, and check invariants.

With `servers` > 1, several servers share a throwaway Postgres and one state directory; requests go to a random live
server, kills hit a random server, and the first killed server stays down long enough for the others to take over
its sessions.

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
import contextlib
import os
import random
import signal
import socket
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from agent_sessions.service import DEPLOYMENT
from rollout.coordination import CoordinationStore
from rollout.core.contracts import RunEventType
from rollout.core.harness import ConversationKey
from rollout.core.testing import read_ledger
from rollout.database import Database, create_database, fetch_all, temporary_postgres
from rollout.durable import RunStore

INSTRUCTIONS = (
    "You coordinate. First create two worker sessions named w1 and w2, instructing each to subscribe to the board "
    "channel 'jobs', claim open tasks there one at a time, do them on their own computer, and resolve each with the "
    "answer. Then post these three tasks to 'jobs': (1) Which Python version does Alpine's apk install? (2) Fetch "
    "https://example.com and report the page's <title>. (3) What is the sha256 of the text 'rollout' (no newline)? "
    "When all three are resolved, send the operator one message summarizing the answers."
)
EXPECTED = ["3.14", "Example Domain", "a4fa034cc780dbd72a36bf51ba5ee7afd509020953aae10021794638543fd997"]
FOLLOW_UP = (
    "Ask w1 to write the three answers, one per line, to answers.txt in its workspace, and to tell the operator "
    "when the file is written."
)


class Server:
    def __init__(
        self,
        work: Path,
        environment: str,
        *,
        database: str | None = None,
        runner_id: str | None = None,
        takeover_after: float = 15,
        providers: str | None = None,
        variables: Mapping[str, str] | None = None,
    ) -> None:
        self.work = work
        self.environment = environment
        self.database = database
        self.runner_id = runner_id
        self.takeover_after = takeover_after
        self.providers = providers
        self.variables = variables
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.process: asyncio.subprocess.Process | None = None
        self.log = (work / f"server{f'-{runner_id}' if runner_id else ''}.log").open("a")

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def start(self) -> None:
        options = ["--database", self.database, "--runner-id", str(self.runner_id)] if self.database else []
        options += ["--takeover-after", str(self.takeover_after)]
        options += ["--providers", self.providers] if self.providers else []
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "agent_sessions.cli", "serve", "--state", str(self.work / "state"),
            "--port", str(self.port), "--model-ledger", str(self.work / "models.jsonl"),
            "--command-ledger", str(self.work / "commands.jsonl"), "--evict-after", "5",
            "--environment", self.environment, *options,
            stdout=self.log, stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, **(self.variables or {})},
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


class Servers:
    """One server, or several sharing a database; requests go to a random live one."""

    def __init__(self, servers: list[Server], rng: random.Random) -> None:
        self.servers = servers
        self.rng = rng

    def url(self) -> str:
        live = [server for server in self.servers if server.alive]
        return self.rng.choice(live).url

    async def request(self, client: httpx.AsyncClient, method: str, path: str, **options: Any) -> httpx.Response:
        """Retried on another live server if the chosen one fails (it may be being killed)."""
        for _ in range(100):
            try:
                response = await client.request(method, f"{self.url()}{path}", **options)
                response.raise_for_status()
                return response
            except (httpx.HTTPError, IndexError):
                await asyncio.sleep(0.2)
        raise RuntimeError(f"no server answered {method} {path}")


async def run_faults(
    *, kills: int = 3, seed: int = 1, environment: str = "namespaces", servers: int = 1, deadline_seconds: float = 420
) -> dict[str, Any]:
    rng = random.Random(seed)
    kill_times = sorted(rng.uniform(4, 45) for _ in range(kills))
    problems: list[str] = []
    report: dict[str, Any] = {"servers": servers, "kills": 0, "kill_seconds": [round(t, 1) for t in kill_times]}
    report["problems"] = problems
    with tempfile.TemporaryDirectory(prefix="session-faults-") as directory, contextlib.ExitStack() as stack:
        work = Path(directory)
        database = None
        if servers > 1:
            server_url = stack.enter_context(temporary_postgres(work / "postgres"))
            database = create_database(server_url, "sessions")
        takeover_after = 6.0
        group = Servers(
            [
                Server(work, environment, database=database, runner_id=f"server-{index}", takeover_after=takeover_after)
                if database
                else Server(work, environment)
                for index in range(servers)
            ],
            rng,
        )
        await asyncio.gather(*(server.start() for server in group.servers))
        started = time.monotonic()
        down_until: dict[int, float] = {}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                await _create_lead(client, group)
                while time.monotonic() - started < deadline_seconds:
                    now = time.monotonic() - started
                    for index, until in list(down_until.items()):
                        if now >= until:
                            del down_until[index]
                            await group.servers[index].start()
                    if kill_times and now >= kill_times[0]:
                        kill_times.pop(0)
                        await _kill_one(group, report, down_until, now, takeover_after)
                    if not kill_times and not down_until and _summary(await _inbox(client, group)):
                        break
                    await asyncio.sleep(0.5)
                # Phase 2: every session is evicted, servers die while they sleep, and a message wakes them.
                report["all_slept"] = await _until_all_sleeping(client, group, seconds=90)
                if not report["all_slept"]:
                    problems.append("the sessions were never all evicted")
                await _kill_one(group, report, None, 0, 0)
                await group.request(client, "POST", "/sessions/lead/messages", json={"text": FOLLOW_UP})
                await asyncio.sleep(rng.uniform(1, 4))  # kill again while they wake
                await _kill_one(group, report, None, 0, 0)
                confirmed = time.monotonic() + 300
                while time.monotonic() < confirmed and not _confirmation(await _inbox(client, group)):  # noqa: ASYNC110 - polling the servers over HTTP
                    await asyncio.sleep(1)
                report["quiet"] = await _until_quiet(client, group, seconds=60)  # messages in flight land
                report["seconds"] = round(time.monotonic() - started, 1)
                inbox = await _inbox(client, group)
        finally:
            for server in group.servers:
                await server.stop()
        _check(report, work, inbox, database)
    report["passed"] = not report["problems"]
    return report


async def _create_lead(client: httpx.AsyncClient, group: Servers) -> None:
    for _ in range(100):
        try:
            response = await client.post(f"{group.url()}/sessions", json={"name": "lead", "instructions": INSTRUCTIONS})
            if response.status_code in (201, 400):  # 400: a retried request that had created it
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.2)
    raise RuntimeError("could not create the lead session")


async def _kill_one(
    group: Servers, report: dict[str, Any], down_until: dict[int, float] | None, now: float, takeover_after: float
) -> None:
    """Kill a random live server. The first kill with several servers keeps it down past the takeover time."""
    live = [index for index, server in enumerate(group.servers) if server.alive]
    if len(group.servers) > 1 and len(live) < 2:
        return  # keep one server up
    index = group.rng.choice(live)
    await group.servers[index].kill()
    report["kills"] += 1
    killed: list[str] = report.setdefault("killed", [])
    killed.append(group.servers[index].runner_id or "server")
    if down_until is not None and len(group.servers) > 1 and "taken_over" not in report:
        report["taken_over"] = group.servers[index].runner_id
        down_until[index] = now + takeover_after + 6
        return
    await group.servers[index].start()


def _check(report: dict[str, Any], work: Path, inbox: list[dict[str, str]], database: str | None) -> None:
    problems: list[str] = report["problems"]
    state = work / "state"
    shared = Database(database) if database else None
    coordination = CoordinationStore(shared or state / "coordination.sqlite")
    runs = RunStore(shared or state / "runs" / "runs.sqlite")
    try:
        summary = _summary(inbox)
        report["operator_messages"] = len(inbox)
        if summary is None:
            problems.append("the operator never received a summary with every answer")
        if _confirmation(inbox) is None:
            problems.append("the woken sessions never confirmed the follow-up")
        workspaces = ("e_*/rootfs/workspace", "e_*/workspace")  # namespaces, local
        written = [
            path.read_text()
            for pattern in workspaces
            for path in (state / "environments").glob(f"{pattern}/answers.txt")
        ]
        report["answers_file_written"] = bool(written) and EXPECTED[2] in written[0]
        if not report["answers_file_written"]:
            problems.append("answers.txt was not written with the answers")

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
        rows = coordination.read(lambda db: fetch_all(db, "SELECT key FROM outbox WHERE recipient != 'operator'"))
        outbox_keys = [str(row[0]) for row in rows]
        duplicated = [key for key in outbox_keys if received[f"outbox:{key}"] > 1]
        missing = [key for key in outbox_keys if received[f"outbox:{key}"] == 0]
        report["messages_delivered"] = len(outbox_keys)
        if duplicated:
            problems.append(f"{len(duplicated)} messages reached their recipient more than once")
        if missing:
            problems.append(f"{len(missing)} messages never reached their recipient")
            details = [
                row
                for row in coordination.read(
                    lambda db: fetch_all(
                        db, "SELECT key, recipient, sender, substr(text, 1, 80), created_at, delivered_at FROM outbox"
                    )
                )
                if row[0] in missing
            ]
            report["missing_messages"] = [
                {
                    "key": k,
                    "to": to,
                    "from": sender,
                    "text": text,
                    "created": created,
                    "delivered": delivered,
                    "recipient_evicted": any(
                        runs.is_evicted(r)
                        for r in runs.conversation_runs(ConversationKey(deployment=DEPLOYMENT, key=str(to)).address)
                    ),
                }
                for k, to, sender, text, created, delivered in details
            ]

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

        report["evictions"] = runs.evictions()
        if report["evictions"] < len(sessions):
            problems.append(f"only {report['evictions']} evictions for {len(sessions)} sessions")
        environments = [path for path in (state / "environments").iterdir() if path.name.startswith("e_")]
        report["environments"] = len(environments)
        if len(environments) != len(sessions):
            problems.append(f"{len(environments)} environments for {len(sessions)} sessions")
    finally:
        coordination.close()
        runs.close()
        if shared is not None:
            shared.close()


def _confirmation(inbox: list[dict[str, str]]) -> str | None:
    for message in inbox:
        if "answers.txt" in message["text"]:
            return message["text"]
    return None


async def _until_all_sleeping(client: httpx.AsyncClient, group: Servers, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        sessions = (await group.request(client, "GET", "/sessions")).json()["sessions"]
        if sessions and all(session["status"] == "sleeping" for session in sessions):
            return True
        await asyncio.sleep(1)
    return False


async def _until_quiet(client: httpx.AsyncClient, group: Servers, seconds: float) -> bool:
    """Until every session has been waiting or sleeping for a few seconds in a row."""
    deadline, calm = time.monotonic() + seconds, 0
    while time.monotonic() < deadline and calm < 3:
        sessions = (await group.request(client, "GET", "/sessions")).json()["sessions"]
        calm = calm + 1 if all(session["status"] in ("waiting", "sleeping") for session in sessions) else 0
        await asyncio.sleep(1.5)
    return calm >= 3


def _summary(inbox: list[dict[str, str]]) -> str | None:
    for message in inbox:
        if message["from"] == "lead" and all(answer in message["text"] for answer in EXPECTED):
            return message["text"]
    return None


async def _inbox(client: httpx.AsyncClient, group: Servers) -> list[dict[str, str]]:
    response = await group.request(client, "GET", "/inbox")
    return response.json()["messages"]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
