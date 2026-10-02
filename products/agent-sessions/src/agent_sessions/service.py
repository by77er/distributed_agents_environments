"""Composes the agent sessions product: a runner, environments, coordination tools and a relay.

Sessions are conversations of the deployment `agents/session`, keyed by session name. The operator is a participant
too: sessions can message them, and those messages land in the operator's inbox.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal

import sqlalchemy as sa
from pydantic import JsonValue

from agent_sessions.coordination import (
    BoardTools,
    CoordinationStore,
    Delivery,
    Post,
    Relay,
    SessionTools,
    post,
    register,
)
from agent_sessions.coordination.store import now
from agent_sessions.session import OPERATOR, AgentSession, SessionAgent
from rollout.contracts import EffectIdentity, RunEvent, RunEventType, Text, new_ulid
from rollout.harness import (
    Address,
    Blobs,
    ConversationKey,
    Deployment,
    DirectModel,
    Envelope,
    EnvironmentService,
    FileBlobStore,
    ModelBinding,
    Priority,
    RunBinding,
    RunHandle,
    Runner,
    RunSpecification,
    SamplingParameters,
    ToolBinding,
    ToolSet,
    agent_program,
)
from rollout.local import EndpointFactory, LocalRunner
from rollout.testing import LedgerEndpoint, LedgerEnvironments
from rollout_computers import ImageStore, LocalEnvironments, NamespaceEnvironments
from rollout_computers.local import HOST_IMAGE
from rollout_durable import DurableRunner, RunStore
from rollout_durable.database import Connection, Database, fetch_all, fetch_one, sql

DEPLOYMENT = "agents/session"


@dataclass(frozen=True)
class Settings:
    state: Path
    """Everything lives here: runs, environments, the coordination store."""
    model: str = "gpt-6-astra"
    reasoning_effort: str | None = "low"
    durable: bool = True
    image_cache: Path = Path.home() / ".cache" / "rollout" / "images"
    model_ledger: Path | None = None
    """If set, every model call is appended here (for fault evaluations)."""
    command_ledger: Path | None = None
    """If set, every command started in an environment is appended here (for fault evaluations)."""
    evict_after: timedelta | None = timedelta(minutes=5)
    """Unload sessions that have waited this long for a message; they wake when messaged (durable runner only)."""
    environment: Literal["namespaces", "local"] = "namespaces"
    """Where sessions' commands run: a private Alpine computer in namespaces, or a workspace directory on the host
    itself, with no sandbox (for uses where convenience matters more than isolation)."""
    database: str | None = None
    """A Postgres URL shared by several servers (each with its own runner); without it, state is SQLite and there
    is one server. Servers sharing a database must share `state` too (a shared disk), which holds environments and
    blobs."""
    runner_id: str | None = None
    """This server's runner among those sharing the database; keep it across restarts."""
    takeover_after: timedelta = timedelta(seconds=15)
    """How long a server may stop heartbeating before another server recovers its sessions."""

    blob_store: str | None = None
    """Where images are kept: `s3://bucket/prefix` on S3 or an S3-compatible service (the `s3` extra; endpoint and
    credentials from the usual AWS_* variables), or None for files in `state`."""

    def blobs(self) -> Blobs:
        """Where images the sessions look at are kept; model endpoints read them from here too."""
        if self.blob_store is not None:
            from rollout_s3 import S3BlobStore

            return S3BlobStore.from_url(self.blob_store)
        return FileBlobStore(self.state / "blobs")


@dataclass(frozen=True)
class SessionInfo:
    name: str
    parent: str | None
    purpose: str
    status: str  # starting | working | waiting | stopped
    created_at: str


class SessionsService:
    def __init__(self, settings: Settings, providers: Mapping[str, EndpointFactory]) -> None:
        self.settings = settings
        settings.state.mkdir(parents=True, exist_ok=True)
        _keep_environment_backend(settings.state / "environments", settings.environment)
        self.environments: EnvironmentService
        if settings.environment == "local":
            self.environments, image = LocalEnvironments(settings.state / "environments"), HOST_IMAGE
        else:
            self.environments = NamespaceEnvironments(settings.state / "environments", ImageStore(settings.image_cache))
            image = "alpine"
        if settings.command_ledger is not None:
            self.environments = LedgerEnvironments(self.environments, settings.command_ledger)
        if settings.model_ledger is not None:
            ledger = settings.model_ledger
            providers = {
                name: (lambda model, factory=factory: LedgerEndpoint(factory(model), ledger))
                for name, factory in providers.items()
            }
        if settings.database is not None and not settings.durable:
            raise ValueError("a shared database needs the durable runner")
        self.database = Database(settings.database) if settings.database else None
        self.coordination = CoordinationStore(self.database or settings.state / "coordination.sqlite")
        self.coordination.database.create(OPERATOR_TABLES)
        self.coordination.write(_register_operator)
        tool_sets: dict[str, ToolSet] = {
            "sessions": SessionTools(self.coordination, self._identify, self.status),
            "board": BoardTools(self.coordination, self._identify),
        }
        self.runner: Runner
        self._run_store: RunStore | None = None
        """Says which sessions a durable runner has evicted."""
        if settings.durable:
            self.runner = durable = DurableRunner(
                settings.state / "runs",
                providers=providers,
                tool_sets=tool_sets,
                environments=self.environments,
                blobs=settings.blobs(),
                evict_after=settings.evict_after,
                eviction_interval=min(5.0, settings.evict_after.total_seconds() / 2) if settings.evict_after else 5.0,
                database=self.database,
                runner_id=settings.runner_id,
                takeover_after=settings.takeover_after,
            )
            self._run_store = durable.store
        else:
            self.runner = LocalRunner(
                providers=providers, tool_sets=tool_sets, environments=self.environments, blobs=settings.blobs()
            )
        provider = next(iter(providers))
        sampling = SamplingParameters(reasoning_effort=settings.reasoning_effort)
        binding = RunBinding(
            models={
                "policy": ModelBinding(direct=DirectModel(provider=provider, model=settings.model, sampling=sampling))
            },
            imports={"sessions": ToolBinding(local="sessions"), "board": ToolBinding(local="board")},
        )
        self.runner.deploy(
            Deployment(
                name=DEPLOYMENT,
                specification=RunSpecification(
                    program=agent_program(
                        AgentSession,
                        SessionAgent,
                        task_parameters={"image": image},
                        agent_configuration={"image": image},
                    ),
                    binding=binding,
                ),
            )
        )
        self.relay = Relay(self.coordination, self._deliver)

    async def start(self) -> None:
        await self.runner.launch()
        self.relay.start()

    async def close(self) -> None:
        await self.relay.stop()
        await self.runner.close()
        self.coordination.close()
        if self.database is not None:
            self.database.close()

    # Operator actions

    async def create(self, name: str, instructions: str) -> None:
        if self.coordination.participant(name) is not None:
            raise ValueError(f"a session named {name!r} already exists")
        self.coordination.write(lambda db: register(db, name, OPERATOR, instructions))
        await self.send(name, instructions)

    async def send(self, name: str, text: str, *, urgent: bool = False, key: str | None = None) -> str:
        if self.coordination.participant(name) is None or name == OPERATOR:
            raise KeyError(name)
        priority = Priority.HIGH if urgent else Priority.NORMAL
        return await self.runner.send(
            self._address(name),
            Envelope(content=[Text(text=text)]),
            priority=priority,
            idempotency_key=key,
            sender=OPERATOR,
        )

    async def stop(self, name: str) -> bool:
        runs = self.runs(name)
        if not runs or runs[-1].done:
            return False
        await self.runner.cancel(runs[-1].run_id, reason="stopped by the operator")
        return True

    def post(self, channel: str, title: str, body: str, kind: str = "note") -> str:
        key = f"operator:{new_ulid()}"
        arguments: dict[str, JsonValue] = {"channel": channel, "title": title, "body": body, "kind": kind}
        result = self.coordination.write(lambda db: post(db, key, OPERATOR, arguments))
        return "".join(block.text for block in result.content if isinstance(block, Text))

    # Views

    def sessions(self) -> list[SessionInfo]:
        return [
            SessionInfo(p.name, p.parent, p.purpose, self.status(p.name), p.created_at)
            for p in self.coordination.participants()
            if p.name != OPERATOR
        ]

    def status(self, name: str) -> str:
        if name == OPERATOR:
            return "human"
        runs = self.runs(name)
        if not runs:
            return "starting"
        if runs[-1].done:
            return "stopped"
        if self._run_store is not None and self._run_store.is_evicted(runs[-1].run_id):
            return "sleeping"  # evicted from memory; wakes when messaged
        events = runs[-1].recorded_events()
        return "waiting" if events and events[-1].type is RunEventType.RUN_SUSPENDED else "working"

    def runs(self, name: str) -> list[RunHandle]:
        return list(self.runner.conversation_runs(DEPLOYMENT, name))

    def board(self, channel: str | None = None, status: str | None = None, limit: int = 30) -> list[Post]:
        return self.coordination.posts(channel, status, limit)

    def inbox(self) -> list[dict[str, str]]:
        rows = self.coordination.read(
            lambda db: fetch_all(db, "SELECT sender, text, at FROM operator_inbox ORDER BY at, key")
        )
        return [{"from": sender, "text": text, "at": at} for sender, text, at in rows]

    def activity(self, name: str) -> list[dict[str, JsonValue]]:
        """A session's activity: messages in, replies out, tool calls and results, in order."""
        return [entry for run in self.runs(name) for event in run.recorded_events() if (entry := describe(event))]

    # Internals

    def _address(self, name: str) -> Address:
        return Address(kind="conversation", value=ConversationKey(deployment=DEPLOYMENT, key=name).address)

    def _identify(self, effect_id: str) -> str | None:
        conversation = self.runner.conversation_of(EffectIdentity.parse(effect_id).run_id)
        return conversation.key if conversation is not None and conversation.deployment == DEPLOYMENT else None

    async def _deliver(self, delivery: Delivery) -> None:
        if delivery.recipient == OPERATOR:
            self.coordination.database.write(
                lambda db: sql(
                    db,
                    "INSERT INTO operator_inbox (key, sender, text, at) VALUES (:key, :sender, :text, :at) "
                    "ON CONFLICT DO NOTHING",
                    {"key": delivery.key, "sender": delivery.sender, "text": delivery.text, "at": now()},
                )
            )
            return
        await self.runner.send(
            self._address(delivery.recipient),
            Envelope(content=[Text(text=delivery.text)]),
            priority=Priority(delivery.priority),
            idempotency_key=f"outbox:{delivery.key}",
            sender=delivery.sender,
        )


def describe(event: RunEvent) -> dict[str, JsonValue] | None:
    """A short, human-readable view of an event, or None for bookkeeping events."""
    data = event.payload if isinstance(event.payload, dict) else {}
    at = event.recorded_at.isoformat(timespec="seconds")
    if event.type is RunEventType.MESSAGE_RECEIVED:
        envelope = data.get("envelope")
        if isinstance(envelope, dict):
            content = envelope.get("content")
            blocks = content if isinstance(content, list) else []
            text = "".join(str(b.get("text", "")) for b in blocks if isinstance(b, dict))
            return {"at": at, "kind": "message", "from": envelope.get("sender") or OPERATOR, "text": text}
    if event.type is RunEventType.OUTPUT_EMITTED and data.get("kind") == "reply":
        return {"at": at, "kind": "reply", "text": data.get("payload")}
    if event.type is RunEventType.EFFECT_COMPLETED:
        completion = data.get("payload")
        if isinstance(completion, dict) and isinstance(completion.get("message"), dict):
            message = completion["message"]
            content = message.get("content") if isinstance(message, dict) else None
            calls = [
                f"{b.get('name')}({json.dumps(b.get('arguments'))[:200]})"
                for b in (content if isinstance(content, list) else [])
                if isinstance(b, dict) and b.get("type") == "tool_call"
            ]
            if calls:
                return {"at": at, "kind": "tools", "text": "; ".join(calls)}
    if event.type in (RunEventType.RUN_FAILED, RunEventType.RUN_CANCELLED, RunEventType.RUN_COMPLETED):
        return {"at": at, "kind": "ended", "text": event.type.value}
    return None


OPERATOR_TABLES = sa.MetaData()
sa.Table(
    "operator_inbox",
    OPERATOR_TABLES,
    sa.Column("key", sa.Text, primary_key=True),
    sa.Column("sender", sa.Text),
    sa.Column("text", sa.Text),
    sa.Column("at", sa.Text),
)


def _register_operator(db: Connection) -> None:
    """Register the operator once, whichever server starts first (writes to the store run one at a time)."""
    if fetch_one(db, "SELECT 1 FROM participants WHERE name = :name", {"name": OPERATOR}) is None:
        register(db, OPERATOR, None, "The person managing all sessions")


def _keep_environment_backend(directory: Path, backend: str) -> None:
    """Existing sessions' computers belong to the backend that made them: refuse to serve them with another."""
    marker = directory / "backend"
    if marker.exists() and (existing := marker.read_text().strip()) != backend:
        raise ValueError(f"the sessions in {directory.parent} use {existing!r} environments, not {backend!r}")
    directory.mkdir(parents=True, exist_ok=True)
    marker.write_text(backend)
