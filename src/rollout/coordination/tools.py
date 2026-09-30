"""Coordination tool sets: sessions (list, create, message) and a board (channels of notes and tasks).

Both are imported tool sets over one `CoordinationStore`. Tools write to the store; messages go into its outbox and a
`Relay` delivers them. Who is calling comes from the call's `effect_id` through an injected `identify` function, so
the same tools serve agent sessions, swarms or multi-agent RL tasks, whatever their addressing.
"""

import re
from collections.abc import Callable, Mapping, Sequence

from pydantic import JsonValue

from rollout.coordination.store import CoordinationStore, enqueue, now
from rollout.core.contracts import RetryClass, Text, ToolAnnotations, ToolResult, ToolSpecification
from rollout.database import Connection, fetch_all, fetch_one, sql

type Identify = Callable[[str], str | None]
"""The participant a call comes from, given its `effect_id`; None if the caller is not a participant."""

type Status = Callable[[str], str]
"""A participant's current state, e.g. `working`, `waiting` or `stopped`."""

NAME = re.compile(r"^[a-z][a-z0-9-]{0,39}$")


def _object(properties: dict[str, JsonValue], required: list[str]) -> dict[str, JsonValue]:
    return {
        "type": "object",
        "properties": properties,
        "required": list[JsonValue](required),
        "additionalProperties": False,
    }


def _text(text: str, *, error: bool = False) -> ToolResult:
    return ToolResult(content=[Text(text=text)], is_error=error)


class _CoordinationTools:
    """Shared plumbing: identity, deduplicated writes, dispatch."""

    deduplicates = True
    """Every write is recorded against its effect_id, so side-effecting tools need no attempt marker."""

    specifications_: Sequence[ToolSpecification] = ()

    def __init__(self, store: CoordinationStore, identify: Identify) -> None:
        self.store = store
        self.identify = identify

    def specifications(self) -> Sequence[ToolSpecification]:
        return self.specifications_

    async def call(
        self, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        caller = self.identify(effect_id)
        if caller is None:
            return _text("only participants can use this tool", error=True)
        handler = getattr(self, f"_{name}", None)
        if handler is None:
            return _text(f"unknown tool {name!r}", error=True)
        try:
            return handler(caller, arguments, effect_id, arguments_digest)
        except (KeyError, ValueError, TypeError) as error:
            return _text(f"{type(error).__name__}: {error}", error=True)


class SessionTools(_CoordinationTools):
    """`list_sessions`, `create_session`, `send_message`."""

    specifications_ = [
        ToolSpecification(
            name="list_sessions",
            description="List every session: its name, who created it, what it is for, and whether it is working.",
            input_schema=_object({}, []),
            annotations=ToolAnnotations(read_only_hint=True),
            retry_class=RetryClass.PURE,
        ),
        ToolSpecification(
            name="create_session",
            description="Start a new session with its own environment. It receives `instructions` as its first "
            "message and knows you created it. Names are lowercase letters, digits and dashes.",
            input_schema=_object(
                {"name": {"type": "string"}, "instructions": {"type": "string"}}, ["name", "instructions"]
            ),
            retry_class=RetryClass.SIDE_EFFECTING,
        ),
        ToolSpecification(
            name="send_message",
            description="Send a message to another session. Normal messages reach it at its next step; urgent ones "
            "interrupt what it is doing.",
            input_schema=_object(
                {"to": {"type": "string"}, "text": {"type": "string"}, "urgent": {"type": "boolean"}}, ["to", "text"]
            ),
            retry_class=RetryClass.SIDE_EFFECTING,
        ),
    ]

    def __init__(self, store: CoordinationStore, identify: Identify, status: Status) -> None:
        super().__init__(store, identify)
        self.status = status

    def _list_sessions(self, caller: str, arguments: Mapping[str, JsonValue], *_: str) -> ToolResult:
        lines = [
            f"{p.name}{' (you)' if p.name == caller else ''}: {self.status(p.name)}; "
            f"created by {p.parent or 'the operator'}; {p.purpose}"
            for p in self.store.participants()
        ]
        return _text("\n".join(lines) or "(no sessions)")

    def _create_session(
        self, caller: str, arguments: Mapping[str, JsonValue], effect_id: str, digest: str
    ) -> ToolResult:
        name, instructions = str(arguments["name"]), str(arguments["instructions"])
        if not NAME.match(name):
            return _text(f"{name!r} is not a valid name: lowercase letters, digits and dashes", error=True)

        def create(db: Connection) -> ToolResult:
            if fetch_one(db, "SELECT 1 FROM participants WHERE name = :name", {"name": name}):
                return _text(f"a session named {name!r} already exists", error=True)
            register(db, name, caller, instructions)
            enqueue(db, effect_id, name, caller, instructions)
            return _text(f"Created session {name!r}. It has received your instructions.")

        return self.store.recorded(effect_id, digest, create)

    def _send_message(self, caller: str, arguments: Mapping[str, JsonValue], effect_id: str, digest: str) -> ToolResult:
        recipient, text = str(arguments["to"]), str(arguments["text"])
        priority = "high" if arguments.get("urgent") else "normal"

        def send(db: Connection) -> ToolResult:
            if not fetch_one(db, "SELECT 1 FROM participants WHERE name = :name", {"name": recipient}):
                return _text(f"there is no session named {recipient!r}", error=True)
            enqueue(db, effect_id, recipient, caller, text, priority)
            return _text(f"Sent to {recipient}.")

        return self.store.recorded(effect_id, digest, send)


class BoardTools(_CoordinationTools):
    """A board of channels holding notes and tasks. Subscribers are told about new posts."""

    specifications_ = [
        ToolSpecification(
            name="post",
            description="Post a note or a task to a channel of the shared board. Subscribers of the channel are told. "
            "Tasks can be claimed by one session and resolved with a result.",
            input_schema=_object(
                {
                    "channel": {"type": "string"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "kind": {"type": "string", "enum": ["note", "task"]},
                },
                ["channel", "title", "body"],
            ),
            retry_class=RetryClass.SIDE_EFFECTING,
        ),
        ToolSpecification(
            name="read_board",
            description="Read recent posts, newest first. Filter by channel and status (open, claimed, done). "
            "Without a channel, also lists the channels.",
            input_schema=_object(
                {
                    "channel": {"type": "string"},
                    "status": {"type": "string", "enum": ["open", "claimed", "done"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                [],
            ),
            annotations=ToolAnnotations(read_only_hint=True),
            retry_class=RetryClass.PURE,
        ),
        ToolSpecification(
            name="claim_task",
            description="Claim an open task so no other session takes it. Fails if someone else claimed it first.",
            input_schema=_object({"post_id": {"type": "integer"}}, ["post_id"]),
            retry_class=RetryClass.SIDE_EFFECTING,
        ),
        ToolSpecification(
            name="resolve_task",
            description="Mark a task you claimed as done, with its result. The task's author is told.",
            input_schema=_object({"post_id": {"type": "integer"}, "result": {"type": "string"}}, ["post_id", "result"]),
            retry_class=RetryClass.SIDE_EFFECTING,
        ),
        ToolSpecification(
            name="subscribe",
            description="Be told about new posts in a channel.",
            input_schema=_object({"channel": {"type": "string"}}, ["channel"]),
            retry_class=RetryClass.IDEMPOTENT,
        ),
        ToolSpecification(
            name="unsubscribe",
            description="Stop being told about new posts in a channel.",
            input_schema=_object({"channel": {"type": "string"}}, ["channel"]),
            retry_class=RetryClass.IDEMPOTENT,
        ),
    ]

    def _post(self, caller: str, arguments: Mapping[str, JsonValue], effect_id: str, digest: str) -> ToolResult:
        return self.store.recorded(effect_id, digest, lambda db: post(db, effect_id, caller, arguments))

    def _read_board(self, caller: str, arguments: Mapping[str, JsonValue], *_: str) -> ToolResult:
        channel = str(arguments["channel"]) if arguments.get("channel") else None
        status = str(arguments["status"]) if arguments.get("status") else None
        limit = arguments.get("limit")
        posts = self.store.posts(channel, status, int(limit) if isinstance(limit, int) else 20)
        lines = [
            f"#{p.id} [{p.channel}] {p.kind} ({p.status}{f', {p.claimed_by}' if p.claimed_by else ''}) by {p.author}: "
            f"{p.title}\n    {p.body}" + (f"\n    result: {p.result}" if p.result else "")
            for p in posts
        ]
        if channel is None:
            channels = self.store.read(
                lambda db: fetch_all(db, "SELECT channel, COUNT(*) FROM posts GROUP BY channel ORDER BY channel")
            )
            lines.insert(0, "Channels: " + (", ".join(f"{name} ({count})" for name, count in channels) or "(none)"))
        return _text("\n".join(lines) or "(no posts)")

    def _claim_task(self, caller: str, arguments: Mapping[str, JsonValue], effect_id: str, digest: str) -> ToolResult:
        post_id = int(str(arguments["post_id"]))

        def claim(db: Connection) -> ToolResult:
            claimed = sql(
                db,
                "UPDATE posts SET status = 'claimed', claimed_by = :caller "
                "WHERE id = :id AND kind = 'task' AND status = 'open'",
                {"caller": caller, "id": post_id},
            ).rowcount
            if claimed:
                return _text(f"You claimed task #{post_id}.")
            row = fetch_one(db, "SELECT kind, status, claimed_by FROM posts WHERE id = :id", {"id": post_id})
            if row is None:
                return _text(f"there is no post #{post_id}", error=True)
            return _text(f"#{post_id} is a {row[0]} that is {row[1]}{f' by {row[2]}' if row[2] else ''}", error=True)

        return self.store.recorded(effect_id, digest, claim)

    def _resolve_task(self, caller: str, arguments: Mapping[str, JsonValue], effect_id: str, digest: str) -> ToolResult:
        post_id, result = int(str(arguments["post_id"])), str(arguments["result"])

        def resolve(db: Connection) -> ToolResult:
            row = fetch_one(db, "SELECT author, title, claimed_by, status FROM posts WHERE id = :id", {"id": post_id})
            if row is None or row[2] != caller or row[3] != "claimed":
                return _text(f"you have not claimed task #{post_id}", error=True)
            sql(
                db,
                "UPDATE posts SET status = 'done', result = :result WHERE id = :id",
                {"result": result, "id": post_id},
            )
            if row[0] != caller:
                enqueue(
                    db,
                    effect_id,
                    row[0],
                    caller,
                    f"[board] Task #{post_id} ({row[1]}) was resolved by {caller}: {result}",
                )
            return _text(f"Resolved task #{post_id}.")

        return self.store.recorded(effect_id, digest, resolve)

    def _subscribe(self, caller: str, arguments: Mapping[str, JsonValue], effect_id: str, digest: str) -> ToolResult:
        channel = str(arguments["channel"])
        return self.store.recorded(effect_id, digest, lambda db: subscribe(db, caller, channel))

    def _unsubscribe(self, caller: str, arguments: Mapping[str, JsonValue], effect_id: str, digest: str) -> ToolResult:
        channel = str(arguments["channel"])

        def unsubscribe(db: Connection) -> ToolResult:
            sql(
                db,
                "DELETE FROM subscriptions WHERE channel = :channel AND participant = :participant",
                {"channel": channel, "participant": caller},
            )
            return _text(f"Unsubscribed from {channel}.")

        return self.store.recorded(effect_id, digest, unsubscribe)


# Writes shared by tools and operators


def register(db: Connection, name: str, parent: str | None, purpose: str) -> None:
    sql(
        db,
        "INSERT INTO participants (name, parent, purpose, created_at) VALUES (:name, :parent, :purpose, :at)",
        {
            "name": name,
            "parent": parent,
            "purpose": purpose.strip().splitlines()[0][:200] if purpose.strip() else "",
            "at": now(),
        },
    )


def post(db: Connection, key: str, author: str, arguments: Mapping[str, JsonValue]) -> ToolResult:
    channel, title, body = str(arguments["channel"]), str(arguments["title"]), str(arguments["body"])
    kind = str(arguments.get("kind") or "note")
    if kind not in ("note", "task"):
        return _text("kind must be note or task", error=True)
    post_id = sql(
        db,
        "INSERT INTO posts (channel, kind, title, body, author, status, created_at) "
        "VALUES (:channel, :kind, :title, :body, :author, 'open', :at) RETURNING id",
        {"channel": channel, "kind": kind, "title": title, "body": body, "author": author, "at": now()},
    ).scalar_one()
    subscribers = fetch_all(
        db,
        "SELECT participant FROM subscriptions WHERE channel = :channel AND participant != :author",
        {"channel": channel, "author": author},
    )
    for (subscriber,) in subscribers:
        claim = f" Claim it with claim_task({post_id})." if kind == "task" else ""
        text = f"[board #{channel}] New {kind} #{post_id} from {author}: {title}\n{body}{claim}"
        enqueue(db, f"{key}:{subscriber}", subscriber, author, text, priority="low")
    return _text(f"Posted #{post_id} to {channel}; {len(subscribers)} subscriber(s) told.")


def subscribe(db: Connection, participant: str, channel: str) -> ToolResult:
    sql(
        db,
        "INSERT INTO subscriptions (channel, participant) VALUES (:channel, :participant) ON CONFLICT DO NOTHING",
        {"channel": channel, "participant": participant},
    )
    return _text(f"Subscribed to {channel}.")
