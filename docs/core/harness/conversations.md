# Conversations and messages

Status: **Proposed** · Layer: core · See [ADR-0019](../../decisions/0019-conversations-and-priority-delivery.md)

Agents that talk to people or to each other receive **messages** over time. This document defines how messages are
addressed, ordered and delivered to runs. It adds no new resource: an addressable agent is a **deployment**, and a
conversation is a **run keyed by a conversation key**.

## Addressing

```python
@dataclass(frozen=True)
class ConversationKey:
    deployment: str                      # e.g. "acme/support-bot"
    key: str                             # caller-chosen, e.g. "slack:T1/C2/171.2" or "user:42"
    origin: Address | None = None        # where replies go by default (e.g. a connector target)

@dataclass(frozen=True)
class Address:
    kind: Literal["conversation", "run", "external"]
    value: str                           # "{deployment}/{key}" | run_id | connector target

@dataclass(frozen=True)
class Envelope:
    kind: str = "message"                # "message" or an application-defined kind
    content: list[Block] = field(default_factory=list)   # canonical content
    data: JsonValue | None = None        # structured payload
    reply_to: Address | None = None
    message_id: str = ""                 # set by the runner: the sender's effect_id or idempotency key
    sender: Principal | None = None      # set by the runner; never trusted from the payload
```

- **One conversation, one live run.** Sending to a conversation starts its run if none is live
  (signal-with-start), otherwise delivers to the live run. A run that ends (`End`, a `WaitFor` timeout, or
  `End(continue_as=…)`) closes that episode; the next message starts a new run on the deployment's current version.
- **Lanes are per conversation.** At most one run consumes a conversation's messages at a time. Different
  conversations of one deployment run in parallel. Anything they share (memory, knowledge, profiles) is a tool and
  handles its own concurrency.
- **Order.** Messages are delivered first-in, first-out per (sender, conversation), deduplicated by `message_id`.

## Priority and delivery mode

```python
class Priority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"

class DeliveryMode(Enum):
    QUEUE = "queue"            # deliver only when the run next waits (WaitFor)
    STEER = "steer"            # add to the next observation at the next turn boundary; cancel nothing
    INTERRUPT = "interrupt"    # cancel the in-flight model sample now; the message becomes the next observation

@dataclass(frozen=True)
class DeliveryPolicy:          # part of RunBinding; deployments may override
    modes: Mapping[Priority, DeliveryMode] = field(default_factory=lambda: {
        Priority.LOW: DeliveryMode.QUEUE,
        Priority.NORMAL: DeliveryMode.STEER,
        Priority.HIGH: DeliveryMode.INTERRUPT,
    })
    max_priority_by_sender: Mapping[str, Priority] = field(default_factory=dict)   # caps per sender class
```

| Run state when the message arrives | `QUEUE` | `STEER` | `INTERRUPT` |
|---|---|---|---|
| Waiting (`WaitFor` of a matching kind) | resumes the run: `Task.resume` | resumes the run | resumes the run |
| Sampling a reply | held until the run next waits | merged by `Task.steer` into the observation that follows this turn | the sample is cancelled; `Task.resume` produces the next observation |
| Executing tools | held | merged after the tools finish | tools in flight finish (side effects cannot be undone); the message is then merged like `STEER` |
| No live run | starts one | starts one | starts one |

Rules:

- An interrupted reply is recorded as an aborted branch in the recorder's session tree and is never trained on.
- A sender's priority is capped by `max_priority_by_sender`, so an external system cannot interrupt when it
  should only queue.
- Control operations (cancel) are not messages; they use `Runner.cancel`.

## Sending

```python
await runner.send(Address(kind="conversation", value="acme/support-bot/slack:T1/C2/171.2"),
                  Envelope(content=[Text("Hi")]), priority=Priority.NORMAL,
                  idempotency_key="slack-event-Ev123")

# from inside a run
await run.send(Address(kind="run", value=peer_run_id), Envelope(kind="task", data={"subtask": "…"}))
```

Replies to people go out as durable outputs: `run.emit("reply", content, to=run.conversation.origin)`; a connector
delivers them (platform layer).

## How runners implement this

| Runner | Waiting | Delivery |
|---|---|---|
| `LocalRunner` | an in-memory future | an in-memory queue per conversation |
| `DurableRunner` | the run ends its current activation and holds no compute | one transaction inserts the envelope and, if the conversation is idle, enqueues a new activation on a queue partitioned by conversation ([durability](../../durability/README.md#conversations)) |
