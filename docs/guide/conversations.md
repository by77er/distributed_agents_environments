# Conversations

Agents that talk with people or with other agents receive messages over time. A task waits for a message with
`WaitFor`, and messages that arrive while the agent is busy are queued, merged into the next observation, or
interrupt the reply in progress, depending on their priority.

The examples on this page deliver messages straight to a run context with `deliver(envelope, mode)`, to show
the loop's behavior. Applications send through the runner instead: see [the last section](#sending-and-replying).

## Messages

A message is an `Envelope`: canonical content blocks, optional structured `data`, and a `kind` that `WaitFor` can
select on.

```python
from rollout.core.contracts import Text
from rollout.core.harness import Envelope

hello = Envelope(content=[Text(text="Hi, can you help me with my order?")])
approval = Envelope(kind="approval", data={"approved": True, "by": "reviewer-7"})
assert hello.kind == "message"
```

The runner sets `message_id` (for deduplication) and `sender`; a message's payload never sets them.

## Waiting for a message

A hook returns `WaitFor(kind, timeout)` to suspend the run. When a message of that kind arrives, the loop calls
`Task.resume(run, envelope)`, whose default turns the message into a USER observation. If the timeout passes
first, the loop uses `WaitFor.on_timeout`, which by default ends the episode as truncated.

`WaitFor` is valid from `start`, from `resume`, and from `respond` after a reply without tool calls.

```python
import asyncio
from datetime import timedelta

from rollout.core.contracts import Message, Role
from rollout.core.harness import Agent, DeliveryMode, Ending, Observation, RunContext, Task, WaitFor, rollout
from rollout.core.testing import local_run


class Support(Task):
    """Answers each message, then waits up to 50 ms for the next one."""

    wait = WaitFor("message", timeout=timedelta(milliseconds=50))

    async def start(self, run: RunContext) -> WaitFor:
        return self.wait  # the first message opens the episode

    async def respond(self, run: RunContext, reply: Message) -> WaitFor:
        return self.wait


async def main() -> None:
    task = Support()
    run, endpoint = local_run(task, replies=["Sure. What is the order number?", "Found it: it ships today."])
    episode = asyncio.create_task(rollout(task, Agent(), run))

    run.deliver(hello, DeliveryMode.QUEUE)        # held until the run waits, then resumes it
    await asyncio.sleep(0.01)
    run.deliver(Envelope(content=[Text(text="It is 4417.")]), DeliveryMode.QUEUE)
    await episode                                 # ends when no message arrives within 50 ms

    users = [m.text for m in run.history.messages() if m.role is Role.USER]
    assert users == ["Hi, can you help me with my order?", "It is 4417."]
    assert run.history.turns[-1].observation.end is Ending.TRUNCATED


asyncio.run(main())
```

A `WaitFor("approval")` takes only messages of kind `approval`; messages of other kinds stay held until a matching
wait.

## Priority and delivery mode

A sender gives a message a `Priority`. The run's `DeliveryPolicy` maps it to a `DeliveryMode`, which decides what
happens when the run is busy:

| Mode (default priority) | Run is waiting | Run is sampling a reply | Run is executing tools |
|---|---|---|---|
| `QUEUE` (`LOW`) | resumes it | held until the run next waits | held |
| `STEER` (`NORMAL`) | resumes it | merged into the observation after this turn by `Task.steer` | merged after the tools finish |
| `INTERRUPT` (`HIGH`) | resumes it | the reply is cancelled; `Task.resume` handles the message | tools finish (side effects cannot be undone), then merged like `STEER` |

```python
from rollout.core.harness import DeliveryPolicy, Priority

policy = DeliveryPolicy(max_priority_by_sender={"webhook": Priority.LOW})
assert policy.mode(Priority.HIGH) is DeliveryMode.INTERRUPT
assert policy.mode(Priority.HIGH, sender="webhook") is DeliveryMode.QUEUE  # capped: an integration cannot interrupt
```

## Steering

Messages delivered with mode `STEER` during a turn are passed to `Task.steer(run, envelopes, observation)` after
`respond` returns, and merged into the next observation. The default appends them as USER messages. Messages that
arrive as the episode ends are not merged into the final observation.

## Interrupting

A message delivered with mode `INTERRUPT` while the agent is acting cancels the reply in progress. The model
endpoint receives a best-effort `cancel(effect_id)`, the run records a `turn.interrupted` event, the partial reply
never enters the history, and the loop calls `Task.resume` with the message.

```python
from rollout.core.contracts import RunEventType, SampleRequest
from rollout.core.testing import events_of, payload


class Chat(Task):
    async def start(self, run: RunContext) -> Observation:
        return Observation("Write a long essay about tides.")

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        return Observation(end=Ending.TERMINATED)


async def interrupted_episode() -> None:
    started = asyncio.Event()

    async def slow_reply(request: SampleRequest) -> Message:
        started.set()
        await asyncio.Event().wait()  # never finishes on its own
        raise AssertionError("unreachable")

    task = Chat()
    run, endpoint = local_run(task, replies=[slow_reply, "Tides in one line: the moon pulls the sea."])
    episode = asyncio.create_task(rollout(task, Agent(), run))

    await started.wait()
    run.deliver(Envelope(content=[Text(text="Stop. One line only.")]), DeliveryMode.INTERRUPT)
    await episode

    cancelled = endpoint.requests[0].effect_id
    assert endpoint.cancelled == [cancelled]
    (interruption,) = events_of(run, RunEventType.TURN_INTERRUPTED)
    assert payload(interruption)["reply_effect_id"] == cancelled
    # The second request sees the original prompt and the interruption, not the cancelled reply.
    assert [m.text for m in endpoint.requests[1].context.append] == [
        "Write a long essay about tides.",
        "Stop. One line only.",
    ]


asyncio.run(interrupted_episode())
```

## Sending and replying

An addressable agent is a **deployment**: a name and a run specification. A message sent to a conversation of the
deployment (`{deployment}/{key}`) starts a run when none is live, and otherwise reaches the live run with the mode
its priority maps to. Messages are deduplicated by `idempotency_key`. Messages a run never consumed start the
conversation's next run. Replies leave a run with `run.emit`, which records an `output.emitted` event that clients
and connectors read.

```python
from rollout.core.harness import Address, Deployment, DirectModel, ModelBinding, RunBinding, RunSpecification
from rollout.core.harness import agent_program
from rollout.core.local import LocalRunner
from rollout.core.testing import ScriptedModelEndpoint


class Replying(Support):
    async def respond(self, run: RunContext, reply: Message) -> WaitFor:
        await run.emit("reply", reply.text, to=run.conversation.origin if run.conversation else None)
        return self.wait


async def through_the_runner() -> None:
    endpoint = ScriptedModelEndpoint(["Hello! What can I do for you?"])
    runner = LocalRunner(providers={"scripted": lambda model: endpoint})
    binding = RunBinding(models={"policy": ModelBinding(direct=DirectModel(provider="scripted", model="script"))})
    runner.deploy(Deployment(name="acme/support", specification=RunSpecification(program=agent_program(Replying),
                                                                                binding=binding)))

    conversation = Address(kind="conversation", value="acme/support/user:42")
    await runner.send(conversation, hello, idempotency_key="chat-event-1")
    await runner.send(conversation, hello, idempotency_key="chat-event-1")   # a retry: deduplicated

    (run,) = runner.conversation_runs("acme/support", "user:42")
    await run.result()                                   # ends when no message arrives within 50 ms
    replies = [event.payload for event in run.context.events if event.type is RunEventType.OUTPUT_EMITTED]
    assert [reply["payload"] for reply in replies] == ["Hello! What can I do for you?"]


asyncio.run(through_the_runner())
```

`run.send(...)` between runs and `run.spawn(...)` for child runs are designed but not built yet. See the design in
[conversations](../core/harness/conversations.md).
