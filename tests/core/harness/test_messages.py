import asyncio
from datetime import timedelta

from rollout.core.contracts import Message, Role, RunEventType, SampleRequest, Text, ToolCall
from rollout.core.harness import (
    Agent,
    DeliveryMode,
    DeliveryPolicy,
    Ending,
    Envelope,
    Observation,
    Priority,
    RunContext,
    Task,
    WaitFor,
    rollout,
    tool,
)
from rollout.core.local import LocalRunContext
from rollout.core.testing import events_of, local_run, payload, tool_call_reply


def envelope(text: str, kind: str = "message") -> Envelope:
    return Envelope(kind=kind, content=[Text(text=text)])


class Held:
    """A scripted reply that stays in flight until released (or cancelled)."""

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, request: SampleRequest) -> Message:
        self.started.set()
        await self.release.wait()
        return Message.assistant(self.text)


MESSAGE = WaitFor("message")


class Conversation(Task):
    """Waits for the first message, answers each message, then waits for the next."""

    def __init__(self, wait: WaitFor = MESSAGE) -> None:
        self.wait = wait

    async def start(self, run: RunContext) -> WaitFor:
        return self.wait

    async def respond(self, run: RunContext, reply: Message) -> Observation | WaitFor:
        return self.wait


def user_texts(run: LocalRunContext) -> list[str]:
    return [message.text for message in run.history.messages() if message.role is Role.USER]


async def until(condition: asyncio.Event) -> None:
    await asyncio.wait_for(condition.wait(), timeout=5)


async def test_wait_for_resumes_on_a_message_and_ends_on_timeout() -> None:
    task = Conversation(WaitFor("message", timeout=timedelta(milliseconds=50)))
    run, endpoint = local_run(task, ["Hello!"])
    episode = asyncio.create_task(rollout(task, Agent(), run))
    await asyncio.sleep(0)
    run.deliver(envelope("hi"), DeliveryMode.QUEUE)
    await asyncio.wait_for(episode, timeout=5)
    assert user_texts(run) == ["hi"]
    assert run.history.turns[-1].observation.end is Ending.TRUNCATED  # pyright: ignore[reportOptionalMemberAccess]
    assert len(events_of(run, RunEventType.RUN_SUSPENDED)) == 2
    assert len(endpoint.requests) == 1


async def test_wait_for_takes_only_its_kind() -> None:
    task = Conversation(WaitFor("approval", timeout=timedelta(milliseconds=50)))
    run, _ = local_run(task, ["Approved, proceeding."])
    run.deliver(envelope("chatter"), DeliveryMode.QUEUE)
    run.deliver(envelope("yes", kind="approval"), DeliveryMode.QUEUE)
    await asyncio.wait_for(rollout(task, Agent(), run), timeout=5)
    assert user_texts(run) == ["yes"]


async def test_queued_messages_wait_for_the_next_wait_for() -> None:
    task = Conversation(WaitFor("message", timeout=timedelta(milliseconds=50)))
    held = Held("First answer.")
    run, endpoint = local_run(task, [held, "Second answer."])
    episode = asyncio.create_task(rollout(task, Agent(), run))
    await asyncio.sleep(0)
    run.deliver(envelope("first"), DeliveryMode.QUEUE)
    await until(held.started)
    run.deliver(envelope("second"), DeliveryMode.QUEUE)
    held.release.set()
    await asyncio.wait_for(episode, timeout=5)
    assert user_texts(run) == ["first", "second"]
    assert [message.text for message in endpoint.requests[1].context.append] == ["first", "First answer.", "second"]


class Chat(Task):
    """Answers until the model says goodbye."""

    def __init__(self) -> None:
        self.lookup_started = asyncio.Event()
        self.lookup_release = asyncio.Event()

    async def start(self, run: RunContext) -> Observation:
        return Observation("hi")

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        if reply.tool_calls:
            return await self.run_tools(run, reply)
        if "bye" in reply.text:
            return Observation(end=Ending.TERMINATED)
        return Observation("go on")

    @tool
    async def lookup(self) -> str:
        """Look something up."""
        self.lookup_started.set()
        await until(self.lookup_release)
        return "found"


async def test_steering_messages_are_merged_after_the_turn() -> None:
    task = Chat()
    held = Held("Working on it.")
    run, endpoint = local_run(task, [held, "bye"])
    episode = asyncio.create_task(rollout(task, Agent(), run))
    await until(held.started)
    run.deliver(envelope("also check the logs"), DeliveryMode.STEER)
    held.release.set()
    await asyncio.wait_for(episode, timeout=5)
    assert endpoint.cancelled == []
    assert user_texts(run) == ["hi", "go on", "also check the logs"]
    assert len(run.history.turns[1].observation.messages) == 2  # pyright: ignore[reportOptionalMemberAccess]


async def test_interrupt_cancels_the_sample_and_resumes_with_the_message() -> None:
    task = Chat()
    held = Held("never finished")
    run, endpoint = local_run(task, [held, "bye"])
    episode = asyncio.create_task(rollout(task, Agent(), run))
    await until(held.started)
    run.deliver(envelope("stop, do this instead"), DeliveryMode.INTERRUPT)
    await asyncio.wait_for(episode, timeout=5)
    interrupted_effect = endpoint.requests[0].effect_id
    assert endpoint.cancelled == [interrupted_effect]
    (interruption,) = events_of(run, RunEventType.TURN_INTERRUPTED)
    assert payload(interruption)["reply_effect_id"] == interrupted_effect
    # The aborted reply never enters the history; the message becomes the next observation.
    assert [message.text for message in endpoint.requests[1].context.append] == ["hi", "stop, do this instead"]
    assert run.turn == 1


async def test_interrupt_during_tools_lets_them_finish_and_merges_like_steer() -> None:
    task = Chat()
    run, endpoint = local_run(task, [tool_call_reply(ToolCall(call_id="c1", name="lookup", arguments={})), "bye"])
    episode = asyncio.create_task(rollout(task, Agent(), run))
    await until(task.lookup_started)
    run.deliver(envelope("hurry up"), DeliveryMode.INTERRUPT)
    task.lookup_release.set()
    await asyncio.wait_for(episode, timeout=5)
    assert endpoint.cancelled == []
    tool_observation = run.history.turns[1].observation
    assert [message.role for message in tool_observation.messages] == [Role.TOOL, Role.USER]  # pyright: ignore[reportOptionalMemberAccess]
    assert tool_observation.messages[1].text == "hurry up"  # pyright: ignore[reportOptionalMemberAccess]


def test_delivery_policy_maps_priorities_and_caps_senders() -> None:
    policy = DeliveryPolicy(max_priority_by_sender={"webhook": Priority.LOW})
    assert policy.mode(Priority.LOW) is DeliveryMode.QUEUE
    assert policy.mode(Priority.NORMAL) is DeliveryMode.STEER
    assert policy.mode(Priority.HIGH) is DeliveryMode.INTERRUPT
    assert policy.mode(Priority.HIGH, sender="webhook") is DeliveryMode.QUEUE
    assert policy.mode(Priority.HIGH, sender="person") is DeliveryMode.INTERRUPT
