"""Memory for long episodes: recent turns as they were, a summary of the older ones, and a context that fits
whatever the model. On a scripted model that counts 100 tokens a message."""

from typing import Any

import pytest

from rollout.contracts import (
    CapabilityContract,
    ContextOverflow,
    FinishReason,
    Message,
    Role,
    SampleRequest,
    SampleResult,
    ToolCall,
    ToolResult,
    ToolResultBlock,
    ToolSpecification,
    Usage,
)
from rollout.harness import (
    CompactingAgent,
    DirectModel,
    Ending,
    Memory,
    ModelBinding,
    Observation,
    RunBinding,
    RunContext,
    RunSpecification,
    RunStatus,
    Task,
    agent_program,
)
from rollout.harness.memory import PROMPT
from rollout.harness.model import EndpointModel
from rollout.local import LocalRunner
from rollout.local.context import LocalRunContext
from rollout.testing import tool_call_reply

LIMIT, OUTPUT = 5_000, 1_400
WAIT = ToolSpecification(name="wait")


class Counting:
    """Calls `wait` every turn; asked what to remember, answers with a numbered summary. It reports 100 tokens of
    input a message; or, with `overflow`, reports nothing and refuses a context of more than that many messages."""

    def __init__(self, overflow: int | None = None) -> None:
        self.requests: list[SampleRequest] = []
        self.summaries = 0
        self.overflow = overflow

    def describe(self, session_id: str) -> CapabilityContract:
        return CapabilityContract(context_limit=LIMIT, max_output_tokens=OUTPUT)

    async def cancel(self, effect_id: str) -> None:
        pass

    async def sample(self, request: SampleRequest) -> SampleResult:
        messages = len(request.context.append)
        compacting = request.context.append[-1].text == PROMPT
        if self.overflow is not None and not compacting and messages > self.overflow:
            raise ContextOverflow(LIMIT)
        self.requests.append(request)
        tokens = None if self.overflow is not None else 100 * messages
        usage = Usage(context_used=tokens or 1, context_limit=LIMIT, input_tokens=tokens)
        if compacting:
            self.summaries += 1
            summary = Message.assistant(f"SUMMARY {self.summaries}")
            return SampleResult(message=summary, finish_reason=FinishReason.STOP, usage=usage)
        call = ToolCall(call_id=f"c{len(self.requests)}", name="wait", arguments={})
        return SampleResult(message=tool_call_reply(call), finish_reason=FinishReason.TOOL_USE, usage=usage)


def model(endpoint: Counting) -> EndpointModel:
    return EndpointModel(endpoint, "r_1/policy", LocalRunContext("r_1", {"policy": endpoint}))


async def play(memory: Memory, policy: EndpointModel, turns: int) -> None:
    """Turns as a program plays them: what is seen in full now, and remembered in brief afterwards."""
    system = Message.system("You walk east.")
    for turn in range(1, turns + 1):
        if memory.turns:
            memory.answer("waited")
        if memory.crowded(policy):
            await memory.compact(policy, system)
        reply = await memory.sample(policy, system=system, current=[Message.user(f"FULL {turn}")], tools=[WAIT])
        memory.remember(Message.user(f"BRIEF {turn}"), reply)


def texts(request: SampleRequest) -> list[str]:
    return [message.text for message in request.context.append]


async def test_memory_is_compacted_when_the_model_says_its_context_is_nearly_full() -> None:
    endpoint, memory = Counting(), Memory()
    await play(memory, model(endpoint), 24)
    acting = [request for request in endpoint.requests if texts(request)[-1] != PROMPT]
    compactions = [request for request in endpoint.requests if texts(request)[-1] == PROMPT]
    assert len(acting) == 24 and len(compactions) == memory.compactions == 2

    # A context grows by a turn (what was seen, the reply, how it went) until one more might cut into the room the
    # model may use to reply; it never does. (A turn adds 300 tokens here.)
    assert [len(request.context.append) for request in acting[:3]] == [2, 5, 8]
    largest = max(len(request.context.append) for request in acting)
    assert 100 * largest <= LIMIT - OUTPUT < 100 * (largest + 6)
    assert all(texts(request)[-1].startswith("FULL") for request in acting)  # what is seen now is always in full
    assert not any(text.startswith("FULL") for request in acting for text in texts(request)[:-1])  # and only that

    # The compaction: the older two thirds are shown once more; no tools are offered; there is room for a summary.
    first = compactions[0]
    turns = (len(first.context.append) - 2) // 3
    assert not first.tools and first.max_output_tokens == LIMIT // 20
    assert texts(first)[1] == "BRIEF 1" and texts(first)[-4] == f"BRIEF {turns}"
    after = next(request for request in acting if "SUMMARY 1" in texts(request)[1])
    assert texts(after)[1].endswith("SUMMARY 1") and texts(after)[2] == f"BRIEF {turns + 1}"

    # The second compaction builds on the first summary, and replaces it.
    assert texts(compactions[1])[1].endswith("SUMMARY 1")
    assert texts(acting[-1])[1].endswith("SUMMARY 2") and memory.summary == "SUMMARY 2"


async def test_each_request_says_how_it_follows_from_a_compaction() -> None:
    endpoint, memory = Counting(), Memory()
    await play(memory, model(endpoint), 24)
    linked = [
        (texts(request)[-1] == PROMPT, [(link.type, link.source) for link in request.links])
        for request in endpoint.requests
    ]
    summaries = [index for index, (compacting, _) in enumerate(linked) if compacting]
    assert len(summaries) == 2
    for index in summaries:  # a summary's request names the latest reply's; the next request names the summary's
        before, summary, after = endpoint.requests[index - 1], endpoint.requests[index], endpoint.requests[index + 1]
        assert linked[index][1] == [("compaction_attempt", before.effect_id)]
        assert linked[index + 1][1] == [("compaction", summary.effect_id)] and after is not summary
    assert all(
        not links for index, (_, links) in enumerate(linked) if index not in {*summaries, *(i + 1 for i in summaries)}
    )


async def test_a_context_the_model_refuses_is_compacted_and_tried_again() -> None:
    endpoint, memory = Counting(overflow=20), Memory()  # says nothing of its tokens; refuses more than 20 messages
    await play(memory, model(endpoint), 12)
    acting = [request for request in endpoint.requests if texts(request)[-1] != PROMPT]
    assert len(acting) == 12 and memory.compactions >= 1
    assert max(len(request.context.append) for request in acting) <= 20
    assert texts(acting[-1])[1].startswith("What you remember from earlier")

    stuck = Memory()
    with pytest.raises(ContextOverflow):  # with nothing to forget, the refusal is the caller's to handle
        current = [Message.user("x")] * 30
        await stuck.sample(model(Counting(overflow=20)), current=current)


def test_every_reply_is_answered_whatever_it_called() -> None:
    seen = Message.user("You are ada.")
    silent = Memory(turns=[[seen, Message.assistant("I wonder.")]])
    silent.answer("You called no tool.")
    assert silent.turns[0][-1].role is Role.USER and silent.turns[0][-1].text == "You called no tool."

    first = ToolCall(call_id="c1", name="mine", arguments={"x": 1})
    second = ToolCall(call_id="c2", name="move", arguments={"direction": "north"})
    eager = Memory(turns=[[seen, tool_call_reply(first).model_copy(update={"content": [first, second]})]])
    eager.answer("mined: stone", others="Only your first call counts.")
    results = [block for block in eager.turns[0][-1].content if isinstance(block, ToolResultBlock)]
    assert [block.call_id for block in results] == ["c1", "c2"]  # each call has its answer
    said: list[Any] = [block.result.content[0] for block in results]
    assert [part.text for part in said] == ["mined: stone", "Only your first call counts."]
    Memory().answer("nothing to answer")  # (no turn yet: nothing happens)


async def test_the_task_loop_gets_the_same_memory_from_a_compacting_agent() -> None:
    class Long(Task):
        async def start(self, run: RunContext) -> Observation:
            return Observation("Step 0.")

        async def respond(self, run: RunContext, reply: Message) -> Observation:
            calls = reply.tool_calls
            answered = Message(role=Role.TOOL, content=[ToolResultBlock(call_id=calls[0].call_id, result=ToolResult())])
            if run.turn >= 39:
                return Observation([answered], reward=1.0, end=Ending.TERMINATED)
            return Observation([answered, Message.user(f"Step {run.turn + 1}.")])

    endpoint = Counting()
    runner = LocalRunner(providers={"scripted": lambda _: endpoint})
    binding = RunBinding(models={"policy": ModelBinding(direct=DirectModel(provider="scripted", model="m"))})
    handle = await runner.start(RunSpecification(program=agent_program(Long, CompactingAgent), binding=binding))
    assert (await handle.result()).status is RunStatus.COMPLETED
    acting = [request for request in endpoint.requests if texts(request)[-1] != PROMPT]
    assert len(acting) == 40 and endpoint.summaries >= 1
    assert max(100 * len(request.context.append) for request in acting) <= LIMIT - OUTPUT
    assert texts(acting[-1])[-1] == "Step 39." and any("SUMMARY" in text for text in texts(acting[-1]))
    for request in acting:  # a tool call and its result are never parted by a compaction
        called = {call.call_id for message in request.context.append for call in message.tool_calls}
        answered = {
            block.call_id
            for message in request.context.append
            for block in message.content
            if isinstance(block, ToolResultBlock)
        }
        assert answered <= called and len(called - answered) == 0
