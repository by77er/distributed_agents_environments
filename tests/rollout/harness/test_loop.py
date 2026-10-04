import pytest

from rollout.contracts import (
    Message,
    Role,
    RunEventType,
    Text,
    ToolCall,
    ToolResult,
    ToolResultBlock,
    context_digests,
)
from rollout.harness import (
    Agent,
    End,
    Ending,
    InvalidObservation,
    ModelSlot,
    Observation,
    RunContext,
    Task,
    rollout,
    tool,
)
from rollout.testing import events_of, local_run, payload, tool_call_reply


class Arithmetic(Task):
    def __init__(self, parameters: dict[str, str]) -> None:
        self.question, self.answer = parameters["question"], parameters["answer"]

    async def start(self, run: RunContext) -> Observation:
        return Observation(self.question)

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        return End(reward=1.0 if reply.text.strip() == self.answer else 0.0)


class Wordle(Task):
    max_turns = 3

    def __init__(self, secret: str) -> None:
        self.secret = secret

    async def start(self, run: RunContext) -> Observation:
        return Observation("Guess the 5-letter word.")

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        guess = reply.text.strip().lower()
        if len(guess) != 5:
            return Observation("Invalid. Reply with one 5-letter word.", reward=-0.1)
        if guess == self.secret:
            return End(reward=1.0)
        return Observation("".join("g" if a == b else "." for a, b in zip(guess, self.secret, strict=True)))


class Lookup(Task):
    """Tool-driven: the default `respond` executes tool calls and ends on a reply without any."""

    def __init__(self) -> None:
        self.lifecycle: list[str] = []

    async def setup(self, run: RunContext) -> None:
        self.lifecycle.append("setup")

    async def start(self, run: RunContext) -> Observation:
        return Observation("What is the capital of France?")

    async def score(self, run: RunContext) -> float | None:
        self.lifecycle.append("score")
        return 1.0 if "Paris" in run.history.turns[-1].reply.text else 0.0  # pyright: ignore[reportOptionalMemberAccess]

    async def teardown(self, run: RunContext) -> None:
        self.lifecycle.append("teardown")

    @tool
    async def search(self, query: str) -> str:
        """Search an encyclopedia."""
        return "Paris is the capital of France."


async def test_single_turn_task() -> None:
    task = Arithmetic({"question": "2 + 3?", "answer": "5"})
    run, endpoint = local_run(task, ["5"])
    await rollout(task, Agent(), run)
    (recorded_start, recorded_end) = events_of(run, RunEventType.OBSERVATION_RECORDED)
    assert payload(recorded_start)["reply_effect_id"] is None
    assert payload(recorded_end)["reward"] == 1.0
    assert payload(recorded_end)["reply_effect_id"] == endpoint.requests[0].effect_id
    assert run.turn == 1
    assert [event.seq for event in run.events] == list(range(len(run.events)))


async def test_multi_step_task_binds_rewards_to_replies_and_truncates_at_max_turns() -> None:
    task = Wordle("crane")
    run, endpoint = local_run(task, ["abc", "crate", "brine"])
    await rollout(task, Agent(), run)
    turns = run.history.turns
    assert [turn.reply.text if turn.reply else None for turn in turns] == [None, "abc", "crate", "brine", None]
    assert turns[1].observation.reward == -0.1  # pyright: ignore[reportOptionalMemberAccess]
    assert turns[-1].observation.end is Ending.TRUNCATED  # pyright: ignore[reportOptionalMemberAccess]
    # The model sees the whole episode each turn: here the third request carries 5 messages.
    assert len(endpoint.requests[2].context.append) == 5
    assert endpoint.requests[2].context.digest == context_digests(endpoint.requests[2].context.append)[-1]


async def test_tool_driven_task_runs_the_lifecycle_in_order() -> None:
    task = Lookup()
    search = ToolCall(call_id="c1", name="search", arguments={"query": "capital of France"})
    run, endpoint = local_run(task, [tool_call_reply(search), "Paris."])
    await rollout(task, Agent(), run)
    assert task.lifecycle == ["setup", "score", "teardown"]
    assert [reward.value for reward in run.rewards] == [1.0]
    assert [tool.name for tool in endpoint.requests[0].tools] == ["search"]
    tool_message = run.history.turns[1].observation.messages[0]  # pyright: ignore[reportOptionalMemberAccess]
    assert tool_message.role is Role.TOOL


class FailingSetup(Lookup):
    async def setup(self, run: RunContext) -> None:
        await super().setup(run)
        raise RuntimeError("no environment")


class FailingRespond(Lookup):
    async def respond(self, run: RunContext, reply: Message) -> Observation:
        raise RuntimeError("step failed")


@pytest.mark.parametrize("task_class", [FailingSetup, FailingRespond])
async def test_teardown_runs_and_score_does_not_after_a_hook_raised(task_class: type[Lookup]) -> None:
    task = task_class()
    run, _ = local_run(task, ["Paris."])
    with pytest.raises(RuntimeError):
        await rollout(task, Agent(), run)
    assert "score" not in task.lifecycle
    assert task.lifecycle[-1] == "teardown"


class Scripted(Task):
    """Returns scripted observations from `respond`."""

    def __init__(self, responses: list[Observation]) -> None:
        self.responses = responses

    async def start(self, run: RunContext) -> Observation:
        return Observation("go")

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        return self.responses.pop(0)


SYSTEM_MESSAGE = Message.system("not allowed")
UNANSWERED_RESULT = Message(role=Role.TOOL, content=[ToolResultBlock(call_id="other", result=ToolResult())])


@pytest.mark.parametrize(
    ("reply", "response"),
    [
        ("hi", Observation(SYSTEM_MESSAGE)),
        ("hi", Observation(Message.assistant("impersonated"))),
        ("hi", Observation()),
        ("hi", Observation(UNANSWERED_RESULT)),
        (tool_call_reply(ToolCall(call_id="c1", name="t", arguments={})), End()),
    ],
)
async def test_invalid_observations_are_rejected(reply: str | Message, response: Observation) -> None:
    task = Scripted([response])
    run, _ = local_run(task, [reply])
    with pytest.raises(InvalidObservation):
        await rollout(task, Agent(), run)


async def test_ending_with_tool_results_is_valid() -> None:
    answer = Message(role=Role.TOOL, content=[ToolResultBlock(call_id="c1", result=ToolResult())])
    task = Scripted([Observation(answer, end=Ending.TERMINATED)])
    run, _ = local_run(task, [tool_call_reply(ToolCall(call_id="c1", name="t", arguments={}))])
    await rollout(task, Agent(), run)
    assert run.turn == 1


class SimulatedUser(Task):
    """A second model inside the environment: the task samples the frozen `user` slot itself."""

    models = {"policy": ModelSlot(), "user": ModelSlot()}

    async def start(self, run: RunContext) -> Observation:
        return Observation("Hello, I need help.")

    async def respond(self, run: RunContext, reply: Message) -> Observation:
        user_turn = await run.models["user"].sample([Message.user(reply.text)])
        if "[resolved]" in user_turn.text:
            return End(reward=1.0)
        return Observation(user_turn.text)


async def test_a_task_can_sample_its_own_model_slots() -> None:
    task = SimulatedUser()
    run, endpoint = local_run(task, ["How can I help?", "My printer is broken.", "Restart it.", "Thanks [resolved]"])
    await rollout(task, Agent(), run)
    assert [request.session_id.split("/")[1] for request in endpoint.requests] == ["policy", "user", "policy", "user"]
    assert run.turn == 2


async def test_agent_system_prompt_leads_the_context() -> None:
    class Terse(Agent):
        system_prompt = "Be terse."

    task = Arithmetic({"question": "1 + 1?", "answer": "2"})
    run, endpoint = local_run(task, ["2"])
    await rollout(task, Terse(), run)
    assert endpoint.requests[0].context.append[0].content == (Text(text="Be terse."),)
