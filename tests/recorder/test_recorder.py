"""The recorder and its renderers: tokens in and out of canonical content, thinking budgets, and recorded turns."""

import math
from collections.abc import Sequence
from typing import Any, cast

import pytest

from rollout.core.contracts import (
    ContextDelta,
    FinishReason,
    Message,
    Reasoning,
    SampleRequest,
    ToolSpecification,
    context_digests,
)
from rollout.core.harness import (
    End,
    ModelBinding,
    Observation,
    RecordedModel,
    RunBinding,
    RunContext,
    RunSpecification,
    Task,
    agent_program,
)
from rollout.core.local import LocalRunner
from rollout.recorder import Channel, Generation, Recorder, renderer_for
from rollout.recorder.renderers import Tokenizer

MODEL = "Qwen/Qwen3.5-9B"
MINE = ToolSpecification(
    name="mine",
    description="Mine a block you can see.",
    input_schema={
        "type": "object",
        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "z": {"type": "integer"}},
        "required": ["x", "y", "z"],
    },
)


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    transformers = pytest.importorskip("transformers")
    try:
        loaded: Any = transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)  # pyright: ignore[reportUnknownMemberType]
    except OSError:
        pytest.skip(f"the {MODEL} tokenizer is not in the local cache")
    return cast(Tokenizer, loaded)


def test_qwen35_renders_an_open_thinking_block_and_tools(tokenizer: Tokenizer) -> None:
    renderer = renderer_for("qwen3.5", tokenizer)
    tokens = renderer.render([Message.system("You mine."), Message.user("Go.")], [MINE])
    text = tokenizer.decode(tokens)
    assert text.endswith("<|im_start|>assistant\n<think>\n")
    assert '"name": "mine"' in text and "<function=example_function_name>" in text
    assert renderer.thinking_end_token_ids() == tokenizer.encode("</think>", add_special_tokens=False)


def test_qwen35_parses_reasoning_and_typed_xml_tool_calls(tokenizer: Tokenizer) -> None:
    renderer = renderer_for("qwen3.5", tokenizer)
    sampled = (
        "The ore is two blocks east.\n</think>\n\nMining it now.\n\n<tool_call>\n<function=mine>\n"
        "<parameter=x>\n3\n</parameter>\n<parameter=y>\n-58\n</parameter>\n<parameter=z>\n7\n</parameter>\n"
        "</function>\n</tool_call><|im_end|>"
    )
    message = renderer.parse(renderer.encode(sampled), [MINE])
    reasoning = [block for block in message.content if isinstance(block, Reasoning)]
    assert reasoning[0].text == "The ore is two blocks east."
    assert message.text == "Mining it now."
    (call,) = message.tool_calls
    assert (call.name, dict(call.arguments)) == ("mine", {"x": 3, "y": -58, "z": 7})


def test_qwen3_parses_json_tool_calls(tokenizer: Tokenizer) -> None:
    renderer = renderer_for("qwen3", tokenizer)
    call_json = '{"name": "mine", "arguments": {"x": 1, "y": 2, "z": 3}}'
    sampled = f"<think>\nok\n</think>\n\n<tool_call>\n{call_json}\n</tool_call><|im_end|>"
    (call,) = renderer.parse(renderer.encode(sampled), [MINE]).tool_calls
    assert dict(call.arguments) == {"x": 1, "y": 2, "z": 3}


def test_unknown_renderers_are_named(tokenizer: Tokenizer) -> None:
    with pytest.raises(KeyError, match=r"qwen3\.5"):
        renderer_for("llama9", tokenizer)


class ScriptedEngine:
    """Answers each generate with the next scripted (text, finish reason); logprobs are -0.5 per token."""

    def __init__(self, tokenizer: Tokenizer, script: list[tuple[str, str]]) -> None:
        self.tokenizer = tokenizer
        self.script = list(script)
        self.prompts: list[list[int]] = []
        self.budgets: list[int] = []

    async def generate(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Sequence[int],
        adapter: str | None,
    ) -> Generation:
        self.prompts.append(list(prompt))
        self.budgets.append(max_tokens)
        text, finish = self.script.pop(0)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
        return Generation(tokens=tokens, logprobs=[-0.5] * len(tokens), finish_reason=finish)


def request(messages: list[Message], effect_id: str = "r_1:0:0") -> SampleRequest:
    return SampleRequest(
        effect_id=effect_id,
        arguments_digest="d",
        session_id="r_1/ada",
        context=ContextDelta(append=messages, digest=context_digests(messages)[-1]),
        tools=[MINE],
    )


async def test_thinking_over_budget_is_closed_unsampled_and_the_answer_follows(tokenizer: Tokenizer) -> None:
    renderer = renderer_for("qwen3.5", tokenizer)
    engine = ScriptedEngine(
        tokenizer,
        [
            ("I should look around first and then decide which way", "length"),  # out of thinking budget
            ("<tool_call>\n<function=mine>\n<parameter=x>\n1\n</parameter>\n<parameter=y>\n2\n</parameter>\n"
             "<parameter=z>\n3\n</parameter>\n</function>\n</tool_call><|im_end|>", "stop"),
        ],
    )  # fmt: skip
    recorder = Recorder({"policy": Channel(engine, renderer, adapter="step-3", adapter_version=3, thinking_budget=8)})
    endpoint = recorder.endpoint(RecordedModel(channel="policy"))
    messages = [Message.system("You mine."), Message.user("Go.")]
    result = await endpoint.sample(request(messages))
    assert result.finish_reason is FinishReason.TOOL_USE
    assert dict(result.message.tool_calls[0].arguments) == {"x": 1, "y": 2, "z": 3}

    (turn,) = recorder.sessions_of("r_1")["ada"]
    forced = renderer.encode(renderer.thinking.forced_close)  # type: ignore[union-attr]
    unsampled = [index for index, sampled in enumerate(turn.loss_mask) if not sampled]
    assert unsampled == list(range(8, 8 + len(forced)))  # exactly the forced close
    assert all(math.isnan(turn.logprobs[index]) for index in unsampled)
    assert all(turn.logprobs[index] == -0.5 for index, sampled in enumerate(turn.loss_mask) if sampled)
    assert (turn.adapter, turn.adapter_version) == ("step-3", 3)
    assert engine.prompts[1] == [*turn.prompt, *turn.completion[: 8 + len(forced)]]  # the answer continues the thought

    again = await endpoint.sample(request(messages))  # a retried effect: recorded, not resampled
    assert again == result and len(recorder.sessions_of("r_1")["ada"]) == 1 and not engine.script


async def test_thinking_that_closes_in_budget_needs_no_forcing(tokenizer: Tokenizer) -> None:
    renderer = renderer_for("qwen3.5", tokenizer)
    engine = ScriptedEngine(tokenizer, [("Short thought.</think>", "stop"), ("\n\nNothing to do.<|im_end|>", "stop")])
    recorder = Recorder({"policy": Channel(engine, renderer, thinking_budget=64)})
    result = await recorder.endpoint(RecordedModel(channel="policy")).sample(request([Message.user("Hi")], "r_1:0:1"))
    (turn,) = recorder.sessions_of("r_1")["ada"]
    assert all(turn.loss_mask) and result.message.text == "Nothing to do."
    assert result.finish_reason is FinishReason.STOP


async def test_a_run_bound_to_a_recorded_channel_is_recorded(tokenizer: Tokenizer) -> None:

    class Ask(Task):
        async def start(self, run: RunContext) -> Observation:
            return Observation("Say hi.")

        async def respond(self, run: RunContext, reply: Message) -> Observation:
            return End(reward=1.0)

    engine = ScriptedEngine(tokenizer, [("ok</think>", "stop"), ("\n\nhi<|im_end|>", "stop")])
    recorder = Recorder({"policy": Channel(engine, renderer_for("qwen3.5", tokenizer))})
    runner = LocalRunner(recorder=recorder)
    binding = RunBinding(models={"policy": ModelBinding(recorded=RecordedModel(channel="policy"))})
    handle = await runner.start(RunSpecification(program=agent_program(Ask), binding=binding))
    await handle.result()
    (turn,) = recorder.sessions_of(handle.run_id)["policy"]
    assert tokenizer.decode(turn.completion).endswith("hi<|im_end|>")


async def test_a_metered_engine_counts_tokens_and_throughput() -> None:
    import asyncio

    from rollout.recorder import MeteredEngine

    class Slow:
        async def generate(self, prompt: Sequence[int], **options: Any) -> Generation:
            await asyncio.sleep(0.2)
            return Generation(tokens=[1] * 50, logprobs=[0.0] * 50, finish_reason="stop")

    metered = MeteredEngine(Slow())
    options: dict[str, Any] = {
        "max_tokens": 50,
        "temperature": 1.0,
        "top_p": 1.0,
        "stop_token_ids": [],
        "adapter": None,
    }
    await asyncio.gather(*(metered.generate([0] * 100, **options) for _ in range(4)))  # four at once
    counts = metered.take()
    assert counts["requests"] == 4 and counts["prompt_tokens"] == 400 and counts["generated_tokens"] == 200
    assert 0.15 < counts["busy_seconds"] < 0.5  # the four overlapped: busy once, not four times
    assert counts["tokens_per_second"] > 3 * counts["tokens_per_second_per_stream"]
    assert 3.0 < counts["mean_concurrency"] <= 4.0
    assert metered.take()["requests"] == 0  # taking resets


async def test_a_long_prompt_leaves_less_room_to_think_so_that_no_turn_is_too_long(tokenizer: Tokenizer) -> None:
    renderer = renderer_for("qwen3.5", tokenizer)
    messages = [Message.user("Say hi.")]
    engine = ScriptedEngine(tokenizer, [("thinking " * 30, "length"), ("\n\nhi<|im_end|>", "stop")])
    limit = len(renderer.render(messages, [MINE])) + 40  # (the request offers that tool)
    channel = Channel(engine, renderer, thinking_budget=64, answer_tokens=16, max_sequence_tokens=limit)
    recorder = Recorder({"policy": channel})
    await recorder.endpoint(RecordedModel(channel="policy")).sample(request(messages))
    prompt = engine.prompts[0]
    forced = len(renderer.encode(renderer.thinking.forced_close))  # type: ignore[union-attr]
    room = limit - len(prompt) - 16 - forced  # what the prompt leaves, after room for the answer and the close
    assert 0 < room < 64 and engine.budgets[0] == room  # not the channel's 64
    (turn,) = recorder.turns["r_1/ada"]
    assert len(turn.prompt) + len(turn.completion) <= limit


async def test_a_request_can_cap_its_output_down_to_no_thinking_at_all(tokenizer: Tokenizer) -> None:
    renderer = renderer_for("qwen3.5", tokenizer)
    engine = ScriptedEngine(tokenizer, [("A summary.<|im_end|>", "stop")])
    recorder = Recorder({"policy": Channel(engine, renderer, thinking_budget=64, answer_tokens=16)})
    capped = request([Message.user("Sum up.")]).model_copy(update={"max_output_tokens": 16})
    result = await recorder.endpoint(RecordedModel(channel="policy")).sample(capped)
    assert result.message.text == "A summary." and engine.budgets == [16]  # one generation: the answer
    (turn,) = recorder.turns["r_1/ada"]
    forced = renderer.encode(renderer.thinking.forced_close)  # type: ignore[union-attr]
    assert turn.completion[: len(forced)] == forced and turn.loss_mask[: len(forced)] == [False] * len(forced)
    assert all(turn.loss_mask[len(forced) :]) and engine.prompts[0] == [*turn.prompt, *forced]

    engine.script = [("thinking " * 30, "length"), ("\n\nhi<|im_end|>", "stop")]
    partly = request([Message.user("Sum up.")], "r_1:0:9").model_copy(update={"max_output_tokens": 24})
    await recorder.endpoint(RecordedModel(channel="policy")).sample(partly)
    assert engine.budgets[1:] == [8, 16]  # what the cap leaves after the answer's room, then the answer


async def test_a_channel_tells_programs_the_engines_own_limit_and_refuses_what_is_over_it(tokenizer: Tokenizer) -> None:
    from rollout.core.contracts import ContextOverflow

    renderer = renderer_for("qwen3.5", tokenizer)
    engine = ScriptedEngine(tokenizer, [("ok</think>", "stop"), ("\n\nhi<|im_end|>", "stop")])
    engine.max_model_len = 8192  # pyright: ignore[reportAttributeAccessIssue]
    channel = Channel(engine, renderer)
    endpoint = Recorder({"policy": channel}).endpoint(RecordedModel(channel="policy"))
    assert endpoint.describe("r_1/ada").context_limit == 8192  # the engine's, not a number of the channel's own
    assert Channel(engine, renderer, max_sequence_tokens=5400).limit == 5400  # or the trainer's, if that is less

    messages = [Message.user("Say hi.")]
    tight = Channel(engine, renderer, answer_tokens=16, context_limit=len(renderer.render(messages, [MINE])) + 15)
    with pytest.raises(ContextOverflow):  # what tells a program to compact and try again
        await Recorder({"policy": tight}).endpoint(RecordedModel(channel="policy")).sample(request(messages))


async def test_a_paused_engine_finishes_what_is_in_flight_and_holds_the_rest_back() -> None:
    import asyncio

    from rollout.recorder import MeteredEngine

    started: list[int] = []

    class Slow:
        async def generate(self, prompt: Sequence[int], **options: Any) -> Generation:
            started.append(prompt[0])
            await asyncio.sleep(0.1)
            return Generation(tokens=[1], logprobs=[0.0], finish_reason="stop")

    metered = MeteredEngine(Slow())
    options: dict[str, Any] = {"max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "stop_token_ids": [], "adapter": None}
    first = asyncio.create_task(metered.generate([1], **options))
    await asyncio.sleep(0.02)
    await metered.pause()  # returns once the request in flight is done
    assert first.done() and started == [1]
    second = asyncio.create_task(metered.generate([2], **options))
    await asyncio.sleep(0.05)
    assert started == [1] and not second.done()  # held back: the engine may sleep now
    metered.resume()
    await second
    assert started == [1, 2]
