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
