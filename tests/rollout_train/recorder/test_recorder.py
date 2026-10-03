"""The recorder and its renderers: tokens in and out of canonical content, thinking budgets, and what sessions
export."""

import pytest

from rollout.contracts import ContextOverflow, FinishReason, Message, Reasoning, ToolSpecification
from rollout.harness import (
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
from rollout.local import LocalRunner
from rollout_qwen import qwen3, qwen35
from rollout_train.recorder import Recorder, Span
from rollout_train.recorder.renderers import Tokenizer
from rollout_train.testing import ScriptedEngine, plain_channel, sample_request
from tests.rollout_qwen.support import channel, qwen_tokenizer

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
    return qwen_tokenizer()


def test_qwen35_renders_an_open_thinking_block_and_tools(tokenizer: Tokenizer) -> None:
    renderer = qwen35(tokenizer)
    tokens = renderer.render([Message.system("You mine."), Message.user("Go.")], [MINE])
    text = tokenizer.decode(tokens)
    assert text.endswith("<|im_start|>assistant\n<think>\n")
    assert '"name": "mine"' in text and "<function=example_function_name>" in text
    assert renderer.thinking_end_token_ids() == tokenizer.encode("</think>", add_special_tokens=False)


def test_qwen35_parses_reasoning_and_typed_xml_tool_calls(tokenizer: Tokenizer) -> None:
    renderer = qwen35(tokenizer)
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


def test_thinking_that_goes_on_after_it_was_closed_is_still_thinking(tokenizer: Tokenizer) -> None:
    renderer = qwen35(tokenizer)
    sampled = "\n</think>\n\nI should note the ore.\n</think>\n\nOre at (1, 2, 3).<|im_end|>"  # closed by force first
    message = renderer.parse(renderer.encode(sampled), [])
    (thought,) = [block.text for block in message.content if isinstance(block, Reasoning)]
    assert message.text == "Ore at (1, 2, 3)." and thought == "I should note the ore."


def test_qwen3_parses_json_tool_calls(tokenizer: Tokenizer) -> None:
    renderer = qwen3(tokenizer)
    call_json = '{"name": "mine", "arguments": {"x": 1, "y": 2, "z": 3}}'
    sampled = f"<think>\nok\n</think>\n\n<tool_call>\n{call_json}\n</tool_call><|im_end|>"
    (call,) = renderer.parse(renderer.encode(sampled), [MINE]).tool_calls
    assert dict(call.arguments) == {"x": 1, "y": 2, "z": 3}


CALL = (
    "<tool_call>\n<function=mine>\n<parameter=x>\n1\n</parameter>\n<parameter=y>\n2\n</parameter>\n"
    "<parameter=z>\n3\n</parameter>\n</function>\n</tool_call><|im_end|>"
)
POLICY = RecordedModel(channel="policy")


async def test_thinking_over_budget_is_closed_unsampled_and_the_answer_follows(tokenizer: Tokenizer) -> None:
    engine = ScriptedEngine(
        tokenizer, [("I should look around first and then decide which way", "length"), (CALL, "stop")]
    )
    served = channel(engine, thinking=8)
    served.adapter, served.version = "step-3", 3
    recorder = Recorder({"policy": served})
    endpoint = recorder.endpoint(POLICY)
    messages = [Message.system("You mine."), Message.user("Go.")]
    result = await endpoint.sample(sample_request(messages, tools=[MINE]))
    assert result.finish_reason is FinishReason.TOOL_USE
    assert dict(result.message.tool_calls[0].arguments) == {"x": 1, "y": 2, "z": 3}

    (segment,) = recorder.export("r_1/ada")
    prompt, forced = engine.prompts[0], served.renderer.encode("\n</think>\n\n")
    answer = len(segment.tokens) - len(prompt) - 8 - len(forced)
    assert segment.tokens[: len(prompt)] == prompt
    # Two sampled spans, around the forced close: the eight tokens of thought, and the answer. Both at version 3.
    assert segment.spans == [
        Span(len(prompt), len(prompt) + 8, 3, "r_1:0:0"),
        Span(len(prompt) + 8 + len(forced), len(segment.tokens), 3, "r_1:0:0"),
    ]
    assert segment.sampled == 8 + answer and segment.logprobs == [-0.5] * (8 + answer)
    assert engine.adapters == ["step-3", "step-3"]
    assert engine.prompts[1] == segment.tokens[: len(prompt) + 8 + len(forced)]  # the answer continues the thought

    again = await endpoint.sample(sample_request(messages, tools=[MINE]))  # a retried effect: not resampled
    assert again == result and len(recorder.export("r_1/ada")) == 1 and not engine.script


async def test_thinking_that_closes_in_budget_needs_no_forcing(tokenizer: Tokenizer) -> None:
    engine = ScriptedEngine(tokenizer, [("Short thought.</think>", "stop"), ("\n\nNothing to do.<|im_end|>", "stop")])
    recorder = Recorder({"policy": channel(engine, thinking=64)})
    result = await recorder.endpoint(POLICY).sample(sample_request([Message.user("Hi")]))
    (segment,) = recorder.export("r_1/ada")
    assert segment.spans == [Span(len(engine.prompts[0]), len(segment.tokens), 0, "r_1:0:0")]  # all of it sampled
    assert result.message.text == "Nothing to do." and result.finish_reason is FinishReason.STOP


async def test_a_run_bound_to_a_recorded_channel_is_recorded(tokenizer: Tokenizer) -> None:
    class Ask(Task):
        async def start(self, run: RunContext) -> Observation:
            return Observation("Say hi.")

        async def respond(self, run: RunContext, reply: Message) -> Observation:
            return End(reward=1.0)

    engine = ScriptedEngine(tokenizer, [("ok</think>", "stop"), ("\n\nhi<|im_end|>", "stop")])
    recorder = Recorder({"policy": channel(engine)})
    runner = LocalRunner(recorder=recorder)
    binding = RunBinding(models={"policy": ModelBinding(recorded=POLICY)})
    handle = await runner.start(RunSpecification(program=agent_program(Ask), binding=binding))
    await handle.result()
    (segment,) = recorder.sessions(handle.run_id)["policy"]
    assert tokenizer.decode(segment.tokens).endswith("hi<|im_end|>")
    recorder.forget(handle.run_id)
    assert recorder.sessions(handle.run_id) == {}


async def test_a_conversation_that_only_grows_is_one_sequence_and_an_edited_one_starts_another() -> None:
    served = plain_channel(always=[("One.\n", "stop"), ("Two.\n", "stop")])
    recorder = Recorder({"policy": served})
    endpoint = recorder.endpoint(POLICY)
    first = [Message.user("Count.")]
    one = await endpoint.sample(sample_request(first, "e1"))
    assert one.message.text == "One."
    served.version = 1  # (weights were published between the turns)
    grown = [*first, one.message, Message.user("Go on.")]
    await endpoint.sample(sample_request(grown, "e2"))
    (segment,) = recorder.export("r_1/ada")  # one segment for both turns
    text = "".join(chr(token) for token in segment.tokens)
    assert text == "user: Count.\nassistant: One.\nuser: Go on.\nassistant: Two.\n"
    assert [text[span.start : span.end] for span in segment.spans] == ["One.\n", "Two.\n"]
    assert [span.version for span in segment.spans] == [0, 1] and segment.logprobs == [-0.5] * 10

    # An edited context (the first exchange summarised away) cannot continue it: a second segment begins.
    edited = [Message.user("You counted to two. Go on.")]
    await endpoint.sample(sample_request(edited, "e3"))
    earlier, later = recorder.export("r_1/ada")
    assert earlier == segment and len(later.spans) == 1

    # The same prompt again (a client that retried) replaces what it sampled before.
    await endpoint.sample(sample_request(edited, "e4"))
    earlier, retried = recorder.export("r_1/ada")
    assert earlier == segment and "".join(chr(token) for token in retried.tokens).endswith("assistant: Two.\n")


async def test_a_long_prompt_leaves_less_room_to_think_so_that_no_turn_is_too_long(tokenizer: Tokenizer) -> None:
    messages = [Message.user("Say hi.")]
    engine = ScriptedEngine(tokenizer, [("thinking " * 30, "length"), ("\n\nhi<|im_end|>", "stop")])
    renderer = qwen35(tokenizer)
    limit = len(renderer.render(messages, [MINE])) + 40
    recorder = Recorder({"policy": channel(engine, thinking=64, answer=16, sequence=limit)})
    await recorder.endpoint(POLICY).sample(sample_request(messages, tools=[MINE]))
    forced = len(renderer.encode("\n</think>\n\n"))
    room = limit - len(engine.prompts[0]) - 16 - forced  # what the prompt leaves, after the answer and the close
    assert 0 < room < 64 and engine.budgets[0] == room  # not the channel's 64
    (segment,) = recorder.export("r_1/ada")
    assert len(segment.tokens) <= limit


async def test_a_request_can_cap_its_output_down_to_no_thinking_at_all(tokenizer: Tokenizer) -> None:
    engine = ScriptedEngine(tokenizer, [("A summary.<|im_end|>", "stop")])
    recorder = Recorder({"policy": channel(engine, thinking=64, answer=16)})
    capped = sample_request([Message.user("Sum up.")]).model_copy(update={"max_output_tokens": 16})
    result = await recorder.endpoint(POLICY).sample(capped)
    assert result.message.text == "A summary." and engine.budgets == [16]  # one generation: the answer
    (segment,) = recorder.export("r_1/ada")
    forced = len(qwen35(tokenizer).encode("\n</think>\n\n"))
    assert segment.spans == [Span(len(engine.prompts[0]), len(segment.tokens), 0, "r_1:0:0")]  # after the forced close
    assert len(engine.prompts[0]) == len(segment.tokens) - segment.sampled  # (the engine was shown the close)
    assert forced > 0

    engine.script = [("thinking " * 30, "length"), ("\n\nhi<|im_end|>", "stop")]
    partly = sample_request([Message.user("Sum up.")], "r_1:0:9").model_copy(update={"max_output_tokens": 24})
    await recorder.endpoint(POLICY).sample(partly)
    assert engine.budgets[1:] == [8, 16]  # what the cap leaves after the answer's room, then the answer


async def test_a_channel_tells_programs_its_limit_and_refuses_what_is_over_it(tokenizer: Tokenizer) -> None:
    engine = ScriptedEngine(tokenizer, [("ok</think>", "stop"), ("\n\nhi<|im_end|>", "stop")])
    engine.max_model_len = 8192
    endpoint = Recorder({"policy": channel(engine)}).endpoint(POLICY)
    assert endpoint.describe("r_1/ada").context_limit == 8192  # the engine's own
    assert channel(engine, sequence=5400).context_limit == 5400  # or the trainer's, if that is less

    messages = [Message.user("Say hi.")]
    tight = channel(engine, answer=16, sequence=len(qwen35(tokenizer).render(messages, [MINE])) + 15)
    with pytest.raises(ContextOverflow):  # what tells a program to compact and try again
        await Recorder({"policy": tight}).endpoint(POLICY).sample(sample_request(messages, tools=[MINE]))
