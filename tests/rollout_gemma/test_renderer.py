"""Gemma 4's token format: what a prompt holds, what ends a turn, and what a sampled turn says."""

from pathlib import Path
from typing import Any, cast

import pytest

from rollout.contracts import (
    FinishReason,
    Message,
    Reasoning,
    ReasoningScope,
    Role,
    Text,
    ToolCall,
    ToolResult,
    ToolResultBlock,
    ToolSpecification,
)
from rollout.harness import RecordedModel
from rollout.harness.blobs import FileBlobStore
from rollout_gemma import arguments, gemma4
from rollout_train.inference import Channel, Limits
from rollout_train.ledger import FileLedger
from rollout_train.recorder.renderers import Tokenizer
from rollout_train.testing import ScriptedEngine, admitted, gateway_endpoints, sample_request

MODEL = "google/gemma-4-12B-it"
MINE = ToolSpecification(
    name="mine",
    description="Mine a block within reach.",
    input_schema={
        "type": "object",
        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "z": {"type": "integer"}},
        "required": ["x", "y", "z"],
    },
)
CHAT = ToolSpecification(
    name="chat",
    description="Say something to your teammates.",
    input_schema={"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
)


@pytest.fixture
def tokenizer() -> Tokenizer:
    transformers = pytest.importorskip("transformers")
    try:
        loaded: Any = transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)  # pyright: ignore[reportUnknownMemberType]
    except OSError:
        pytest.skip(f"the {MODEL} tokenizer is not in the local cache")
    return cast(Tokenizer, loaded)


def test_a_prompt_declares_the_tools_and_opens_the_thought_channel(tokenizer: Tokenizer) -> None:
    renderer = gemma4(tokenizer)
    thought = Reasoning(scope=ReasoningScope.PORTABLE, text="Ore to the west.")
    called = Message(
        role=Role.ASSISTANT,
        content=[thought, ToolCall(call_id="c1", name="mine", arguments={"x": 3, "y": -46, "z": 2})],
    )
    answered = Message(
        role=Role.TOOL,
        content=[ToolResultBlock(call_id="c1", result=ToolResult(content=[Text(text="mined: 1 diamond")]))],
    )
    messages = [
        Message.system("You play Minecraft."),
        Message.user("You see ore."),
        called,
        answered,
        Message.user("Now?"),
    ]
    text = renderer.decode(renderer.render(messages, [MINE, CHAT]))
    assert text.startswith("<bos><|turn>system\n<|think|>\nYou play Minecraft.<|tool>declaration:mine{")
    assert "<|tool_call>call:mine{x:3,y:-46,z:2}<tool_call|>" in text
    assert '<|tool_response>response:mine{value:<|"|>mined: 1 diamond<|"|>}<tool_response|>' in text
    assert "Ore to the west." not in text  # (an earlier turn's thinking is not shown again)
    assert text.endswith("<|turn>user\nNow?<turn|>\n<|turn>model\n<|channel>thought\n")
    stops = renderer.stop_token_ids()
    assert [tokenizer.decode([stop]) for stop in stops] == ["<turn|>", "<|tool_response>", "<eos>"]
    assert renderer.thinking_end_token_ids() == [tokenizer.convert_tokens_to_ids("<channel|>")]


def test_a_sampled_turn_is_its_thinking_its_words_and_its_typed_calls(tokenizer: Tokenizer) -> None:
    renderer = gemma4(tokenizer)
    sampled = (
        "The ore is two blocks west.\n<channel|>Mining it."
        '<|tool_call>call:mine{x:<|"|>-3<|"|>,y:-46,z:2}<tool_call|>'
        '<|tool_call>call:chat{message:<|"|>Got it, {all} of it<|"|>}<tool_call|><|tool_response>'
    )
    message = renderer.parse(renderer.encode(sampled), [MINE, CHAT])
    (reasoning,) = [block for block in message.content if isinstance(block, Reasoning)]
    assert reasoning.text == "The ore is two blocks west."
    assert message.text == "Mining it."
    assert [(call.name, dict(call.arguments)) for call in message.tool_calls] == [
        ("mine", {"x": -3, "y": -46, "z": 2}),  # (a quoted number its schema says is an integer)
        ("chat", {"message": "Got it, {all} of it"}),
    ]


def test_arguments_read_strings_numbers_flags_objects_and_lists() -> None:
    said = 'name:<|"|>a, b: c<|"|>,n:3,f:1.5,ok:true,no:null,nested:{k:<|"|>v<|"|>,m:[1,<|"|>x<|"|>]},items:[{a:1},[2]]'
    assert arguments(said) == {
        "name": "a, b: c", "n": 3, "f": 1.5, "ok": True, "no": None,
        "nested": {"k": "v", "m": [1, "x"]}, "items": [{"a": 1}, [2]],
    }  # fmt: skip
    assert arguments("") == {}


async def test_a_thinking_budget_closes_the_channel_and_the_turn_is_recorded(
    tokenizer: Tokenizer, tmp_path: Path
) -> None:
    engine = ScriptedEngine(tokenizer, [("I could go on thinking", "length"), ("Done.<turn|>", "stop")])
    channel = Channel("policy", [engine], gemma4(tokenizer), Limits(thinking=8, answer=32))
    recorder = gateway_endpoints(
        channel, ledger=FileLedger(tmp_path / "ledger"), blobs=FileBlobStore(tmp_path / "blobs")
    )
    await admitted(recorder, "r_1")
    result = await recorder.endpoint(RecordedModel(channel="policy")).sample(sample_request([Message.user("Hi")]))
    assert result.message.text == "Done." and result.finish_reason is FinishReason.STOP
    (segment,) = (await recorder.sessions("train", "r_1"))["ada"]
    assert tokenizer.decode(segment.tokens).endswith("I could go on thinking\n<channel|>Done.<turn|>")
