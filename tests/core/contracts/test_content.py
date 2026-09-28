import pytest
from pydantic import TypeAdapter, ValidationError

from rollout.core.contracts import (
    BlobReference,
    Block,
    Media,
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

BLOB = BlobReference(uri="s3://bucket/key", sha256="00" * 32, size=3, media_type="image/png")


def test_messages_round_trip_through_json() -> None:
    messages = [
        Message.system("Be terse."),
        Message.user("Add 2 and 3."),
        Message(
            role=Role.ASSISTANT,
            content=[
                Reasoning(scope=ReasoningScope.PORTABLE, text="Use the tool."),
                ToolCall(call_id="c1", name="add", arguments={"a": 2, "b": 3, "note": None}),
            ],
        ),
        Message(role=Role.TOOL, content=[ToolResultBlock(call_id="c1", result=ToolResult(content=[Text(text="5")]))]),
        Message(role=Role.USER, content=[Media(media_type="image/png", source=BLOB)], name="alice", meta={"k": "v"}),
    ]
    for message in messages:
        assert Message.model_validate_json(message.model_dump_json()) == message


def test_blocks_parse_by_type_discriminator() -> None:
    adapter: TypeAdapter[Block] = TypeAdapter(Block)
    block = adapter.validate_python({"type": "tool_call", "call_id": "c1", "name": "add", "arguments": {}})
    assert isinstance(block, ToolCall)
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "unknown"})


def test_content_is_immutable() -> None:
    message = Message(role=Role.USER, content=[Text(text="hi")])
    assert isinstance(message.content, tuple)
    with pytest.raises(ValidationError):
        message.role = Role.ASSISTANT  # pyright: ignore[reportAttributeAccessIssue]


def test_unknown_fields_are_preserved() -> None:
    raw = {"role": "user", "content": [{"type": "text", "text": "hi", "future": 1}], "future_field": "x"}
    message = Message.model_validate(raw)
    assert message.model_dump(mode="json", exclude_none=True)["future_field"] == "x"
    assert message.model_dump(mode="json")["content"][0]["future"] == 1


@pytest.mark.parametrize(
    ("role", "block"),
    [
        (Role.USER, ToolResultBlock(call_id="c1", result=ToolResult())),
        (Role.TOOL, Text(text="not a result")),
        (Role.USER, ToolCall(call_id="c1", name="add", arguments={})),
        (Role.TOOL, ToolCall(call_id="c1", name="add", arguments={})),
    ],
)
def test_blocks_must_match_role(role: Role, block: Block) -> None:
    with pytest.raises(ValidationError):
        Message(role=role, content=[block])


def test_text_and_tool_calls_accessors() -> None:
    call = ToolCall(call_id="c1", name="add", arguments={})
    message = Message(role=Role.ASSISTANT, content=[Text(text="a"), call, Text(text="b")])
    assert message.text == "ab"
    assert message.tool_calls == [call]


def test_reasoning_fields_match_scope() -> None:
    Reasoning(scope=ReasoningScope.POLICY, producer="qwen3@1", opaque=BLOB)
    with pytest.raises(ValidationError):
        Reasoning(scope=ReasoningScope.PORTABLE)
    with pytest.raises(ValidationError):
        Reasoning(scope=ReasoningScope.POLICY, text="visible")


def test_tool_specification_validation() -> None:
    with pytest.raises(ValidationError):
        ToolSpecification(name="has space")
    with pytest.raises(ValidationError):
        ToolSpecification(name="t", input_schema={"type": "string"})
    specification = ToolSpecification(name="t", description="d", timeout_ms=5)
    assert specification.model_visible() == {
        "name": "t",
        "description": "d",
        "input_schema": {"type": "object", "properties": {}},
    }
