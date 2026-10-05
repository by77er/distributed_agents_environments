"""The Messages API endpoint: canonical messages and tool specifications map to the API's turns and tools; each model
thinks and samples as its catalog entry says; a reply's text, tool calls, thinking and usage come back as canonical
content; errors map to the contract. Against a fake Messages API on this machine: nothing is sent elsewhere."""

from typing import Any

import pytest

from rollout.contracts import (
    CapabilityContract,
    ContextDelta,
    ContextOverflow,
    FinishReason,
    Message,
    ModelEndpointError,
    NamedToolChoice,
    Overloaded,
    Reasoning,
    ReasoningScope,
    Role,
    SampleRequest,
    Text,
    ToolCall,
    ToolChoiceMode,
    ToolResult,
    ToolResultBlock,
    ToolSpecification,
    context_digests,
)
from rollout.harness import SamplingParameters
from rollout_anthropic import MessagesEndpoint, MessagesOptions, hosted
from tests.hosted_apis import FakeApi, Refusal, Said, anthropic_api

CONTEXT = 200_000


def request(messages: list[Message], **options: Any) -> SampleRequest:
    return SampleRequest(
        effect_id="r_x:0:0",
        arguments_digest="d",
        session_id="r_x/policy",
        context=ContextDelta(append=messages, digest=context_digests(messages)[-1]),
        **options,
    )


CONVERSATION = [
    Message.system("Be terse."),
    Message.user("Weather in Lisbon?"),
    Message(
        role=Role.ASSISTANT,
        content=[
            Reasoning(scope=ReasoningScope.PORTABLE, text="Look it up."),
            Text(text="Looking."),
            ToolCall(call_id="call.1", name="weather", arguments={"city": "Lisbon"}),
        ],
    ),
    Message(
        role=Role.TOOL, content=[ToolResultBlock(call_id="call.1", result=ToolResult(content=[Text(text="18 C")]))]
    ),
    Message.user("And tomorrow?"),
]
WEATHER = ToolSpecification(name="weather", description="Look up the weather.")


def endpoint(options: dict[str, Any] | None = None, url: str = "http://127.0.0.1:9", key: str = "test-key") -> Any:
    return hosted(
        "claude-test", api_key=key, context_limit=CONTEXT, max_output_tokens=64_000, options=options, base_url=url
    )


def test_canonical_messages_and_tools_map_to_turns() -> None:
    body = endpoint().request_body(request(CONVERSATION, tools=[WEATHER], tool_choice=NamedToolChoice(name="weather")))
    assert body["system"] == "Be terse."
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Weather in Lisbon?"}]},
        {"role": "assistant", "content": [  # (reasoning is left out: it is replayed only with its signature)
            {"type": "text", "text": "Looking."},
            {"type": "tool_use", "id": "call_1", "name": "weather", "input": {"city": "Lisbon"}},
        ]},
        {"role": "user", "content": [  # (the tool's result opens the user turn that follows)
            {"type": "tool_result", "tool_use_id": "call_1", "content": [{"type": "text", "text": "18 C"}]},
            {"type": "text", "text": "And tomorrow?"},
        ]},
    ]  # fmt: skip
    assert body["tools"] == [
        {"name": "weather", "description": "Look up the weather.", "input_schema": {"type": "object", "properties": {}}}
    ]
    assert body["tool_choice"] == {"type": "tool", "name": "weather"}
    assert body["max_tokens"] == 64_000 and "thinking" not in body
    unforced = endpoint({"forced_tool_choice": False})
    for choice in (NamedToolChoice(name="weather"), ToolChoiceMode.REQUIRED):
        assert unforced.request_body(request(CONVERSATION, tools=[WEATHER], tool_choice=choice))["tool_choice"] == {
            "type": "auto"
        }


def test_each_model_thinks_and_samples_as_its_catalog_entry_says() -> None:
    asked = request(CONVERSATION[:2])
    sampling = SamplingParameters(temperature=0.5, top_p=0.9, reasoning_effort="high", thinking_tokens=2048,
                                  answer_tokens=512)  # fmt: skip
    adaptive = endpoint({"thinking": "adaptive", "sampling": False, "effort": "low"})
    body = adaptive.request_body(asked, sampling=sampling)
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}  # (the binding's, over the model's own)
    assert body["max_tokens"] == 2560 and "temperature" not in body and "top_p" not in body
    assert adaptive.request_body(asked)["output_config"] == {"effort": "low"}
    budget = endpoint({"thinking": "budget"})
    body = budget.request_body(asked, sampling=sampling)
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "temperature" not in body  # (the API takes no temperature beside a thinking budget)
    small = budget.request_body(asked, sampling=SamplingParameters(temperature=0.5, thinking_tokens=512))
    assert "thinking" not in small and small["temperature"] == 0.5  # (too few to think with)
    plain = endpoint().request_body(asked, sampling=SamplingParameters(top_p=0.9))
    assert plain["top_p"] == 0.9 and "thinking" not in plain
    assert endpoint().request_body(request(CONVERSATION[:2], max_output_tokens=300))["max_tokens"] == 300
    with pytest.raises(ValueError, match="thinking is one of"):
        MessagesOptions.of({"thinking": "always"})


async def test_text_tool_calls_thinking_and_usage_come_back_as_canonical_content() -> None:
    reply = Said(text="Sunny.", calls=(("toolu_1", "weather", {"city": "Porto"}),), thinking="Check Porto.",
                 input_tokens=1200, cached_input_tokens=1000, output_tokens=80, thinking_tokens=50)  # fmt: skip
    with anthropic_api(FakeApi(replies=[reply])) as api:
        result = await endpoint({"thinking": "adaptive"}, api.url).sample(request(CONVERSATION, tools=[WEATHER]))
    assert list(result.message.content) == [
        Reasoning(scope=ReasoningScope.PORTABLE, text="Check Porto."),
        Text(text="Sunny."),
        ToolCall(call_id="toolu_1", name="weather", arguments={"city": "Porto"}),
    ]
    assert result.finish_reason is FinishReason.TOOL_USE
    usage = result.usage
    assert (usage.input_tokens, usage.cached_input_tokens, usage.output_tokens, usage.thinking_tokens) == (
        1200, 1000, 80, 50,
    )  # fmt: skip
    assert usage.context_used == 1280 and usage.context_limit == CONTEXT
    assert api.headers[0]["x-api-key"] == "test-key" and api.requests[0]["model"] == "claude-test"


async def test_a_reply_cut_at_its_most_output_finishes_with_length() -> None:
    with anthropic_api(FakeApi(replies=[Said(text="Part", cut=True)])) as api:
        result = await endpoint(url=api.url).sample(request(CONVERSATION[:2]))
    assert result.finish_reason is FinishReason.LENGTH and result.message.text == "Part"


async def test_the_sampling_a_model_takes_reaches_the_api() -> None:
    with anthropic_api() as api:
        await endpoint(url=api.url).sample(request(CONVERSATION[:2]), sampling=SamplingParameters(temperature=0.4))
        await endpoint({"sampling": False}, api.url).sample(
            request(CONVERSATION[:2]), sampling=SamplingParameters(temperature=0.4)
        )
    assert api.requests[0]["temperature"] == 0.4 and "temperature" not in api.requests[1]


@pytest.mark.parametrize(
    ("refusal", "error"),
    [
        (Refusal(429, '{"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}}',
                 {"retry-after": "3"}), Overloaded),
        (Refusal(529, '{"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}'), Overloaded),
        (Refusal(401, '{"type": "error", "error": {"type": "authentication_error", "message": "bad key"}}'),
         PermissionError),
        (Refusal(400, '{"type": "error", "error": {"type": "invalid_request_error", "message": '
                 '"prompt is too long: 300000 tokens > 200000 maximum"}}'), ContextOverflow),
        (Refusal(400, '{"type": "error", "error": {"type": "invalid_request_error", "message": "bad field"}}'),
         ModelEndpointError),
    ],
)  # fmt: skip
async def test_errors_map_to_the_contract(refusal: Refusal, error: type[Exception]) -> None:
    with anthropic_api(FakeApi(replies=[refusal])) as api, pytest.raises(error) as raised:
        await endpoint(url=api.url).sample(request(CONVERSATION[:2]))
    if isinstance(raised.value, Overloaded) and refusal.status == 429:
        assert raised.value.retry_after == 3.0
    if error is ModelEndpointError:
        assert type(raised.value) is ModelEndpointError and "bad field" in str(raised.value)
    assert len(api.requests) == 1  # (the SDK retries nothing: the holder decides)


def test_an_endpoint_without_a_key_is_refused() -> None:
    with pytest.raises(PermissionError, match="api_key_env"):
        endpoint(key="")
    made = endpoint()
    assert isinstance(made, MessagesEndpoint)
    assert made.describe("r_x/policy") == CapabilityContract(context_limit=CONTEXT, max_output_tokens=64_000)
