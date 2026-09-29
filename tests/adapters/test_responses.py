import base64
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from rollout.adapters.responses import ApiKey, CodexLogin, ResponsesEndpoint
from rollout.core.contracts import (
    ContextDelta,
    ContextOverflow,
    FinishReason,
    Media,
    Message,
    NamedToolChoice,
    Overloaded,
    Role,
    SampleRequest,
    Text,
    ToolCall,
    ToolResult,
    ToolResultBlock,
    ToolSpecification,
    context_digests,
)
from rollout.core.harness import FileBlobStore, SamplingParameters


def events(*items: dict[str, Any], status: str = "completed", usage: dict[str, int] | None = None) -> str:
    lines = [f"data: {json.dumps({'type': 'response.output_item.done', 'item': item})}" for item in items]
    response = {"status": status, "output": list(items), "usage": usage or {"input_tokens": 12, "output_tokens": 3}}
    lines.append(f"data: {json.dumps({'type': f'response.{status}', 'response': response})}")
    return "\n\n".join(lines) + "\n\n"


def endpoint(
    handler: Callable[[httpx.Request], httpx.Response], credentials: Any = None, blobs: FileBlobStore | None = None
) -> ResponsesEndpoint:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ResponsesEndpoint(
        credentials or ApiKey("test-key"),
        "gpt-test",
        sampling=SamplingParameters(reasoning_effort="low"),
        client=client,
        blobs=blobs,
    )


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
    Message(role=Role.ASSISTANT, content=[ToolCall(call_id="c1", name="weather", arguments={"city": "Lisbon"})]),
    Message(role=Role.TOOL, content=[ToolResultBlock(call_id="c1", result=ToolResult(content=[Text(text="18 C")]))]),
]
WEATHER = ToolSpecification(name="weather", description="Look up the weather.")


def test_canonical_messages_render_to_responses_items() -> None:
    body = endpoint(lambda _: httpx.Response(500)).request_body(
        request(CONVERSATION, tools=[WEATHER], tool_choice=NamedToolChoice(name="weather"))
    )
    assert body["instructions"] == "Be terse."
    assert body["input"] == [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Weather in Lisbon?"}]},
        {"type": "function_call", "call_id": "c1", "name": "weather", "arguments": '{"city": "Lisbon"}'},
        {"type": "function_call_output", "call_id": "c1", "output": "18 C"},
    ]
    assert body["tools"] == [
        {
            "type": "function",
            "name": "weather",
            "description": "Look up the weather.",
            "parameters": {"type": "object", "properties": {}},
            "strict": False,
        }
    ]
    assert body["tool_choice"] == {"type": "function", "name": "weather"}
    assert body["reasoning"] == {"effort": "low"}
    assert body["store"] is False and body["stream"] is True


async def test_images_are_sent_as_input_images_from_the_blob_store(tmp_path: Path) -> None:
    blobs = FileBlobStore(tmp_path)
    picture = Media(media_type="image/png", source=await blobs.put(b"png bytes", "image/png"))
    audio = Media(media_type="audio/wav", source=await blobs.put(b"wav bytes", "audio/wav"))
    messages = [
        Message(role=Role.USER, content=[Text(text="What is this?"), picture]),
        Message(role=Role.ASSISTANT, content=[ToolCall(call_id="c1", name="read_image", arguments={"path": "a.png"})]),
        Message(
            role=Role.TOOL,
            content=[ToolResultBlock(call_id="c1", result=ToolResult(content=[Text(text="a.png"), picture, audio]))],
        ),
    ]
    sent: list[dict[str, Any]] = []

    def handler(incoming: httpx.Request) -> httpx.Response:
        sent.append(json.loads(incoming.content))
        return httpx.Response(200, text=events({"type": "message", "content": [{"type": "output_text", "text": "A"}]}))

    with pytest.raises(ValueError, match="no blob store"):
        await endpoint(handler).sample(request(messages))
    await endpoint(handler, blobs=blobs).sample(request(messages))
    image = {"type": "input_image", "image_url": f"data:image/png;base64,{base64.b64encode(b'png bytes').decode()}"}
    user, _, output = sent[0]["input"]
    assert user["content"] == [{"type": "input_text", "text": "What is this?"}, image]
    assert output["output"] == [
        {"type": "input_text", "text": "a.png"},
        image,
        {"type": "input_text", "text": "[audio/wav content omitted: this model reads only images]"},
    ]


async def test_text_and_tool_calls_come_back_as_canonical_content() -> None:
    stream = events(
        {"type": "reasoning", "summary": []},
        {"type": "message", "content": [{"type": "output_text", "text": "Checking."}]},
        {"type": "function_call", "call_id": "c2", "name": "weather", "arguments": '{"city": "Porto"}'},
    )
    seen: list[httpx.Request] = []

    def handler(incoming: httpx.Request) -> httpx.Response:
        seen.append(incoming)
        return httpx.Response(200, text=stream)

    result = await endpoint(handler).sample(request(CONVERSATION[:2]))
    assert result.message.text == "Checking."
    assert result.message.tool_calls == [ToolCall(call_id="c2", name="weather", arguments={"city": "Porto"})]
    assert result.finish_reason is FinishReason.TOOL_USE
    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.context_used) == (12, 3, 15)
    assert seen[0].headers["authorization"] == "Bearer test-key"


async def test_an_incomplete_response_finishes_with_length() -> None:
    stream = events({"type": "message", "content": [{"type": "output_text", "text": "Partial"}]}, status="incomplete")
    result = await endpoint(lambda _: httpx.Response(200, text=stream)).sample(request(CONVERSATION[:2]))
    assert result.finish_reason is FinishReason.LENGTH


@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (429, "slow down", Overloaded),
        (400, '{"error": {"code": "context_length_exceeded"}}', ContextOverflow),
    ],
)
async def test_errors_map_to_the_contract(status: int, body: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        await endpoint(lambda _: httpx.Response(status, text=body, headers={"retry-after": "2"})).sample(
            request(CONVERSATION[:2])
        )


def fake_token(expires_in: float) -> str:
    def part(value: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    claims = {"exp": time.time() + expires_in, "iss": "https://auth.test", "client_id": "app_test"}
    return f"{part({'alg': 'none'})}.{part(claims)}.signature"


def codex_file(tmp_path: Path, expires_in: float) -> Path:
    path = tmp_path / "auth.json"
    tokens = {
        "id_token": "id",
        "access_token": fake_token(expires_in),
        "refresh_token": "refresh-1",
        "account_id": "acct",
    }
    path.write_text(json.dumps({"auth_mode": "chatgpt", "OPENAI_API_KEY": None, "tokens": tokens, "last_refresh": "x"}))
    return path


async def test_codex_login_sends_account_headers(tmp_path: Path) -> None:
    login = CodexLogin(path=codex_file(tmp_path, expires_in=3600))
    seen: list[httpx.Request] = []

    def handler(incoming: httpx.Request) -> httpx.Response:
        seen.append(incoming)
        return httpx.Response(200, text=events({"type": "message", "content": [{"type": "output_text", "text": "ok"}]}))

    await endpoint(handler, login).sample(request(CONVERSATION[:2]))
    assert seen[0].url == httpx.URL("https://chatgpt.com/backend-api/codex/responses")
    assert seen[0].headers["chatgpt-account-id"] == "acct"


async def test_an_expired_codex_login_is_refreshed_and_written_back(tmp_path: Path) -> None:
    path = codex_file(tmp_path, expires_in=-60)
    new_access = fake_token(3600)

    def handler(incoming: httpx.Request) -> httpx.Response:
        if incoming.url.path == "/oauth/token":
            sent = json.loads(incoming.content)
            assert (sent["client_id"], sent["refresh_token"]) == ("app_test", "refresh-1")
            return httpx.Response(
                200, json={"access_token": new_access, "refresh_token": "refresh-2", "id_token": "id2"}
            )
        assert incoming.headers["authorization"] == f"Bearer {new_access}"
        return httpx.Response(200, text=events({"type": "message", "content": [{"type": "output_text", "text": "ok"}]}))

    await endpoint(handler, CodexLogin(path=path)).sample(request(CONVERSATION[:2]))
    stored = json.loads(path.read_text())
    assert stored["tokens"]["refresh_token"] == "refresh-2"
    assert stored["tokens"]["account_id"] == "acct"
    assert stored["auth_mode"] == "chatgpt"
    assert oct(path.stat().st_mode & 0o777) == "0o600"


@pytest.mark.skipif(os.environ.get("ROLLOUT_LIVE") != "1", reason="set ROLLOUT_LIVE=1 to call the real model")
async def test_live_codex_round_trip() -> None:
    live = ResponsesEndpoint(CodexLogin(), os.environ.get("ROLLOUT_MODEL", "gpt-6-astra"))
    result = await live.sample(request([Message.system("Reply with one word."), Message.user("Say: ready")]))
    assert "ready" in result.message.text.lower()
