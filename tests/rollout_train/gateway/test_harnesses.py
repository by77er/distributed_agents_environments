"""Claude Code's and Codex's own requests, as they sent them to a gateway (`harnesses/`, recorded from Claude Code
2.1.288 and codex-cli 0.157.1 and scrubbed): each is answered with a stream its harness's API reads, means the
conversation it is, and counts its tokens with the channel's renderer, so the formats keep working without the
binaries."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import TypeAdapter

from rollout.contracts import Message, Reasoning, ReasoningScope, Role, Text, ToolSpecification
from rollout_train.gateway import create_app
from rollout_train.inference import Channel, Limits
from rollout_train.recorder import Renderer
from rollout_train.recorder.compat import messages, responses
from rollout_train.recorder.renderers import Tokenizer
from rollout_train.testing import Characters, PlainRenderer, ScriptedEngine
from tests.rollout_train.gateway.support import client, gateway_over, grant, keyring, stores

pytest.importorskip("anthropic")
pytest.importorskip("openai")
from anthropic.types.beta import BetaRawMessageStreamEvent
from openai.types.responses import ResponseStreamEvent

RECORDED = Path(__file__).parent / "harnesses"


def recorded(name: str) -> dict[str, Any]:
    """A recorded request: `{"harness", "path", "headers", "body"}`."""
    return json.loads((RECORDED / f"{name}.json").read_text())


class ThinkingRenderer(PlainRenderer):
    """The plain format, with reasoning written before a `~`: `assistant: thought~answer`."""

    def parse(self, completion: Sequence[int], tools: Sequence[ToolSpecification]) -> Message:
        message = super().parse(completion, tools)
        if "~" not in message.text:
            return message
        thought, answer = message.text.split("~", 1)
        reasoning = Reasoning(scope=ReasoningScope.PORTABLE, text=thought)
        return Message(role=Role.ASSISTANT, content=[reasoning, Text(text=answer)])

    @staticmethod
    def _said(message: Message) -> str:
        thought = "".join(block.text for block in message.content if isinstance(block, Reasoning))
        return (f"{thought}~" if thought else "") + PlainRenderer._said(message)


async def served(tmp_path: Path, script: Sequence[tuple[str, str]]) -> tuple[Any, str, httpx.AsyncClient]:
    """A gateway whose one channel answers with `script` (reasoning before a `~`), a key for it, and a client."""
    engine = ScriptedEngine(cast(Tokenizer, Characters()), always=script)
    channel = Channel("policy", [engine], cast(Renderer, ThinkingRenderer()), Limits(thinking=64, answer=64))
    ledger, blobs = stores(tmp_path)
    gateway = gateway_over(channel, ledger, blobs)
    return gateway, keyring().mint(await grant(ledger)), client(create_app(gateway))


def events(stream: str) -> list[dict[str, Any]]:
    """The data of each server-sent event."""
    return [json.loads(line.removeprefix("data: ")) for line in stream.splitlines() if line.startswith("data: {")]


async def replayed(http: httpx.AsyncClient, request: dict[str, Any], key: str) -> httpx.Response:
    """A recorded request sent as its harness sent it, under `key`: Claude Code's key as a bearer token
    (`ANTHROPIC_AUTH_TOKEN`), Codex's too (its provider's `env_key`)."""
    headers = request["headers"] | {"authorization": f"Bearer {key}"}
    return await http.post(request["path"], json=request["body"], headers=headers)


@pytest.mark.parametrize("name", ["messages-first", "messages-tool-results", "messages-compact"])
async def test_claude_codes_requests_are_answered_with_a_stream_it_reads(tmp_path: Path, name: str) -> None:
    gateway, key, http = await served(tmp_path, [("I knew it~It was five.\n", "stop")])
    async with http:
        answer = await replayed(http, recorded(f"claude-code-{name}"), key)
    assert answer.status_code == 200, answer.text
    adapter: TypeAdapter[Any] = TypeAdapter(BetaRawMessageStreamEvent)
    parsed = [adapter.validate_python(event) for event in events(answer.text)]
    kinds = [event.type for event in parsed]
    assert kinds[0] == "message_start" and kinds[-2:] == ["message_delta", "message_stop"]
    deltas = [event.delta for event in parsed if event.type == "content_block_delta"]
    assert [delta.type for delta in deltas] == ["thinking_delta", "signature_delta", "text_delta"]
    assert deltas[0].thinking == "I knew it" and deltas[1].signature and deltas[2].text == "It was five."
    assert len(await gateway.store.turns("train", "r_1")) == 1


def test_what_claude_code_sends_reads_as_its_conversation() -> None:
    first = messages.read(recorded("claude-code-messages-first")["body"])
    assert [message.role for message in first.messages] == [Role.SYSTEM, Role.USER, Role.SYSTEM]
    system, _, environment = (message.text for message in first.messages)
    assert system.startswith("You are a Claude agent") and messages.BILLING not in system  # (attribution, not prompt)
    assert environment.startswith("# Environment")  # a system message mid-conversation
    assert [tool.name for tool in first.tools] == ["Edit", "Glob", "Grep", "Read", "Write"]
    later = messages.read(recorded("claude-code-messages-tool-results")["body"])
    roles = [message.role for message in later.messages]
    assert roles == [Role.SYSTEM, Role.USER, Role.SYSTEM, Role.ASSISTANT, Role.TOOL, Role.TOOL, Role.SYSTEM]
    said = later.messages[3]
    assert isinstance(said.content[0], Reasoning) and [call.name for call in said.tool_calls] == ["Read", "Write"]


def test_thinking_signatures_are_taken_whatever_they_are_and_never_change_what_is_rendered() -> None:
    body = recorded("claude-code-messages-tool-results")["body"]
    renderer = ThinkingRenderer()

    def rendered(given: dict[str, Any]) -> list[int]:
        prompt = messages.read(given)
        return renderer.render(prompt.messages, prompt.tools)

    other = json.loads(json.dumps(body))
    (said,) = [entry for entry in other["messages"] if entry["role"] == "assistant"]
    thinking = said["content"][0]
    text = renderer.decode(rendered(body))
    assert thinking["thinking"][:80] in text and thinking["signature"] not in text  # the thought, not its signature
    thinking["signature"] = "EqQBCkgIARABGAIiQL2l+opaque+signature+from+elsewhere=="
    said["content"].insert(1, {"type": "redacted_thinking", "data": "EmwKAhgBEgy3va3pzix/LafPsn4aDFIT2Xlxh0L5L8rLVyIx"})
    assert rendered(other) == rendered(body)


@pytest.mark.parametrize("name", ["responses-first", "responses-function-outputs"])
async def test_codexs_requests_are_answered_with_a_stream_it_reads(tmp_path: Path, name: str) -> None:
    gateway, key, http = await served(tmp_path, [("I knew it~It was five.\n", "stop")])
    async with http:
        answer = await replayed(http, recorded(f"codex-{name}"), key)
    assert answer.status_code == 200, answer.text
    adapter: TypeAdapter[Any] = TypeAdapter(ResponseStreamEvent)
    parsed = [adapter.validate_python(event) for event in events(answer.text)]
    assert parsed[0].type == "response.created" and parsed[-1].type == "response.completed"
    final = parsed[-1].response
    assert [item.type for item in final.output] == ["reasoning", "message"] and final.output_text == "It was five."
    assert len(await gateway.store.turns("train", "r_1")) == 1


def test_what_codex_sends_reads_as_its_conversation() -> None:
    first = responses.read(recorded("codex-responses-first")["body"])
    roles = [message.role for message in first.messages]
    assert roles == [Role.SYSTEM, Role.SYSTEM, Role.USER, Role.USER]  # instructions, then a developer message
    assert [tool.name for tool in first.tools] == ["exec_command", "write_stdin", "request_user_input", "view_image"]
    later = responses.read(recorded("codex-responses-function-outputs")["body"])
    roles = [message.role for message in later.messages]
    assert roles == [Role.SYSTEM, Role.SYSTEM, Role.USER, Role.USER, *[Role.ASSISTANT, Role.TOOL, Role.TOOL] * 2]
    turn = later.messages[4]  # its reasoning and its two calls, one assistant message
    assert isinstance(turn.content[0], Reasoning) and len(turn.tool_calls) == 2


async def test_claude_codes_token_counts_are_the_renderers_and_record_nothing(tmp_path: Path) -> None:
    gateway, key, http = await served(tmp_path, [("I knew it~It was five.\n", "stop")])
    request = recorded("claude-code-count-tokens")
    async with http:
        answer = await replayed(http, request, key)
        refused = await http.post(request["path"], json=request["body"], headers={"x-api-key": "not a key"})
    prompt = messages.prompt(request["body"])
    assert answer.json() == {"input_tokens": len(ThinkingRenderer().render(prompt.messages, prompt.tools))}
    assert refused.status_code == 401 and refused.json()["error"]["type"] == "authentication_error"
    assert await gateway.store.turns("train", "r_1") == []
