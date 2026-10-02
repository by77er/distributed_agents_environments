"""The recorder over HTTP: a harness with its own loop talks Chat Completions, and its samples are recorded."""

import json
from typing import Any

import httpx
import pytest

from rollout.core.harness import RecordedModel
from rollout.recorder import Recorder
from tests.support import plain_channel

pytest.importorskip("starlette")
from rollout.recorder.compat import create_app

GUESS = {
    "type": "function",
    "function": {
        "name": "guess",
        "description": "Guess the number.",
        "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
    },
}


def served(**limits: Any) -> tuple[Recorder, httpx.AsyncClient]:
    script = [('call guess {"n": 5}\n', "stop"), ("It was five.\n", "stop")]
    recorder = Recorder({"policy": plain_channel(always=script, **limits)}, base_url="http://recorder/v1")
    address = recorder.endpoint(RecordedModel(channel="policy")).address("r_1/policy")
    assert address.base_url == "http://recorder/v1" and address.model == "policy"
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(recorder)),
        base_url=address.base_url,
        headers={"Authorization": f"Bearer {address.api_key}"},
    )
    return recorder, client


async def test_a_harness_plays_a_whole_exchange_over_chat_completions_and_it_is_one_recorded_sequence() -> None:
    recorder, client = served()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Guess the number."},
        {"role": "user", "content": [{"type": "text", "text": "Go."}]},
    ]
    first = (await client.post("/chat/completions", json={"model": "x", "messages": messages, "tools": [GUESS]})).json()
    choice = first["choices"][0]
    assert choice["finish_reason"] == "tool_calls" and choice["message"]["content"] is None
    (call,) = choice["message"]["tool_calls"]
    assert call["function"] == {"name": "guess", "arguments": '{"n": 5}'}
    assert first["usage"]["completion_tokens"] == len('call guess {"n": 5}\n')

    messages += [choice["message"], {"role": "tool", "tool_call_id": call["id"], "content": "Right."}]
    second = (
        await client.post("/chat/completions", json={"model": "x", "messages": messages, "tools": [GUESS]})
    ).json()
    assert second["choices"][0]["message"] == {"role": "assistant", "content": "It was five."}
    assert second["choices"][0]["finish_reason"] == "stop"

    (epoch,) = recorder.export("r_1/policy")  # the harness only ever appended: one sequence, two sampled spans
    text = "".join(chr(token) for token in epoch.tokens)
    assert [text[span.start : span.end] for span in epoch.spans] == ['call guess {"n": 5}\n', "It was five.\n"]
    assert "tool: Right.\n" in text and text.startswith("tools: guess\nsystem: Guess the number.\nuser: Go.\n")
    assert (await client.get("/models")).json()["data"] == [{"id": "policy", "object": "model"}]


async def test_a_stream_carries_the_same_reply() -> None:
    _, client = served()
    body = {"model": "x", "stream": True, "messages": [{"role": "user", "content": "Go."}], "tools": [GUESS]}
    response = await client.post("/chat/completions", json=body)
    assert response.headers["content-type"].startswith("text/event-stream")
    lines = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    said, ended, used = (json.loads(line) for line in lines[:-1])
    (call,) = said["choices"][0]["delta"]["tool_calls"]
    assert call["index"] == 0 and call["function"]["name"] == "guess" and said["object"] == "chat.completion.chunk"
    assert ended["choices"][0] == {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
    assert used["usage"]["prompt_tokens"] > 0


async def test_a_key_names_one_session_and_a_full_context_is_refused_the_way_harnesses_expect() -> None:
    recorder, client = served(answer=8, sequence=40)
    stranger = await client.post(
        "/chat/completions", json={"messages": []}, headers={"Authorization": "Bearer not-a-key"}
    )
    assert stranger.status_code == 401 and stranger.json()["error"]["code"] == "invalid_api_key"
    long = {"model": "x", "messages": [{"role": "user", "content": "word " * 40}]}
    refused = await client.post("/chat/completions", json=long)
    assert refused.status_code == 400 and refused.json()["error"]["code"] == "context_length_exceeded"
    assert (await client.post("/chat/completions", json={"model": "x"})).status_code == 400  # no messages at all
    recorder.forget("r_1")  # the run is over: its key no longer names anything
    gone = await client.post("/chat/completions", json={"messages": [{"role": "user", "content": "Go."}]})
    assert gone.status_code == 401
