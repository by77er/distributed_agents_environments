"""The gateway over HTTP, as a harness with its own loop sees it: it talks Chat Completions, Responses or Messages
through the official clients, and its samples are recorded."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from rollout.harness import RecordedModel
from rollout.harness.blobs import FileBlobStore
from rollout_train.gateway import GatewayEndpoints, create_app
from rollout_train.inference import Channel, Limits
from rollout_train.ledger import FileLedger
from rollout_train.recorder import Renderer
from rollout_train.recorder.renderers import Tokenizer
from rollout_train.testing import Characters, ScriptedEngine, admitted, gateway_endpoints

pytest.importorskip("starlette")
pytest.importorskip("openai")
pytest.importorskip("anthropic")
import anthropic
import openai
from anthropic.types import Message as AnthropicMessage
from openai.types.chat import ChatCompletion
from openai.types.responses import Response as OpenAIResponse
from starlette.applications import Starlette

from tests.rollout_train.support import ThinkingRenderer

GUESS: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "guess",
        "description": "Guess the number.",
        "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
    },
}
SCRIPT = [('call guess {"n": 5}\n', "stop"), ("I knew it~It was five.\n", "stop")]
"""A tool call, then reasoning (before the `~`) and an answer."""


async def served(
    directory: Path, script: Sequence[tuple[str, str]] = SCRIPT, **limits: Any
) -> tuple[GatewayEndpoints, str, httpx.AsyncClient, Starlette]:
    """A gateway serving one session (recording in `directory`), the key that names it, an HTTP client that reaches
    the app in process, and the app."""
    engine = ScriptedEngine(cast(Tokenizer, Characters()), always=script)
    channel = Channel("policy", [engine], cast(Renderer, ThinkingRenderer()), Limits(**limits))
    ledger, blobs = FileLedger(directory / "ledger"), FileBlobStore(directory / "blobs")
    endpoints = gateway_endpoints(channel, ledger=ledger, blobs=blobs, url="http://recorder")
    await admitted(endpoints, "r_1")
    address = endpoints.endpoint(RecordedModel(channel="policy")).address("r_1/policy")
    assert address.base_url == "http://recorder/v1" and address.model == "policy"
    assert endpoints.gateway is not None
    app = create_app(endpoints.gateway)
    return endpoints, address.api_key, httpx.AsyncClient(transport=httpx.ASGITransport(app=app)), app


def gpt(http: httpx.AsyncClient, key: str, **options: Any) -> openai.AsyncOpenAI:
    """OpenAI's client, reaching the app in process."""
    return openai.AsyncOpenAI(base_url="http://recorder/v1", api_key=key, http_client=cast(Any, http), **options)


def claude(app: Starlette, key: str, **options: Any) -> anthropic.AsyncAnthropic:
    """Anthropic's client, reaching `app` in process (newer versions of it are built on `httpx2`)."""
    library: Any = httpx
    try:
        import httpx2

        library = httpx2
    except ImportError:
        pass
    http = library.AsyncClient(transport=library.ASGITransport(app=app))
    return anthropic.AsyncAnthropic(base_url="http://recorder", api_key=key, http_client=http, **options)


async def sampled(recorder: GatewayEndpoints) -> list[str]:
    """What the session's one segment holds that the policy sampled."""
    (segment,) = await exported(recorder)  # the harness only ever appended: one segment
    text = "".join(chr(token) for token in segment.tokens)
    return [text[span.start : span.end] for span in segment.spans]


async def exported(recorder: GatewayEndpoints) -> list[Any]:
    return (await recorder.sessions("train", "r_1"))["policy"]


# Chat Completions ---------------------------------------------------------------------------------------------------


async def test_a_harness_plays_a_whole_exchange_over_chat_completions_and_it_is_one_recorded_sequence(
    tmp_path: Path,
) -> None:
    recorder, key, http, _app = await served(tmp_path)
    client = gpt(http, key)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Guess the number."},
        {"role": "user", "content": [{"type": "text", "text": "Go."}]},
    ]
    first = await client.chat.completions.create(model="x", messages=cast(Any, messages), tools=cast(Any, [GUESS]))
    choice = first.choices[0]
    assert choice.finish_reason == "tool_calls" and choice.message.content is None
    (call,) = choice.message.tool_calls or []
    assert call.type == "function" and (call.function.name, call.function.arguments) == ("guess", '{"n": 5}')
    assert first.usage is not None and first.usage.completion_tokens == len('call guess {"n": 5}\n')

    messages += [
        choice.message.model_dump(exclude_none=True),
        {"role": "tool", "tool_call_id": call.id, "content": "Right."},
    ]
    raw = await http.post(
        "http://recorder/v1/chat/completions",
        json={"model": "x", "messages": messages, "tools": [GUESS]},
        headers={"Authorization": f"Bearer {key}"},
    )
    second = ChatCompletion.model_validate(raw.json())  # strictly what the API's own types say
    said = second.choices[0].message
    assert (said.content, said.model_extra) == ("It was five.", {"reasoning_content": "I knew it"})
    assert second.choices[0].finish_reason == "stop"

    assert await sampled(recorder) == ['call guess {"n": 5}\n', "I knew it~It was five.\n"]
    (segment,) = await exported(recorder)
    text = "".join(chr(token) for token in segment.tokens)
    assert "tool: Right.\n" in text and text.startswith("tools: guess\nsystem: Guess the number.\nuser: Go.\n")


async def test_a_chat_completions_stream_carries_the_same_reply(tmp_path: Path) -> None:
    _, key, http, _app = await served(tmp_path)
    client = gpt(http, key)
    chunks = [
        chunk
        async for chunk in await client.chat.completions.create(
            model="x", stream=True, messages=[{"role": "user", "content": "Go."}], tools=cast(Any, [GUESS])
        )
    ]
    said, ended, used = chunks
    (call,) = said.choices[0].delta.tool_calls or []
    assert call.index == 0 and call.function is not None and call.function.name == "guess"
    assert ended.choices[0].finish_reason == "tool_calls" and not used.choices
    assert used.usage is not None and used.usage.prompt_tokens > 0


# Responses ----------------------------------------------------------------------------------------------------------


async def test_a_harness_plays_a_whole_exchange_over_responses_and_it_is_one_recorded_sequence(tmp_path: Path) -> None:
    recorder, key, http, _app = await served(tmp_path)
    client = gpt(http, key)
    tool = {"type": "function", **GUESS["function"]}
    web = {"type": "web_search"}  # not a function: never offered to the model
    items: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "input_text", "text": "Go."}]}]
    first = await client.responses.create(
        model="x", instructions="Guess the number.", input=cast(Any, items), tools=cast(Any, [tool, web]), store=False
    )
    (call,) = first.output
    assert call.type == "function_call" and (call.name, call.arguments) == ("guess", '{"n": 5}')
    assert first.status == "completed" and first.usage is not None and first.usage.output_tokens == 20

    items += [
        call.model_dump(exclude_none=True),
        {"type": "function_call_output", "call_id": call.call_id, "output": "Right."},
    ]
    raw = await http.post(
        "http://recorder/v1/responses",
        json={"model": "x", "instructions": "Guess the number.", "input": items, "tools": [tool]},
        headers={"Authorization": f"Bearer {key}"},
    )
    second = OpenAIResponse.model_validate(raw.json())  # strictly what the API's own types say
    thought, message = second.output
    assert thought.type == "reasoning" and [part.text for part in thought.content or []] == ["I knew it"]
    assert message.type == "message" and second.output_text == "It was five."

    items += [item.model_dump(exclude_none=True) for item in second.output]  # the reasoning goes back as it came
    items.append({"role": "user", "content": "Again."})
    await client.responses.create(
        model="x", instructions="Guess the number.", input=cast(Any, items), tools=cast(Any, [tool])
    )
    assert await sampled(recorder) == ['call guess {"n": 5}\n', "I knew it~It was five.\n", 'call guess {"n": 5}\n']


async def test_a_responses_stream_has_the_events_the_official_client_builds_the_response_from(tmp_path: Path) -> None:
    _, key, http, _app = await served(tmp_path, [("I knew it~It was five.\n", "stop")])
    client = gpt(http, key)
    async with client.responses.stream(model="x", input="Go.") as stream:
        kinds = [event.type async for event in stream]
        final = await stream.get_final_response()
    assert final.output_text == "It was five." and [item.type for item in final.output] == ["reasoning", "message"]
    assert kinds[:2] == ["response.created", "response.in_progress"] and kinds[-1] == "response.completed"
    assert "response.reasoning_text.delta" in kinds and "response.output_text.delta" in kinds


async def test_a_responses_reply_cut_short_is_incomplete_and_stateful_requests_are_refused(tmp_path: Path) -> None:
    _, key, http, _app = await served(tmp_path, [("It was five, or six.\n", "length")])
    client = gpt(http, key)
    cut = await client.responses.create(model="x", input="Go.", max_output_tokens=4)
    assert cut.status == "incomplete" and cut.incomplete_details is not None
    assert cut.incomplete_details.reason == "max_output_tokens" and cut.output_text == "It w"
    with pytest.raises(openai.BadRequestError, match="previous_response_id"):
        await client.responses.create(model="x", input="Go.", previous_response_id=cut.id)


# Messages -----------------------------------------------------------------------------------------------------------


async def test_a_harness_plays_a_whole_exchange_over_messages_and_it_is_one_recorded_sequence(tmp_path: Path) -> None:
    recorder, key, http, app = await served(tmp_path)
    client = claude(app, key)
    tool = {"name": "guess", "description": "Guess the number.", "input_schema": GUESS["function"]["parameters"]}
    search = {"type": "web_search_20250305", "name": "web_search"}  # a server tool: never offered to the model
    messages: list[dict[str, Any]] = [{"role": "user", "content": "Go."}]
    first = await client.messages.create(
        model="x",
        max_tokens=1000,
        system="Guess the number.",
        messages=cast(Any, messages),
        tools=cast(Any, [tool, search]),
    )
    (use,) = first.content
    assert use.type == "tool_use" and (use.name, use.input) == ("guess", {"n": 5})
    assert first.stop_reason == "tool_use" and first.usage.output_tokens == 20

    messages += [
        {"role": "assistant", "content": [block.model_dump() for block in first.content]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use.id, "content": "Right."}]},
    ]
    raw = await http.post(
        "http://recorder/v1/messages",
        json={"model": "x", "max_tokens": 1000, "system": "Guess the number.", "messages": messages, "tools": [tool]},
        headers={"x-api-key": key},
    )
    second = AnthropicMessage.model_validate(raw.json())  # strictly what the API's own types say
    thought, answer = second.content
    assert thought.type == "thinking" and thought.thinking == "I knew it" and thought.signature
    assert answer.type == "text" and answer.text == "It was five." and second.stop_reason == "end_turn"

    messages += [
        {"role": "assistant", "content": [block.model_dump() for block in second.content]},
        {"role": "user", "content": [{"type": "text", "text": "Again."}]},
    ]
    await client.messages.create(
        model="x", max_tokens=1000, system="Guess the number.", messages=cast(Any, messages), tools=cast(Any, [tool])
    )
    assert await sampled(recorder) == ['call guess {"n": 5}\n', "I knew it~It was five.\n", 'call guess {"n": 5}\n']


async def test_a_messages_stream_has_the_events_the_official_client_builds_the_message_from(tmp_path: Path) -> None:
    _, key, _http, app = await served(tmp_path)
    client = claude(app, key)
    tool = {"name": "guess", "description": "Guess the number.", "input_schema": GUESS["function"]["parameters"]}
    async with client.messages.stream(
        model="x", max_tokens=100, messages=[{"role": "user", "content": "Go."}], tools=cast(Any, [tool])
    ) as stream:
        call = await stream.get_final_message()
    assert [(block.type, getattr(block, "input", None)) for block in call.content] == [("tool_use", {"n": 5})]
    async with client.messages.stream(
        model="x", max_tokens=100, messages=[{"role": "user", "content": "Go."}]
    ) as stream:
        kinds = [event.type async for event in stream]
        answer = await stream.get_final_message()
    assert [block.type for block in answer.content] == ["thinking", "text"] and answer.stop_reason == "end_turn"
    assert kinds[0] == "message_start" and kinds[-1] == "message_stop" and "content_block_delta" in kinds


# What every format shares -------------------------------------------------------------------------------------------


async def test_a_key_names_one_session_and_a_full_context_is_refused_the_way_each_api_refuses_one(
    tmp_path: Path,
) -> None:
    recorder, key, http, app = await served(tmp_path, answer=8, sequence=40)
    chat = gpt(http, key)
    anthropic_client = claude(app, key)
    long = "word " * 40
    with pytest.raises(openai.BadRequestError) as refused:
        await chat.chat.completions.create(model="x", messages=[{"role": "user", "content": long}])
    assert refused.value.code == "context_length_exceeded"
    with pytest.raises(openai.BadRequestError) as refused:
        await chat.responses.create(model="x", input=long)
    assert refused.value.code == "context_length_exceeded"
    with pytest.raises(anthropic.BadRequestError, match="prompt is too long"):
        await anthropic_client.messages.create(model="x", max_tokens=8, messages=[{"role": "user", "content": long}])

    stranger = gpt(http, "not-a-key", max_retries=0)
    with pytest.raises(openai.AuthenticationError) as denied:
        await stranger.responses.create(model="x", input="Go.")
    assert denied.value.code == "invalid_api_key"
    unknown = claude(app, "not-a-key", max_retries=0)
    with pytest.raises(anthropic.AuthenticationError):
        await unknown.messages.create(model="x", max_tokens=8, messages=[{"role": "user", "content": "Go."}])

    headers = {"Authorization": f"Bearer {key}"}
    unreadable_bodies: list[tuple[str, dict[str, Any]]] = [
        ("chat/completions", {}),
        ("responses", {"input": []}),
        ("messages", {"messages": [{"role": "user", "content": "Go."}]}),  # (without max_tokens)
    ]
    for path, body in unreadable_bodies:
        assert (await http.post(f"http://recorder/v1/{path}", json=body, headers=headers)).status_code == 400, path
    assert (await http.post("http://recorder/v1/responses", content=b"{", headers=headers)).status_code == 400

    await recorder.store.ledger.take("tests/r_1")  # another attempt took its fence: its key records no more
    with pytest.raises(openai.AuthenticationError):
        await chat.with_options(max_retries=0).chat.completions.create(
            model="x", messages=[{"role": "user", "content": "Go."}]
        )


async def test_a_request_repeated_under_its_idempotency_key_gets_the_recorded_reply_in_every_format(
    tmp_path: Path,
) -> None:
    recorder, key, http, _app = await served(tmp_path, [("one\n", "stop"), ("two\n", "stop"), ("three\n", "stop")])
    requests = [
        ("chat/completions", {"messages": [{"role": "user", "content": "Go."}]}, "one"),
        ("responses", {"input": "Go."}, "two"),
        ("messages", {"max_tokens": 50, "messages": [{"role": "user", "content": "Go."}]}, "three"),
    ]
    for number, (path, body, said) in enumerate(requests):
        headers = {"Authorization": f"Bearer {key}", "Idempotency-Key": f"turn-{number}"}
        first, again = [
            (await http.post(f"http://recorder/v1/{path}", json=body, headers=headers)).json() for _ in "12"
        ]
        assert said in json.dumps(first)
        assert {key: value for key, value in first.items() if key not in ("created", "created_at")} == {
            key: value for key, value in again.items() if key not in ("created", "created_at")
        }
    assert recorder.gateway is not None
    engine = cast(ScriptedEngine, recorder.gateway.channels["policy"].engines[0])
    assert len(engine.prompts) == 3  # each turn sampled once; its repeats were answered from the record


async def test_the_channels_are_listed_as_models_to_both_clients(tmp_path: Path) -> None:
    _, key, http, app = await served(tmp_path)
    listed = await gpt(http, key).models.list()
    assert [model.id for model in listed.data] == ["policy"]
    anthropic_client = claude(app, key)
    assert [model.id async for model in anthropic_client.models.list()] == ["policy"]
