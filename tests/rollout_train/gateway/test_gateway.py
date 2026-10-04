"""The gateway in this process: what it records is what the in-process recorder records, a retried request is not
sampled again, keys are verified with nothing but the secret, and a turn samples the newest checkpoint an endpoint
serves, one checkpoint for the whole turn."""

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from rollout.contracts import ContextDelta, Message, ModelEndpointError, SampleRequest, context_digests
from rollout_train.checkpoints import Checkpoint, Checkpoints, new_id
from rollout_train.gateway import (
    Gateway,
    Grant,
    Keyring,
    create_app,
)
from rollout_train.inference import Channel, Generation, Limits, Route, Routes, Unserved
from rollout_train.ledger import Fence
from rollout_train.presence import FilePresence
from rollout_train.record import scope
from rollout_train.recorder import Renderer
from rollout_train.recorder.renderers import ThinkingFormat
from rollout_train.serving import Serving, record_serving
from rollout_train.testing import Characters, PlainRenderer, ScriptedEngine
from tests.rollout_train.gateway.support import (
    EchoEngine,
    bearer,
    client,
    converse,
    echo_channel,
    gateway_over,
    grant,
    keyring,
    stores,
)
from tests.rollout_train.machines import MODEL, engine_host

pytest.importorskip("openai")
pytest.importorskip("anthropic")
import anthropic
import openai


async def test_the_gateway_records_a_conversation_as_its_segments(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    gateway = gateway_over(echo_channel(), ledger, blobs)
    granted = await grant(ledger)
    async with client(create_app(gateway)) as http:
        replies = await converse(http, keyring().mint(granted))
    segments = (await gateway.store.sessions("train", "r_1"))["policy"]
    trained = [list(dict.fromkeys(span.effect_id for span in segment.spans)) for segment in segments]
    assert trained == [["e0", "e1", "e2"], ["e3", "e4", "e-again-later"]]  # (an edit begins one; a retry replaces)
    assert {reply["model"] for reply in replies} == {"anything"}  # (the reply echoes the name it was asked by)


async def test_the_official_clients_are_served_and_recorded(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    gateway = gateway_over(echo_channel(), ledger, blobs)
    key = keyring().mint(await grant(ledger))
    app = create_app(gateway)
    gpt = openai.AsyncOpenAI(base_url="http://gateway/v1", api_key=key, http_client=cast(Any, client(app)))
    chat = await gpt.chat.completions.create(
        model="x", messages=[{"role": "user", "content": "Go."}], extra_headers={"Idempotency-Key": "o1"}
    )
    assert chat.choices[0].message.content or chat.choices[0].message.tool_calls
    raw = await gpt.chat.completions.with_raw_response.create(
        model="x", messages=[{"role": "user", "content": "Go."}], extra_headers={"Idempotency-Key": "o1"}
    )
    assert raw.headers["x-rollout-replayed"] == "true" and raw.parse().choices[0].message == chat.choices[0].message
    response = await gpt.responses.create(model="x", input="Go on.", extra_headers={"Idempotency-Key": "o2"})
    assert response.status == "completed" and response.output
    library: Any = httpx
    try:
        import httpx2

        library = httpx2
    except ImportError:
        pass
    http = library.AsyncClient(transport=library.ASGITransport(app=app))
    claude = anthropic.AsyncAnthropic(base_url="http://gateway", api_key=key, http_client=http)
    message = await claude.messages.create(
        model="x", max_tokens=200, messages=[{"role": "user", "content": "Go, Claude."}]
    )
    assert message.content and message.stop_reason in ("end_turn", "tool_use")
    async with claude.messages.stream(model="x", max_tokens=200, messages=[{"role": "user", "content": "Again."}]) as s:
        streamed = await s.get_final_message()
    assert [block.type for block in streamed.content] == [block.type for block in message.content] or streamed.content
    turns = await gateway.store.turns("train", "r_1")
    assert [turn.effect_id for turn in turns][:2] == ["o1", "o2"] and len(turns) == 4  # (o1 recorded once)


async def test_a_retried_request_is_answered_with_the_recorded_reply_and_not_sampled_again(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    channel = echo_channel(delay=0.05)
    engine = cast(EchoEngine, channel.engines[0])
    one, two = gateway_over(channel, ledger, blobs), gateway_over(channel, ledger, blobs)  # (two replicas)
    key = keyring().mint(await grant(ledger))
    body = {"model": "x", "messages": [{"role": "user", "content": "Go."}]}
    async with client(create_app(one)) as first, client(create_app(two)) as second:
        answered = await first.post("/v1/chat/completions", json=body, headers=bearer(key, "same"))
        again = await second.post("/v1/chat/completions", json=body, headers=bearer(key, "same"))
        assert len(engine.prompts) == 1
        assert again.headers["x-rollout-replayed"] == "true" and answered.headers["x-rollout-replayed"] == "false"
        assert again.json()["choices"] == answered.json()["choices"]
        at_once = await asyncio.gather(  # (sent to both at once: both sample, one turn is recorded)
            first.post("/v1/chat/completions", json=body, headers=bearer(key, "racing")),
            second.post("/v1/chat/completions", json=body, headers=bearer(key, "racing")),
        )
    assert at_once[0].json()["choices"] == at_once[1].json()["choices"]
    assert [turn.effect_id for turn in await one.store.turns("train", "r_1")] == ["same", "racing"]


async def test_a_key_that_is_expired_forged_or_of_a_superseded_attempt_is_refused(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    gateway = gateway_over(echo_channel(), ledger, blobs)
    body = {"model": "x", "messages": [{"role": "user", "content": "Go."}]}
    expired = keyring().mint(await grant(ledger, expires=time.time() - 3600))
    good = keyring().mint(await grant(ledger, run_id="r_2"))
    head, payload, signature = good.rsplit(".", 2)
    other = json.dumps({**json.loads(_decoded(payload)), "slot": "rival"}).encode()
    tampered = f"{head}.{_encoded(other)}.{signature}"
    stranger = Keyring.parse([("k2", "someone-else-s-secret-of-32-bytes!!")]).mint(await grant(ledger))
    async with client(create_app(gateway)) as http:
        for key, why in [(expired, "expired"), (tampered, "does not match"), (stranger, "does not match")]:
            answer = await http.post("/v1/chat/completions", json=body, headers=bearer(key))
            assert answer.status_code == 401 and why in answer.json()["error"]["message"]
        answer = await http.post("/v1/messages", json=body | {"max_tokens": 50}, headers={"x-api-key": "sk-nonsense"})
        assert answer.status_code == 401 and answer.json()["error"]["type"] == "authentication_error"
        await ledger.take("runs/train/episodes/1/1")  # a newer attempt of the episode: r_2's key is stale now
        answer = await http.post("/v1/chat/completions", json=body, headers=bearer(good))
        assert answer.status_code == 401 and "taken over" in answer.json()["error"]["message"]
    assert await gateway.store.turns("train", "r_2") == []


def test_keys_rotate_a_new_secret_signs_and_an_old_one_still_verifies() -> None:
    old = Keyring.parse([("k1", "an-older-secret-of-thirty-two-bytes!")])
    rotated = keyring()  # k2 first, k1 kept
    granted = Grant("train", "r_1", "policy", "policy", fence=_fence(), expires=time.time() + 60)
    assert rotated.verify(old.mint(granted)) == granted
    assert rotated.mint(granted).split(".")[1] == "k2"
    with pytest.raises(Exception, match="does not have"):
        old.verify(rotated.mint(granted))
    assert Keyring.from_environment({"ROLLOUT_GATEWAY_KEYS": "k2:a-newer-secret-of-thirty-two-bytes!!"}).signing == "k2"
    with pytest.raises(ValueError):
        Keyring.parse([("short", "too short")])


class Publishing(ScriptedEngine):
    """Thinks until its budget runs out, then answers; the first time it is asked, a new adapter is published on its
    channel while it thinks. `refusing`: the first that many generations say the weights are not served there."""

    def __init__(self, refusing: int = 0, answering_as: str | None = None) -> None:
        super().__init__(cast(Any, Characters()), always=[("<hmm", "length"), ("ok\n", "stop")])
        self.channel: Channel | None = None
        self.refusing = refusing
        self.answering_as = answering_as

    async def generate(self, prompt: Sequence[int], **options: Any) -> Generation:
        if self.refusing:
            self.refusing -= 1
            raise Unserved("not loaded here")
        if self.channel is not None and self.channel.serving == "kkk":
            await self.channel.publish("lll", "/checkpoints/lll", 2)  # (while the turn is between its phases)
        generation = await super().generate(prompt, **options)
        return replace(generation, model=self.answering_as) if self.answering_as else generation


async def thinking_channel(engine: Publishing) -> Channel:
    """A channel whose model opens a thinking block (its turns have two phases), serving adapter `kkk` at depth 1."""

    class Thinking(PlainRenderer):
        thinking = ThinkingFormat(open="<", close=">", prompt_opens=False, forced_close=">")

        def thinking_end_token_ids(self) -> list[int]:
            return [ord(">")]

    channel = Channel("policy", [engine], cast(Renderer, Thinking()), Limits(thinking=4, answer=20))
    await channel.publish("kkk", "/checkpoints/kkk", 1)
    return channel


def asked(effect: str) -> SampleRequest:
    messages = [Message.user("Go.")]
    return SampleRequest(
        effect_id=effect,
        arguments_digest="d",
        session_id="r_1/policy",
        context=ContextDelta(append=messages, digest=context_digests(messages)[-1]),
    )


async def test_a_turn_samples_one_checkpoint_through_both_phases_of_its_thinking(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    engine = Publishing()
    engine.channel = await thinking_channel(engine)
    gateway = gateway_over(engine.channel, ledger, blobs)
    granted = await grant(ledger)
    first = await gateway.sample(granted, asked("r_1:0:0"))
    assert engine.adapters == ["kkk", "kkk"]  # (lll was published between the phases: the turn stayed on kkk)
    second = await gateway.sample(granted, asked("r_1:0:1"))
    assert (first.checkpoint, first.depth, second.checkpoint, second.depth) == ("kkk", 1, "lll", 2)
    turns = await gateway.store.turns("train", "r_1")
    assert [turn.mask.count(False) for turn in turns] == [1, 1]  # the close forced after thinking ran out


async def test_a_turn_whose_weights_stop_being_served_is_sampled_again_and_a_misstamped_answer_is_refused(
    tmp_path: Path,
) -> None:
    ledger, blobs = stores(tmp_path)
    engine = Publishing(refusing=1)
    gateway = gateway_over(await thinking_channel(engine), ledger, blobs)
    granted = await grant(ledger)
    await gateway.sample(granted, asked("r_1:0:0"))
    (turn,) = await gateway.store.turns("train", "r_1")
    assert turn.timings["attempt"] == 2 and turn.checkpoint == "kkk"

    wrong = Publishing(answering_as="someone-else")  # (a server that answers as another checkpoint)
    gateway = gateway_over(await thinking_channel(wrong), ledger, blobs)
    with pytest.raises(ModelEndpointError, match="answered for kkk"):
        await gateway.sample(await grant(ledger, run_id="r_2"), SampleRequest.model_validate(
            asked("r_2:0:0").model_dump() | {"session_id": "r_2/policy"}
        ))  # fmt: skip
    assert await gateway.store.turns("train", "r_2") == []


async def made(checkpoints: Checkpoints, fence: Fence, tmp_path: Path, parent: Checkpoint | None = None) -> Checkpoint:
    """A checkpoint of run `train`, its weights one small file, that the run says its channel serves from now on."""
    id = new_id()
    weights = tmp_path / "trained" / id
    weights.mkdir(parents=True)
    (weights / "adapter.bin").write_text(id)
    made = await checkpoints.add(
        fence, id, weights=weights, run="train", base=MODEL, parents=[parent.id] if parent else []
    )
    said = Serving("policy", made.id, made.depth, made.kind, made.weights, model=MODEL)
    await record_serving(checkpoints.ledger, "train", said, fence)
    return made


async def test_a_routed_channel_samples_the_newest_checkpoint_its_server_has_within_the_lag_allowed(
    tmp_path: Path,
) -> None:
    ledger, blobs = stores(tmp_path)
    checkpoints, presence = Checkpoints(ledger, blobs), FilePresence(ledger.directory)
    fence = await ledger.take(scope("train"))
    first = await made(checkpoints, fence, tmp_path)
    async with engine_host("gpu-1", checkpoints, "train", tmp_path / "gpu-1", presence, following=False) as host:
        await host.follower.follow()  # (the server loads the first)
        second = await made(checkpoints, fence, tmp_path, first)  # (and has not loaded the second yet)
        route = Route(cast(Renderer, PlainRenderer()), MODEL, (host.address,))
        routes = Routes({"policy": route}, ledger, every=0.05, patience=0.5)
        gateway = gateway_over(None, ledger, blobs, routes)
        key = keyring().mint(await grant(ledger))
        body = {"model": "x", "messages": [{"role": "user", "content": "Go."}]}
        async with client(create_app(gateway)) as http:
            behind = await http.post("/v1/chat/completions", json=body, headers=bearer(key, "a"))
            assert (behind.headers["x-rollout-checkpoint"], behind.headers["x-rollout-depth"]) == (first.id, "1")
            await host.follower.follow()
            await routes.channel("train", "policy").refresh(now=True)
            newest = await http.post("/v1/chat/completions", json=body, headers=bearer(key, "b"))
            assert (newest.headers["x-rollout-checkpoint"], newest.headers["x-rollout-depth"]) == (second.id, "2")
            third = await made(checkpoints, fence, tmp_path, second)
            fourth = await made(checkpoints, fence, tmp_path, third)  # (two ahead of what the server has)
            await routes.channel("train", "policy").refresh(now=True)
            stale = await http.post("/v1/chat/completions", json=body, headers=bearer(key, "c"))
            assert stale.status_code == 503 and "within 1" in stale.json()["error"]["message"]
        assert fourth.depth == 4
        turns = await gateway.store.turns("train", "r_1")
        assert [(turn.effect_id, turn.checkpoint, turn.depth) for turn in turns] == [
            ("a", first.id, 1), ("b", second.id, 2)
        ]  # fmt: skip


async def test_links_a_request_declares_are_recorded_and_can_decide_what_is_trained(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    gateway = gateway_over(echo_channel(), ledger, blobs)
    key = keyring().mint(await grant(ledger))
    messages = [{"role": "user", "content": "Go."}]
    links = json.dumps([{"type": "compaction_attempt", "source": "a"}])
    async with client(create_app(gateway)) as http:
        first = await http.post("/v1/chat/completions", json={"messages": messages}, headers=bearer(key, "a"))
        said = {name: value for name, value in first.json()["choices"][0]["message"].items() if value is not None}
        summarized = [*messages, said, {"role": "tool", "tool_call_id": "-", "content": "Summarize."}]
        headers = bearer(key, "b", **{"X-Rollout-Links": links})
        assert (await http.post("/v1/chat/completions", json={"messages": summarized}, headers=headers)).is_success
        bad = await http.post(
            "/v1/chat/completions", json={"messages": messages}, headers=bearer(key, "c", **{"X-Rollout-Links": "{}"})
        )
        assert bad.status_code == 400
    turns = await gateway.store.turns("train", "r_1")
    assert [list(turn.links) for turn in turns] == [[], [_link("compaction_attempt", "a")]]
    every = (await gateway.store.sessions("train", "r_1"))["policy"]
    accepted = (await gateway.store.sessions("train", "r_1", accepted_only=True))["policy"]
    assert [span.effect_id for span in every[0].spans] == ["a", "b"]
    assert [span.effect_id for span in accepted[0].spans] == ["a"]


async def test_health_and_readiness(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    gateway = gateway_over(echo_channel(), ledger, blobs)
    async with client(create_app(gateway)) as http:
        assert (await http.get("/healthz")).json() == {"status": "ok"}
        assert (await http.get("/readyz")).json() == {"status": "ready"}
        assert [model["id"] for model in (await http.get("/v1/models")).json()["data"]] == ["policy"]
    broken = Gateway(gateway.store, gateway.keyring, gateway.channels, None, gateway.models)
    broken.store.blobs = _Unreachable()  # pyright: ignore[reportAttributeAccessIssue]
    async with client(create_app(broken)) as http:
        answer = await http.get("/readyz")
    assert answer.status_code == 503 and "blobs" in answer.json()


class _Unreachable:
    async def put(self, data: bytes, media_type: str) -> Any:
        raise ConnectionError("no route to the store")


def _link(kind: str, source: str) -> Any:
    from rollout_train.gateway import Link

    return Link(kind, source)


def _fence() -> Any:
    from rollout_train.ledger import Fence

    return Fence("runs/train/episodes/1/1", 1)


def _encoded(data: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _decoded(text: str) -> bytes:
    import base64

    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
