"""Scoring through the gateway: a client holding a run's key asks a channel for the logprobs of tokens it hands over
(`POST /v1/scores`). The scores are recorded as a turn of their own use before they are returned, answered again from
the record under the same request id, counted as tokens in, and never trained on."""

from pathlib import Path
from typing import Any, cast

from rollout_train.checkpoints import Checkpoints, new_id
from rollout_train.gateway import create_app
from rollout_train.gateway.turns import SCORE
from rollout_train.inference import Limits, Route, Routes
from rollout_train.presence import FilePresence
from rollout_train.record import scope
from rollout_train.recorder import Renderer
from rollout_train.serving import Serving, record_serving
from rollout_train.testing import PlainRenderer
from tests.rollout_train.gateway.support import (
    EchoEngine,
    bearer,
    client,
    converse,
    echo_channel,
    echoed,
    gateway_over,
    grant,
    keyring,
    stores,
)
from tests.rollout_train.machines import MODEL, engine_host


async def test_a_key_holder_scores_a_sampled_segment_and_the_scores_are_recorded_and_not_trained(
    tmp_path: Path,
) -> None:
    ledger, blobs = stores(tmp_path)
    channel = echo_channel()
    gateway = gateway_over(channel, ledger, blobs)
    student = await grant(ledger)
    teacher = await grant(ledger, slot="teacher", fence=student.fence)
    async with client(create_app(gateway)) as http:
        await converse(http, keyring().mint(student), turns=2)
        before = (await gateway.store.sessions("train", "r_1"))["policy"]
        segment = before[0]
        start = segment.spans[0].start
        body = {"effect_id": "s1", "session_id": "r_1/teacher", "tokens": segment.tokens, "start": start, "top": 2}
        answer = await http.post("/v1/scores", json=body, headers=bearer(keyring().mint(teacher)))
        assert answer.status_code == 200, answer.text
        said: dict[str, Any] = answer.json()
        assert said["start"] == start and said["logprobs"] == [echoed(each) for each in segment.tokens[start:]]
        assert said["top_tokens"] == [[each, each + 1] for each in segment.tokens[start:]]
        assert said["top_logprobs"] == [[echoed(each), echoed(each + 1)] for each in segment.tokens[start:]]
        headers = answer.headers
        assert (headers["x-rollout-checkpoint"], headers["x-rollout-depth"]) == ("base", "0")
        assert (headers["x-rollout-request-id"], headers["x-rollout-replayed"]) == ("s1", "false")

        again = await http.post("/v1/scores", json=body, headers=bearer(keyring().mint(teacher)))
        assert again.json() == said and again.headers["x-rollout-replayed"] == "true"  # (from the record)
    engine = cast(EchoEngine, channel.engines[0])
    assert engine.scored == [segment.tokens]  # (scored once)

    assert await gateway.store.sessions("train", "r_1") == {"policy": before}  # (the scoring turn is not trained)
    turns = await gateway.store.turns("train", "r_1")
    (scored,) = [turn for turn in turns if turn.use == SCORE]
    assert (scored.slot, list(scored.prompt), scored.completion, scored.mask) == ("teacher", segment.tokens, [], [])
    assert scored.scores is not None and scored.scores.logprobs == said["logprobs"]
    assert scored.scores.top_tokens == said["top_tokens"] and scored.scores.top_logprobs == said["top_logprobs"]
    usage = scored.result.usage
    assert (usage.input_tokens, usage.output_tokens) == (len(segment.tokens), 0)  # (counted as tokens in)
    entry = (await gateway.store.index("train", "r_1"))["s1"]
    assert isinstance(entry, dict) and (entry["use"], entry["prompt"], entry["sampled"]) == (
        SCORE, len(segment.tokens), 0
    )  # fmt: skip
    assert channel.take()["prompt_tokens"] > len(segment.tokens)  # (the samples' prompts, and the tokens scored)


async def test_a_score_request_the_gateway_cannot_take_is_refused(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    gateway = gateway_over(echo_channel(), ledger, blobs)
    limited = gateway_over(echo_channel(sequence=10), ledger, blobs)  # (turns of 10 tokens at most)
    minted = keyring().mint(await grant(ledger, slot="teacher"))
    key = bearer(minted)

    def asked(**changes: Any) -> dict[str, Any]:
        return {"effect_id": "s1", "session_id": "r_1/teacher", "tokens": [65, 66, 67], "start": 1} | changes

    async with client(create_app(gateway)) as http:
        other = await http.post("/v1/scores", json=asked(session_id="r_2/teacher"), headers=key)
        assert other.status_code == 401 and "this key is for session r_1/teacher" in other.json()["error"]["message"]
        first = await http.post("/v1/scores", json=asked(start=0), headers=key)
        assert first.status_code == 400  # (the first token has nothing before it to be scored by)
        beyond = await http.post("/v1/scores", json=asked(end=4), headers=key)
        assert beyond.status_code == 400 and "not a range" in beyond.json()["error"]["message"]
        many = await http.post("/v1/scores", json=asked(top=6), headers=key)
        assert many.status_code == 400 and "top is 0 to 5" in many.json()["error"]["message"]
        unknown = await http.post("/v1/scores", json=asked(extra=1), headers=key)
        assert unknown.status_code == 400
        sampled = {"model": "x", "messages": [{"role": "user", "content": "Go."}]}
        assert (await http.post("/v1/chat/completions", json=sampled, headers=bearer(minted, "c"))).status_code == 200
        reused = await http.post("/v1/scores", json=asked(effect_id="c"), headers=key)
        assert reused.status_code == 400 and "used by another request" in reused.json()["error"]["message"]
    async with client(create_app(limited)) as http:
        long = await http.post("/v1/scores", json=asked(tokens=list(range(65, 75))), headers=key)
        error = long.json()["error"]  # (vLLM generates a token after the tokens scored: room for 9)
        assert long.status_code == 400 and (error["type"], error["context_limit"]) == ("ContextOverflow", 9)
    assert [turn.effect_id for turn in await gateway.store.turns("train", "r_1")] == ["c"]  # (nothing else recorded)


async def test_a_routed_channel_scores_with_the_checkpoint_its_run_serves(tmp_path: Path) -> None:
    ledger, blobs = stores(tmp_path)
    checkpoints, presence = Checkpoints(ledger, blobs), FilePresence(ledger.directory)
    fence = await ledger.take(scope("train"))
    id = new_id()
    weights = tmp_path / "trained" / id
    weights.mkdir(parents=True)
    (weights / "adapter.bin").write_text(id)
    made = await checkpoints.add(fence, id, weights=weights, run="train", base=MODEL)
    await record_serving(ledger, "train", Serving("policy", made.id, made.depth, made.kind, made.weights, model=MODEL),
                         fence)  # fmt: skip
    async with engine_host("gpu-1", checkpoints, "train", tmp_path / "gpu-1", presence, following=False) as host:
        await host.follower.follow()
        route = Route(cast(Renderer, PlainRenderer()), MODEL, (host.address,), Limits())
        routes = Routes({"policy": route}, ledger, every=0.05, patience=0.5)
        gateway = gateway_over(None, ledger, blobs, routes)
        key = keyring().mint(await grant(ledger, slot="teacher"))
        body = {"effect_id": "s1", "session_id": "r_1/teacher", "tokens": [65, 66, 67], "start": 1, "top": 1}
        async with client(create_app(gateway)) as http:
            answer = await http.post("/v1/scores", json=body, headers=bearer(key))
        assert answer.status_code == 200, answer.text
        assert (answer.headers["x-rollout-checkpoint"], answer.headers["x-rollout-depth"]) == (made.id, "1")
        assert answer.json()["top_tokens"] == [[66], [67]] and host.engine.adapters[-1] == made.id
    (turn,) = await gateway.store.turns("train", "r_1")
    assert (turn.use, turn.checkpoint, turn.depth, turn.channel) == (SCORE, made.id, 1, "policy")
