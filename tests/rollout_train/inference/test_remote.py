"""Replicas served over HTTP, followed from the ledger, and routed to: what a run's channel should serve is written
down, engine hosts follow it, and a runner samples from whichever replica is close enough, stamping each token with the
version of the weights that sampled it."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

import pytest

from rollout.contracts import Message
from rollout.harness import RecordedModel
from rollout.harness.blobs import FileBlobStore
from rollout_train.checkpoints import Checkpoint, Checkpoints, new_id
from rollout_train.following import Follower
from rollout_train.inference import Connection, RemoteEngine, Route, Routes, Unserved, serve_engines
from rollout_train.inference.remote import ENGINES, NoReplica, Unreachable
from rollout_train.ledger import Fence, FileLedger
from rollout_train.presence import FilePresence
from rollout_train.record import scope
from rollout_train.recorder import Recorder, Renderer
from rollout_train.serving import Serving, qualified, record_serving, wanted
from rollout_train.testing import PlainRenderer, sample_request
from tests.rollout_train.machines import Saying, engine_host, passing_on, saying_channel, served


async def until(condition: Callable[[], bool | Awaitable[bool]], seconds: float = 5.0) -> None:
    async with asyncio.timeout(seconds):
        while True:
            met = condition()
            if met if isinstance(met, bool) else await met:
                return
            await asyncio.sleep(0.02)


async def made(checkpoints: Checkpoints, fence: Fence, tmp_path: Path, parent: Checkpoint | None = None) -> Checkpoint:
    """A checkpoint of run `r`, its weights one small file."""
    id = new_id()
    weights = tmp_path / "trained" / id
    weights.mkdir(parents=True)
    (weights / "adapter.bin").write_text(id)
    parents = [parent.id] if parent else []
    return await checkpoints.add(fence, id, weights=weights, run="r", base="m", parents=parents)


async def serve(checkpoints: Checkpoints, fence: Fence, checkpoint: Checkpoint) -> None:
    """What the training loop writes when it serves a checkpoint."""
    said = Serving("policy", checkpoint.id, checkpoint.depth, checkpoint.kind, checkpoint.weights, model="m")
    await record_serving(checkpoints.ledger, "r", said, fence)


def shared(tmp_path: Path) -> tuple[Checkpoints, FilePresence]:
    """The ledger, blob store and heartbeats every machine of a test reaches."""
    ledger = FileLedger(tmp_path / "ledger")
    return Checkpoints(ledger, FileBlobStore(tmp_path / "blobs")), FilePresence(ledger.directory)


async def said(recorder: Recorder, session: str, turns: int = 1) -> list[tuple[str, int]]:
    """A session's turns on run `r`'s channel (those sampled before are not sampled again): what each turn's replica
    said it sampled from, and the version its tokens are stamped with."""
    endpoint = recorder.endpoint(RecordedModel(channel=qualified("r", "policy")))
    messages = [Message.user("Say.")]
    for _ in range(turns):
        effect = f"{session}:{len(messages)}"
        result = await endpoint.sample(sample_request(messages, effect_id=effect, session_id=session))
        messages += [result.message, Message.user("Again.")]
    (segment,) = recorder.export(session)
    text = "".join(map(chr, segment.tokens))
    return [(text[span.start : span.end].strip(), span.version) for span in segment.spans]


async def test_what_a_channel_should_serve_is_its_record_of_the_greatest_depth(tmp_path: Path) -> None:
    checkpoints, _ = shared(tmp_path)
    fence = await checkpoints.ledger.take(scope("r"))
    assert await wanted(checkpoints.ledger, "r", "policy") is None
    await record_serving(checkpoints.ledger, "r", Serving("policy", model="m", sequence=900), fence)
    assert (base := await wanted(checkpoints.ledger, "r", "policy")) is not None and base.checkpoint is None
    first = await made(checkpoints, fence, tmp_path)
    second = await made(checkpoints, fence, tmp_path, first)
    await serve(checkpoints, fence, second)
    await serve(checkpoints, fence, first)  # (written late: a channel does not go back)
    assert not await record_serving(checkpoints.ledger, "r", Serving("policy", second.id, 2), fence)  # (once)
    now = await wanted(checkpoints.ledger, "r", "policy")
    assert now is not None and (now.checkpoint, now.depth, now.files) == (second.id, 2, second.weights)
    assert await wanted(checkpoints.ledger, "r", "judge") is None


async def test_a_replica_served_over_http_samples_and_refuses_weights_it_does_not_serve() -> None:
    channel = saying_channel(replicas=2)
    await channel.publish("adapter-1", "/files/a1", 1)
    async with served(serve_engines({"policy": channel}, "gpu-1")) as (base, _):
        engine = RemoteEngine("m", address=base, replica="gpu-1.policy.1")
        state = await engine.state()
        assert (state["replica"], state["adapter"], state["version"]) == ("gpu-1.policy.1", "adapter-1", 1)
        assert state["loaded"] == {"adapter-1": 1} and engine.max_model_len == channel.engines[1].max_model_len
        options: dict[str, Any] = {"max_tokens": 20, "temperature": 1.0, "top_p": 1.0, "stop_token_ids": [10]}
        reply = await engine.generate([65], adapter="adapter-1", version=1, request="e:1/1/1", **options)
        assert "".join(map(chr, reply.tokens)) == "adapter-1\n" and reply.logprobs == [-0.5] * 10
        assert (reply.replica, reply.version) == ("gpu-1.policy.1", 1)  # (who sampled it, and at what version)
        second = cast(Saying, channel.engines[1])
        assert second.prompts == [[65]] and cast(Saying, channel.engines[0]).prompts == []  # (that replica only)
        again = await engine.generate([65], adapter="adapter-1", version=1, request="e:1/1/1", **options)
        assert again == reply and second.prompts == [[65]]  # (a request sent again is answered once)
        with pytest.raises(Unserved, match="version 2"):  # (what the caller knew is out of date)
            await engine.generate([65], adapter="adapter-1", version=2, **options)
        with pytest.raises(Unreachable):  # (a replica the server does not have)
            await RemoteEngine(address=base, replica="gpu-2.policy.0").state()
        await engine.load_adapter("other", "/files/other")
        await engine.remove_adapter("other")
        assert second.told[-2:] == ["load other", "remove other"]
        engine.close()


async def test_a_follower_serves_what_the_run_says_and_beats_with_each_replicas_address(tmp_path: Path) -> None:
    checkpoints, presence = shared(tmp_path)
    fence = await checkpoints.ledger.take(scope("r"))
    channel = saying_channel(replicas=2)
    follower = Follower("gpu-1", checkpoints, "r", {"policy": channel}, tmp_path / "gpu-1", address="http://gpu-1:8820",
                        presence=presence)  # fmt: skip
    assert not await follower.follow()  # (the run says nothing yet: the model's own weights)
    first = await made(checkpoints, fence, tmp_path)
    await serve(checkpoints, fence, first)
    assert await follower.follow() and not await follower.follow()  # (once)
    assert (channel.serving, channel.version) == (first.id, 1)
    assert (tmp_path / "gpu-1" / first.id / "weights" / "adapter.bin").read_text() == first.id  # (from the blobs)
    second = await made(checkpoints, fence, tmp_path, first)
    third = await made(checkpoints, fence, tmp_path, second)
    await serve(checkpoints, fence, third)  # (a checkpoint it never served is skipped: the newest is what it wants)
    assert await follower.follow() and (channel.serving, channel.version) == (third.id, 3)
    assert sorted(each.name for each in (tmp_path / "gpu-1").iterdir()) == sorted([first.id, third.id])
    await follower.beat()
    (beat,) = await presence.beats()
    assert beat.about["kind"] == ENGINES and beat.about["follows"] == "r"
    (entry,) = cast(list[dict[str, object]], beat.about["channels"])
    assert (entry["channel"], entry["adapter"], entry["version"]) == ("policy", third.id, 3)
    assert entry["replicas"] == [
        {"replica": f"gpu-1.policy.{index}", "address": "http://gpu-1:8820", "adapter": third.id, "serving": third.id,
         "version": 3}
        for index in (0, 1)
    ]  # fmt: skip


async def test_sessions_go_to_replicas_close_enough_stay_there_and_move_when_one_dies(tmp_path: Path) -> None:
    checkpoints, presence = shared(tmp_path)
    fence = await checkpoints.ledger.take(scope("r"))
    first = await made(checkpoints, fence, tmp_path)
    await serve(checkpoints, fence, first)
    prompt, lagging = saying_channel(), saying_channel()
    async with (
        engine_host("gpu-1", checkpoints, "r", {"policy": prompt}, tmp_path / "gpu-1", presence) as (_, one),
        engine_host("gpu-2", checkpoints, "r", {"policy": lagging}, tmp_path / "gpu-2", presence, following=False) as (
            stuck,
            _,
        ),
    ):
        await until(lambda: prompt.serving == first.id)
        route = Route(cast(Renderer, PlainRenderer()))
        routes = Routes({"policy": route}, checkpoints.ledger, presence, every=0.05, patience=0.5)
        recorder = Recorder({}, routes=routes)
        assert await routes.reaches("r", "policy") and not await routes.reaches("s", "policy")  # (by run)
        # One checkpoint behind (gpu-2 serves the model's own weights) is close enough to be given new sessions, and
        # every turn is stamped with the version of the weights that sampled it.
        before = {session: await said(recorder, session) for session in [f"r_{number}/policy" for number in range(16)]}
        assert set(map(tuple, before.values())) == {((first.id, 1),), (("base", 0),)}
        # Two behind is not: every turn goes to gpu-1, of new sessions and of those that were on gpu-2.
        second = await made(checkpoints, fence, tmp_path, first)
        await serve(checkpoints, fence, second)
        await until(lambda: prompt.serving == second.id)
        await routes.channel("r", "policy").refresh(now=True)
        after = [await said(recorder, f"r_{number}/policy") for number in range(16, 24)]
        assert after == [[(second.id, 2)]] * 8
        on_gpu_2 = next(session for session, turns in before.items() if turns == [("base", 0)])
        assert await said(recorder, on_gpu_2, turns=2) == [("base", 0), (second.id, 2)]  # (one segment, two versions)
        replicas = {each["replica"]: each for each in routes.channel("r", "policy").replicas()}
        assert (replicas["gpu-1.policy.0"]["behind"], replicas["gpu-1.policy.0"]["taking"]) == (0, True)
        assert (replicas["gpu-2.policy.0"]["behind"], replicas["gpu-2.policy.0"]["taking"]) == (2, False)
        # gpu-2 catches up and gpu-1 dies: a session on gpu-1 goes on on gpu-2, from the start of its turn.
        catching_up = asyncio.create_task(stuck.serve())
        await until(lambda: lagging.serving == second.id)
        on_gpu_1 = next(session for session, turns in before.items() if turns == [(first.id, 1)])
        one.should_exit = True
        await asyncio.sleep(0.5)  # (the server stops within a tenth of a second; its host's beats go on)
        assert await said(recorder, on_gpu_1, turns=2) == [(first.id, 1), (second.id, 2)]
        catching_up.cancel()
    with pytest.raises(NoReplica):  # (no replica answers: a new session waits, then gives up)
        await said(recorder, "r_99/policy")
    routes.close()


async def test_replicas_are_reached_directly_or_through_a_proxy_with_a_token_and_a_cache_cannot_misstamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints, presence = shared(tmp_path)
    fence = await checkpoints.ledger.take(scope("r"))
    first = await made(checkpoints, fence, tmp_path)
    await serve(checkpoints, fence, first)
    monkeypatch.setenv("ROLLOUT_TEST_TOKEN", "s3cret")
    token = Connection(token_env="ROLLOUT_TEST_TOKEN")
    one, two = saying_channel(), saying_channel()
    async with (
        engine_host("gpu-1", checkpoints, "r", {"policy": one}, tmp_path / "gpu-1", presence, token="s3cret") as gpu_1,
        engine_host("gpu-2", checkpoints, "r", {"policy": two}, tmp_path / "gpu-2", presence, token="s3cret") as gpu_2,
    ):
        await until(lambda: one.serving == two.serving == first.id)
        hosts = {"gpu-1": str(gpu_1[0].address), "gpu-2": str(gpu_2[0].address)}
        async with served(passing_on(hosts)) as (proxy, _), served(passing_on(hosts, caching=True)) as (caching, _):
            direct = routed(checkpoints, presence, connection=token)
            through = routed(checkpoints, presence, connection=token, via=proxy)
            for each, sessions in ((direct, range(6)), (through, range(6, 12))):  # (the same replicas, either way)
                assert [await said(each, f"r_{number}/policy") for number in sessions] == [[(first.id, 1)]] * 6
            assert len(cast(Saying, one.engines[0]).prompts) + len(cast(Saying, two.engines[0]).prompts) == 12
            assert not await routes_of(routed(checkpoints, presence)).reaches("r", "policy")  # (no token: refused)
            # A proxy that answers from a cache gives the version of the first answer: refused, and never recorded.
            stale = routed(checkpoints, presence, connection=token, via=caching)
            assert await said(stale, "r_20/policy") == [(first.id, 1)]
            second = await made(checkpoints, fence, tmp_path, first)
            await serve(checkpoints, fence, second)
            await until(lambda: one.serving == two.serving == second.id)
            await routes_of(stale).channel("r", "policy").refresh(now=True)
            with pytest.raises(Unserved, match="sampled at version 1, not 2"):
                await said(stale, "r_20/policy", turns=2)
            assert await said(through, "r_21/policy") == [(second.id, 2)]


def routed(checkpoints: Checkpoints, presence: FilePresence, **route: Any) -> Recorder:
    """A runner's recorder, whose channel `policy` is routed as `route` says (from the hosts' beats)."""
    routes = Routes(
        {"policy": Route(cast(Renderer, PlainRenderer()), **route)}, checkpoints.ledger, presence, every=0.05
    )
    return Recorder({}, routes=routes)


def routes_of(recorder: Recorder) -> Routes:
    assert isinstance(recorder.routes, Routes)
    return recorder.routes


async def test_an_eval_plays_on_its_training_runs_replicas_and_only_while_they_serve_its_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoints, presence = shared(tmp_path)
    fence = await checkpoints.ledger.take(scope("r"))
    first = await made(checkpoints, fence, tmp_path)
    await serve(checkpoints, fence, first)
    second = await made(checkpoints, fence, tmp_path, first)
    evaluated = await checkpoints.ledger.take(scope("e"))  # (the eval of `second`, served by run r's channel)
    said = Serving("policy", second.id, second.depth, served_by=qualified("r", "policy"), max_lag=0)
    await record_serving(checkpoints.ledger, "e", said, evaluated)
    channel = saying_channel()
    async with engine_host("gpu-1", checkpoints, "r", {"policy": channel}, tmp_path / "gpu-1", presence):
        await until(lambda: channel.serving == first.id)
        routes = routes_of(routed(checkpoints, presence))
        assert await routes.reaches("r", "policy")  # (it serves what run r says)
        assert not await routes.reaches("e", "policy")  # (one behind the eval's checkpoint: an eval plays no other)
        await serve(checkpoints, fence, second)
        await until(lambda: channel.serving == second.id)
        await routes.channel("e", "policy").refresh(now=True)
        assert await routes.reaches("e", "policy")
