"""Engines on other machines: what a run's channel should serve is written down, engine hosts load it into their vLLM
servers by the checkpoint's id, and a runner asks for the newest checkpoint a server has, close enough to what the run
says, stamping each token with the depth of the checkpoint its answer names. The servers here are fakes of vLLM's
OpenAI-compatible server."""

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from rollout.contracts import Message, ModelEndpointError
from rollout.harness import RecordedModel
from rollout.harness.blobs import FileBlobStore
from rollout.testing import until
from rollout_train.checkpoints import Checkpoint, Checkpoints, new_id
from rollout_train.following import Follower
from rollout_train.gateway import GatewayEndpoints
from rollout_train.inference import Channel, Connection, RemoteEngine, Route, Routes
from rollout_train.inference.remote import ENGINES, NotLoaded
from rollout_train.ledger import Fence, FileLedger
from rollout_train.presence import FilePresence
from rollout_train.record import scope
from rollout_train.recorder import Renderer
from rollout_train.serving import Serving, qualified, record_serving, wanted
from rollout_train.testing import PlainRenderer, admitted, recording, sample_request
from tests.rollout_train.machines import MODEL, Saying, engine_host, fake_vllm, passing_on, served

OPTIONS: dict[str, Any] = {"max_tokens": 20, "temperature": 1.0, "top_p": 1.0, "stop_token_ids": [10]}


async def made(checkpoints: Checkpoints, fence: Fence, tmp_path: Path, parent: Checkpoint | None = None) -> Checkpoint:
    """A checkpoint of run `r`, its weights one small file."""
    id = new_id()
    weights = tmp_path / "trained" / id
    weights.mkdir(parents=True)
    (weights / "adapter.bin").write_text(id)
    parents = [parent.id] if parent else []
    return await checkpoints.add(fence, id, weights=weights, run="r", base=MODEL, parents=parents)


async def serve(checkpoints: Checkpoints, fence: Fence, checkpoint: Checkpoint | None) -> None:
    """What the training loop writes when it serves a checkpoint (or, with none, starts from the base model)."""
    if checkpoint is None:
        said = Serving("policy", model=MODEL)
    else:
        said = Serving("policy", checkpoint.id, checkpoint.depth, checkpoint.kind, checkpoint.weights, model=MODEL)
    await record_serving(checkpoints.ledger, "r", said, fence)


def shared(tmp_path: Path) -> tuple[Checkpoints, FilePresence]:
    """The ledger, blob store and heartbeats every machine of a test reaches."""
    ledger = FileLedger(tmp_path / "ledger")
    return Checkpoints(ledger, FileBlobStore(tmp_path / "blobs")), FilePresence(ledger.directory)


def routed(checkpoints: Checkpoints, *servers: str, **route: Any) -> GatewayEndpoints:
    """A runner's recorder, whose channel `policy` is sampled on `servers` as `route` says."""
    channel = Route(cast(Renderer, PlainRenderer()), MODEL, servers, **route)
    routes = Routes({"policy": channel}, checkpoints.ledger, every=0.05, patience=0.5)
    return recording(ledger=checkpoints.ledger, blobs=checkpoints.blobs, routes=routes)


def routes_of(recorder: GatewayEndpoints) -> Routes:
    assert isinstance(recorder.routes, Routes)
    return recorder.routes


async def said(recorder: GatewayEndpoints, session: str, turns: int = 1, run: str = "r") -> list[tuple[str, int]]:
    """A session's turns on a run's channel (those sampled before are not sampled again): which checkpoint each turn's
    server said it sampled from, and the version its tokens are stamped with."""
    run_id, slot = session.split("/")
    try:
        recorder.attempt(run_id)
    except RuntimeError:
        await admitted(recorder, run_id, run)
    endpoint = recorder.endpoint(RecordedModel(channel=qualified(run, "policy")))
    messages = [Message.user("Say.")]
    for _ in range(turns):
        effect = f"{session}:{len(messages)}"
        result = await endpoint.sample(sample_request(messages, effect_id=effect, session_id=session))
        messages += [result.message, Message.user("Again.")]
    (segment,) = (await recorder.sessions(run, run_id))[slot]
    text = "".join(map(chr, segment.tokens))
    return [(text[span.start : span.end].split()[0], span.version) for span in segment.spans]


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


async def test_a_remote_engine_samples_on_a_vllm_server_by_the_name_of_the_checkpoint() -> None:
    engine = Saying()
    async with served(fake_vllm(engine)) as (address, _):
        remote = RemoteEngine(MODEL, address=address)
        assert set(await remote.models()) == {MODEL} and remote.max_model_len == engine.max_model_len  # (as it says)
        reply = await remote.generate([65], adapter=None, session="r_1/policy", request="e/1/1", **OPTIONS)
        assert "".join(map(chr, reply.tokens)) == "base\n" and reply.logprobs == [-0.5] * 5  # (the stop token too)
        assert (reply.finish_reason, reply.model) == ("stop", MODEL)  # (the model that sampled it, as the server says)
        with pytest.raises(NotLoaded, match="does not exist"):
            await remote.generate([65], adapter="kpqxwlmrtsnvoyzu", **OPTIONS)
        await remote.load_adapter("kpqxwlmrtsnvoyzu", "/checkpoints/kpqxwlmrtsnvoyzu/weights")
        assert set(await remote.models()) == {MODEL, "kpqxwlmrtsnvoyzu"}
        reply = await remote.generate([65], adapter="kpqxwlmrtsnvoyzu", **OPTIONS)
        assert "".join(map(chr, reply.tokens)) == "kpqxwlmrtsnvoyzu\n" and reply.model == "kpqxwlmrtsnvoyzu"
        await remote.remove_adapter("kpqxwlmrtsnvoyzu")
        assert engine.told == ["load kpqxwlmrtsnvoyzu", "remove kpqxwlmrtsnvoyzu"]
        with pytest.raises(NotImplementedError, match="full weights"):
            await remote.load_weights("/checkpoints/full")
        remote.close()


async def test_a_follower_loads_what_the_run_says_into_its_server_and_beats(tmp_path: Path) -> None:
    checkpoints, presence = shared(tmp_path)
    fence = await checkpoints.ledger.take(scope("r"))
    async with engine_host("gpu-1", checkpoints, "r", tmp_path / "gpu-1", presence, following=False) as host:
        follower = host.follower
        assert not await follower.follow()  # (the run says nothing yet: the model's own weights)
        first = await made(checkpoints, fence, tmp_path)
        await serve(checkpoints, fence, first)
        assert await follower.follow() and not await follower.follow()  # (once)
        assert (host.channel.serving, host.channel.version) == (first.id, 1)
        loaded = tmp_path / "gpu-1" / first.id / "weights"
        assert (loaded / "adapter.bin").read_text() == first.id and host.engine.told[-1] == f"load {first.id}"
        second = await made(checkpoints, fence, tmp_path, first)
        third = await made(checkpoints, fence, tmp_path, second)
        await serve(checkpoints, fence, third)  # (a checkpoint it never served is skipped: the newest is what it wants)
        assert await follower.follow() and (host.channel.serving, host.channel.version) == (third.id, 3)
        assert sorted(each.name for each in (tmp_path / "gpu-1").iterdir()) == sorted([first.id, third.id])
        assert set(await RemoteEngine(MODEL, address=host.address).models()) == {MODEL, first.id, third.id}
        await follower.beat()
        (beat,) = await presence.beats()
        assert beat.about["kind"] == ENGINES and beat.about["follows"] == "r"
        (entry,) = cast(list[dict[str, object]], beat.about["channels"])
        assert (entry["channel"], entry["adapter"], entry["version"]) == ("policy", third.id, 3)
        adapters = [
            {"run": "r", "channel": "policy", "checkpoint": first.id, "depth": 1},
            {"run": "r", "channel": "policy", "checkpoint": third.id, "depth": 3},
        ]  # (every adapter the server holds, so that the gateway can send a turn to one that holds its checkpoint)
        assert entry["engines"] == [{"address": host.address, "serving": third.id, "version": 3, "adapters": adapters}]


async def test_a_follower_holds_its_view_to_what_its_server_says_it_has(tmp_path: Path) -> None:
    checkpoints, presence = shared(tmp_path)
    fence = await checkpoints.ledger.take(scope("r"))
    async with engine_host("gpu-1", checkpoints, "r", tmp_path / "gpu-1", presence, following=False) as host:
        first = await made(checkpoints, fence, tmp_path)
        await serve(checkpoints, fence, first)
        assert await host.follower.follow()
        behind = RemoteEngine(MODEL, address=host.address)
        await behind.remove_adapter(first.id)  # the server lost it (it started again): loaded again at the next look
        assert await host.follower.follow() and host.engine.told[-2:] == [f"remove {first.id}", f"load {first.id}"]
        await behind.load_adapter("strayadapter", str(tmp_path))  # one nobody serves (a follower before this one's)
        assert not await host.follower.follow() and host.engine.told[-1] == "remove strayadapter"

        # A follower started again finds the server holding what the run serves: it is kept, and taken as loaded.
        channel = Channel("policy", [behind], cast(Renderer, PlainRenderer()))
        again = Follower("gpu-1", checkpoints, "r", {"policy": channel}, tmp_path / "again", presence=presence)
        assert await again.follow() and channel.serving == first.id
        assert set(await behind.models()) == {MODEL, first.id} and host.engine.told[-1] == "remove strayadapter"
        behind.close()


async def test_turns_sample_the_newest_checkpoint_a_server_has_within_bounds_and_go_on_where_one_dies(
    tmp_path: Path,
) -> None:
    checkpoints, presence = shared(tmp_path)
    fence = await checkpoints.ledger.take(scope("r"))
    await serve(checkpoints, fence, None)
    first = await made(checkpoints, fence, tmp_path)
    await serve(checkpoints, fence, first)
    async with (
        engine_host("gpu-1", checkpoints, "r", tmp_path / "gpu-1", presence) as prompt,
        engine_host("gpu-2", checkpoints, "r", tmp_path / "gpu-2", presence, following=False) as lagging,
    ):
        await until(lambda: prompt.channel.serving == first.id)
        recorder = routed(checkpoints, prompt.address, lagging.address)
        routes = routes_of(recorder)
        assert await routes.reaches("r", "policy")
        # One checkpoint behind (gpu-2 has the model's own weights alone) is close enough, and every turn is stamped
        # with the depth of the checkpoint its server's answer names.
        before = {session: await said(recorder, session) for session in [f"r_{number}/policy" for number in range(16)]}
        assert set(map(tuple, before.values())) == {((first.id, 1),), (("base", 0),)}
        # Two behind is not: every turn goes to gpu-1, of new sessions and of those that were on gpu-2.
        second = await made(checkpoints, fence, tmp_path, first)
        await serve(checkpoints, fence, second)
        await until(lambda: prompt.channel.serving == second.id)
        await routes.channel("r", "policy").refresh(now=True)
        assert [await said(recorder, f"r_{number}/policy") for number in range(16, 24)] == [[(second.id, 2)]] * 8
        on_gpu_2 = next(session for session, turns in before.items() if turns == [("base", 0)])
        assert await said(recorder, on_gpu_2, turns=2) == [("base", 0), (second.id, 2)]  # (one segment, two versions)
        servers = {each["address"]: each for each in routes.channel("r", "policy").servers()}
        assert (servers[prompt.address]["serving"], servers[prompt.address]["behind"]) == (second.id, 0)
        assert servers[lagging.address]["serving"] is None  # (it has nothing close enough)
        # gpu-2 catches up and gpu-1 dies: a session on gpu-1 goes on on gpu-2, from the start of its turn.
        catching_up = asyncio.create_task(lagging.follower.serve())
        await until(lambda: lagging.channel.serving == second.id)
        on_gpu_1 = next(session for session, turns in before.items() if turns == [(first.id, 1)])
        prompt.server.should_exit = True
        await asyncio.sleep(0.5)  # (the server stops within a tenth of a second)
        assert await said(recorder, on_gpu_1, turns=2) == [(first.id, 1), (second.id, 2)]
        catching_up.cancel()
    with pytest.raises(ModelEndpointError, match="no server"):  # (no server answers: a turn waits, then gives up)
        await said(recorder, "r_99/policy")
    routes.close()


async def test_a_server_that_has_not_loaded_the_newest_yet_samples_the_one_before(tmp_path: Path) -> None:
    checkpoints, presence = shared(tmp_path)
    fence = await checkpoints.ledger.take(scope("r"))
    first = await made(checkpoints, fence, tmp_path)
    await serve(checkpoints, fence, first)
    async with engine_host("gpu-1", checkpoints, "r", tmp_path / "gpu-1", presence, following=False) as host:
        await host.follower.follow()
        second = await made(checkpoints, fence, tmp_path, first)
        await serve(checkpoints, fence, second)
        # Behind a router that lists the newest, but sends this turn to a server without it: the server says so, and
        # the turn is sampled again from the one before.
        async with served(passing_on([host.address])) as (router, _):
            recorder = routed(checkpoints, router)
            channel = routes_of(recorder).channel("r", "policy")
            await channel.refresh(now=True)
            channel._has[router].add(second.id)  # pyright: ignore[reportPrivateUsage]  (as a router that lists it)
            assert await said(recorder, "r_1/policy") == [(first.id, 1)]
            await host.follower.follow()
            await channel.refresh(now=True)
            assert await said(recorder, "r_2/policy") == [(second.id, 2)]


async def test_servers_are_reached_directly_or_through_a_proxy_with_a_token_and_a_cache_cannot_misstamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints, presence = shared(tmp_path)
    fence = await checkpoints.ledger.take(scope("r"))
    first = await made(checkpoints, fence, tmp_path)
    await serve(checkpoints, fence, first)
    monkeypatch.setenv("ROLLOUT_TEST_TOKEN", "s3cret")
    token = Connection(token_env="ROLLOUT_TEST_TOKEN")
    hosting: dict[str, Any] = {"token": "s3cret", "connection": token}
    async with (
        engine_host("gpu-1", checkpoints, "r", tmp_path / "gpu-1", presence, **hosting) as one,
        engine_host("gpu-2", checkpoints, "r", tmp_path / "gpu-2", presence, **hosting) as two,
    ):
        await until(lambda: one.channel.serving == two.channel.serving == first.id)
        upstreams = [one.address, two.address]
        async with (
            served(passing_on(upstreams)) as (proxy, _),
            served(passing_on(upstreams, caching=True)) as (cache, _),
        ):
            direct = routed(checkpoints, one.address, two.address, connection=token)
            through = routed(checkpoints, proxy, connection=token)
            for each, sessions in ((direct, range(6)), (through, range(6, 12))):  # (the same servers, either way)
                assert [await said(each, f"r_{number}/policy") for number in sessions] == [[(first.id, 1)]] * 6
            assert len(one.engine.prompts) + len(two.engine.prompts) == 12
            assert not await routes_of(routed(checkpoints, proxy)).reaches("r", "policy")  # (no token: refused)
            # A proxy that answers from a cache names the checkpoint of its first answer: refused, never recorded.
            stale = routed(checkpoints, cache, connection=token)
            assert await said(stale, "r_20/policy") == [(first.id, 1)]
            second = await made(checkpoints, fence, tmp_path, first)
            await serve(checkpoints, fence, second)
            await until(lambda: one.channel.serving == two.channel.serving == second.id)
            await routes_of(stale).channel("r", "policy").refresh(now=True)
            with pytest.raises(ModelEndpointError, match=f"{second.id} was asked for, and {first.id} answered"):
                await said(stale, "r_20/policy", turns=2)
            assert await said(through, "r_21/policy") == [(second.id, 2)]


async def test_an_eval_plays_its_checkpoint_and_no_other(tmp_path: Path) -> None:
    checkpoints, presence = shared(tmp_path)
    fence = await checkpoints.ledger.take(scope("r"))
    first = await made(checkpoints, fence, tmp_path)
    await serve(checkpoints, fence, first)
    second = await made(checkpoints, fence, tmp_path, first)
    evaluated = await checkpoints.ledger.take(scope("e"))  # (the eval of `second`, played on run r's servers)
    eval_of = Serving("policy", second.id, second.depth, max_lag=0)
    await record_serving(checkpoints.ledger, "e", eval_of, evaluated)
    async with engine_host("gpu-1", checkpoints, "r", tmp_path / "gpu-1", presence) as host:
        await until(lambda: host.channel.serving == first.id)
        routes = routes_of(recorder := routed(checkpoints, host.address))
        assert await routes.reaches("r", "policy")  # (it has what run r says)
        assert not await routes.reaches("e", "policy")  # (one behind the eval's checkpoint: an eval plays no other)
        await serve(checkpoints, fence, second)
        await until(lambda: host.channel.serving == second.id)
        await routes.channel("e", "policy").refresh(now=True)
        assert await said(recorder, "r_1/policy", run="e") == [(second.id, 2)]
