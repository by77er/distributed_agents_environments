"""Every channel a run's start names, served by one gateway: the trained channel, a channel that follows it some
records behind, and a fixed one (a judge on a base model, or a pinned checkpoint), each reached by the key of the
slot the run binds to it. The servers here are scripted: they hold checkpoints by name and say which one sampled."""

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from rollout.contracts import Message
from rollout.harness import ModelSlot, Program, ProgramReference, RunBinding, RunContext, bind, register
from rollout.harness.blobs import FileBlobStore
from rollout_train.checkpoints import Checkpoint, Checkpoints, new_id
from rollout_train.following import Follower
from rollout_train.gateway import ChannelDirectory, Gateway, GatewayEndpoints, Provided, TurnStore, create_app
from rollout_train.inference import Generation, Limits
from rollout_train.inference.channel import Channel, NotLoaded
from rollout_train.ledger import Fence, FileLedger
from rollout_train.record import STARTS, scope, start_header, table
from rollout_train.recorder import Renderer
from rollout_train.run_settings import RunSettings, recorded
from rollout_train.serving import Serving, record_serving, serving_of, wanted
from rollout_train.testing import Characters, PlainRenderer, ScriptedEngine, admitted, sample_request
from tests.rollout_train.gateway.support import client, keyring
from tests.rollout_train.test_validation import the_cluster

RENDERER = "rollout_train.testing:plain_renderer"


class Held:
    """A server that holds models by name (its base model, and the checkpoints loaded into it) and answers with the
    name of the one asked for."""

    max_model_len = 4096

    def __init__(self, address: str, model: str) -> None:
        self.address, self.model = address, model
        self.held = {model}

    async def models(self, within: float = 2.0) -> dict[str, Any]:
        return {name: {"id": name, "max_model_len": self.max_model_len} for name in self.held}

    async def generate(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Sequence[int],
        adapter: str | None,
        session: str = "",
        request: str | None = None,
    ) -> Generation:
        name = adapter or self.model
        if name not in self.held:
            raise NotLoaded(f"{self.address} holds no {name}")
        tokens = [ord(each) for each in f"{name}\n"]
        return Generation(tokens, [-0.5] * len(tokens), "stop", model=name)

    def close(self) -> None: ...


class Matched(Program):
    """A player, an opponent that is not trained, and a judge."""

    def model_slots(self) -> Mapping[str, ModelSlot]:
        return {
            "player": ModelSlot(),
            "opponent": ModelSlot(trained=False),
            "judge": ModelSlot(trained=False, judge=True),
        }

    async def main(self, run: RunContext) -> None:
        raise NotImplementedError


SETTINGS: dict[str, JsonValue] = {
    "kind": "train",
    "environment": "tests:matched",
    "trainer.provider": "local-lora",
    "channels.policy.provider": "local",
    "channels.policy.model": "base",
    "channels.policy.renderer": RENDERER,
    "channels.opponent.provider": "local",
    "channels.opponent.model": "base",
    "channels.opponent.renderer": RENDERER,
    "channels.opponent.mode": "follows",
    "channels.opponent.follows": "policy",
    "channels.opponent.lag": 1,
    "channels.judge.provider": "judging",
    "channels.judge.model": "judge-model",
    "channels.judge.renderer": RENDERER,
    "channels.judge.mode": "fixed",
    "slots.opponent": "opponent",
    "slots.judge": "judge",
}


async def started(ledger: FileLedger, run: str, settings: Mapping[str, JsonValue]) -> Fence:
    """A run's start, recording its settings as a launch's does."""
    fence = await ledger.take(scope(run))
    said = {**start_header(), "run_settings": recorded(RunSettings(settings))}
    await ledger.append(table(run, STARTS), str(fence.number), cast(JsonValue, said), fence)
    return fence


async def made(checkpoints: Checkpoints, fence: Fence, tmp_path: Path, parent: Checkpoint | None = None) -> Checkpoint:
    id = new_id()
    (weights := tmp_path / "trained" / id).mkdir(parents=True)
    (weights / "adapter.bin").write_text(id)
    return await checkpoints.add(
        fence, id, weights=weights, run="r", base="base", parents=[parent.id] if parent else []
    )


async def served(checkpoints: Checkpoints, fence: Fence, checkpoint: Checkpoint | None) -> None:
    """What the training loop writes when it serves a checkpoint (with none, the base model)."""
    said = Serving("policy", model="base")
    if checkpoint is not None:
        said = Serving("policy", checkpoint.id, checkpoint.depth, checkpoint.kind, checkpoint.weights, model="base")
    await record_serving(checkpoints.ledger, "r", said, fence)


async def asked(endpoints: GatewayEndpoints, binding: RunBinding, slot: str, turn: int = 0) -> str:
    """Which model sampled a slot's turn, as the reply says."""
    model = binding.models[slot].recorded
    assert model is not None
    request = sample_request([Message.user(f"Turn {turn}.")], f"r_1:{slot}:{turn}", session_id=f"r_1/{slot}")
    return (await endpoints.endpoint(model).sample(request)).message.text.strip()


async def test_one_gateway_serves_the_trained_a_following_and_a_fixed_channel_of_a_run(tmp_path: Path) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    checkpoints = Checkpoints(ledger, blobs)
    fence = await started(ledger, "r", SETTINGS)
    first = await made(checkpoints, fence, tmp_path)
    second = await made(checkpoints, fence, tmp_path, first)
    for each in (None, first, second):
        await served(checkpoints, fence, each)
    local, judging = Held("local", "base"), Held("judging", "judge-model")
    local.held |= {first.id, second.id}  # (what its follower loaded)
    providers = {"local": Provided((local,)), "judging": Provided((judging,))}
    directory = ChannelDirectory(ledger, providers, every=0.01, patience=1.0)
    gateway = Gateway(TurnStore(ledger, blobs), keyring(), directory=directory)
    endpoints = GatewayEndpoints.of(gateway)
    binding = bind(
        ProgramReference(program=register(Matched)), "policy", slots={"opponent": "opponent", "judge": "judge"}
    )
    assert await endpoints.reaches("r", binding)
    assert set(await directory.load("r")) == {"policy", "opponent", "judge"}
    await admitted(endpoints, "r_1", "r")
    said = {slot: await asked(endpoints, binding, slot) for slot in ("player", "opponent", "judge")}
    assert said == {"player": second.id, "opponent": first.id, "judge": "judge-model"}  # (follows one record behind)
    turns = {turn.slot: turn for turn in await gateway.store.turns("r", "r_1")}
    assert {slot: (turn.channel, turn.depth, turn.trained) for slot, turn in turns.items()} == {
        "player": ("policy", 2, True),
        "opponent": ("opponent", 1, False),
        "judge": ("judge", 0, False),
    }
    third = await made(checkpoints, fence, tmp_path, second)
    local.held.add(third.id)
    await served(checkpoints, fence, third)
    await asyncio.sleep(0.05)  # (each channel asks again what it should serve)
    assert [await asked(endpoints, binding, slot, 1) for slot in ("player", "opponent")] == [third.id, second.id]
    directory.close()


async def test_a_runner_learns_a_runs_channels_from_a_gateway_elsewhere(tmp_path: Path) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    await started(ledger, "r", SETTINGS)
    providers = {"local": Provided((Held("local", "base"),)), "judging": Provided((Held("judging", "judge-model"),))}
    gateway = Gateway(TurnStore(ledger, blobs), keyring(), directory=ChannelDirectory(ledger, providers, every=0.01))
    async with client(create_app(gateway)) as http:
        listed = (await http.get("/v1/models", params={"run": "r"})).json()["data"]
        assert {each["id"] for each in listed} == {"r/policy", "r/opponent", "r/judge"}
        remote = GatewayEndpoints("http://gateway", keyring(), TurnStore(ledger, blobs), http=http)
        binding = bind(ProgramReference(program=register(Matched)), "policy", slots={"judge": "judge"})
        judge = binding.models["judge"].recorded
        assert judge is not None and await remote.reaches("r", RunBinding(models={"judge": binding.models["judge"]}))
        await admitted(remote, "r_1", "r")
        request = sample_request([Message.user("Score it.")], "r_1:judge:0", session_id="r_1/judge")
        assert remote.endpoint(judge).describe("r_1/judge").context_limit == 4096
        assert (await remote.endpoint(judge).sample(request)).message.text.strip() == "judge-model"


async def test_a_following_channel_serves_the_followed_records_lag_behind_and_a_fixed_one_its_checkpoint(
    tmp_path: Path,
) -> None:
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    checkpoints = Checkpoints(ledger, blobs)
    fence = await started(ledger, "pinned", {})  # (a run whose first checkpoint another run pins)
    pinned = await made(checkpoints, fence, tmp_path)
    settings = {**SETTINGS, "channels.opponent.lag": 2, "channels.judge.checkpoint": pinned.id}
    fence = await started(ledger, "r", settings)
    assert await wanted(ledger, "r", "opponent") is None  # (nothing to follow yet: the base model)
    first = await made(checkpoints, fence, tmp_path)
    second = await made(checkpoints, fence, tmp_path, first)
    for each in (None, first, second):
        await served(checkpoints, fence, each)
    following = await serving_of(ledger, "r", "opponent")
    assert [(each.channel, each.checkpoint) for each in following] == [("opponent", None)]  # (two behind: the base)
    third = await made(checkpoints, fence, tmp_path, second)
    await served(checkpoints, fence, third)
    now = await wanted(ledger, "r", "opponent")
    assert now is not None and (now.channel, now.checkpoint, now.files) == ("opponent", first.id, first.weights)
    judge = await wanted(ledger, "r", "judge")
    assert judge is not None and (judge.checkpoint, judge.depth, judge.max_lag) == (pinned.id, 1, 0)
    assert (await wanted(ledger, "r", "policy")).checkpoint == third.id  # type: ignore[union-attr]
    engine = ScriptedEngine(Characters())  # type: ignore[arg-type]
    channels = {
        name: Channel(name, [engine], cast(Renderer, PlainRenderer()), Limits()) for name in ("opponent", "judge")
    }
    follower = Follower("engines", checkpoints, "r", channels, tmp_path / "following")
    assert await follower.follow()  # (an engine host loads what each channel serves)
    assert {f"load {first.id}", f"load {pinned.id}"} <= set(engine.told)


def test_a_directory_over_the_cluster_config_knows_the_providers_reached_at_endpoints(tmp_path: Path) -> None:
    cluster = the_cluster()
    directory = ChannelDirectory.of(cluster, FileLedger(tmp_path / "ledger"))
    assert "lab" in directory.providers and not {"openai", "tinker"} & set(directory.providers)
    assert directory.providers["lab"].servers == cluster.inference["lab"].endpoints
