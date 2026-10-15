"""A runner of its own (`rollout runner`, `rollout_train.runners`): it plays the runs it is given, and their evals',
beside their drivers' runners, sampling through a gateway of its own over the channels each run's start names; it plays
no run whose channels its driver alone reaches, says why, and plays nothing where it can play none; two share a run's
work as any runners do, each attempt played once; the pools it makes are named apart from the run's driver's, so
neither's keeper fences or sweeps the other's; and taken over under its name, it ends with status 1, saying so."""

import asyncio
import contextlib
import logging
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import JsonValue

from rollout.harness import PoolBinding, RunBinding, SandboxPool, agent_program
from rollout.testing import until
from rollout_train.gateway import ChannelDirectory, Provided
from rollout_train.jobs import Run
from rollout_train.ledger import FileLedger, Ledger
from rollout_train.pods.pools import PodPools
from rollout_train.presence import presence_of
from rollout_train.record import GROUPS, STARTS, scope, start_header, table
from rollout_train.registry import Entry
from rollout_train.rollouts import Plan, episodes_of, plan
from rollout_train.rollouts.scheduler import CLAIMS, runner_scope
from rollout_train.run_settings import RunSettings, recorded
from rollout_train.runners import Played, played, pools_of, runner_of, unreachable
from rollout_train.sandboxes import ClusterPools, leases_of
from rollout_train.stores import Stores
from tests.rollout_train.clusters import POLICY, WORDS, a_cluster
from tests.rollout_train.gateway.test_directory import Held
from tests.rollout_train.rollouts.games import BOXES, GATES, BoxedGuess, Gated
from tests.rollout_train.support import BINDING, ask

BOXED = "tests.rollout_train.rollouts.games:boxed"
SERVERS = """
[inference.servers]
kind = "vllm-servers"
addresses = ["http://127.0.0.1:9"]
auth = "none"
[inference.servers.models.tiny]
context = 4096

[sandboxes.fake]
provider = "tests.rollout_train.rollouts.games:boxes"
size = 2
"""
ON_SERVERS: dict[str, JsonValue] = {**POLICY, "channels.policy.provider": "servers"}
"""A training run whose channel is on servers at addresses, which any machine that reaches them samples."""


def scripted(ledger: Ledger) -> ChannelDirectory:
    """The cluster's directory, its `servers` scripted: one server, which holds the base model and says its name."""
    return ChannelDirectory(ledger, {"servers": Provided((Held("servers", "tiny"),))}, every=0.01, patience=1.0)


async def asked_for(
    stores: Stores,
    name: str,
    settings: Mapping[str, JsonValue],
    groups: Mapping[int, tuple[JsonValue, int]],
    *,
    environment: str = WORDS,
    task: type = Gated,
) -> Entry:
    """A training run as its driver leaves it in the ledger: registered, its start recording its settings, and its
    groups asked for (`ask`)."""
    entry = await stores.registry.create(name)
    fence = await stores.ledger.take(scope(entry.id))
    said = recorded(RunSettings({**settings, "kind": "train", "name": name, "environment": environment}))
    start = cast(JsonValue, {**start_header(), "run_settings": said})
    await stores.ledger.append(table(entry.id, STARTS), str(fence.number), start, fence)
    await ask(stores.ledger, entry.id, groups, task)
    return entry


def opened(gate: str) -> dict[str, JsonValue]:
    """A guess of the word the scripted server says, past a gate opened already."""
    GATES.setdefault(gate, asyncio.Event()).set()
    return {"word": "tiny", "gate": gate}


async def test_a_runner_of_its_own_plays_a_runs_episodes_and_its_evals_each_recorded_once(tmp_path: Path) -> None:
    cluster = a_cluster(tmp_path, more=SERVERS)
    stores = Stores.open(cluster)
    ledger = stores.ledger
    run = (await asked_for(stores, "far", ON_SERVERS, {1: (opened("far"), 4)})).id
    evaluated = f"{run}-eval-2"  # (a scheduled eval: its start says only the run it is played for)
    fence = await ledger.take(scope(evaluated))
    start = cast(JsonValue, {**start_header(), "by": run, "step": 2})
    await ledger.append(table(evaluated, STARTS), str(fence.number), start, fence)
    await ask(ledger, evaluated, {1: (opened("far"), 2)}, Gated)
    lines: list[str] = []
    failures: list[BaseException] = []
    async with contextlib.AsyncExitStack() as stack:
        runner = await runner_of(
            cluster, stores, ["far"], stack, failures.append, name="far-away", directory=scripted(ledger),
            told=lines.append,
        )  # fmt: skip
        async with asyncio.timeout(30):
            episodes = await episodes_of(ledger, stores.blobs, run, 1, 4, every=0.01)
            evals = await episodes_of(ledger, stores.blobs, evaluated, 1, 2, every=0.01)
        heard = presence_of(ledger)
        assert heard is not None
        (beat,) = [each for each in await heard.beats() if each.runner == "far-away"]
    assert failures == [] and runner.places == 4  # (the run's episodes_at_once)
    assert [episode.reward for episode in [*episodes, *evals]] == [1.0] * 6
    for each in (run, evaluated):
        claims = await ledger.read(table(each, CLAIMS))
        assert all(key.endswith("/1") for key in claims)  # (each attempt once)
        assert {cast(dict[str, Any], claim)["runner"] for claim in claims.values()} == {"far-away"}
    assert lines[0] == f"far: played here, as {run}" and "a larger episodes_at_once" in lines[1]
    assert beat.about["runs"] == [run] and "run" not in beat.about  # (not taken for the run's own process)


async def test_a_runner_of_its_own_plays_no_run_whose_channels_its_driver_alone_reaches(tmp_path: Path) -> None:
    cluster = a_cluster(tmp_path, more=SERVERS)
    stores = Stores.open(cluster)
    ledger = stores.ledger
    near = (await asked_for(stores, "near", POLICY, {1: (opened("near"), 2)})).id  # (on engine hosts of its driver's)
    lines: list[str] = []
    async with contextlib.AsyncExitStack() as stack:
        with pytest.raises(ValueError, match="it can play none of near, nowhere"):
            await runner_of(cluster, stores, ["near", "nowhere"], stack, print, directory=scripted(ledger),
                            told=lines.append)  # fmt: skip
    assert lines == [
        "near: not played here: its channel policy is served by engine hosts in its driver's Ray cluster",
        "nowhere: not played here: there is no run nowhere",
    ]
    far = (await asked_for(stores, "far", ON_SERVERS, {1: (opened("far"), 2)})).id
    async with contextlib.AsyncExitStack() as stack:
        runner = await runner_of(cluster, stores, ["near", "far"], stack, print, directory=scripted(ledger),
                                 told=lines.append)  # fmt: skip
        runs = runner.runs
        assert isinstance(runs, Played) and near not in runs and far in runs
        assert not await runner.recorder.reaches(near, BINDING)  # (nor would it claim near's, were it given it)
        async with asyncio.timeout(30):
            await episodes_of(ledger, stores.blobs, far, 1, 2, every=0.01)
    assert await ledger.read(table(near, CLAIMS)) == {}


def test_what_a_runner_of_its_own_cannot_reach_is_said_channel_by_channel() -> None:
    from tests.rollout_train.test_validation import the_cluster

    cluster = the_cluster()
    policy: dict[str, JsonValue] = {"kind": "train", "channels.policy.model": "m", "channels.policy.renderer": "r"}
    judge: dict[str, JsonValue] = {"channels.judge.provider": "openai", "channels.judge.model": "gpt"}

    def said(provider: str, *more: Mapping[str, JsonValue]) -> list[str]:
        values = {**policy, "channels.policy.provider": provider}
        for each in more:
            values |= each
        return unreachable(cluster, RunSettings(values))

    assert said("lab") == [] and said("lab", judge) == []  # (servers at addresses; a hosted API it spends on freely)
    assert said("local-vllm") == ["its channel policy is served by engine hosts in its driver's Ray cluster"]
    assert said("tinker") == ["its channel policy is sampled by Tinker's engines in its driver's process"]
    assert said("elsewhere") == ["its channel policy is on elsewhere, which this cluster config lacks"]
    assert said("lab", judge, {"limits.spend": 5}) == [
        "its channel judge samples a hosted API, and its limits.spend is counted by its driver's gateway, from the "
        "turns that gateway records"
    ]


def test_a_runner_of_its_own_plays_the_runs_it_is_given_and_their_evals() -> None:
    given = Played(["run_a", "run_b"])
    assert list(given) == ["run_a", "run_b"] and len(given) == 2
    assert all(each in given for each in ("run_a", "run_a-eval-3", "run_a-eval-3-2", "run_b-1"))
    assert not any(each in given for each in ("run_c", "run_ab", "run_a-evil", "run_a-eval-", "run_a/1"))
    assert "run_a" not in Played([])


async def _claimed(ledger: Ledger, run: str, count: int) -> bool:
    return len(await ledger.read(table(run, CLAIMS))) == count


async def test_two_runners_of_their_own_share_a_runs_work_and_never_play_one_attempt_twice(tmp_path: Path) -> None:
    cluster = a_cluster(tmp_path, more=SERVERS)
    stores = Stores.open(cluster)
    ledger = stores.ledger
    GATES.pop("shared", None)
    run = (await asked_for(stores, "shared", ON_SERVERS, {1: ({"word": "tiny", "gate": "shared"}, 4)})).id
    failures: list[BaseException] = []
    async with contextlib.AsyncExitStack() as stack:
        for name in ("one", "two"):
            await runner_of(cluster, stores, ["shared"], stack, failures.append, name=name, places=2,
                            directory=scripted(ledger), told=lambda line: None)  # fmt: skip
        await until(lambda: _claimed(ledger, run, 4), 30)
        claims = await ledger.read(table(run, CLAIMS))
        assert Counter(cast(dict[str, Any], each)["runner"] for each in claims.values()) == {"one": 2, "two": 2}
        GATES.setdefault("shared", asyncio.Event()).set()
        async with asyncio.timeout(30):
            episodes = await episodes_of(ledger, stores.blobs, run, 1, 4, every=0.01)
    assert sorted(episode.number for episode in episodes) == [1, 2, 3, 4] and failures == []
    assert sorted(await ledger.read(table(run, CLAIMS))) == ["1/1/1", "1/2/1", "1/3/1", "1/4/1"]  # (each attempt once)


async def _fenced(ledger: Ledger, scopes: set[str]) -> bool:
    return scopes <= set(await ledger.fences())


async def _none_leased(stores: Stores) -> bool:
    leases = leases_of(stores.ledger)
    return leases is not None and await leases.all() == []


async def test_a_runner_of_its_own_makes_its_pools_apart_from_its_runs_drivers(tmp_path: Path) -> None:
    cluster = a_cluster(tmp_path, more=SERVERS)
    stores = Stores.open(cluster)
    ledger = stores.ledger
    entry = await asked_for(stores, "boxed", ON_SERVERS, {}, environment=BOXED)
    run = entry.id
    fence = await ledger.take(scope(run))
    binding = RunBinding(models=BINDING.models, pools={"fake": PoolBinding(local="fake")})
    await plan(ledger, run, Plan(agent_program(BoxedGuess), binding), fence)
    record: JsonValue = {"parameters": {"word": "tiny"}, "episodes": 2, "decided": 1.0}
    await ledger.append(table(run, GROUPS), "1", record, fence)
    BOXES.clear()
    lines: list[str] = []
    async with contextlib.AsyncExitStack() as stack:
        settings = RunSettings({**ON_SERVERS, "kind": "train", "environment": BOXED, "name": "boxed"})
        driver = Run(cluster, stores, settings, entry, directory=tmp_path / "run", sandboxes=frozenset({"fake"}))
        drivers = (await driver._pools(stack))["fake"]  # pyright: ignore[reportPrivateUsage]
        runner = await runner_of(cluster, stores, ["boxed"], stack, print, name="far-away",
                                 directory=scripted(ledger), told=lines.append)  # fmt: skip
        theirs = runner.pools["fake"]
        assert isinstance(drivers, SandboxPool) and isinstance(theirs, SandboxPool)
        assert (drivers.name, theirs.name) == (f"fake@{run}", f"fake@{run}/far-away")
        assert lines[0] == f"boxed: played here, as {run}; its sandboxes: fake from fake@{run}/far-away"
        async with asyncio.timeout(30):
            episodes = await episodes_of(ledger, stores.blobs, run, 1, 2, every=0.01)
        await until(lambda: _fenced(ledger, {f"pools/fake@{run}", f"pools/fake@{run}/far-away"}))  # (each its own)
        await until(lambda: _none_leased(stores))  # (released with each episode, by the pool that leased it)
        mine, its = BOXES  # (the driver's provider, then the runner's)
        assert mine.made == [] and len(its.made) == 2 and sorted(its.deleted) == sorted(its.made)
    assert all(str(episode.info["key"]).startswith(f"{run}/1/") for episode in episodes)


HOST = """
[tls]
ca = "~/ca.pem"
certificate = "~/gateway.crt"
key = "~/gateway.key"
[inference.h100]
kind = "runpod-host"
image = "ghcr.io/by77er/rollout-host@sha256:0"
gpu_types = ["NVIDIA H100 80GB HBM3"]
sandboxes = ["fake"]
[inference.h100.models."m"]
context = 8192
[sandboxes.fake]
provider = "tests.rollout_train.rollouts.games:boxes"
url = "http://sandboxes-fake-{pool}.sandboxes-fake-pools:8710"
pools = 2
on_pods = true
"""


async def test_the_pools_of_a_runs_pods_and_of_several_pools_are_named_apart_for_a_runner_of_its_own(
    tmp_path: Path,
) -> None:
    cluster = a_cluster(tmp_path, more=HOST)
    ledger = FileLedger(tmp_path / "ledger")
    failures: list[BaseException] = []
    async with contextlib.AsyncExitStack() as stack:
        for apart, named in ((None, "fake@r"), ("runner/x", "fake@r/runner/x")):
            pools, bindings = pools_of(cluster, ledger, "r", {"fake"}, tmp_path / named, stack, failures.append,
                                       held={"h100"}, apart=apart)  # fmt: skip
            pool = pools["fake"]
            assert isinstance(pool, PodPools) and isinstance(pool.fallback, PodPools)
            assert (pool.name, pool.fallback.name) == (named, f"{named}/cluster")  # (each keeper its own fence)
            assert bindings == {"fake": PoolBinding(local="fake")}
            alone, _ = pools_of(cluster, ledger, "s", {"fake"}, tmp_path / "s" / named, stack, failures.append,
                                apart=apart)  # fmt: skip
            spread = alone["fake"]  # (no pod of the run's serves it: the cluster's pools alone)
            assert isinstance(spread, PodPools) and isinstance(spread.discover, ClusterPools)
            assert spread.name == named.replace("@r", "@s")
    assert failures == []


async def test_a_runner_of_its_own_taken_over_under_its_name_ends_with_status_1_saying_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cluster = a_cluster(tmp_path, more=SERVERS)
    stores = Stores.open(cluster)
    ledger = stores.ledger
    run = (await asked_for(stores, "far", ON_SERVERS, {})).id
    playing = asyncio.ensure_future(played(cluster, ["far"], name="taken", stores=stores, directory=scripted(ledger)))
    try:
        await until(lambda: _fenced(ledger, {runner_scope("taken")}), 30)
        await ledger.take(runner_scope("taken"))  # (another runner started under its name)
        await ask(ledger, run, {1: (opened("taken"), 1)}, Gated)  # (its claim of this is refused: it serves no more)
        with caplog.at_level(logging.ERROR, logger="rollout_train.runners"):
            assert await asyncio.wait_for(asyncio.shield(playing), 30) == 1
    finally:
        playing.cancel()
        await asyncio.gather(playing, return_exceptions=True)
    assert "the runner taken stopped" in caplog.text and "Fenced" in caplog.text


def test_rollout_runner_takes_a_run_at_least_and_exits_saying_why_where_it_plays_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rollout_train import runners
    from rollout_train.cli import main

    a_cluster(tmp_path, more=SERVERS)
    path = str(tmp_path / "cluster.toml")
    refused = {"the following arguments are required: --run": [], "at least 1 episode at once, not 0": ["--run", "a",
                                                                                         "--places", "0"]}  # fmt: skip
    for said, given in refused.items():
        monkeypatch.setattr("sys.argv", ["rollout", "runner", "--cluster", path, *given])
        with pytest.raises(SystemExit) as exited:
            main()
        assert exited.value.code == 2 and said in capsys.readouterr().err
    monkeypatch.setattr("sys.argv", ["rollout", "runner", "--cluster", path, "--run", "nowhere", "--run", "gone"])
    with pytest.raises(SystemExit, match="it can play none of nowhere, gone"):
        main()
    out = capsys.readouterr().out
    assert "nowhere: not played here: there is no run nowhere" in out and "gone: not played here" in out
    with pytest.raises(SystemExit, match="it can play none of nowhere"):
        runners.main(["--cluster", path, "--run", "nowhere"])
