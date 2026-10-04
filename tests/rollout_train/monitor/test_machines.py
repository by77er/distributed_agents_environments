"""The Machines page's endpoint: every machine that beats and the roles on it, from fake heartbeats and the ledger: the
runners and the episodes their claims hold, the pools and their leases, an engine host and how far behind its run's
wanted checkpoint each engine is, a launcher and its launches going, a gateway; each alive or gone by the store's
clock."""

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from rollout.harness.sandboxes import Lease
from rollout_train.gateway.beats import GATEWAY
from rollout_train.inference.remote import ENGINES
from rollout_train.launcher import LAUNCHER
from rollout_train.launches import Asked, launches_of
from rollout_train.ledger import FileLedger
from rollout_train.monitor.machines import machines
from rollout_train.monitor.system import System
from rollout_train.presence import STALE, Beat, presence_of
from rollout_train.record import GROUPS, STARTS, scope, table
from rollout_train.rollouts.scheduler import CLAIMS, runner_scope
from rollout_train.sandboxes import POOL, FileLeases
from rollout_train.serving import Serving, record_serving

pytest.importorskip("starlette")
from tests.rollout_train.support import monitor_client

GPU: dict[str, Any] = {"name": "RTX", "used": 2**30, "total": 2**34, "busy": 0.5}
MACHINE: dict[str, Any] = {"memory": {"available": 2**33, "total": 2**34}, "accelerators": [GPU], "disk": None}


async def scratch(tmp_path: Path) -> FileLedger:
    """A ledger with a run whose episodes a runner claimed, a pool's leases, what its channel should serve, a
    launch going, and the beats of a runner, a pool on a machine of its own, an engine host, a launcher and a
    gateway."""
    ledger = FileLedger(tmp_path / "ledger")
    fence = await ledger.take(scope("train"))
    await ledger.append(table("train", STARTS), "1", {"started": time.time(), "host": "gpu-1"}, fence)
    await ledger.append(table("train", GROUPS), "1", {"task": "t", "decided": time.time(), "episodes": 2}, fence)
    await record_serving(ledger, "train", Serving(channel="policy", checkpoint="ck-2", depth=2), fence)
    runner = await ledger.take(runner_scope("gpu-1/train"))
    for key, run_id in (("1/1/1", "r_one"), ("1/2/1", "r_two")):
        claim: JsonValue = {"runner": "gpu-1/train", "fence": runner.number, "at": time.time(), "run_id": run_id}
        await ledger.append(table("train", CLAIMS), key, claim, runner)
    leases = FileLeases(ledger.directory)
    pool = "minecraft@gpu-1/train"
    now = time.time()
    await leases.put(Lease(key="train/1/1/1/world", kind="minecraft", pool=pool, handle="s-1", at=now, seconds=600))
    await leases.put(Lease(key="train/1/9/1/world", kind="minecraft", pool=pool, handle="s-2", at=now, lost=True))
    await leases.put(Lease(key="by-hand/world", kind="docker", pool="docker@far", handle="s-3", at=time.time()))
    launches = launches_of(ledger)
    assert launches is not None
    going = await launches.ask(Asked(profile="one-gpu", environment="c:c", name="diamonds"))
    await launches.claim(going.id, "launcher/gpu-1")
    await launches.ask(Asked(profile="one-gpu", environment="c:c", name="not claimed"))

    beats = presence_of(ledger)
    assert beats is not None
    pools: JsonValue = {"minecraft": {"size": 4, "leased": 1, "free": 3}}
    await beats.beat("gpu-1/train", {"host": "gpu-1", "run": "train", "machine": MACHINE, "places": 6, "playing": 2,
                                     "pools": pools, "channels": []})  # fmt: skip
    await beats.beat("pools/docker@far", {"kind": POOL, "host": "far", "pool": "docker@far", "sandboxes": "docker",
                                          "size": 8, "leased": 1, "free": 7})  # fmt: skip
    engines: list[JsonValue] = [{"address": "http://gpu-2:8000", "serving": "ck-1", "version": 1}]
    channel: JsonValue = {"channel": "policy", "adapter": "ck-1", "version": 1, "requests": 3,
                          "tokens_per_second": 90.0, "mean_concurrency": 2.0, "engines": engines}  # fmt: skip
    await beats.beat("gpu-2", {"kind": ENGINES, "host": "gpu-2", "follows": "train", "machine": MACHINE,
                               "channels": [channel]})  # fmt: skip
    offered: JsonValue = [{"profile": "one-gpu", "path": "one.toml", "model": "m", "weights": "lora", "settings": {}}]
    await beats.beat("launcher/gpu-1", {"kind": LAUNCHER, "host": "gpu-1", "machine": MACHINE, "profiles": offered,
                                        "environments": ["c:c"], "at_once": 1, "playing": 1})  # fmt: skip
    await beats.beat("gateway/edge/0.0.0.0:8443", {"kind": GATEWAY, "host": "edge", "listen": "0.0.0.0:8443",
                                                   "channels": []})  # fmt: skip
    return ledger


async def test_every_role_on_every_machine_with_what_it_holds(tmp_path: Path) -> None:
    ledger = await scratch(tmp_path)
    shown = await System(ledger=ledger).machines()

    assert [host["host"] for host in shown["hosts"]] == ["edge", "far", "gpu-1", "gpu-2"]
    gpu_1 = next(host for host in shown["hosts"] if host["host"] == "gpu-1")
    assert gpu_1["alive"] and gpu_1["machine"]["accelerators"] == [GPU] and len(gpu_1["history"]) == 1
    assert sorted((role["kind"], role["name"]) for role in gpu_1["roles"]) == [
        ("launchers", "launcher/gpu-1"), ("pools", "minecraft@gpu-1/train"), ("runners", "gpu-1/train"),
    ]  # fmt: skip

    (runner,) = shown["runners"]
    assert runner == runner | {"name": "gpu-1/train", "host": "gpu-1", "alive": True, "run": "train"}
    assert (runner["places"], runner["playing"], runner["free"]) == (6, 2, 4)
    assert [(claim["run"], claim["group"], claim["episode"], claim["run_id"]) for claim in runner["claims"]] == [
        ("train", 1, 1, "r_one"), ("train", 1, 2, "r_two"),
    ]  # fmt: skip
    assert runner["pools"] == ["minecraft@gpu-1/train"]

    pools = {pool["name"]: pool for pool in shown["pools"]}
    assert set(pools) == {"minecraft@gpu-1/train", "docker@far"}
    within = pools["minecraft@gpu-1/train"]
    assert within == within | {"kind": "minecraft", "host": "gpu-1", "runner": "gpu-1/train", "alive": True}
    assert (within["size"], within["leased"], within["free"]) == (4, 1, 3)
    held, lost = within["leases"]  # (the lost one last)
    assert held == held | {"key": "train/1/1/1/world", "run": "train", "group": 1, "episode": 1, "attempt": 1}
    assert held == held | {"sandbox": "world", "run_id": "r_one", "holds": True, "seconds": 600, "lost": False}
    assert lost["lost"] and lost["holds"] is None  # (no claim of its key)
    apart = pools["docker@far"]
    assert apart == apart | {"kind": "docker", "host": "far", "runner": None, "size": 8, "leased": 1, "free": 7}
    (by_hand,) = apart["leases"]
    assert by_hand["run"] is None and by_hand["holds"] is None  # (a key that names no run's episode)

    (host,) = shown["engines"]
    assert host["follows"] == "train"
    (channel,) = host["channels"]
    assert channel == channel | {"serving": "ck-1", "version": 1, "wanted": "ck-2", "behind": 1}
    assert channel["tokens_per_second"] == 90.0 and channel["engines"] == [
        {"address": "http://gpu-2:8000", "serving": "ck-1", "version": 1, "behind": 1}
    ]

    (launcher,) = shown["launchers"]
    assert launcher["profiles"] == [{"profile": "one-gpu", "model": "m", "weights": "lora"}]
    assert launcher["environments"] == ["c:c"] and [each["name"] for each in launcher["launches"]] == ["diamonds"]

    (gateway,) = shown["gateways"]
    assert gateway == gateway | {"name": "gateway/edge/0.0.0.0:8443", "host": "edge", "listen": "0.0.0.0:8443"}

    async with monitor_client(str(tmp_path / "ledger"), beat=0.0) as client:
        answer = await client.get("/api/machines")
        assert answer.status_code == 200 and [each["name"] for each in answer.json()["runners"]] == ["gpu-1/train"]
        again = await client.get("/api/machines", headers={"If-None-Match": answer.headers["ETag"]})
        assert again.status_code == 304  # (nothing changed: when it was read is not part of its version)


def beat(name: str, about: Mapping[str, JsonValue], *, age: float, at: float) -> Beat:
    return Beat(name, at, about, [{"at": at, "machine": MACHINE}], age=age)


def test_alive_or_gone_is_judged_by_the_stores_clock() -> None:
    """The store's stamps may be far from the monitor's clock: whether a beat is alive is its age by the store's
    clock, and when it beat is shown by the monitor's (its age before now)."""
    now = 1_000_000.0
    beats = [
        beat("gpu-1/train", {"host": "gpu-1", "pools": {"docker": {"size": 2, "leased": 2}}}, age=5.0, at=1.0),
        beat("gpu-2/train", {"host": "gpu-2", "places": 2, "machine": MACHINE}, age=STALE + 1, at=now + 3600),
    ]
    shown = machines(beats, now=now)
    alive = {runner["name"]: (runner["alive"], runner["at"]) for runner in shown["runners"]}
    assert alive == {"gpu-1/train": (True, now - 5), "gpu-2/train": (False, now - STALE - 1)}
    assert [runner["name"] for runner in shown["runners"]] == ["gpu-1/train", "gpu-2/train"]  # (the alive first)
    (pool,) = shown["pools"]
    assert pool == pool | {"name": "docker@gpu-1/train", "alive": True, "size": 2, "leased": 2, "free": 0}
    hosts = {host["host"]: host["alive"] for host in shown["hosts"]}
    assert hosts == {"gpu-1": True, "gpu-2": False}
    assert machines([], now=now) == {
        "now": now, "hosts": [], "runners": [], "pools": [], "engines": [], "launchers": [], "gateways": [],
    }  # fmt: skip
