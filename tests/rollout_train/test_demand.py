# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (Ray's state API is partly untyped.)
"""A run's demand: what its scheduled parts need, from its settings and the cluster config alone (local LoRA beside
local vLLM sharing a card; the acceptance run's Tinker trainer with a local engine host and the Tinker to PEFT bridge;
Tinker alone), only scheduled parts counted; and, on the session's Ray, the placement group a run reserves for it,
with every actor placed in its bundle and asking for exactly what the demand counted."""

import tomllib
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from rollout.names import named
from rollout_train.cluster import parsed
from rollout_train.demand import BRIDGE, HEADROOM, SUBMITTER, TRAINER, Resources, demand, pods, requested
from rollout_train.jobs import Run, started
from rollout_train.run_settings import RunSettings
from rollout_train.stores import Stores
from tests.local_ray import LocalRay
from tests.rollout_train.clusters import POLICY, WORDS, a_cluster

ROOT = Path(__file__).resolve().parents[2]
CLUSTER = parsed(tomllib.loads((ROOT / "deploy" / "clusters" / "example.toml").read_text()))
GSM8K = "rollout_verifiers.environments:gsm8k"
CHANNEL: dict[str, JsonValue] = {
    "channels.policy.provider": "local-vllm",
    "channels.policy.model": "Qwen/Qwen3.5-4B",
    "channels.policy.renderer": "rollout_qwen:qwen35",
}
ACCEPTANCE: dict[str, JsonValue] = {"kind": "train", "environment": GSM8K, "trainer.provider": "tinker-lora", **CHANNEL}
"""The acceptance run's shape: Tinker trains, one local engine host serves through the peft-from-tinker bridge."""
LOCAL_LORA: dict[str, JsonValue] = {**ACCEPTANCE, "trainer.provider": "local-lora"}
TINKER: dict[str, JsonValue] = {**ACCEPTANCE, "channels.policy.provider": "tinker"}


def parts(settings: dict[str, JsonValue], **given: Any) -> list[dict[str, Resources]]:
    asked = demand(RunSettings(settings), CLUSTER, **given)
    return [{part.name: part.asks for part in bundle.parts} for bundle in asked.bundles]


def test_local_lora_beside_local_vllm_shares_one_bundle_on_the_drivers_node() -> None:
    asked = demand(RunSettings(LOCAL_LORA), CLUSTER)
    half = Resources(cpus=1, gpus=0.5)
    assert parts(LOCAL_LORA) == [{TRAINER: half, "engine/policy/0": half}, {BRIDGE: Resources(1, 1)}]  # (verbatim)
    assert asked.bundles[0].on_driver and not asked.bundles[1].on_driver
    assert asked.driver == Resources(cpus=2, memory_gib=2)  # (the loop, and one runner of 8 places for 6 at once)
    assert asked.total == Resources(cpus=5, memory_gib=3, gpus=1) and asked.strategy == "PACK"


def test_the_acceptance_runs_tinker_trainer_is_metered_and_its_engine_host_and_bridge_are_scheduled() -> None:
    asked = demand(RunSettings(ACCEPTANCE), CLUSTER)
    assert parts(ACCEPTANCE) == [{"engine/policy/0": Resources(cpus=1, gpus=1)}, {BRIDGE: Resources(2, 1)}]
    assert asked.asks(TRAINER) is None  # (Tinker's: bounded by spend, not placed)
    assert asked.total == Resources(cpus=5, memory_gib=3, gpus=1)
    assert asked.bundles[0].resources.bundle() == {"CPU": 1, "GPU": 1}
    assert asked.bundles[1].resources.bundle() == {"CPU": 2, "memory": 2**30}
    assert requested(asked, kubernetes=True) == asked.total + HEADROOM + SUBMITTER


def test_tinker_alone_asks_ray_for_its_driver_only() -> None:
    asked = demand(RunSettings(TINKER), CLUSTER)
    assert asked.bundles == () and asked.total == Resources(cpus=2, memory_gib=2)


def test_the_driver_counts_its_runners_and_the_sandbox_pools_its_environment_declares() -> None:
    many = demand(RunSettings({**TINKER, "episodes_at_once": 20}), CLUSTER, sandboxes={"minecraft", "unknown"})
    assert many.driver == Resources(cpus=1 + 3 + 6 * 2, memory_gib=2 + 6 * 1.5)  # (3 runners; 6 worlds of 2 CPUs)


def test_an_eval_of_a_checkpoint_has_a_bridge_bundle_and_a_check_none() -> None:
    evaluation = {**CHANNEL, "kind": "eval", "environment": GSM8K, "start": "run:3"}
    assert parts(evaluation) == [{"engine/policy/0": Resources(cpus=1, gpus=1)}, {BRIDGE: Resources(2, 1)}]
    assert parts({**CHANNEL, "kind": "eval", "environment": GSM8K}) == [{"engine/policy/0": Resources(1, gpus=1)}]
    assert parts({**CHANNEL, "kind": "check", "environment": GSM8K}) == [{"engine/policy/0": Resources(1, gpus=1)}]


def test_replicas_are_bundles_of_their_own_and_a_bridge_asks_what_the_cluster_says() -> None:
    import dataclasses

    from rollout_train.cluster import BridgeSection

    bigger = dataclasses.replace(CLUSTER, bridges={"peft-from-tinker": BridgeSection("peft-from-tinker", 4, 8)})
    asked = demand(RunSettings({**ACCEPTANCE, "channels.policy.replicas": 2}), bigger)
    names = [[part.name for part in bundle.parts] for bundle in asked.bundles]
    assert names == [["engine/policy/0"], ["engine/policy/1"], [BRIDGE]]
    assert asked.asks(BRIDGE) == Resources(cpus=4, memory_gib=8)


def test_a_run_too_big_for_one_pod_gets_a_worker_pod_for_each_engine_host() -> None:
    asked = demand(RunSettings({**ACCEPTANCE, "channels.policy.replicas": 2}), CLUSTER)
    (one,) = pods(asked)
    assert one.ray == Resources(cpus=6, memory_gib=3, gpus=2) and one.requests == one.ray + HEADROOM
    head, engines = pods(asked, Resources(memory_gib=14, gpus=1), known=("memory_gib", "gpus"))
    assert (head.group, head.ray) == ("head", Resources(cpus=4, memory_gib=3))  # (the driver and the bridge)
    assert (engines.group, engines.replicas, engines.ray) == ("engines-0", 2, Resources(cpus=1, gpus=1))


async def test_a_runs_placement_group_reserves_its_demand_with_each_actor_in_its_bundle(
    tmp_path: Path, local_ray: LocalRay
) -> None:
    from ray.util import placement_group_table
    from ray.util.state import list_actors

    cluster = a_cluster(tmp_path)
    stores = Stores.open(cluster)
    entry = await stores.registry.create("placed")
    run = Run(cluster, stores, RunSettings({**POLICY, "kind": "train", "environment": WORDS, "name": "placed"}), entry)
    run.environment = named(WORDS)
    async with started(run, training=True) as live:
        asked, group = live.demand, live.group
        assert asked is not None and group is not None
        table = placement_group_table(group)
        assert (table["state"], table["strategy"], table["name"]) == ("CREATED", "PACK", f"run/{run.run.id}")
        bundles = list(table["bundles"].values())
        assert [{key: value for key, value in each.items() if not key.startswith("node:")} for each in bundles] == [
            each.resources.bundle() for each in asked.bundles
        ]
        assert any(key.startswith("node:") for key in bundles[0])  # (the trainer's: on the driver's node)
        assert [[part.name for part in each.parts] for each in asked.bundles] == [[TRAINER, "engine/policy/0"],
                                                                                 [BRIDGE]]  # fmt: skip
        for part in (TRAINER, "engine/policy/0"):
            found = list_actors(
                address=local_ray.dashboard, detail=True, filters=[("name", "=", f"run/{run.run.id}/{part}")]
            )
            (actor,) = cast(list[Any], found)
            assert actor.placement_group_id == group.id.hex()
            wanted = asked.asks(part)
            assert wanted is not None and _unplaced(actor.required_resources) == wanted.bundle()
    assert placement_group_table(group)["state"] == "REMOVED"  # (removed with the run)


def _unplaced(required: dict[str, float]) -> dict[str, float]:
    """An actor's resources as it asked for them: Ray names a placed one's `CPU_group_INDEX_GROUP`, and adds a share of
    its bundle's own (`bundle_group_INDEX_GROUP`)."""
    found: dict[str, float] = {}
    for key, value in required.items():
        if key.startswith("bundle_group_"):
            continue
        if "_group_" in key:
            name, _, rest = key.partition("_group_")
            if rest.count("_") == 1:  # (`CPU_group_0_ID`: the one in its bundle, beside `CPU_group_ID`)
                found[name] = value
        else:
            found[key] = value
    return found
