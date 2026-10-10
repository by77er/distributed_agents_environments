"""A run on RunPod's pods, checked before it is asked for: a `runpod-host` pod is one pod for the run's trainer and
its trained channel, charged once; a step's spend on pods is their hourly price for as long as a step took here lately
(not known: said so); a run is refused more pods than a provider's `max_pods`, pods with no step-ca, or a cluster
whose pods cannot reach the ledger service; and pods are not the cluster's GPUs."""

import dataclasses
import io
import json
import tomllib
import zipfile
from pathlib import Path
from typing import Any, cast

import pytest

from rollout.contracts import BlobReference
from rollout.harness.blobs import FileBlobStore
from rollout_train.cluster import Cluster, KubernetesSection, parsed
from rollout_train.pods.leasing import PodNeed, needs_of, serving, with_sandboxes
from rollout_train.pods.sources import SandboxSource, local_projects, sources_of
from rollout_train.providers import pod_table
from rollout_train.published import EnvironmentVersion
from rollout_train.run_settings import RunSettings
from rollout_train.validation import EnvironmentFacts, LedgerFacts, check, spend_of

STEP_CA = '{ url = "https://ca.example.com", provisioner = "launcher", key_file = "~/p.jwk", root = "~/root.crt" }'


def cluster_of(ledger: str = 'token_env = "LEDGER_TOKEN"\npublic = "https://ledger.example.com"',
               host: str = f'step_ca = {STEP_CA}\nstore = "r2"') -> Cluster:  # fmt: skip
    return parsed(
        tomllib.loads(f"""
name = "test"
[ledger]
url = "sqlite:///~/ledger.db"
{ledger}
[stores.r2]
kind = "rollout_s3:S3BlobStore"
bucket = "rollout"
[tls]
ca = "~/ca.pem"
certificate = "~/gateway.crt"
key = "~/gateway.key"
[inference.h100]
kind = "runpod-host"
image = "ghcr.io/by77er/rollout-host@sha256:0"
gpu_types = ["NVIDIA H100 80GB HBM3"]
price = 2.69
{host}
[inference.h100.models."Qwen/Qwen3.5-9B"]
context = 8192
options = {{ max_lora_rank = 32 }}
[trainers.h100-lora]
kind = "runpod-trainer"
colocate_with = "h100"
models = ["Qwen/Qwen3.5-9B"]
""")
    )


SETTINGS = {
    "kind": "train", "environment": "gridworld.environment:environment", "trainer.provider": "h100-lora",
    "channels.policy.provider": "h100", "channels.policy.model": "Qwen/Qwen3.5-9B",
    "channels.policy.renderer": "rollout_qwen:qwen35", "channels.policy.answer_tokens": 256, "trainer.rank": 16,
}  # fmt: skip
ENVIRONMENT = EnvironmentFacts("gridworld.environment:environment", episodes_per_group=4, turns_per_episode=10,
                               prompt_tokens=500)  # fmt: skip


def test_a_host_pod_takes_the_runs_steps_and_serves_its_trained_channel() -> None:
    (need,) = needs_of(RunSettings(SETTINGS), cluster_of())
    assert (need.provider, need.role, need.count, need.channel) == ("h100", "host", 1, "policy")
    assert need.settings["implementation"] == "rollout_lora:LoraTrainer" and need.settings["model"] == "Qwen/Qwen3.5-9B"
    assert dict(need.settings["trainer"])["rank"] == 16  # type: ignore[arg-type]
    assert isinstance(need, PodNeed)


def test_a_steps_spend_on_pods_is_their_price_for_as_long_as_a_step_took_here_lately() -> None:
    cluster, settings = cluster_of(), RunSettings(SETTINGS)
    unknown = spend_of(settings, cluster, ENVIRONMENT, LedgerFacts())
    assert unknown.dollars is None and "how long a step takes here is not known yet" in unknown.why
    known = spend_of(settings, cluster, ENVIRONMENT, LedgerFacts(step_seconds=120.0))
    assert known.dollars == pytest.approx(2.69 * 120 / 3600) and set(known.parts) == {"h100"}  # (one pod, once)


def test_a_run_is_refused_what_its_pods_cannot_be_given() -> None:
    def refused(cluster: Cluster, settings: RunSettings) -> list[str]:
        return [each.reason for each in check(settings, cluster, ENVIRONMENT) if each.refuses]

    assert not [each for each in refused(cluster_of(), RunSettings(SETTINGS)) if "pod" in each]
    two = RunSettings({**SETTINGS, "channels.policy.replicas": 2})
    assert any("needs 2 pods of h100, which has at most 1" in each for each in refused(cluster_of(), two))
    no_ca = cluster_of(host='store = "r2"')
    assert any("get their certificates from step-ca" in each for each in refused(no_ca, RunSettings(SETTINGS)))
    assert any("reach the ledger service at [ledger] public" in each
               for each in refused(cluster_of(ledger=""), RunSettings(SETTINGS)))  # fmt: skip
    files = cluster_of(host=f"step_ca = {STEP_CA}")  # (no store: the cluster's [blobs], files)
    assert any("cannot read a store of files" in each for each in refused(files, RunSettings(SETTINGS)))
    gpus = check(RunSettings(SETTINGS), cluster_of(), ENVIRONMENT, LedgerFacts(gpus=0.0))
    assert not [each for each in gpus if each.refuses and "GPUs" in each.reason]  # (pods are not the cluster's GPUs)


SERVED = f"""step_ca = {STEP_CA}
store = "r2"
sandboxes = ["minecraft"]
[sandboxes.minecraft]
provider = "minecraft_team.worlds:worlds"
on_pods = {{ settings = {{ cache = "/workspace/minecraft" }} }}
[environments."gridworld.environment:environment"]
python = "platform"
"""


def test_on_kubernetes_sandboxes_the_runs_pods_serve_need_no_pool_of_their_own() -> None:
    worlds = dataclasses.replace(ENVIRONMENT, sandboxes=frozenset({"minecraft"}))
    on_kubernetes = KubernetesSection("rollout", "rayjob.yaml")

    def refused(cluster: Cluster) -> list[str]:
        found = check(RunSettings(SETTINGS), dataclasses.replace(cluster, kubernetes=on_kubernetes), worlds)
        return [each.reason for each in found if each.refuses and "sandboxes of kind" in each.reason]

    assert refused(cluster_of(host=SERVED)) == []  # (the run leases an h100 pod, which serves them)
    elsewhere = SERVED.replace('sandboxes = ["minecraft"]\n', "") + (
        '[inference.a100]\nkind = "runpod-host"\nimage = "ghcr.io/by77er/rollout-host@sha256:0"\n'
        'gpu_types = ["NVIDIA A100 80GB PCIe"]\nsandboxes = ["minecraft"]\n[inference.a100.models."m"]\ncontext = 8\n'
    )
    (said,) = refused(cluster_of(host=elsewhere))
    assert "serves from the pods of a100, and the run leases none of them" in said
    with_url = elsewhere.replace("[sandboxes.minecraft]\n", '[sandboxes.minecraft]\nurl = "http://sandboxes:8710"\n')
    assert refused(cluster_of(host=with_url)) == []  # (the cluster's own pool serves them)


async def test_a_host_pod_is_given_the_sources_of_the_kinds_it_serves_for_its_run(tmp_path: Path) -> None:
    cluster = cluster_of(host=SERVED.replace("on_pods = {", 'heap = "1G"\non_pods = {'))
    host = cluster.inference["h100"]
    assert serving(cluster, "minecraft") == ["h100"] and pod_table(host.kind, host.settings).sandboxes == ("minecraft",)
    blobs = FileBlobStore(tmp_path / "blobs")
    sources = await sources_of(cluster, {"minecraft", "elsewhere"}, blobs)
    (source,) = sources.values()
    assert (source.kind, source.provider, source.python) == ("minecraft", "minecraft_team.worlds:worlds", "3.13")
    assert source.settings == {"heap": "1G", "cache": "/workspace/minecraft"}  # (the section's, under on_pods')
    assert [(each.name, each.extras) for each in source.projects] == [("minecraft-team", ()), ("rollout", ("http",))]
    for project in source.projects:  # (each a zip in the pods' store, packed as an imported environment is)
        names = zipfile.ZipFile(
            io.BytesIO(await blobs.read(BlobReference.model_validate(dict(project.blob))))
        ).namelist()
        assert "pyproject.toml" in names and not [name for name in names if "node_modules" in name]
    assert any(each.startswith("pyyaml==") for each in source.pins)  # (minecraft-team's, at the platform's)
    assert any(each.startswith("uvicorn==") for each in source.pins)  # (what serves the pool)
    assert (await sources_of(cluster, {"minecraft"}, blobs))["minecraft"].digest == source.digest
    needs = needs_of(RunSettings(SETTINGS), cluster)
    (given,) = with_sandboxes(needs, cluster, {"minecraft": source.to_json()})
    assert given.settings["sandboxes"] == {"minecraft": source.to_json()} and given.settings["model"]
    assert SandboxSource.from_json(source.to_json()) == source


class Versions:
    """Published versions, by `NAME@VERSION`."""

    def __init__(self, *versions: EnvironmentVersion) -> None:
        self.versions = {each.reference: each for each in versions}

    async def get(self, reference: str) -> EnvironmentVersion | None:
        return self.versions.get(reference)


async def test_a_published_versions_source_ships_its_zip_and_every_project_it_needs_and_pins_the_rest(
    tmp_path: Path,
) -> None:
    published, pods = FileBlobStore(tmp_path / "published"), FileBlobStore(tmp_path / "pods")
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.writestr("pyproject.toml", '[project]\nname = "boxes-world"\nversion = "0"\n')
    blob = (await published.put(zipped.getvalue(), "application/zip")).model_dump(mode="json")
    version = EnvironmentVersion(
        name="boxes-world", version="abc", source="https://example.com/boxes.git", ref=None, commit="0",
        subdirectory="",
        entry_point="boxes_world.environment:environment", blob=blob, runtime_env={},
        dependencies=("rollout", "minecraft-team", "pyyaml>=6"),
    )  # fmt: skip
    text = SERVED.replace('provider = "minecraft_team.worlds:worlds"', 'provider = "boxes_world.worlds:worlds"')
    cluster = cluster_of(host=text.replace("on_pods = { ", 'on_pods = { version = "boxes-world@abc", '))
    (source,) = (await sources_of(cluster, {"minecraft"}, pods, cast(Any, Versions(version)),
                                  published=published)).values()  # fmt: skip
    assert source.provider == "boxes_world.worlds:worlds"  # (the platform holds no such module: never imported)
    names = [each.name for each in source.projects]
    assert names == ["minecraft-team", "rollout", "boxes-world"]  # (minecraft-team shipped, never fetched by name)
    code = next(each for each in source.projects if each.name == "boxes-world")
    assert code.sha256 == blob["sha256"] and await pods.read(BlobReference.model_validate(dict(code.blob)))
    assert any(each.startswith("pyyaml==") for each in source.pins)
    unknown = dataclasses.replace(version, version="def", dependencies=("rollout", "no-such-distribution-here"))
    with pytest.raises(ValueError, match="no-such-distribution-here is needed, and the platform holds it neither"):
        await sources_of(cluster_of(host=text.replace("on_pods = { ", 'on_pods = { version = "boxes-world@def", ')),
                         {"minecraft"}, pods, cast(Any, Versions(unknown)), published=published)  # fmt: skip


def test_only_a_distribution_installed_from_a_directory_is_a_project_of_the_platforms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rollout_train.pods import sources

    said: dict[str, dict[str, object]] = {
        "from-a-directory": {"url": f"file://{tmp_path}", "dir_info": {"editable": True}},
        "from-a-wheel": {"url": f"file://{tmp_path}/a-0-py3-none-any.whl", "archive_info": {}},
    }

    class Installed:
        def __init__(self, name: str) -> None:
            self.name = name

        def read_text(self, file: str) -> str:
            return json.dumps(said[self.name])

    monkeypatch.setattr(sources.metadata, "distribution", Installed)
    assert sources._directory_of("from-a-directory") == tmp_path  # pyright: ignore[reportPrivateUsage]
    assert sources._directory_of("from-a-wheel") is None  # pyright: ignore[reportPrivateUsage]


def test_a_provider_in_no_project_of_its_own_is_not_packed() -> None:
    with pytest.raises(ValueError, match="in no project the platform holds"):  # (the workspace's root is no project)
        local_projects("tests.rollout_train.pods.sandbox_kinds:boxes")
    projects, _ = local_projects("minecraft_horizons.worlds:worlds")
    assert [each.name for each in projects] == ["minecraft-horizons", "minecraft-team", "rollout"]


def test_sandboxes_whose_specs_name_model_slots_are_not_served_from_pods() -> None:
    harnessed = dataclasses.replace(ENVIRONMENT, sandboxes=frozenset({"minecraft"}), slotted=frozenset({"minecraft"}))
    refused = [each.reason for each in check(RunSettings(SETTINGS), cluster_of(host=SERVED), harnessed)
               if each.refuses and "model slots" in each.reason]  # fmt: skip
    assert refused == [
        "gridworld.environment:environment's sandboxes of kind minecraft name model slots: a harness inside reaches "
        "its model at the run's gateway, which a pod cannot reach, so they are not served from pods "
        "([sandboxes.minecraft] on_pods)"
    ]
