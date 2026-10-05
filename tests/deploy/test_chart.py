"""The Helm chart (deploy/chart/rollout), rendered: its cluster config and profiles read as the code reads them and name
the cluster's stores; every container says what it needs and the most memory it may take; every volume is of the
class `storageClass` names. Skipped where helm is not
installed."""

import shutil
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from rollout_train.cluster import parsed
from rollout_train.profile import Profile

ROOT = Path(__file__).resolve().parent.parent.parent
CHART = ROOT / "deploy" / "chart" / "rollout"
HELM = shutil.which("helm") or next((str(each) for each in [Path.home() / ".local/bin/helm"] if each.exists()), None)
LEDGER = "postgresql://rollout@postgres.rollout:5432/rollout"
BLOBS = {"kind": "rollout_s3:S3BlobStore", "bucket": "rollout-blobs", "prefix": "blobs/"}

pytestmark = pytest.mark.skipif(HELM is None, reason="helm is not installed")


def render(*options: str) -> list[dict[str, Any]]:
    import subprocess

    assert HELM is not None
    out = subprocess.run(
        [HELM, "template", "rollout", str(CHART), "--namespace", "rollout", *options],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [each for each in yaml.safe_load_all(out) if each]


@pytest.fixture(scope="module")
def rendered() -> list[dict[str, Any]]:
    return render()


def config_of(rendered: list[dict[str, Any]]) -> dict[str, str]:
    (config,) = [each for each in rendered if each["kind"] == "ConfigMap" and each["metadata"]["name"] == "rollout"]
    return config["data"]


def test_the_cluster_config_reads_and_names_the_stores(rendered: list[dict[str, Any]]) -> None:
    cluster = parsed(tomllib.loads(config_of(rendered)["cluster.toml"]))
    assert cluster.name == "k3s" and cluster.ledger.url == LEDGER and cluster.ledger.url_secret is None
    assert cluster.blobs.kind == BLOBS["kind"] and dict(cluster.blobs.settings) == {
        "bucket": "rollout-blobs",
        "prefix": "blobs/",
    }
    assert (
        cluster.gateway.url == "http://gateway.rollout:8900" and cluster.ray.jobs == "http://ray-head-svc.rollout:8265"
    )
    assert set(cluster.environments) == {
        "minecraft_team.environment:environment",
        "gridworld.environment:environment",
        "rollout_verifiers.environments:gsm8k",
    }


def test_every_profile_reads_and_names_the_stores(rendered: list[dict[str, Any]], tmp_path: Path) -> None:
    profiles = {key: text for key, text in config_of(rendered).items() if key.startswith("profiles_")}
    assert set(profiles) == {
        "profiles_gsm8k_gsm8k_tinker.toml",
        "profiles_minecraft_one-gpu.toml",
        "profiles_minecraft_tinker.toml",
        "profiles_gridworld_qwen3-0.6b.toml",
    }
    for key, text in profiles.items():
        path = tmp_path / key
        path.write_text(text)
        profile = Profile.load(path)
        assert profile.ledger == {"kind": "rollout_train.database:DatabaseLedger", "url": LEDGER}, key
        assert profile.blobs == BLOBS, key
        assert str(profile.directory).startswith("/root/.cache/rollout/runs/"), key


def test_every_container_asks_for_what_it_needs_and_is_held_to_a_memory_limit(rendered: list[dict[str, Any]]) -> None:
    def pods(each: dict[str, Any]) -> list[dict[str, Any]]:
        if each["kind"] == "RayCluster":
            spec = each["spec"]
            return [
                spec["headGroupSpec"]["template"]["spec"],
                *(g["template"]["spec"] for g in spec["workerGroupSpecs"]),
            ]
        if each["kind"] in ("Deployment", "StatefulSet", "Job"):
            return [each["spec"]["template"]["spec"]]
        return []

    containers = [container for each in rendered for pod in pods(each) for container in pod["containers"]]
    assert len(containers) == 12  # the stores, the bucket job, Ray (3), the launchers, the gateway, the monitors
    for container in containers:
        assert container["resources"]["requests"]["memory"] and container["resources"]["limits"]["memory"], container[
            "name"
        ]
    (ray,) = [each for each in rendered if each["kind"] == "RayCluster"]
    (gpu,) = [group for group in ray["spec"]["workerGroupSpecs"] if group["groupName"] == "gpu"]
    assert (gpu["minReplicas"], gpu["maxReplicas"], gpu["template"]["spec"]["runtimeClassName"]) == (0, 1, "nvidia")
    assert ray["spec"]["enableInTreeAutoscaling"] is True


def test_every_volume_is_of_the_class_storage_class_names(rendered: list[dict[str, Any]]) -> None:
    def classes(rendered: list[dict[str, Any]]) -> list[str]:
        claims = [each["spec"] for each in rendered if each["kind"] == "PersistentVolumeClaim"]
        claims += [
            template["spec"]
            for each in rendered
            if each["kind"] == "StatefulSet"
            for template in each["spec"]["volumeClaimTemplates"]
        ]
        return [claim["storageClassName"] for claim in claims]

    assert classes(rendered) == ["local-path"] * 3  # the state volume, the ledger's, the blob store's
    retain = yaml.safe_load((ROOT / "deploy" / "k3s" / "storage-class.yaml").read_text())
    assert (retain["provisioner"], retain["reclaimPolicy"]) == ("rancher.io/local-path", "Retain")
    assert classes(render("--set", f"storageClass={retain['metadata']['name']}")) == ["local-path-retain"] * 3


def containers_of(rendered: list[dict[str, Any]], app: str) -> list[dict[str, Any]]:
    return [
        container
        for each in rendered
        if each["kind"] == "Deployment" and each["spec"]["template"]["metadata"]["labels"].get("app") == app
        for container in each["spec"]["template"]["spec"]["containers"]
    ]


def test_a_launcher_offers_the_gridworld_with_its_profile_on_the_gpu(rendered: list[dict[str, Any]]) -> None:
    (gridworld,) = [each for each in containers_of(rendered, "launcher") if "gridworld" in each["command"]]
    command = gridworld["command"]
    assert command[command.index("--profiles") + 1] == "/etc/rollout/profiles/gridworld"
    assert command[command.index("--environment") + 1] == "gridworld.environment:environment"
    assert command[command.index("--gpus") + 1] == "1" and "--ray" in command
