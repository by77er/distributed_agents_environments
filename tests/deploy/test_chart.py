"""The Helm chart (deploy/chart/rollout), rendered: its cluster config reads as the code reads it, names the cluster's
stores and makes each run's job a RayJob from its template, which renders into a run's RayJob; its presets are run
settings that cluster takes; the monitors' account may make RayJobs and, with Kueue, read the queue; every container
says what it needs and the most memory it may take; every volume is of the class `storageClass` names. Skipped where
helm is not installed."""

import shutil
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from rollout_train.cluster import parsed
from rollout_train.launches import TRAIN, Asked, new_launch
from rollout_train.run_settings import RunSettings, flattened
from rollout_train.submitting import rendered as made_from
from rollout_train.validation import check

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


def test_the_cluster_config_reads_names_the_stores_and_makes_runs_rayjobs(rendered: list[dict[str, Any]]) -> None:
    cluster = parsed(tomllib.loads(config_of(rendered)["cluster.toml"]))
    assert cluster.name == "k3s" and cluster.ledger.url == LEDGER and cluster.ledger.url_secret is None
    assert cluster.blobs.kind == BLOBS["kind"] and dict(cluster.blobs.settings) == {
        "bucket": "rollout-blobs",
        "prefix": "blobs/",
    }
    assert (
        cluster.gateway.url == "http://gateway.rollout:8900" and cluster.ray.jobs == "http://ray-head-svc.rollout:8265"
    )
    assert cluster.kubernetes is not None
    assert (cluster.kubernetes.namespace, cluster.kubernetes.rayjob) == ("rollout", "/etc/rollout/rayjob.yaml")
    assert set(cluster.environments) == {
        "minecraft_team.environment:environment",
        "gridworld.environment:environment",
        "rollout_verifiers.environments:gsm8k",
    }
    gsm8k = cluster.environments["rollout_verifiers.environments:gsm8k"]
    assert gsm8k.runs_in == "/opt/rollout/verifiers/bin/python"


def test_a_runs_rayjob_is_made_from_the_charts_template(rendered: list[dict[str, Any]]) -> None:
    template = yaml.safe_load(config_of(rendered)["rayjob.yaml"])
    launch = new_launch(Asked(TRAIN, "team 8", {"environment": "minecraft_team.environment:environment"}), "run_1")
    made = made_from(template, launch, f"python -m rollout_train.jobs {launch.id}", {"env_vars": {}}, "rollout")
    assert made["kind"] == "RayJob" and made["metadata"]["namespace"] == "rollout"
    spec = made["spec"]
    assert spec["shutdownAfterJobFinishes"] is True and spec["backoffLimit"] == 2 and spec["entrypointNumCpus"] == 1
    head = spec["rayClusterSpec"]["headGroupSpec"]
    assert head["rayStartParams"]["num-gpus"] == "1"
    pod = head["template"]["spec"]
    (container,) = pod["containers"]
    assert container["image"] == "localhost:30500/rollout-platform:dev" and pod["runtimeClassName"] == "nvidia"
    assert {"ROLLOUT_CLUSTER", "PGPASSWORD", "AWS_ACCESS_KEY_ID"} <= {each["name"] for each in container["env"]}
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert {each["mountPath"] for each in container["volumeMounts"]} >= {"/etc/rollout", "/root/.cache/rollout"}


def test_every_preset_is_run_settings_the_cluster_takes(rendered: list[dict[str, Any]]) -> None:
    config = config_of(rendered)
    cluster = parsed(tomllib.loads(config["cluster.toml"]))
    presets = {key: text for key, text in config.items() if key.startswith("presets_")}
    assert set(presets) == {
        "presets_minecraft-one-gpu.toml",
        "presets_minecraft-tinker.toml",
        "presets_gridworld-qwen3-0.6b.toml",
        "presets_gsm8k-tinker.toml",
    }
    for key, text in presets.items():
        settings = flattened(tomllib.loads(text))
        kind = "train" if "trainer.provider" in settings else "check"  # (gsm8k-tinker's channels play its evals)
        refused = [each for each in check(RunSettings({**settings, "kind": kind}), cluster) if each.refuses]
        assert refused == [], key
    (job,) = [each for each in rendered if each["kind"] == "Job" and each["metadata"]["name"] == "presets"]
    assert job["metadata"]["annotations"]["helm.sh/hook"] == "post-install,post-upgrade"
    (container,) = job["spec"]["template"]["spec"]["containers"]
    assert container["command"] == ["rollout", "preset", "load", "/etc/rollout/presets", "--cluster"]


def test_the_monitors_may_make_read_and_delete_rayjobs(rendered: list[dict[str, Any]]) -> None:
    (role,) = [each for each in rendered if each["kind"] == "Role" and each["metadata"]["name"] == "monitor"]
    (rayjobs,) = [rule for rule in role["rules"] if rule["resources"] == ["rayjobs"]]
    assert {"create", "get", "list", "delete"} <= set(rayjobs["verbs"]) and rayjobs["apiGroups"] == ["ray.io"]
    (binding,) = [each for each in rendered if each["kind"] == "RoleBinding" and each["metadata"]["name"] == "monitor"]
    assert binding["subjects"] == [{"kind": "ServiceAccount", "name": "monitor", "namespace": "rollout"}]
    monitors = [
        each["spec"]["template"]["spec"]
        for each in rendered
        if each["kind"] == "Deployment" and each["spec"]["template"]["metadata"]["labels"].get("app") == "monitor"
    ]
    assert monitors and all(pod["serviceAccountName"] == "monitor" for pod in monitors)


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
    assert len(containers) == 10  # the stores, the bucket job, the presets job, Ray (3), the ledger, gateway, monitor
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


def test_the_gateway_serves_the_cluster_configs_channels(rendered: list[dict[str, Any]]) -> None:
    (gateway,) = containers_of(rendered, "gateway")
    assert gateway["command"][:3] == ["rollout", "gateway", "--cluster"]


def test_the_ledger_service_serves_the_cluster_configs_ledger_reachable_from_outside_when_asked(
    rendered: list[dict[str, Any]],
) -> None:
    (ledger,) = containers_of(rendered, "ledger")
    assert ledger["command"][:4] == ["rollout", "ledger", "serve", "--cluster"]
    assert {"PGPASSWORD", "ROLLOUT_LEDGER_TOKEN"} <= {each["name"] for each in ledger["env"]}
    cluster = parsed(tomllib.loads(config_of(rendered)["cluster.toml"]))
    assert cluster.ledger.token is not None and cluster.ledger.token.env == "ROLLOUT_LEDGER_TOKEN"
    assert cluster.ledger.public is None and not [each for each in rendered if each["kind"] == "Ingress"
                                                  and each["metadata"]["name"] == "ledger"]  # fmt: skip
    (service,) = [each for each in rendered if each["kind"] == "Service" and each["metadata"]["name"] == "ledger"]
    assert service["spec"]["type"] == "ClusterIP"
    exposed = render("--set", "ledger.public=https://ledger.example.com", "--set", "ledger.ingress.enabled=true",
                     "--set", "ledger.ingress.tlsSecret=ledger-tls")  # fmt: skip
    assert parsed(tomllib.loads(config_of(exposed)["cluster.toml"])).ledger.public == "https://ledger.example.com"
    (ingress,) = [each for each in exposed if each["kind"] == "Ingress" and each["metadata"]["name"] == "ledger"]
    assert ingress["spec"]["tls"][0]["secretName"] == "ledger-tls"
    noded = render("--set", "ledger.service.type=NodePort", "--set", "ledger.service.nodePort=30840")
    (service,) = [each for each in noded if each["kind"] == "Service" and each["metadata"]["name"] == "ledger"]
    assert service["spec"]["type"] == "NodePort" and service["spec"]["ports"][0]["nodePort"] == 30840


def test_every_monitor_asks_for_runs_and_imports_with_the_cluster_config_and_reaches_ray_with_its_token(
    rendered: list[dict[str, Any]],
) -> None:
    monitors = containers_of(rendered, "monitor")
    assert len(monitors) == 1
    for monitor in monitors:
        assert monitor["command"][-1] == "--cluster"
        names = {each["name"] for each in monitor["env"]}
        assert {"ROLLOUT_CLUSTER", "RAY_AUTH_MODE", "RAY_AUTH_TOKEN", "AWS_ACCESS_KEY_ID"} <= names


def test_with_kueue_runs_are_admitted_whole_through_a_queue_the_chart_makes(rendered: list[dict[str, Any]]) -> None:
    assert not [each for each in rendered if each["apiVersion"].startswith("kueue.x-k8s.io")]  # (off by default)
    on = render("--set", "kueue.enabled=true")
    kinds = {each["kind"]: each for each in on if each["apiVersion"] == "kueue.x-k8s.io/v1beta2"}
    assert set(kinds) == {"ResourceFlavor", "ClusterQueue", "LocalQueue"}
    queue = kinds["ClusterQueue"]["spec"]
    assert queue["namespaceSelector"] == {"matchLabels": {"kubernetes.io/metadata.name": "rollout"}}
    (group,) = queue["resourceGroups"]
    assert group["coveredResources"] == ["cpu", "memory", "nvidia.com/gpu"]
    (flavor,) = group["flavors"]
    assert flavor["name"] == kinds["ResourceFlavor"]["metadata"]["name"]
    assert {each["name"]: each["nominalQuota"] for each in flavor["resources"]} == {
        "cpu": "12",
        "memory": "16Gi",
        "nvidia.com/gpu": "1",
    }
    local = kinds["LocalQueue"]
    assert local["metadata"]["namespace"] == "rollout"
    assert local["spec"]["clusterQueue"] == kinds["ClusterQueue"]["metadata"]["name"]
    cluster = parsed(tomllib.loads(config_of(on)["cluster.toml"]))
    assert cluster.kubernetes is not None and cluster.kubernetes.queue == local["metadata"]["name"]
    assert cluster.capacity is not None
    assert (cluster.capacity.cpus, cluster.capacity.memory_gib, cluster.capacity.gpus) == (12, 16, 1)
    (role,) = [each for each in on if each["kind"] == "Role" and each["metadata"]["name"] == "monitor"]
    (workloads,) = [rule for rule in role["rules"] if rule["resources"] == ["workloads"]]
    assert workloads["apiGroups"] == ["kueue.x-k8s.io"] and "list" in workloads["verbs"]
    (local_queue,) = [rule for rule in role["rules"] if rule["resources"] == ["localqueues"]]
    assert local_queue["resourceNames"] == [local["metadata"]["name"]] and local_queue["verbs"] == ["get"]
    for key, text in config_of(on).items():  # (every preset fits the queue's quota)
        if key.startswith("presets_"):
            settings = flattened(tomllib.loads(text))
            kind = "train" if "trainer.provider" in settings else "check"
            assert [each for each in check(RunSettings({**settings, "kind": kind}), cluster) if each.refuses] == [], key


def test_with_kueue_the_monitors_may_read_the_queue_and_its_pending_order(rendered: list[dict[str, Any]]) -> None:
    def cluster_wide(rendered: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [each for each in rendered if each["kind"] in ("ClusterRole", "ClusterRoleBinding")]

    assert cluster_wide(rendered) == []  # (without Kueue the monitors read nothing outside their namespace)
    (role,) = [each for each in rendered if each["kind"] == "Role" and each["metadata"]["name"] == "monitor"]
    assert not [rule for rule in role["rules"] if "kueue.x-k8s.io" in rule["apiGroups"]]
    on = render("--set", "kueue.enabled=true", "--set", "kueue.clusterQueue=runs-queue")
    roles = {each["kind"]: each for each in cluster_wide(on)}
    assert set(roles) == {"ClusterRole", "ClusterRoleBinding"}
    assert {(tuple(rule["apiGroups"]), tuple(rule["resources"]), tuple(rule["resourceNames"]), tuple(rule["verbs"]))
            for rule in roles["ClusterRole"]["rules"]} == {
        (("kueue.x-k8s.io",), ("clusterqueues",), ("runs-queue",), ("get",)),
        (("visibility.kueue.x-k8s.io",), ("clusterqueues/pendingworkloads",), ("runs-queue",), ("get",)),
    }  # fmt: skip
    binding = roles["ClusterRoleBinding"]
    assert binding["subjects"] == [{"kind": "ServiceAccount", "name": "monitor", "namespace": "rollout"}]
    assert binding["roleRef"] == {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole",
                                  "name": roles["ClusterRole"]["metadata"]["name"]}  # fmt: skip


def provider_keys(env: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The hosted APIs' keys among a container's environment, each by the Secret it is read from."""
    return {each["name"]: each["valueFrom"]["secretKeyRef"] for each in env if each["name"].endswith("_API_KEY")
            and each["name"] != "TINKER_API_KEY"}  # fmt: skip


def test_the_hosted_apis_keys_come_from_a_secret_in_the_gateway_and_runs_and_never_the_config(
    rendered: list[dict[str, Any]],
) -> None:
    config = config_of(rendered)
    cluster = parsed(tomllib.loads(config["cluster.toml"]))
    hosted = {name: each for name, each in cluster.inference.items() if each.kind == "api"}
    assert {name: (each.allocation, str(each.secrets["api_key"])) for name, each in hosted.items()} == {
        "openai": ("metered", "$OPENAI_API_KEY"), "anthropic": ("metered", "$ANTHROPIC_API_KEY"),
    }  # fmt: skip
    assert all(offer.cost.get("input") and offer.cost.get("output") for each in hosted.values()
               for offer in each.models.values())  # fmt: skip
    wanted = {
        name: {"name": "providers", "key": name, "optional": True} for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    }
    (gateway,) = containers_of(rendered, "gateway")
    assert provider_keys(gateway["env"]) == wanted
    template = yaml.safe_load(config["rayjob.yaml"])
    (head,) = template["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"]
    assert provider_keys(head["env"]) == wanted  # (each run's job: its driver samples its channels on hosted APIs)
    (monitor,) = containers_of(rendered, "monitor")
    assert provider_keys(monitor["env"]) == {}  # (only what samples them has the keys)
    renamed = render("--set", "secrets.providers=api-keys")
    (gateway,) = containers_of(renamed, "gateway")
    assert {each["name"] for each in provider_keys(gateway["env"]).values()} == {"api-keys"}
