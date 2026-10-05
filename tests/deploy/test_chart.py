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
from rollout_train.demand import SUBMITTER
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
    submitter = spec["submitterPodTemplate"]["spec"]
    (submits,) = submitter["containers"]
    assert submitter["restartPolicy"] == "Never" and "command" not in submits  # (KubeRay gives it `ray job submit`)
    requests = submits["resources"]["requests"]
    assert (float(requests["cpu"].removesuffix("m")) / 1000, int(requests["memory"].removesuffix("Mi")) / 1024) == (
        SUBMITTER.cpus,
        SUBMITTER.memory_gib,
    )  # (what Kueue counts of it, as a run's demand says)


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
    assert len(containers) == 11  # the stores, the two jobs, Ray (3), the ledger, gateway, monitor, a pool
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


def test_the_minecraft_worlds_are_served_from_a_pod_of_their_own_that_runs_reach_at_its_url(
    rendered: list[dict[str, Any]],
) -> None:
    cluster = parsed(tomllib.loads(config_of(rendered)["cluster.toml"]))
    pool = cluster.sandboxes["minecraft"]
    assert (pool.provider, pool.size, pool.url) == (
        "minecraft_team.worlds:worlds",
        4,
        "http://sandboxes-minecraft.rollout:8710",
    )
    (deployment,) = [each for each in rendered if each["kind"] == "Deployment"
                     and each["metadata"]["name"] == "sandboxes-minecraft"]  # fmt: skip
    assert deployment["spec"]["replicas"] == 1 and deployment["spec"]["strategy"] == {"type": "Recreate"}
    (container,) = deployment["spec"]["template"]["spec"]["containers"]
    assert container["command"] == ["rollout", "pool", "--kind", "minecraft", "--cluster", "--host", "0.0.0.0",
                                    "--port", "8710"]  # fmt: skip
    resources = container["resources"]
    assert (resources["requests"]["memory"], resources["limits"]["memory"]) == ("7680Mi", "10Gi")  # (1.75 GiB a world)
    (service,) = [each for each in rendered if each["kind"] == "Service"
                  and each["metadata"]["name"] == "sandboxes-minecraft"]  # fmt: skip
    assert service["spec"]["selector"] == deployment["spec"]["selector"]["matchLabels"]
    off = render("--set", "sandboxes.minecraft.enabled=false")
    assert "minecraft" not in parsed(tomllib.loads(config_of(off)["cluster.toml"])).sandboxes
    assert not [each for each in off if each["metadata"]["name"] == "sandboxes-minecraft"]


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
        assert "--cluster" in monitor["command"]
        names = {each["name"] for each in monitor["env"]}
        assert {"ROLLOUT_CLUSTER", "RAY_AUTH_MODE", "RAY_AUTH_TOKEN", "AWS_ACCESS_KEY_ID"} <= names
        assert {"R2_WRITER_ACCESS_KEY_ID", "R2_WRITER_SECRET_ACCESS_KEY"} <= names  # (a second store's, optional)
        assert not {"R2_READER_ACCESS_KEY_ID", "TINKER_API_KEY", "ROLLOUT_LEDGER_TOKEN"} & names  # (it reads none)


def test_the_monitors_ask_for_a_token_the_chart_makes_and_answer_only_under_their_names(
    rendered: list[dict[str, Any]],
) -> None:
    (secret,) = [each for each in rendered if each["kind"] == "Secret" and each["metadata"]["name"] == "monitor-token"]
    assert secret["metadata"]["annotations"]["helm.sh/resource-policy"] == "keep"  # (uninstalling leaves the token)
    assert len(secret["data"]["ROLLOUT_MONITOR_TOKEN"]) >= 43 and set(secret["data"]) == {"ROLLOUT_MONITOR_TOKEN"}
    cluster = parsed(tomllib.loads(config_of(rendered)["cluster.toml"]))
    assert cluster.monitor.token is not None and cluster.monitor.token.env == "ROLLOUT_MONITOR_TOKEN"
    (monitor,) = containers_of(rendered, "monitor")
    env = {each["name"]: each for each in monitor["env"]}
    assert env["ROLLOUT_MONITOR_TOKEN"]["valueFrom"]["secretKeyRef"] == {"name": "monitor-token",
                                                                         "key": "ROLLOUT_MONITOR_TOKEN"}  # fmt: skip
    command = monitor["command"]
    allowed = [command[at + 1] for at, each in enumerate(command) if each == "--allow-host"]
    assert allowed == ["monitor-main", "monitor-main.rollout", "monitor-main.rollout.svc",
                       "monitor-main.rollout.svc.cluster.local"]  # fmt: skip
    ingresses = {each["metadata"]["name"] for each in rendered if each["kind"] == "Ingress"}
    assert "monitor-main" not in ingresses  # (off unless asked for: the page is opened through a port-forward)
    on = render("--set", "monitors.main.ingress=true", "--set", "monitors.main.hosts={monitor.example.com}")
    (ingress,) = [each for each in on if each["kind"] == "Ingress" and each["metadata"]["name"] == "monitor-main"]
    assert ingress["spec"]["rules"][0]["host"] == "monitor.localhost"
    (monitor,) = containers_of(on, "monitor")
    command = monitor["command"]
    allowed = [command[at + 1] for at, each in enumerate(command) if each == "--allow-host"]
    assert allowed[-2:] == ["monitor.localhost", "monitor.example.com"]
    named = render("--set", "secrets.monitor=page-token")
    assert [each["metadata"]["name"] for each in named if each["kind"] == "Secret"] == ["page-token"]


def workloads(rendered: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Every pod the chart makes, by its workload's name (`ray/GROUP` for the long-lived Ray cluster's, `rayjob/head`
    and `rayjob/submitter` for each run's, from files/rayjob.yaml), as its pod spec."""
    found: dict[str, dict[str, Any]] = {}
    for each in rendered:
        kind, name = each["kind"], each["metadata"]["name"]
        if kind in ("Deployment", "StatefulSet", "Job"):
            found[name] = each["spec"]["template"]
        elif kind == "CronJob":
            found[name] = each["spec"]["jobTemplate"]["spec"]["template"]
        elif kind == "RayCluster":
            found["ray/head"] = each["spec"]["headGroupSpec"]["template"]
            for group in each["spec"]["workerGroupSpecs"]:
                found[f"ray/{group['groupName']}"] = group["template"]
    job = yaml.safe_load(config_of(rendered)["rayjob.yaml"])["spec"]
    found["rayjob/head"] = job["rayClusterSpec"]["headGroupSpec"]["template"]
    found["rayjob/submitter"] = job["submitterPodTemplate"]
    return found


def secrets_of(pod: dict[str, Any]) -> set[str]:
    """The Secrets a pod references: `NAME:KEY` for each variable read from one, `NAME/` for each mounted."""
    spec: Any = pod["spec"]
    env: list[Any] = [each for container in spec["containers"] for each in container.get("env") or list[Any]()]
    references: list[Any] = [(each.get("valueFrom") or dict[str, Any]()).get("secretKeyRef") for each in env]
    found = {f"{ref['name']}:{ref['key']}" for ref in references if ref}
    volumes: list[Any] = spec.get("volumes") or []
    found |= {f"{each['secret']['secretName']}/" for each in volumes if "secret" in each}
    return found


R2_WRITER = {"r2:WRITER_ACCESS_KEY_ID", "r2:WRITER_SECRET_ACCESS_KEY"}
R2 = R2_WRITER | {"r2:READER_ACCESS_KEY_ID", "r2:READER_SECRET_ACCESS_KEY"}
STORE_KEYS = {"stores:ROOT_ACCESS_KEY_ID", "stores:ROOT_SECRET_ACCESS_KEY"}
DATABASE = {"stores:POSTGRES_PASSWORD"}
PROVIDERS = {"providers:OPENAI_API_KEY", "providers:ANTHROPIC_API_KEY"}
GIVEN: dict[str, set[str]] = {
    "ledger": DATABASE | {"ledger:ROLLOUT_LEDGER_TOKEN"},
    "gateway": DATABASE | STORE_KEYS | PROVIDERS | {"gateway-keys/", "gateway-tls/"},
    "monitor-main": DATABASE | STORE_KEYS | R2_WRITER | {"ray:auth_token", "monitor-token:ROLLOUT_MONITOR_TOKEN"},
    "sandboxes-minecraft": DATABASE,
    "presets": DATABASE,
    "pods-reaper": DATABASE | {"runpod:RUNPOD_API_KEY", "step-ca/"},
    "ray/head": set(),
    "ray/gpu": set(),
    "ray/cpu": set(),
    "rayjob/head": DATABASE | STORE_KEYS | R2 | PROVIDERS | {
        "tinker:TINKER_API_KEY", "ledger:ROLLOUT_LEDGER_TOKEN", "runpod:RUNPOD_API_KEY", "gateway-keys/", "tinker/",
        "step-ca/", "gateway-tls/",
    },
    "rayjob/submitter": set(),
    "postgres": {"stores:POSTGRES_PASSWORD"},
    "s3": STORE_KEYS,
    "buckets": STORE_KEYS,
    "step-ca": {"step-ca-password:password"},
    "pki-publish": {"step-ca-password:password"},
    "pki-publish-now": {"step-ca-password:password"},
    "tunnel": {"tunnel:token"},
}  # fmt: skip
"""What each workload is given, Secret by Secret: what its code reads (`Cluster.secrets_of` its role), no more."""


def test_each_role_is_given_only_the_secrets_its_code_reads() -> None:
    every = render("--set", "runpod.reaper=true", "--set", "stepCa.enabled=true", "--set", "tunnel.enabled=true")
    given = {name: secrets_of(pod) for name, pod in workloads(every).items()}
    assert given == GIVEN
    for name in ("ray/head", "ray/gpu", "ray/cpu"):  # (where code imported from anywhere is checked)
        assert not workloads(every)[name]["spec"].get("volumes"), name
    cluster = parsed(tomllib.loads(config_of(every)["cluster.toml"]))
    roles = {"gateway": "gateway", "monitor-main": "monitor", "ledger": "ledger", "sandboxes-minecraft": "pool",
             "pods-reaper": "reaper", "rayjob/head": "run"}  # fmt: skip
    pods = workloads(every)
    for workload, role in roles.items():
        spec: Any = pods[workload]["spec"]
        containers: list[Any] = spec["containers"]
        env: list[Any] = [each for container in containers for each in container.get("env") or list[Any]()]
        mounts: list[Any] = [each for container in containers for each in container.get("volumeMounts") or list[Any]()]
        names: set[str] = {each["name"] for each in env}
        paths: set[str] = {each["mountPath"] for each in mounts}
        for where, secret in cluster.secrets_of(role).items():  # (each secret the role reads, it is given)
            if secret.env is not None:
                assert secret.env in names, (workload, where)
            else:
                assert any(str(secret.file).startswith(path) for path in paths), (workload, where)


def test_nothing_reaches_a_pod_but_the_roles_that_use_it() -> None:
    every = render("--set", "stepCa.enabled=true", "--set", "tunnel.enabled=true")
    policies = {each["metadata"]["name"]: each["spec"] for each in every if each["kind"] == "NetworkPolicy"}
    assert policies["default-deny-ingress"] == {"podSelector": {}, "policyTypes": ["Ingress"]}

    def sources(name: str) -> set[str]:
        found: set[str] = set()
        for rule in policies[name].get("ingress", []):
            for peer in rule.get("from", []):
                selector = peer.get("podSelector", {})
                if "namespaceSelector" in peer:
                    found.add(peer["namespaceSelector"]["matchLabels"]["kubernetes.io/metadata.name"])
                elif "ipBlock" in peer:
                    found.add(peer["ipBlock"]["cidr"])
                else:
                    labels = selector.get("matchLabels", {})
                    found |= (
                        {labels.get("app") or f"ray.io/cluster={labels.get('ray.io/cluster')}"} if labels else set()
                    )
                    for expression in selector.get("matchExpressions", []):
                        found |= set(expression["values"])
        return found

    assert sources("postgres") == {"gateway", "monitor", "ledger", "sandboxes", "presets", "pods-reaper", "run"}
    assert sources("s3") == {"gateway", "monitor", "run", "buckets"}
    assert sources("gateway") == {"run", "sandboxes", "kube-system"}  # (and its Ingress, through the controller)
    assert sources("ledger") == {"tunnel"}
    assert sources("step-ca") == {"tunnel", "pki", "run", "pods-reaper"}
    assert sources("monitors") == {"monitor"}  # (people reach it through a port-forward, which no policy stops)
    assert sources("sandboxes-minecraft") == {"run"}
    assert sources("ray") == {"ray.io/cluster=ray", "kuberay", "monitor", "kube-system"}
    assert sources("runs") == {"run", "run-submitter", "gateway", "kuberay"}
    labelled = {name: pod["metadata"]["labels"] for name, pod in workloads(every).items()}
    selected = [policy["podSelector"] for name, policy in policies.items() if name != "default-deny-ingress"]

    def matches(selector: dict[str, Any], labels: dict[str, str]) -> bool:
        wanted = selector.get("matchLabels", {})
        expressions = selector.get("matchExpressions", [])
        return all(labels.get(key) == value for key, value in wanted.items()) and all(
            labels.get(each["key"]) in each["values"] for each in expressions
        )

    reached = {"postgres", "s3", "gateway", "ledger", "step-ca", "monitor-main", "sandboxes-minecraft", "rayjob/head",
               "rayjob/submitter"}  # fmt: skip
    for name in reached:  # (each pod something reaches is let reached by some policy)
        assert any(matches(selector, labelled[name]) for selector in selected), name
    egress = {name: spec for name, spec in policies.items() if "Egress" in spec["policyTypes"]}
    assert set(egress) == {"ray-workers-egress", "sandboxes-minecraft-egress"}
    for spec in egress.values():
        (internet,) = [rule for rule in spec["egress"] if "ipBlock" in rule["to"][0]]
        assert internet["to"][0]["ipBlock"] == {"cidr": "0.0.0.0/0", "except": [
            "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "169.254.0.0/16"]}  # fmt: skip
    assert egress["ray-workers-egress"]["podSelector"]["matchLabels"] == {"ray.io/cluster": "ray",
                                                                          "ray.io/node-type": "worker"}  # fmt: skip
    opened = render("--set", "monitors.main.ingress=true", "--set", "ledger.ingress.enabled=true",
                    "--set", "ledger.service.type=NodePort")  # fmt: skip
    policies = {each["metadata"]["name"]: each["spec"] for each in opened if each["kind"] == "NetworkPolicy"}
    assert sources("monitors") == {"monitor", "kube-system"} and sources("ledger") == {"0.0.0.0/0"}
    assert not [each for each in render("--set", "networkPolicies.enabled=false") if each["kind"] == "NetworkPolicy"]
    plain = render("--set", "networkPolicies.egress=false")
    assert not [each for each in plain if each["kind"] == "NetworkPolicy" and "Egress" in each["spec"]["policyTypes"]]


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
        "memory": "9Gi",
        "nvidia.com/gpu": "1",
    }
    local = kinds["LocalQueue"]
    assert local["metadata"]["namespace"] == "rollout"
    assert local["spec"]["clusterQueue"] == kinds["ClusterQueue"]["metadata"]["name"]
    cluster = parsed(tomllib.loads(config_of(on)["cluster.toml"]))
    assert cluster.kubernetes is not None and cluster.kubernetes.queue == local["metadata"]["name"]
    assert cluster.capacity is not None
    assert (cluster.capacity.cpus, cluster.capacity.memory_gib, cluster.capacity.gpus) == (12, 9, 1)
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
            and each["name"] not in ("TINKER_API_KEY", "RUNPOD_API_KEY")}  # fmt: skip


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


def test_the_pods_reaper_runs_every_minute_when_asked_with_runpods_key(rendered: list[dict[str, Any]]) -> None:
    assert not [each for each in rendered if each["kind"] == "CronJob"]  # (off by default)
    (reaper,) = [each for each in render("--set", "runpod.reaper=true") if each["kind"] == "CronJob"]
    assert reaper["spec"]["schedule"] == "* * * * *" and reaper["spec"]["concurrencyPolicy"] == "Forbid"
    (container,) = reaper["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"]
    assert container["command"] == ["rollout", "pods", "reap", "--cluster"]
    assert {"RUNPOD_API_KEY", "ROLLOUT_CLUSTER", "PGPASSWORD"} <= {each["name"] for each in container["env"]}
    template = yaml.safe_load(config_of(rendered)["rayjob.yaml"])
    (head,) = template["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"]
    assert "RUNPOD_API_KEY" in {each["name"] for each in head["env"]}  # (a run's driver leases its pods)
    mounted = {each["mountPath"] for each in head["volumeMounts"]}
    assert {"/etc/rollout-secrets/step-ca", "/etc/rollout-secrets/tls"} <= mounted


def test_step_ca_and_the_publishing_of_its_root_key_and_the_gateways_certificate_when_asked(
    rendered: list[dict[str, Any]],
) -> None:
    assert not [each for each in rendered if each["metadata"]["name"] in ("step-ca", "pki-publish", "tunnel")]
    on = render("--set", "stepCa.enabled=true", "--set", "stepCa.dnsNames={ca.example.com}")
    (ca,) = [each for each in on if each["kind"] == "StatefulSet" and each["metadata"]["name"] == "step-ca"]
    (container,) = ca["spec"]["template"]["spec"]["containers"]
    env = {each["name"]: each for each in container["env"]}
    assert env["DOCKER_STEPCA_INIT_PROVISIONER_NAME"]["value"] == "launcher"
    assert "ca.example.com" in env["DOCKER_STEPCA_INIT_DNS_NAMES"]["value"].split(",")
    assert env["DOCKER_STEPCA_INIT_PASSWORD"]["valueFrom"]["secretKeyRef"] == {
        "name": "step-ca-password",
        "key": "password",
    }
    (cron,) = [each for each in on if each["kind"] == "CronJob" and each["metadata"]["name"] == "pki-publish"]
    (now,) = [each for each in on if each["kind"] == "Job" and each["metadata"]["name"] == "pki-publish-now"]
    for pod in (cron["spec"]["jobTemplate"]["spec"]["template"]["spec"], now["spec"]["template"]["spec"]):
        (publish,) = pod["containers"]
        assert (
            publish["command"][:3] == ["rollout", "pki", "publish"]
            and "https://step-ca.rollout:9000" in publish["command"]
        )
        assert pod["serviceAccountName"] == "pki" and pod["volumes"][0]["persistentVolumeClaim"]["readOnly"] is True
    (role,) = [each for each in on if each["kind"] == "Role" and each["metadata"]["name"] == "pki"]
    assert role["rules"][0]["resourceNames"] == ["step-ca", "gateway-tls"]


def test_a_tunnel_carries_the_ledger_and_step_ca_hostnames_when_asked() -> None:
    on = render("--set", "tunnel.enabled=true", "--set", "tunnel.hostnames.ledger=ledger.example.com",
                "--set", "tunnel.hostnames.stepCa=ca.example.com")  # fmt: skip
    (config,) = [each for each in on if each["kind"] == "ConfigMap" and each["metadata"]["name"] == "tunnel"]
    rules = yaml.safe_load(config["data"]["config.yaml"])["ingress"]
    assert rules[0] == {"hostname": "ledger.example.com", "service": "http://ledger.rollout:8840"}
    assert rules[1]["service"] == "https://step-ca.rollout:9000" and rules[-1] == {"service": "http_status:404"}
    (tunnel,) = [each for each in on if each["kind"] == "Deployment" and each["metadata"]["name"] == "tunnel"]
    (container,) = tunnel["spec"]["template"]["spec"]["containers"]
    assert container["env"][0]["valueFrom"]["secretKeyRef"] == {"name": "tunnel", "key": "token"}
