"""The pods' images, read as files: Envoy passes on only the requests each pod's role needs, from the gateway's
certificate alone, with limits; the certificates' paths agree across Envoy and the script that writes them; the images
are built from the workspace's vLLM and PyTorch, and what they run is checked for their Python; every variable the pods
read is documented. (CI runs `envoy --mode validate` on both configurations and builds the images.)"""

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from rollout_train.pods import GATEWAY_IDENTITY
from rollout_train.pods.environment import PORT

ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES = ROOT / "deploy" / "images"
ROLES = ["inference", "trainer", "host"]
ALLOWED = {
    "inference": {("POST", "/v1/completions"): "vllm", ("GET", "/v1/models"): "vllm"},
    "trainer": {
        ("POST", "/v1/steps"): "trainer",
        ("GET", "/v1/steps/kmnopqrstuvwxyzk"): "trainer",
        ("GET", "/v1/trainer"): "trainer",
    },
}
ALLOWED["host"] = {**ALLOWED["inference"], **ALLOWED["trainer"]}
REFUSED = [
    ("GET", "/v1/completions"),
    ("POST", "/v1/models"),
    ("POST", "/v1/load_lora_adapter"),
    ("POST", "/v1/unload_lora_adapter"),
    ("POST", "/v1/chat/completions"),
    ("POST", "/sleep"),
    ("POST", "/wake_up"),
    ("GET", "/metrics"),
    ("GET", "/healthz"),
    ("GET", "/readyz"),
    ("GET", "/v1/steps"),
    ("POST", "/v1/steps/kmnopqrstuvwxyzk"),
    ("GET", "/v1/steps/kmnop/qrstuvwxyzk"),
    ("GET", "/v1/steps/"),
    ("DELETE", "/v1/steps/kmnopqrstuvwxyzk"),
    ("GET", "/v1/completions/x"),
    ("GET", "/"),
]


def envoy(role: str) -> dict[str, Any]:
    return yaml.safe_load((IMAGES / role / "envoy.yaml").read_text())


def manager(config: dict[str, Any]) -> dict[str, Any]:
    (listener,) = config["static_resources"]["listeners"]
    (chain,) = listener["filter_chains"]
    (http,) = chain["filters"]
    assert http["name"] == "envoy.filters.network.http_connection_manager"
    return http["typed_config"]


def routed(config: dict[str, Any], method: str, path: str) -> str | int:
    """Where Envoy sends a request, by the first route it matches (as Envoy does): a cluster's name, or the status of a
    direct response."""
    (host,) = manager(config)["route_config"]["virtual_hosts"]
    for route in host["routes"]:
        match = route["match"]
        if "path" in match and path != match["path"]:
            continue
        if "prefix" in match and not path.startswith(match["prefix"]):
            continue
        if "safe_regex" in match and not re.fullmatch(match["safe_regex"]["regex"], path):
            continue
        methods = [each["string_match"]["exact"] for each in match.get("headers", []) if each["name"] == ":method"]
        if methods and method not in methods:
            continue
        return route["route"]["cluster"] if "route" in route else route["direct_response"]["status"]
    return 404  # (no route: Envoy answers 404 itself)


@pytest.mark.parametrize("role", ROLES)
def test_only_the_requests_the_role_needs_are_passed_on(role: str) -> None:
    config = envoy(role)
    for (method, path), cluster in ALLOWED[role].items():
        assert routed(config, method, path) == cluster, (method, path)
    for method, path in REFUSED:
        if (method, path) not in ALLOWED[role]:
            assert routed(config, method, path) == 404, (method, path)
    (host,) = manager(config)["route_config"]["virtual_hosts"]
    assert host["routes"][-1] == {"name": "refused", "match": {"prefix": "/"}, "direct_response": {"status": 404}}
    assert all("timeout" in route["route"] for route in host["routes"] if "route" in route)


@pytest.mark.parametrize("role", ROLES)
def test_only_the_gateway_s_certificate_is_taken(role: str) -> None:
    config = envoy(role)
    (listener,) = config["static_resources"]["listeners"]
    assert listener["address"]["socket_address"] == {"address": "0.0.0.0", "port_value": PORT}
    (chain,) = listener["filter_chains"]
    tls = chain["transport_socket"]["typed_config"]
    assert tls["@type"].endswith("DownstreamTlsContext") and tls["require_client_certificate"] is True
    context = tls["common_tls_context"]
    assert context["tls_params"]["tls_minimum_protocol_version"] in ("TLSv1_2", "TLSv1_3")
    validation = context["combined_validation_context"]
    sans = validation["default_validation_context"]["match_typed_subject_alt_names"]
    assert sans == [{"san_type": "URI", "matcher": {"exact": GATEWAY_IDENTITY}}]
    secrets = [context["tls_certificate_sds_secret_configs"][0], validation["validation_context_sds_secret_config"]]
    for secret in secrets:  # (each read from a file that names the published certificate)
        path = Path(secret["sds_config"]["path_config_source"]["path"])
        (resource,) = yaml.safe_load((IMAGES / "common" / "sds" / path.name).read_text())["resources"]
        assert resource["name"] == secret["name"]
    assert config["admin"]["address"]["socket_address"]["address"] == "127.0.0.1"
    for cluster in config["static_resources"]["clusters"]:  # (the pod's servers listen on its loopback interface)
        for endpoints in cluster["load_assignment"]["endpoints"]:
            for endpoint in endpoints["lb_endpoints"]:
                assert endpoint["endpoint"]["address"]["socket_address"]["address"] == "127.0.0.1"


@pytest.mark.parametrize("role", ROLES)
def test_requests_are_limited_timed_and_logged_without_bodies(role: str) -> None:
    http = manager(envoy(role))
    filters = [each["name"] for each in http["http_filters"]]
    assert filters == ["envoy.filters.http.local_ratelimit", "envoy.filters.http.buffer", "envoy.filters.http.router"]
    buffer = http["http_filters"][1]["typed_config"]
    assert 0 < buffer["max_request_bytes"] <= 8 * 2**20
    bucket = http["http_filters"][0]["typed_config"]["token_bucket"]
    assert bucket["max_tokens"] > 0 and bucket["tokens_per_fill"] > 0
    assert http["request_headers_timeout"] and http["stream_idle_timeout"] and http["max_request_headers_kb"] <= 16
    assert http["normalize_path"] and http["merge_slashes"] and http["path_with_escaped_slashes_action"]
    (log,) = http["access_log"]
    fields = log["typed_config"]["log_format"]["json_format"]
    assert fields["peer"] == "%DOWNSTREAM_PEER_URI_SAN%" and fields["status"] == "%RESPONSE_CODE%"
    assert not any("BODY" in value or "DYNAMIC_METADATA" in value for value in fields.values())


def test_the_certificates_envoy_reads_are_those_the_script_publishes() -> None:
    script = (IMAGES / "common" / "pki.sh").read_text()
    for name in ("certificate.yaml", "ca.yaml"):
        (resource,) = yaml.safe_load((IMAGES / "common" / "sds" / name).read_text())["resources"]
        secret = resource.get("tls_certificate") or resource["validation_context"]
        sources: list[Any] = list(secret.values())
        files: list[str] = [each["filename"] for each in sources if isinstance(each, dict) and "filename" in each]
        assert files and all(Path(each).parent == Path("/certs/current") for each in files)
        assert secret["watched_directory"] == {"path": "/certs"}
        for each in files:
            assert Path(each).name in script
    assert 'mv -T "$CERTS/.current.next" "$CERTS/current"' in script  # (swapped in one rename, which Envoy watches)
    assert 'ln -sfn "$CERTS" /certs' in script


@pytest.mark.parametrize("role", ROLES)
def test_an_entrypoint_uses_the_token_once_and_ends_with_its_processes(role: str) -> None:
    entrypoint = (IMAGES / role / "entrypoint.sh").read_text()
    assert (IMAGES / role / "entrypoint.sh").stat().st_mode & 0o111
    bootstrap, unset, renew = (
        entrypoint.index(each) for each in ("pki.sh bootstrap", "unset STEP_TOKEN", "pki.sh renew")
    )
    assert bootstrap < unset < renew  # (no process the pod starts sees the token)
    assert entrypoint.rstrip().endswith("supervise")


def test_the_inference_image_is_the_workspace_s_vllm() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    version = next(package["version"] for package in lock["package"] if package["name"] == "vllm")
    dockerfile = (IMAGES / "inference" / "Dockerfile").read_text()
    assert f"ARG VLLM_VERSION={version}" in dockerfile and "FROM vllm/vllm-openai:v${VLLM_VERSION}" in dockerfile
    assert "--host 127.0.0.1" in (IMAGES / "inference" / "entrypoint.sh").read_text()
    host = (IMAGES / "host" / "Dockerfile").read_text()
    assert f"ARG VLLM_VERSION={version}" in host and "rollout-lora" in host  # (vLLM and the trainer, one image)
    assert "--host 127.0.0.1" in (IMAGES / "host" / "entrypoint.sh").read_text()
    workflow = (ROOT / ".github" / "workflows" / "images.yml").read_text()
    for role in ROLES:
        dockerfile = (IMAGES / role / "Dockerfile").read_text()
        (envoy_image,) = re.findall(r"FROM (envoyproxy/envoy:\S+) AS envoy", dockerfile)
        assert f"ENVOY_IMAGE: {envoy_image}" in workflow  # (validated with the Envoy the image runs)
        assert f"EXPOSE {PORT}" in dockerfile


def test_the_trainer_image_is_pytorch_s_of_the_workspace_s_torch() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    torch = next(package["version"] for package in lock["package"] if package["name"] == "torch")
    dockerfile = (IMAGES / "trainer" / "Dockerfile").read_text()
    (image,) = re.findall(r"ARG TORCH_IMAGE=(\S+)", dockerfile)
    assert image.startswith(f"pytorch/pytorch:{torch}-") and image.endswith("-runtime")
    assert "FROM ${TORCH_IMAGE} AS base" in dockerfile
    assert "vllm" not in dockerfile.lower()  # (the trainer carries no vLLM)


@pytest.mark.parametrize("role", ROLES)
def test_what_a_pod_runs_is_checked_for_the_python_of_its_image(role: str) -> None:
    """The packages a pod installs run on its base image's Python (3.12): each says it does, and ruff and pyright check
    them for it. Dependencies come from uv.lock alone, apart from the source, so that a change to the source rebuilds
    only the last layer."""
    dockerfile = (IMAGES / role / "Dockerfile").read_text().replace("\\\n", " ")
    image = dockerfile.split("\nFROM base\n")[1]  # (the image itself, after the stage that exports the lock)
    (packages,) = [line for line in image.splitlines() if "install.sh packages" in line]
    directories = packages.split("install.sh packages")[1].split()
    assert "libraries/rollout-train" in directories
    root = tomllib.loads((ROOT / "pyproject.toml").read_text())
    environments = root["tool"]["pyright"]["executionEnvironments"]
    checked = {each["root"] for each in environments if each["pythonVersion"] == "3.12"}
    targets = root["tool"]["ruff"]["per-file-target-version"]
    for directory in directories:
        project = tomllib.loads((ROOT / directory / "pyproject.toml").read_text())["project"]
        assert project["requires-python"] == ">=3.12", directory
        assert f"{directory}/src" in checked and targets[f"{directory}/**"] == "py312", directory
    order = [image.index(each) for each in ("install.sh dependencies", "COPY --chmod=755", "install.sh packages")]
    assert order == sorted(order)


def _read(path: Path) -> set[str]:
    """The variables a script or module reads."""
    text = path.read_text()
    if path.suffix == ".py":
        found: list[str] = re.findall(r'environ(?:\.get)?\(\s*"([A-Z0-9_]+)"', text)
        found += re.findall(r'environ, "([A-Z0-9_]+)"', text)
        found += re.findall(r'"(RUNPOD_[A-Z_]+)"', text)
        return set(found)
    expanded: list[str] = re.findall(r"\$\{([A-Z][A-Z0-9_]+)[:}]", text)
    named: list[str] = re.findall(r"\b(STEP_[A-Z_]+|ROLLOUT_[A-Z_]+)\b", text)
    return {*expanded, *named}


@pytest.mark.parametrize("role", ROLES)
def test_every_variable_a_pod_reads_is_documented(role: str) -> None:
    pods = ROOT / "libraries" / "rollout-train" / "src" / "rollout_train" / "pods"
    modules = {"inference": ["inference.py"], "trainer": ["training.py"], "host": ["inference.py", "training.py"]}[role]
    read: set[str] = set[str]().union(*(_read(each) for each in (
        *(pods / module for module in modules), pods / "environment.py", IMAGES / role / "entrypoint.sh",
        IMAGES / "common" / "pki.sh",
    )))  # fmt: skip
    readmes = {"host": ["host", "inference", "trainer"]}.get(role, [role])
    documented = (
        "".join((IMAGES / each / "README.md").read_text() for each in readmes) + (IMAGES / "README.md").read_text()
    )
    missing = sorted(name for name in read if f"`{name}`" not in documented and not name.startswith("RUNPOD_TCP_PORT"))
    assert missing == []


def test_no_image_is_built_with_this_machines_files_or_keys() -> None:
    ignored = {line.strip() for line in (ROOT / ".dockerignore").read_text().splitlines()}
    assert {"local", "**/*.env", ".claude"} <= ignored  # (local/ holds keys: the platform image copies the tree)
