"""The cluster config: found by `--cluster`, then `ROLLOUT_CLUSTER`, then `~/.config/rollout/cluster.toml`; read
strictly (unknown keys, unknown kinds, a trainer colocated with what is not a vllm provider, a provider reached with
no auth away from this machine are errors); and holding secrets only by name, never by value."""

import dataclasses
import json
from pathlib import Path
from typing import Any, cast

import pytest

from rollout_train.cluster import Cluster, ClusterError, find, inspect, load, parsed
from rollout_train.providers import Secret

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "deploy" / "clusters" / "example.toml"
SMALL = """
name = "small"
[ledger]
url = "sqlite:///~/ledger.db"
[inference.local]
kind = "vllm"
[inference.local.models."m"]
context = 4096
"""


def cluster_of(text: str) -> Cluster:
    import tomllib

    return parsed(tomllib.loads(text))


def test_the_example_describes_this_machine() -> None:
    cluster = load(EXAMPLE)
    assert cluster.name == "home" and cluster.namespace == "rollout-home"
    assert cluster.ledger.url == "sqlite:///~/.cache/rollout/ledger.db" and cluster.ledger.url_secret is None
    assert cluster.blobs.kind == "files" and cluster.blobs.settings == {"directory": "~/.cache/rollout/blobs"}
    assert set(cluster.inference) == {"local-vllm", "tinker", "openai", "anthropic"}
    local = cluster.inference["local-vllm"]
    assert local.kind == "vllm" and local.gpus == 1 and local.local and local.auth.kind == "none"
    assert local.models["Qwen/Qwen3.5-4B"].max_lora_rank == 64
    assert local.models["cyankiwi/Qwen3.5-9B-AWQ-4bit"].base == "Qwen/Qwen3.5-9B"
    assert cluster.inference["tinker"].auth.kind == "vendor"
    assert cluster.inference["tinker"].auth.key == Secret(env="TINKER_API_KEY")
    assert set(cluster.trainers) == {"local-lora", "local-full", "tinker-lora"}
    assert cluster.trainers["local-lora"].colocate_with == "local-vllm"
    assert cluster.trainers["tinker-lora"].capabilities.format == "tinker"
    assert cluster.trainers["local-full"].capabilities.produces == "full"
    assert cluster.sandboxes["minecraft"].size == 6
    assert cluster.environments["minecraft_team.environment:environment"].python == "platform"
    verifiers = cluster.environments["rollout_verifiers.environments:gsm8k"]
    assert verifiers.python is None and verifiers.project is not None
    assert verifiers.project.endswith("implementations/rollout-verifiers")
    assert cluster.guards.runs_gib == 6 and cluster.guards.training_gib == 4


def test_the_cluster_configs_the_guide_shows_load() -> None:
    import re

    page = (ROOT / "docs" / "guide" / "cluster.md").read_text()
    blocks = re.findall(r'^```toml title="cluster.toml"\n(.*?)^```$', page, re.MULTILINE | re.DOTALL)
    assert blocks
    for block in blocks:
        assert cluster_of(block).inference


def write(path: Path, text: str = SMALL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_the_config_is_found_by_flag_then_environment_then_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    home = write(tmp_path / ".config" / "rollout" / "cluster.toml")
    named = write(tmp_path / ".config" / "rollout" / "clusters" / "lab.toml")
    elsewhere = write(tmp_path / "elsewhere" / "big.toml")
    assert find(environ={}) == home
    assert find(environ={"ROLLOUT_CLUSTER": "lab"}) == named
    assert find(environ={"ROLLOUT_CLUSTER": str(elsewhere)}) == elsewhere
    assert find("lab", environ={"ROLLOUT_CLUSTER": str(elsewhere)}) == named  # the flag first
    assert find(str(elsewhere), environ={"ROLLOUT_CLUSTER": "lab"}) == elsewhere


def test_a_config_named_and_missing_says_where_it_looked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(ClusterError, match=r"--cluster names 'lab'.*clusters/lab\.toml"):
        find("lab", environ={})
    with pytest.raises(ClusterError, match="ROLLOUT_CLUSTER names"):
        find(environ={"ROLLOUT_CLUSTER": str(tmp_path / "none.toml")})
    with pytest.raises(ClusterError, match="no cluster config: pass --cluster"):
        find(environ={})


@pytest.mark.parametrize(
    ("change", "says"),
    [
        ('\n[gateway]\nlisten = "x"\nport = 3\n', r"\[gateway\] has no port"),
        ("\nsurprise = 1\n", r'models."m" has no surprise'),
        ('\n[inference.other]\nkind = "sglang"\n', "kind is one of vllm, vllm-servers, tinker, api, runpod-inference"),
        ('\n[trainers.t]\nkind = "rollout_lora:LoraTrainer"\nmodels = ["m"]\n', "named by its kind, not module:name"),
        ('\n[trainers.t]\nkind = "lora"\nmodels = ["m"]\ncolocate_with = "elsewhere"\n', "not a vllm provider"),
        ('\n[trainers.t]\nkind = "lora"\nmodels = []\n', "models is a list"),
        ('\n[inference.local.models."n"]\n', "needs context"),
        (
            '\n[inference.local.models."n"]\ncontext = 8\noptions = { max_logprobs = 5 }\n',
            r"max_logprobs is the provider's \(\[inference.local\] max_logprobs\)",
        ),
        ("\n[placement.everyone]\nresources = { a = 1 }\n", "the roles are"),
        ('\n[environments."e:e"]\npython = "platform"\nproject = "p"\n', 'python = "platform" or a project'),
        ('\n[sandboxes.box]\nprovider = "p:p"\npython = "nowhere"\n', "python is platform or an environment"),
        ('\n[trainers.t]\nkind = "lora"\nmodels = ["m"]\nimage = "x"\n', r"has no image \(a lora trainer has"),
    ],
)
def test_what_a_config_cannot_say_is_refused(change: str, says: str) -> None:
    with pytest.raises(ClusterError, match=says):
        cluster_of(SMALL + change)


def test_a_name_is_lowercase_for_the_namespace() -> None:
    with pytest.raises(ClusterError, match="lowercase"):
        cluster_of(SMALL.replace('name = "small"', 'name = "My Cluster"'))
    with pytest.raises(ClusterError, match="the cluster config has no surprise"):
        cluster_of("surprise = 1\n" + SMALL)


SECRET = "sk-very-secret-value-0123"


@pytest.mark.parametrize(
    ("change", "says"),
    [
        (f'\n[inference.openai]\nkind = "api"\nendpoint = "e:e"\napi_key = "{SECRET}"\n', "api_key_env"),
        (
            f'\n[tools.search]\nurl = "https://s.example"\nauth = {{ kind = "bearer", token = "{SECRET}" }}\n',
            "token_env",
        ),
        (f'\n[blobs]\nkind = "s3:Store"\nsecret_access_key = "{SECRET}"\n', "secret_access_key_env"),
    ],
)
def test_a_secret_written_down_is_refused(change: str, says: str) -> None:
    with pytest.raises(ClusterError, match=says) as raised:
        cluster_of(SMALL + change)
    assert SECRET not in str(raised.value)


def test_a_ledger_url_with_a_password_is_refused() -> None:
    with pytest.raises(ClusterError, match="url_env") as raised:
        cluster_of(SMALL.replace("sqlite:///~/ledger.db", f"postgresql://rollout:{SECRET}@db/rollout"))
    assert SECRET not in str(raised.value)


def test_the_ledger_service_is_reached_with_the_platforms_token_named() -> None:
    service = SMALL.replace('url = "sqlite:///~/ledger.db"', 'url = "https://ledger.example.com"')
    with pytest.raises(ClusterError, match="token_env or token_file"):
        cluster_of(service)
    cluster = cluster_of(service.replace("[inference.local]", 'token_env = "LEDGER_TOKEN"\n[inference.local]'))
    assert (
        cluster.ledger.token == Secret(env="LEDGER_TOKEN") and cluster.secrets()["ledger.token"] == cluster.ledger.token
    )
    public = SMALL.replace("[inference.local]", 'public = "https://ledger.example.com"\n[inference.local]')
    assert cluster_of(public).ledger.public == "https://ledger.example.com"
    with pytest.raises(ClusterError, match="public is the ledger service's address"):
        cluster_of(public.replace("https://ledger.example.com", "ledger.example.com"))
    with pytest.raises(ClusterError, match="looks like a secret"):
        cluster_of(SMALL.replace("[inference.local]", 'token = "written-down"\n[inference.local]'))


def every_value(value: Any) -> list[str]:
    """Every string anywhere in a dataclass, mapping or sequence."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [each for field in dataclasses.fields(value) for each in every_value(getattr(value, field.name))]
    if isinstance(value, dict):
        table = cast(dict[object, object], value)
        return [each for key, item in table.items() for each in [str(key), *every_value(item)]]
    if isinstance(value, list | tuple | frozenset | set):
        return [each for item in cast(list[object], value) for each in every_value(item)]
    return [str(value)]


def test_secrets_are_names_never_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("TINKER_API_KEY", "TINKER_PROJECT_ID", "OPENAI_API_KEY", "LEDGER_URL", "ENGINES_TOKEN"):
        monkeypatch.setenv(name, SECRET)
    keys = write(tmp_path / "gateway.keys", SECRET)
    text = load(EXAMPLE).described  # (the example, and more secrets beside it)
    assert text
    extra = f"""
name = "secretive"
[ledger]
url_env = "LEDGER_URL"
[gateway]
keys_file = "{keys}"
[inference.openai]
kind = "api"
endpoint = "rollout_openai:ResponsesEndpoint"
auth = {{ kind = "vendor", key_env = "OPENAI_API_KEY" }}
[inference.openai.models."gpt-5"]
context = 400000
[inference.lab]
kind = "vllm-servers"
addresses = ["https://gpu-1.lab.example:8000"]
auth = {{ kind = "bearer", token_env = "ENGINES_TOKEN" }}
[inference.lab.models."m"]
context = 8192
[inference.tinker]
kind = "tinker"
project_env = "TINKER_PROJECT_ID"
[inference.tinker.models."m"]
context = 8192
"""
    cluster = load(write(tmp_path / "cluster.toml", extra))
    assert SECRET not in repr(cluster)
    assert SECRET not in json.dumps(cluster.described)
    assert all(SECRET not in each for each in every_value(cluster))
    assert set(cluster.secrets()) == {
        "ledger.url", "gateway.keys", "inference.openai.auth.key", "inference.lab.auth.token",
        "inference.tinker.auth.key", "inference.tinker.project",
    }  # fmt: skip
    # Resolved where they are used, at the moment they are needed:
    assert cluster.ledger.url_secret is not None and cluster.ledger.url_secret.resolve() == SECRET
    assert cluster.gateway.keys is not None and cluster.gateway.keys.resolve() == SECRET
    assert inspect(cluster) == []
    monkeypatch.delenv("ENGINES_TOKEN")
    assert inspect(cluster) == ["inference.lab.auth.token: $ENGINES_TOKEN is not set on this node"]
    assert inspect(cluster, role="monitor") == []  # (a monitor reads no server's token)


def test_each_role_reads_only_the_secrets_its_code_uses(tmp_path: Path) -> None:
    text = (
        SMALL
        + """
[tls]
ca = "~/ca.pem"
certificate = "~/gateway.crt"
key = "~/gateway.key"
[gateway]
keys_file = "~/gateway.keys"
[monitor]
token_env = "MONITOR_TOKEN"
[stores.r2]
kind = "rollout_s3:S3BlobStore"
bucket = "b"
access_key_id_env = "R2_WRITER_ID"
secret_access_key_env = "R2_WRITER_SECRET"
reader = { access_key_id_env = "R2_READER_ID", secret_access_key_env = "R2_READER_SECRET" }
[inference.openai]
kind = "api"
endpoint = "rollout_openai:hosted"
api_key_env = "OPENAI_API_KEY"
[inference.openai.models."gpt"]
context = 1000
[inference.tinker]
kind = "tinker"
[inference.tinker.models."m"]
context = 8192
[inference.pods]
kind = "runpod-inference"
image = "ghcr.io/x/y@sha256:0"
gpu_types = ["NVIDIA H100 80GB HBM3"]
api_key_env = "RUNPOD_KEY"
[inference.pods.models."m"]
context = 8192
"""
    )
    cluster = load(write(tmp_path / "cluster.toml", text.replace("[ledger]", '[ledger]\ntoken_env = "LEDGER_TOKEN"')))
    every = set(cluster.secrets())
    assert set(cluster.secrets_of("run")) == every - {"monitor.token"}
    assert set(cluster.secrets_of("gateway")) == {"gateway.keys", "inference.openai.api_key"}
    assert set(cluster.secrets_of("monitor")) == {
        "stores.r2.access_key_id",
        "stores.r2.secret_access_key",
        "monitor.token",
    }  # (a store's read-only key is a pod's: never the monitor's)
    assert set(cluster.secrets_of("ledger")) == {"ledger.token"}
    assert set(cluster.secrets_of("reaper")) == {"inference.pods.api_key"}
    assert cluster.secrets_of("pool") == {}
    with pytest.raises(ValueError, match="a role is one of"):
        cluster.secrets_of("everyone")


def test_the_parsed_config_is_handed_on_as_json(tmp_path: Path) -> None:
    cluster = load(EXAMPLE)
    again = parsed(json.loads(json.dumps(cluster.described)))
    assert again == cluster


def test_a_project_is_found_from_the_config_file_and_checked_for_a_lock(tmp_path: Path) -> None:
    (tmp_path / "project").mkdir()
    text = SMALL + '\n[environments."e:e"]\nproject = "project"\n'
    cluster = load(write(tmp_path / "cluster.toml", text))
    assert cluster.environments["e:e"].project == str(tmp_path / "project")
    assert inspect(cluster) == [f'environments."e:e": {tmp_path / "project"} has no uv.lock to build its Python from']
    (tmp_path / "project" / "uv.lock").write_text("")
    assert inspect(cluster) == []


def test_auth_none_is_only_for_this_machine() -> None:
    servers = '\n[inference.lab]\nkind = "vllm-servers"\naddresses = ["{address}"]\nauth = "none"\n'
    servers += '[inference.lab.models."m"]\ncontext = 8192\n'
    assert cluster_of(SMALL + servers.format(address="http://127.0.0.1:8000")).inference["lab"].local
    with pytest.raises(ClusterError, match="auth none is only for localhost"):
        cluster_of(SMALL + servers.format(address="http://gpu-1:8000"))
    with pytest.raises(ClusterError, match="says how it is reached"):
        cluster_of(SMALL + servers.format(address="http://gpu-1:8000").replace('auth = "none"\n', ""))
    with pytest.raises(ClusterError, match="auth none is only for localhost"):
        cluster_of(SMALL + '\n[inference.local2]\nkind = "vllm"\nlisten = "0.0.0.0"\n[inference.local2.models."m"]\n'
                   "context = 1\n")  # fmt: skip


def test_mutual_tls_needs_the_clusters_ca_and_certificate() -> None:
    pods = """
[inference.pods]
kind = "runpod-inference"
image = "ghcr.io/by77er/rollout-inference@sha256:0"
gpu_types = ["NVIDIA GeForce RTX 4090"]
max_pods = 2
api_key_env = "RUNPOD_API_KEY"
secrets = { AWS_ACCESS_KEY_ID = "r2_key_id" }
[inference.pods.models."m"]
context = 8192
options = { max_lora_rank = 96 }
"""
    with pytest.raises(ClusterError, match=r"needs \[tls\] ca, certificate and key"):
        cluster_of(SMALL + pods)
    tls = '\n[tls]\nca = "~/ca.pem"\ncertificate = "~/gateway.crt"\nkey = "~/gateway.key"\n'
    cluster = cluster_of(SMALL + tls + pods)
    provider = cluster.inference["pods"]
    assert provider.auth.kind == "mtls" and provider.auth.identity == "leased"
    assert provider.capabilities.bills == "hours" and provider.capabilities.loads == {"peft"}
    assert provider.secrets == {"api_key": Secret(env="RUNPOD_API_KEY")}
    assert provider.settings["secrets"] == {"AWS_ACCESS_KEY_ID": "r2_key_id"}  # (console secrets, by name)
    with pytest.raises(ClusterError, match="auth is one of mtls"):
        cluster_of(SMALL + tls + pods.replace("max_pods = 2", 'max_pods = 2\nauth = "none"'))


def test_a_runpod_trainer_takes_the_capabilities_of_the_trainer_it_runs() -> None:
    tls = '\n[tls]\nca = "~/ca.pem"\ncertificate = "~/gateway.crt"\nkey = "~/gateway.key"\n'
    trainer = (
        '\n[trainers.pods]\nkind = "runpod-trainer"\ntrainer = "{runs}"\nmodels = ["m"]\nsegment_tokens = 16000\n'
        'image = "ghcr.io/by77er/rollout-trainer@sha256:0"\ngpu_types = ["NVIDIA H100 80GB HBM3"]\n'
    )
    lora = cluster_of(SMALL + tls + trainer.format(runs="lora")).trainers["pods"]
    full = cluster_of(SMALL + tls + trainer.format(runs="full")).trainers["pods"]
    assert (lora.capabilities.produces, lora.capabilities.format) == ("lora", "peft")
    assert (full.capabilities.produces, full.capabilities.format) == ("full", "full")
    with pytest.raises(ClusterError, match="trainer is lora or full"):
        cluster_of(SMALL + tls + trainer.format(runs="tinker"))
    assert lora.settings.get("cuda_versions", ("13.0",)) in (("13.0",), ["13.0"])  # (the images' CUDA, by default)
    with pytest.raises(ClusterError, match="cuda_versions are RunPod's"):
        cluster_of(SMALL + tls + trainer.format(runs="lora") + 'cuda_versions = ["14.0"]\n')


TLS = '\n[tls]\nca = "~/ca.pem"\ncertificate = "~/gateway.crt"\nkey = "~/gateway.key"\n'
HOST = """
[inference.h100]
kind = "runpod-host"
image = "ghcr.io/by77er/rollout-host@sha256:0"
gpu_types = ["NVIDIA H100 80GB HBM3"]
cloud = "community"
regions = ["US-KS-2"]
price = 2.69
idle_stop = 300
step_ca = { url = "https://ca.example.com", provisioner = "launcher", key_file = "~/p.jwk", root = "~/root.crt" }
[inference.h100.models."m"]
context = 8192
[trainers.h100-lora]
kind = "runpod-trainer"
colocate_with = "h100"
models = ["m"]
"""


def test_a_runpod_providers_table_says_what_its_pods_are() -> None:
    from rollout_train.providers import pod_table

    cluster = cluster_of(SMALL + TLS + HOST)
    host = cluster.inference["h100"]
    said = pod_table(host.kind, host.settings)
    assert (said.cloud, said.regions, said.price, said.idle_stop, said.max_pods) == ("COMMUNITY", ("US-KS-2",), 2.69,
                                                                                    300.0, 1)  # fmt: skip
    assert said.memory_fraction == 0.42 and not said.sleep and said.start_timeout == 1200.0
    assert host.secrets["api_key"] == Secret(env="RUNPOD_API_KEY")
    assert cluster.trainers["h100-lora"].colocate_with == "h100"
    for change, says in (
        (('cloud = "community"', 'cloud = "cheap"'), "cloud is secure or community"),
        (('image = "ghcr.io/by77er/rollout-host@sha256:0"\n', ""), "image names the image"),
        (("price = 2.69", "price = -1"), "price is a number"),
        (("idle_stop = 300", "idle_stop = 300\nmax_pods = 0"), "max_pods is a whole number"),
        (('provisioner = "launcher", ', ""), "step_ca says url, provisioner"),
        (("idle_stop = 300", "idle_stop = 300\nmemory_fraction = 0.99"), "memory_fraction is a share"),
        (('colocate_with = "h100"', 'colocate_with = "local"'), "not a runpod-host provider"),
        (('colocate_with = "h100"', 'colocate_with = "h100"\nmax_pods = 2'), "takes its steps on h100's pods"),
        (("idle_stop = 300", 'idle_stop = 300\nstore = "r2"'), "store names 'r2', which is no"),
        (("idle_stop = 300", "idle_stop = 300\nmin_vcpus_per_gpu = 0"), "min_vcpus_per_gpu is a whole number"),
        (("idle_stop = 300", 'idle_stop = 300\nmin_memory_gb_per_gpu = "lots"'), "min_memory_gb_per_gpu is a whole"),
        (("idle_stop = 300", 'idle_stop = 300\nsandboxes = ["minecraft"]'), "which is no .sandboxes.minecraft. with"),
    ):
        with pytest.raises(ClusterError, match=says):
            cluster_of(SMALL + TLS + HOST.replace(*change))
    assert (said.min_vcpus_per_gpu, said.min_memory_gb_per_gpu, said.sandboxes) == (None, None, ())
    asking = cluster_of(SMALL + TLS + HOST.replace("idle_stop = 300", "idle_stop = 300\nmin_vcpus_per_gpu = 16\n"
                                                   "min_memory_gb_per_gpu = 128"))  # fmt: skip
    table = pod_table("runpod-host", asking.inference["h100"].settings)
    assert (table.min_vcpus_per_gpu, table.min_memory_gb_per_gpu) == (16, 128)


WORLDS = '\n[sandboxes.minecraft]\nprovider = "minecraft_team.worlds:worlds"\n'


def test_a_kind_of_sandbox_may_be_served_from_the_pods_whose_provider_lists_it() -> None:
    from rollout_train.cluster import OnPods
    from rollout_train.providers import pod_table

    serving = HOST.replace("idle_stop = 300", 'idle_stop = 300\nsandboxes = ["minecraft"]')
    cluster = cluster_of(SMALL + TLS + serving + WORLDS + 'on_pods = true\nheap = "1G"\n')
    section = cluster.sandboxes["minecraft"]
    assert section.on_pods == OnPods() and section.settings == {"heap": "1G"} and section.url is None
    assert pod_table("runpod-host", cluster.inference["h100"].settings).sandboxes == ("minecraft",)
    told = cluster_of(SMALL + TLS + serving + WORLDS + 'on_pods = { size = 8, cpus = 1.5, memory_gib = 3, share = 0.5, '
                      'settings = { heap = "2G" }, version = "minecraft-team@abc" }\n'
                      'url = "http://sandboxes-minecraft:8710"\n')  # fmt: skip
    said = OnPods(size=8, cpus=1.5, memory_gib=3.0, share=0.5, settings={"heap": "2G"}, version="minecraft-team@abc")
    assert told.sandboxes["minecraft"].on_pods == said
    two = HOST.replace("idle_stop = 300", 'idle_stop = 300\nsandboxes = ["minecraft", "boxes"]')
    boxes = '\n[sandboxes.boxes]\nprovider = "tests.rollout_train.pods.sandbox_kinds:boxes"\non_pods = { size = 4 }\n'
    cluster_of(SMALL + TLS + two + WORLDS + "on_pods = { share = 0.5 }\n" + boxes)  # (each says how much it takes)
    for text, says in (
        (SMALL + TLS + HOST + WORLDS + "on_pods = true\n", "no runpod-host provider's pods serve minecraft"),
        (SMALL + TLS + serving + WORLDS, "which is no .sandboxes.minecraft. with on_pods"),
        (SMALL + TLS + serving + WORLDS + "on_pods = { cpus = 0 }\n", "cpus and memory_gib are more than 0"),
        (SMALL + TLS + serving + WORLDS + "on_pods = { gpus = 1 }\n", "on_pods has no gpus"),
        (SMALL + TLS + serving + WORLDS + 'on_pods = "yes"\n', "on_pods is true, or a table"),
        (SMALL + TLS + serving + WORLDS + "on_pods = { share = 1.5 }\n", "share is a part of a pod's spare"),
        (SMALL + TLS + serving + WORLDS + 'on_pods = { version = "abc" }\n', "version names a published version"),
        (SMALL + TLS + two + WORLDS + "on_pods = true\n" + boxes, "each says its size or its share in on_pods, and "
         "minecraft says neither"),
        (SMALL + TLS + two + WORLDS + "on_pods = { share = 0.8 }\n" + boxes.replace("size = 4", "share = 0.5"),
         "shares add up to more than 1"),
        (SMALL + TLS + HOST.replace("idle_stop = 300", 'sandboxes = ["Mine_craft"]') + WORLDS.replace(
         "minecraft]", "Mine_craft]") + "on_pods = true\n", "named with lowercase letters, digits and hyphens"),
        (SMALL + TLS + serving + '\n[sandboxes.minecraft]\nurl = "http://x:1"\non_pods = true\n', "names its provider"),
        (SMALL + TLS + HOST.replace("idle_stop = 300", 'sandboxes = ["minecraft"]').replace('kind = "runpod-host"',
         'kind = "runpod-inference"') + WORLDS + "on_pods = true\n", "has no sandboxes|a runpod-host's pods alone"),
    ):  # fmt: skip
        with pytest.raises(ClusterError, match=says):
            cluster_of(text)


def test_each_provider_and_trainer_is_metered_or_scheduled_by_its_kind_unless_it_says() -> None:
    cluster = load(EXAMPLE)
    assert {name: each.allocation for name, each in cluster.inference.items()} == {
        "local-vllm": "scheduled", "tinker": "metered", "openai": "metered", "anthropic": "metered",
    }  # fmt: skip
    assert {name: each.allocation for name, each in cluster.trainers.items()} == {
        "local-lora": "scheduled", "local-full": "scheduled", "tinker-lora": "metered",
    }  # fmt: skip
    hosted = cluster_of(
        SMALL
        + '\n[inference.openai]\nkind = "api"\nendpoint = "rollout_openai:ResponsesEndpoint"\nconcurrency = 16\n'
        + 'auth = { kind = "vendor", key_env = "OPENAI_API_KEY" }\n[inference.openai.models."gpt-5"]\ncontext = 8\n'
        + '\n[inference.lab]\nkind = "vllm-servers"\naddresses = ["http://127.0.0.1:8000"]\nauth = "none"\n'
        + 'allocation = "metered"\n'
        + '[inference.lab.models."m"]\ncontext = 8\n'
    )
    assert (hosted.inference["openai"].allocation, hosted.inference["openai"].concurrency) == ("metered", 16)
    assert hosted.inference["lab"].allocation == "metered" and hosted.inference["local"].concurrency is None


@pytest.mark.parametrize(
    ("table", "says"),
    [
        ('[inference.more]\nkind = "vllm"\nallocation = "spot"\n', "allocation is metered or scheduled, not 'spot'"),
        ('[inference.more]\nkind = "vllm"\nallocation = "metered"\n', "asks for GPUs of the cluster's, so it is"),
        ('[inference.more]\nkind = "vllm"\nconcurrency = 4\n', "concurrency bounds a metered provider"),
        ('[trainers.more]\nkind = "lora"\nmodels = ["m"]\ngpus = 1\nallocation = "metered"\n', "so it is scheduled"),
        ('[trainers.more]\nkind = "tinker"\nmodels = ["m"]\nconcurrency = 0\n', "concurrency is a whole number"),
    ],
)
def test_an_allocation_is_metered_or_scheduled_and_only_a_metered_one_has_a_concurrency(table: str, says: str) -> None:
    model = '[inference.more.models."m"]\ncontext = 8\n' if table.startswith("[inference") else ""
    with pytest.raises(ClusterError, match=says):
        cluster_of(f"{SMALL}\n{table}{model}")
