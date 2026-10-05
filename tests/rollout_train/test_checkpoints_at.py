"""Where a training run's checkpoints go, as the New run form's check says it (`checkpoints_at`): the cluster's
`[blobs]` for a run here, the store its RunPod providers name for one on RunPod's pods, Tinker's archive for a Tinker
trainer (with the bridge's copies in `[blobs]` where a provider here serves them), and the store a run's start
recorded; and the offers that say which trainer and provider pairs share one machine."""

import tomllib
from pathlib import Path

from rollout_train.cluster import Cluster, parsed
from rollout_train.launching import checkpoints_at, offers
from rollout_train.ledger import FileLedger
from rollout_train.run_settings import RunSettings
from rollout_train.stores import blobs_at

STEP_CA = '{ url = "https://ca.example.com", provisioner = "launcher", key_file = "~/p.jwk", root = "~/root.crt" }'
MODEL = "Qwen/Qwen3.5-4B"
CLUSTER = f"""
name = "test"
[ledger]
url = "sqlite:///~/ledger.db"
[blobs]
kind = "rollout_s3:S3BlobStore"
bucket = "local"
prefix = "blobs/"
[stores.r2]
kind = "rollout_s3:S3BlobStore"
bucket = "rollout"
prefix = "blobs/"
endpoint_url = "https://account.r2.cloudflarestorage.com"
access_key_id_env = "R2_WRITER_ACCESS_KEY_ID"
secret_access_key_env = "R2_WRITER_SECRET_ACCESS_KEY"
[tls]
ca = "~/ca.pem"
certificate = "~/gateway.crt"
key = "~/gateway.key"

[inference.local-vllm]
kind = "vllm"
gpus = 1
[inference.local-vllm.models."{MODEL}"]
context = 8192
[inference.tinker]
kind = "tinker"
[inference.tinker.models."{MODEL}"]
context = 8192
[inference.h100]
kind = "runpod-host"
image = "ghcr.io/by77er/rollout-host@sha256:0"
gpu_types = ["NVIDIA H100 80GB HBM3"]
price = 2.69
step_ca = {STEP_CA}
store = "r2"
[inference.h100.models."{MODEL}"]
context = 8192

[trainers.local-lora]
kind = "lora"
gpus = 1
colocate_with = "local-vllm"
models = ["{MODEL}"]
[trainers.tinker-lora]
kind = "tinker"
models = ["{MODEL}"]
[trainers.h100-lora]
kind = "runpod-trainer"
colocate_with = "h100"
models = ["{MODEL}"]
[trainers.pods]
kind = "runpod-trainer"
image = "ghcr.io/by77er/rollout-trainer@sha256:0"
gpu_types = ["NVIDIA H100 80GB HBM3"]
step_ca = {STEP_CA}
store = "r2"
models = ["{MODEL}"]
"""
LOCAL = {"name": None, "kind": "rollout_s3:S3BlobStore", "bucket": "local", "prefix": "blobs/", "directory": None}
R2 = {"name": "r2", "kind": "rollout_s3:S3BlobStore", "bucket": "rollout", "prefix": "blobs/", "directory": None}


def a_cluster() -> Cluster:
    return parsed(tomllib.loads(CLUSTER))


def training(trainer: str, provider: str) -> RunSettings:
    return RunSettings({
        "kind": "train", "trainer.provider": trainer, "channels.policy.provider": provider,
        "channels.policy.model": MODEL,
    })  # fmt: skip


def test_checkpoints_go_to_the_clusters_store_or_the_one_its_runpod_providers_name() -> None:
    cluster = a_cluster()
    together = checkpoints_at(training("local-lora", "local-vllm"), cluster)
    assert together == {"store": LOCAL, "tinker": False, "bridges": [], "bridged": None}
    host = checkpoints_at(training("h100-lora", "h100"), cluster)
    assert host == {"store": R2, "tinker": False, "bridges": [], "bridged": None}
    pods = checkpoints_at(training("pods", "local-vllm"), cluster)  # (a trainer on RunPod, served here)
    assert pods == {"store": R2, "tinker": False, "bridges": [], "bridged": None}
    assert "access_key_id_env" not in str(pods) and "endpoint_url" not in str(pods)  # (nothing of its keys)


def test_tinker_keeps_the_weights_and_a_bridge_writes_copies_to_the_clusters_store() -> None:
    cluster = a_cluster()
    sampled = checkpoints_at(training("tinker-lora", "tinker"), cluster)
    assert sampled == {"store": LOCAL, "tinker": True, "bridges": [], "bridged": None}
    bridged = checkpoints_at(training("tinker-lora", "local-vllm"), cluster)
    assert bridged == {"store": LOCAL, "tinker": True, "bridges": ["peft-from-tinker"], "bridged": LOCAL}


def test_a_started_run_says_the_store_its_start_recorded_and_an_eval_makes_none() -> None:
    cluster = a_cluster()
    recorded = checkpoints_at(training("local-lora", "local-vllm"), cluster, written=blobs_at(cluster, "r2"))
    assert recorded is not None and recorded["store"] == R2
    elsewhere = checkpoints_at(training("local-lora", "local-vllm"), cluster, written={"kind": "x:Y", "bucket": "b"})
    assert elsewhere is not None and elsewhere["store"] == {**LOCAL, "kind": "x:Y", "bucket": "b", "prefix": None}
    assert checkpoints_at(RunSettings({"kind": "eval", "channels.policy.provider": "tinker"}), cluster) is None


async def test_the_offers_say_which_pairs_share_a_machine_and_which_trainers_train_apart(tmp_path: Path) -> None:
    offered = await offers(a_cluster(), FileLedger(tmp_path / "ledger"))
    together = {(each["trainer"], each["inference"]) for each in offered["pairs"] if each["together"]}
    assert together == {("local-lora", "local-vllm"), ("h100-lora", "h100")}
    separate = {each["name"]: each["separate"] for each in offered["trainers"]}
    assert separate == {"local-lora": True, "tinker-lora": True, "h100-lora": False, "pods": True}
    pairs = {(each["trainer"], each["inference"]): each for each in offered["pairs"]}
    bridged = pairs["tinker-lora", "local-vllm"]
    assert not bridged["together"] and bridged["bridge"] == ["peft-from-tinker"]
