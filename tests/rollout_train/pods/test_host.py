"""A host pod (`runpod-host`): one pod leased for a run's trainer and its trained channel, given the store's writing
key, vLLM's share of the GPU's memory, and the cluster's root pinned; ready once vLLM serves what the run says and the
training service beside it holds the run's trainer; and the checkpoints the service makes loaded from the pod's disk."""

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from rollout.harness.blobs import FileBlobStore
from rollout_train.checkpoints import Checkpoints
from rollout_train.cluster import Cluster, parsed
from rollout_train.database import DatabaseLedger
from rollout_train.pods.environment import CachedBlobs
from rollout_train.pods.inference import InferencePod
from rollout_train.pods.leasing import Pods, needs_of
from rollout_train.run_settings import RunSettings
from tests.rollout_runpod.fake import KEY, FakeRunPod
from tests.rollout_train.machines import MODEL, Saying, fake_vllm
from tests.rollout_train.pods.authority import Authority, served_tls
from tests.rollout_train.pods.test_leasing import ENVIRON, StandIns

KEYS = {
    "R2_WRITER_ACCESS_KEY_ID": "writer", "R2_WRITER_SECRET_ACCESS_KEY": "writer-secret",
    "R2_READER_ACCESS_KEY_ID": "reader", "R2_READER_SECRET_ACCESS_KEY": "reader-secret",
}  # fmt: skip


def cluster_of(directory: Path, root: Path) -> Cluster:
    return parsed(
        tomllib.loads(f"""
name = "test"
[ledger]
url = "sqlite:///{directory / "ledger.db"}"
token_env = "ROLLOUT_LEDGER_TOKEN"
public = "https://ledger.example.com"
[tls]
ca = "~/ca.pem"
certificate = "~/gateway.crt"
key = "~/gateway.key"
[stores.r2]
kind = "rollout_s3:S3BlobStore"
bucket = "rollout"
endpoint_url = "https://ACCOUNT_ID.r2.cloudflarestorage.com"
region = "auto"
access_key_id_env = "R2_WRITER_ACCESS_KEY_ID"
secret_access_key_env = "R2_WRITER_SECRET_ACCESS_KEY"
reader = {{ access_key_id_env = "R2_READER_ACCESS_KEY_ID", secret_access_key_env = "R2_READER_SECRET_ACCESS_KEY" }}
[inference.h100]
kind = "runpod-host"
image = "ghcr.io/by77er/rollout-host@sha256:0"
gpu_types = ["NVIDIA H100 80GB HBM3"]
price = 2.69
store = "r2"
step_ca = {{ url = "https://ca.example.com", provisioner = "launcher", key_file = "~/p.jwk", root = "{root}", \
trust = "system" }}
[inference.h100.models."{MODEL}"]
context = 8192
options = {{ max_lora_rank = 32 }}
[trainers.h100-lora]
kind = "runpod-trainer"
colocate_with = "h100"
models = ["{MODEL}"]
""")
    )


class Authority1:
    """step-ca as a run's driver uses it: a one-time token for each pod's identity."""

    def pod_token(self, identity: str) -> str:
        return f"one-time token for {identity}"


async def test_a_host_pod_is_one_pod_for_trainer_and_channel_with_the_stores_writing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    root = Authority(tmp_path / "ca").root
    ledger = DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}")
    stand_ins = StandIns(ledger)
    fake = FakeRunPod(key=KEY, created=stand_ins.created, deleted=stand_ins.deleted)
    cluster = cluster_of(tmp_path, root)
    settings = RunSettings({
        "kind": "train", "trainer.provider": "h100-lora", "channels.policy.provider": "h100",
        "channels.policy.model": MODEL, "channels.policy.renderer": "r:r", "trainer.rank": 8,
    })  # fmt: skip
    pods = Pods("run_1", cluster, ledger, api=lambda provider: fake.client(), ca=lambda table: Authority1(),  # type: ignore[arg-type,return-value]
                environ={**ENVIRON, **KEYS}, look=0.05)  # fmt: skip
    try:
        (lease,) = await pods.claim(needs_of(settings, cluster))
        assert lease.role == "host" and lease.settings["implementation"] == "rollout_lora:LoraTrainer"
        assert len(pods.of("trainer")) == 1 and len(pods.of("inference", channel="policy")) == 1  # (the one pod)
        (body,) = fake.created_bodies()
        env = body["env"]
        assert env["ROLLOUT_ROLE"] == "host" and env["ROLLOUT_TRAINER_MODEL"] == MODEL
        assert "--gpu-memory-utilization 0.42" in env["VLLM_ARGS"]
        assert json.loads(env["ROLLOUT_BLOBS"])["access_key_id_env"] == "R2_WRITER_ACCESS_KEY_ID"  # (it writes)
        assert env["R2_WRITER_ACCESS_KEY_ID"] == "writer" and "R2_READER_ACCESS_KEY_ID" not in env
        assert env["STEP_ROOT"] == root.read_text() and env["STEP_CA_TRUST"] == "system"
        assert env["STEP_TOKEN"] == f"one-time token for spiffe://rollout/pod/{lease.pod}"
    finally:
        await pods.release()
        for task in stand_ins.tasks.values():
            task.cancel()


def trainer_service(holds: dict[str, Any]) -> Starlette:
    async def trainer(request: Request) -> JSONResponse:
        return JSONResponse({"run": holds["run"]})

    return Starlette(routes=[Route("/v1/trainer", trainer)])


async def test_a_host_pod_is_ready_once_its_training_service_holds_the_runs_trainer_too(tmp_path: Path) -> None:
    from rollout_train.pods.leases import HELD, PodLease, pod_leases_of

    ledger = DatabaseLedger(f"sqlite:///{tmp_path / 'ledger.db'}")
    store = pod_leases_of(ledger)
    assert store is not None
    await store.put(PodLease("host-1", "h100", 0, "host", "image", MODEL, "gpu", 2.69, run="run_1", channel="policy",
                             state=HELD), expect=None)  # fmt: skip
    holds: dict[str, Any] = {"run": None}
    async with served_tls(fake_vllm(Saying())) as vllm, served_tls(trainer_service(holds)) as trainer:
        pod = InferencePod("host-1", Checkpoints(ledger, FileBlobStore(tmp_path / "blobs")), None, None, MODEL,
                           tmp_path / "pod", vllm=vllm, trainer=trainer, role="host")  # fmt: skip
        await pod.follow()
        assert not pod.ready and "holds the trainer of no run" in pod.why
        holds["run"] = "run_1"
        await pod.follow()
        assert pod.ready and pod.about_pod()["pod"]["role"] == "host"  # type: ignore[index]


async def test_what_a_pods_processes_put_is_read_from_its_disk(tmp_path: Path) -> None:
    store, cache = FileBlobStore(tmp_path / "bucket"), FileBlobStore(tmp_path / "disk")
    blobs = CachedBlobs(store, cache)
    reference = await blobs.put(b"a checkpoint's weights", "application/octet-stream")
    assert await store.read(reference) == await cache.read(reference) == b"a checkpoint's weights"
    other = CachedBlobs(store, FileBlobStore(tmp_path / "another-disk"))  # (another process, its disk empty)
    assert await other.read(reference) == b"a checkpoint's weights"
    assert await other.cache.read(reference) == b"a checkpoint's weights"  # (kept)
