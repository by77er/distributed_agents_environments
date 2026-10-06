"""What a trainer needs of each GPU, and the check that refuses one whose GPUs cannot hold it: the estimate shards
weights, gradients and the optimizer's state over the GPUs; a model's size is read from its files; a full-weight trainer
of an 8B model is refused one 80 GB GPU and told how many would hold it; a trainer on several GPUs is a whole number of
them, never sharing an engine's, and a host's pod has one."""

import json
import struct
import tomllib
from pathlib import Path

import pytest
from pydantic import JsonValue

from rollout_train.cluster import ClusterError, parsed
from rollout_train.demand import TRAINER, demand
from rollout_train.memory import ModelFacts, gpu_memory_gib, model_facts, trainer_memory
from rollout_train.run_settings import RunSettings
from rollout_train.validation import check

EIGHT = ModelFacts(file_bytes=16_400_000_000, parameters=8.2e9, hidden=4096, layers=36, vocabulary=151_936, tied=False)
"""An 8B text model's (Qwen3-8B's shape)."""
STEP_CA = '{ url = "https://ca.example.com", provisioner = "launcher", key_file = "~/p.jwk", root = "~/root.crt" }'


def test_full_weights_are_sharded_over_the_gpus_and_an_adapters_model_is_whole_where_it_fits() -> None:
    one = trainer_memory(EIGHT, weights="full", gpus=1)
    assert one.weights + one.gradients + one.optimizer == pytest.approx(16 * 8.2e9 / 2**30)  # (16 bytes a weight)
    four = trainer_memory(EIGHT, weights="full", gpus=4)
    assert four.optimizer == pytest.approx(one.optimizer / 4) and four.total < 80 < one.total
    lora = trainer_memory(EIGHT, weights="lora", gpus=2, gpu_gib=80)
    assert lora.whole_base and lora.weights > EIGHT.file_bytes / 2**30  # (whole, and its shard beside it)
    sharded = trainer_memory(EIGHT, weights="lora", gpus=2, gpu_gib=80, whole_base=False)
    assert not sharded.whole_base and sharded.weights < lora.weights
    longer = trainer_memory(EIGHT, weights="lora", gpus=1, segment_tokens=32_768)
    assert longer.activations > trainer_memory(EIGHT, weights="lora", gpus=1).activations
    assert "GiB a GPU (weights" in four.said()


def test_a_gpus_memory_is_known_by_its_runpod_id() -> None:
    assert gpu_memory_gib(("NVIDIA H100 80GB HBM3",)) == 80
    assert gpu_memory_gib(("NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA H100 80GB HBM3")) == 80  # (least)
    assert gpu_memory_gib(("NVIDIA GeForce RTX 4090",)) == 24 and gpu_memory_gib(("Some GPU",)) is None


def test_a_models_size_is_read_from_its_files(tmp_path: Path) -> None:
    config = {"architectures": ["Qwen3ForCausalLM"], "hidden_size": 64, "num_hidden_layers": 2, "vocab_size": 100,
              "tie_word_embeddings": True}  # fmt: skip
    (tmp_path / "config.json").write_text(json.dumps(config))
    header = json.dumps({"a": {"dtype": "BF16", "shape": [100, 64], "data_offsets": [0, 12_800]}}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(12_800))
    found = model_facts(str(tmp_path))
    assert found == ModelFacts(12_800, 6_400.0, 64, 2, 100, True)
    assert model_facts("someone/not-here", environ={"HF_HUB_OFFLINE": "1"}) is None  # (offline: the Hub is not asked)


def cluster_with(trainer: str) -> str:
    return f"""
name = "test"
[ledger]
url = "sqlite:///~/ledger.db"
token_env = "LEDGER_TOKEN"
public = "https://ledger.example.com"
[stores.r2]
kind = "rollout_s3:S3BlobStore"
bucket = "rollout"
[tls]
ca = "~/ca.pem"
certificate = "~/gateway.crt"
key = "~/gateway.key"
[inference.vllm]
kind = "vllm"
gpus = 1
[inference.vllm.models."Qwen/Qwen3-8B"]
context = 8192
[trainers.pods]
models = ["Qwen/Qwen3-8B"]
{trainer}
"""


def memory_findings(trainer: str, **settings: JsonValue) -> list[tuple[str, bool]]:
    cluster = parsed(tomllib.loads(cluster_with(trainer)))
    said = RunSettings({
        "kind": "train", "environment": "gridworld.environment:environment", "trainer.provider": "pods",
        "channels.policy.provider": "vllm", "channels.policy.model": "Qwen/Qwen3-8B",
        "channels.policy.renderer": "rollout_qwen:qwen3", **settings,
    })  # fmt: skip
    return [(each.reason, each.refuses) for each in check(said, cluster, model=EIGHT) if each.rule == "memory"]


POD = f"""kind = "runpod-trainer"
trainer = "full"
image = "i"
gpu_types = ["NVIDIA H100 80GB HBM3"]
step_ca = {STEP_CA}
store = "r2"
"""


def test_a_trainer_whose_gpus_cannot_hold_it_is_refused_and_told_how_many_would() -> None:
    ((reason, refuses),) = memory_findings(POD)
    assert refuses and "every weight of Qwen/Qwen3-8B on one GPU of 80 GiB: it would fit on 2 GPUs" in reason
    assert memory_findings(POD + "gpu_count = 4") == []
    assert memory_findings('kind = "lora"\ngpus = 1\ngpu_memory_gib = 96') == []  # (an adapter: the model fits)
    ((reason, refuses),) = memory_findings('kind = "lora"\ngpus = 1\ngpu_memory_gib = 16')
    assert refuses and "an adapter over Qwen/Qwen3-8B on one GPU of 16 GiB" in reason
    assert memory_findings('kind = "full"\ngpus = 1') == []  # (its GPUs' memory not known: nothing is said)


@pytest.mark.parametrize(
    ("table", "said"),
    [
        ('kind = "lora"\ngpus = 1.5', "gpus above one is a whole number"),
        ('kind = "lora"\ngpus = 2\ncolocate_with = "vllm"', "so it steps on one"),
        ('kind = "lora"\ngpus = 1\ngpu_memory_gib = -1', "gpu_memory_gib is the memory of each of its GPUs"),
    ],
)
def test_a_trainer_on_several_gpus_takes_them_whole_and_its_own(table: str, said: str) -> None:
    with pytest.raises(ClusterError, match=said):
        parsed(tomllib.loads(cluster_with(table)))


def test_a_hosts_pod_has_one_gpu() -> None:
    host = f"""
[inference.host]
kind = "runpod-host"
image = "i"
gpu_types = ["NVIDIA H100 80GB HBM3"]
gpu_count = 2
step_ca = {STEP_CA}
[inference.host.models."m"]
context = 8192
"""
    with pytest.raises(ClusterError, match="gpu_count is 1 for a host"):
        parsed(tomllib.loads(cluster_with('kind = "lora"\ngpus = 1') + host))


def test_a_local_trainer_on_two_gpus_asks_for_both() -> None:
    cluster = parsed(tomllib.loads(cluster_with('kind = "lora"\ngpus = 2')))
    said = RunSettings({
        "kind": "train", "trainer.provider": "pods", "channels.policy.provider": "vllm",
        "channels.policy.model": "Qwen/Qwen3-8B", "channels.policy.renderer": "rollout_qwen:qwen3",
    })  # fmt: skip
    asked = demand(said, cluster)
    trainer = asked.asks(TRAINER)
    assert trainer is not None and trainer.gpus == 2
    index = asked.index(TRAINER)
    assert index is not None and asked.bundles[index].resources.bundle()["GPU"] == 2  # (its own: the engine's apart)


async def test_a_launch_reads_the_trained_models_size_from_its_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rollout_train import launching, memory

    monkeypatch.setattr(memory, "model_facts", model_facts)  # (the files' own: tests know no size by default)
    test_a_models_size_is_read_from_its_files(tmp_path)
    cluster = parsed(tomllib.loads(cluster_with('kind = "full"\ngpus = 1\ngpu_memory_gib = 16')))
    said = RunSettings({"kind": "train", "trainer.provider": "pods", "trainer.model": str(tmp_path)})
    found = await launching.trained_model_facts(said, cluster)
    assert found is not None and found.hidden == 64
    unknown = parsed(tomllib.loads(cluster_with('kind = "full"\ngpus = 1')))  # (its GPUs' memory not known)
    assert await launching.trained_model_facts(said, unknown) is None
