"""A cluster config the tests of runs share (`a_cluster`): one machine, a database of files, a vLLM provider whose
engines are scripted, a trainer that trains nothing on the same card, and the test games; and the run settings of a
training run on it (`POLICY`)."""

from pathlib import Path

from pydantic import JsonValue

from rollout_train.cluster import Cluster, load

WORDS = "tests.rollout_train.rollouts.games:words"
JUDGED = "tests.rollout_train.rollouts.games:judged"
CLUSTER = """
name = "test"

[ledger]
url = "sqlite:///{root}/ledger.db"

[blobs]
directory = "{root}/blobs"

[scratch]
directory = "{root}/scratch"

[inference.local]
kind = "vllm"
engine = "rollout_train.testing:scripted_engine"
gpus = {gpus}
[inference.local.models.tiny]
context = 4096

[trainers.steps]
kind = "lora"
implementation = "rollout_train.testing:ScriptedTrainer"
gpus = 0.5
colocate_with = "local"
models = ["tiny"]
segment_tokens = 900

[environments."{words}"]
python = "platform"

[environments."{judged}"]
python = "platform"
"""
POLICY: dict[str, JsonValue] = {
    "trainer.provider": "steps",
    "channels.policy.provider": "local",
    "channels.policy.model": "tiny",
    "channels.policy.renderer": "rollout_train.testing:plain_renderer",
    "channels.policy.thinking_tokens": 64,
    "groups": 2,
    "groups_per_step": 1,
    "trainer.segments_per_step": 3,
    "episodes_at_once": 4,
}
"""A training run's settings on `a_cluster`, less its kind, name and environment."""


def a_cluster(root: Path, *, gpus: float = 0.5, more: str = "") -> Cluster:
    """The cluster config of `CLUSTER` under `root` (its provider's replicas each asking for `gpus`), with `more`
    tables written after it."""
    path = root / "cluster.toml"
    path.write_text(CLUSTER.format(root=root, gpus=gpus, words=WORDS, judged=JUDGED) + more)
    return load(path)
