"""A run on RunPod's pods, checked before it is asked for: a `runpod-host` pod is one pod for the run's trainer and
its trained channel, charged once; a step's spend on pods is their hourly price for as long as a step took here lately
(not known: said so); a run is refused more pods than a provider's `max_pods`, pods with no step-ca, or a cluster
whose pods cannot reach the ledger service; and pods are not the cluster's GPUs."""

import tomllib

import pytest

from rollout_train.cluster import Cluster, parsed
from rollout_train.pods.leasing import PodNeed, needs_of
from rollout_train.run_settings import RunSettings
from rollout_train.validation import EnvironmentFacts, LedgerFacts, check, spend_of

STEP_CA = '{ url = "https://ca.example.com", provisioner = "launcher", key_file = "~/p.jwk", root = "~/root.crt" }'


def cluster_of(ledger: str = 'token_env = "LEDGER_TOKEN"\npublic = "https://ledger.example.com"',
               host: str = f"step_ca = {STEP_CA}") -> Cluster:  # fmt: skip
    return parsed(
        tomllib.loads(f"""
name = "test"
[ledger]
url = "sqlite:///~/ledger.db"
{ledger}
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
    assert any(
        "get their certificates from step-ca" in each for each in refused(cluster_of(host=""), RunSettings(SETTINGS))
    )
    assert any("reach the ledger service at [ledger] public" in each
               for each in refused(cluster_of(ledger=""), RunSettings(SETTINGS)))  # fmt: skip
    gpus = check(RunSettings(SETTINGS), cluster_of(), ENVIRONMENT, LedgerFacts(gpus=0.0))
    assert not [each for each in gpus if each.refuses and "GPUs" in each.reason]  # (pods are not the cluster's GPUs)
