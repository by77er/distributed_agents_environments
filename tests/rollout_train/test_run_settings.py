"""Run settings: the schema's keys and what each takes, settings given in layers (defaults, a preset, a file, flags),
what `--set` and a settings file say, a full copy split into fixed and changeable, and what changed between two."""

import dataclasses
import json
from pathlib import Path

import pytest

from rollout_train.cluster import Cluster, GatewaySection, GuardsSection, MonitorSection, RaySection, RunnersSection
from rollout_train.providers import INFERENCE_KINDS, TRAINER_KINDS, InferenceProvider, TrainerProvider, settings_of
from rollout_train.run_settings import (
    KEYS,
    Change,
    RunSettings,
    diff,
    from_file,
    from_flags,
    key_of,
    layered,
    recorded,
    shortcuts,
)


def test_keys_with_a_name_in_them_are_found() -> None:
    assert key_of("channels.policy.model") is key_of("channels.opponent.model")
    found = key_of("channels.policy.model")
    assert found is not None and found.pattern == "channels.*.model"
    assert key_of("slots.agent-2") is not None
    assert key_of("channels.policy") is None and key_of("channels..model") is None
    assert key_of("trainer.rank") is None  # (the trainer's own: its dataclass says)
    assert key_of("trainer.provider") is not None


def test_keys_that_kept_their_meaning_kept_their_names() -> None:
    patterns = {each.pattern for each in KEYS}
    assert {
        "groups_per_step",
        "max_lag",
        "evals.suite",
        "evals.every",
        "evals.episodes",
        "episodes_at_once",
    } <= patterns
    assert {"channels.*.thinking_tokens", "channels.*.answer_tokens", "channels.*.model", "channels.*.renderer"} <= (
        patterns
    )
    changeable = {each.pattern for each in KEYS if each.changeable}
    assert changeable == {
        "groups_per_step", "max_lag", "evals.suite", "evals.every", "evals.episodes", "limits.spend", "share",
    }  # fmt: skip


@pytest.mark.parametrize(
    ("key", "value", "problem"),
    [
        ("groups", 0, "is at least 1, not 0"),
        ("groups", 2.5, "is a whole number, not 2.5"),
        ("groups", "ten", 'is a whole number, not "ten"'),
        ("max_lag", -1, "is at least 0, not -1"),
        ("channels.policy.routing", "random", "is one of spill, weighted, not 'random'"),
        ("channels.policy.thinking_tokens", True, "is a whole number or null, not true"),
        ("share", 0, "is above 0, not 0"),
        ("limits.spend", -2, "is at least 0, not -2"),
    ],
)
def test_a_key_says_what_is_wrong_with_a_value(key: str, value: object, problem: str) -> None:
    found = key_of(key)
    assert found is not None
    assert found.problem(value) == problem  # pyright: ignore[reportArgumentType]


def test_a_key_takes_what_it_says() -> None:
    for key, value in [("groups", 3), ("groups", 3.0), ("limits.spend", 2), ("limits.spend", None), ("share", 0.5)]:
        found = key_of(key)
        assert found is not None and found.problem(value) is None, key


def test_set_flags_read_json_then_toml_then_text() -> None:
    said = from_flags([
        "trainer.rank=16", "trainer.learning_rate=1e-4", "evals.suite=null",
        "channels.policy.renderer=rollout_qwen:qwen35",
        'channels.policy.providers=["local-vllm", "pods"]', "channels.policy.weights={ local-vllm = 3, pods = 1 }",
        "limits.spend=2", "name=gsm8k tinker", "trainer.rank=32",
    ])  # fmt: skip
    assert said == {
        "trainer.rank": 32,  # (a later flag wins)
        "trainer.learning_rate": 1e-4,
        "evals.suite": None,
        "channels.policy.renderer": "rollout_qwen:qwen35",
        "channels.policy.providers": ["local-vllm", "pods"],
        "channels.policy.weights": {"local-vllm": 3, "pods": 1},
        "limits.spend": 2,
        "name": "gsm8k tinker",
    }
    with pytest.raises(ValueError, match="KEY=VALUE"):
        from_flags(["trainer.rank"])


def test_a_settings_file_of_dotted_keys_or_tables(tmp_path: Path) -> None:
    toml = tmp_path / "run.toml"
    toml.write_text(
        'environment = "rollout_verifiers.environments:gsm8k"\n"trainer.rank" = 16\n'
        "[trainer]\nlearning_rate = 1e-4\n"
        '[channels.policy]\nprovider = "local-vllm"\nweights = { local-vllm = 1 }\n'
    )
    assert from_file(toml) == {
        "environment": "rollout_verifiers.environments:gsm8k",
        "trainer.rank": 16,
        "trainer.learning_rate": 1e-4,
        "channels.policy.provider": "local-vllm",
        "channels.policy.weights": {"local-vllm": 1},  # (a table the schema takes stays one)
    }
    document = tmp_path / "run.json"
    document.write_text(json.dumps({"evals": {"suite": None}, "limits.spend": 2}))
    assert from_file(document) == {"evals.suite": None, "limits.spend": 2}


def test_layers_go_defaults_then_preset_then_file_then_flags() -> None:
    preset = {"trainer.rank": 32, "groups": 30, "trainer.learning_rate": 5e-5}
    file = {"groups": 12, "seed": 3}
    flags = from_flags(["groups=6"]) | shortcuts(model="Qwen/Qwen3.5-4B", provider="local-vllm", trainer="tinker-lora")
    settings = layered(preset, file, flags)
    assert settings["groups"] == 6 and settings["seed"] == 3 and settings["trainer.rank"] == 32
    assert settings["groups_per_step"] == 4  # (the schema's default)
    assert settings["channels.policy.model"] == "Qwen/Qwen3.5-4B" and settings["trainer.provider"] == "tinker-lora"
    assert settings.get("trainer.unknown", "fallback") == "fallback"
    with pytest.raises(KeyError):
        settings["nonsense"]


def test_shortcuts_set_the_channel_they_name() -> None:
    assert shortcuts(model="gpt-5", provider="openai", channel="judge") == {
        "channels.judge.model": "gpt-5",
        "channels.judge.provider": "openai",
    }


def test_settings_answer_what_a_run_is() -> None:
    settings = RunSettings({
        "channels.policy.model": "cyankiwi/Qwen3.5-9B-AWQ-4bit", "channels.policy.providers": ["local-vllm", "pods"],
        "channels.opponent.provider": "local-vllm", "channels.opponent.mode": "follows",
        "channels.opponent.follows": "policy", "channels.judge.provider": "openai",
    })  # fmt: skip
    assert settings.kind == "train" and settings.trained == "policy"
    assert settings.channels == ["policy", "opponent", "judge"]
    assert settings.providers("policy") == ("local-vllm", "pods") and settings.providers("judge") == ("openai",)
    assert settings.trainer_model == "cyankiwi/Qwen3.5-9B-AWQ-4bit"
    assert RunSettings({**settings.values, "trainer.model": "Qwen/Qwen3.5-9B"}).trainer_model == "Qwen/Qwen3.5-9B"
    assert [settings.mode(each) for each in settings.channels] == ["trained", "follows", "fixed"]
    assert RunSettings({"kind": "eval"}).trained is None


def test_a_run_records_a_full_copy_of_its_settings_and_the_preset_version() -> None:
    settings = layered(
        {"environment": "e:e", "trainer.provider": "local-lora", "trainer.learning_rate": 1e-4},
        shortcuts(model="Qwen/Qwen3-0.6B", provider="local-vllm"),
    )
    record = recorded(settings, settings_of(TRAINER_KINDS["lora"]), preset="minecraft-one-gpu@3")
    fixed, changeable = record["fixed"], record["changeable"]
    assert isinstance(fixed, dict) and isinstance(changeable, dict)
    assert record["preset"] == "minecraft-one-gpu@3"
    assert fixed["environment"] == "e:e" and fixed["groups"] == 100 and fixed["trainer.rank"] == 32
    assert fixed["channels.policy.model"] == "Qwen/Qwen3-0.6B" and fixed["channels.policy.bridge"] == "auto"
    assert changeable["trainer.learning_rate"] == 1e-4 and changeable["trainer.clip_low"] == 0.2
    assert changeable["groups_per_step"] == 4 and changeable["limits.spend"] is None
    assert "eval.suite" not in fixed and "imitation.passes" not in fixed  # (keys of other kinds of run)
    assert not set(fixed) & set(changeable)


def test_what_changed_between_two_settings() -> None:
    before = {"trainer.rank": 32, "groups": 30, "seed": 0}
    after = {"trainer.rank": 16, "groups": 30, "limits.spend": 2}
    assert diff(before, after) == [
        Change("limits.spend", after=2, added=True),
        Change("seed", before=0, removed=True),
        Change("trainer.rank", 32, 16),
    ]
    assert diff(before, before) == []


PROFILE_FIELDS = {
    # What each field of a profile became (delete with rollout_train.profile): `Class.field` of the cluster config,
    # or a run setting.
    "Profile.directory": "Cluster.scratch",
    "Profile.channels": "channels.*.provider",
    "Profile.trainer": "trainer.provider",
    "Profile.runner": "RunnersSection.durable",
    "Profile.serve": "GatewaySection.listen",
    "Profile.address": "GatewaySection.url",
    "Profile.tools": "Cluster.tools",
    "Profile.pools": "Cluster.sandboxes",
    "Profile.ledger": "Cluster.ledger",
    "Profile.blobs": "Cluster.blobs",
    "Profile.runs_gib": "GuardsSection.runs_gib",
    "Profile.training_gib": "GuardsSection.training_gib",
    "Profile.episodes_at_once": "episodes_at_once",
    "Profile.feed_runs": "MonitorSection.feed_episodes",
    "Profile.ray": "RaySection.address",
    "Profile.name": "name",
    "Profile.evals": "evals.suite",
    "Profile.gateway": "Cluster.gateway",
    "ChannelSpec.model": "channels.*.model",
    "ChannelSpec.renderer": "channels.*.renderer",
    "ChannelSpec.engine": "InferenceProvider.kind",
    "ChannelSpec.engines": "InferenceProvider.replicas",
    "ChannelSpec.thinking_tokens": "channels.*.thinking_tokens",
    "ChannelSpec.answer_tokens": "channels.*.answer_tokens",
    "ChannelSpec.reshard": "channels.*.bridge",
    "ChannelSpec.max_lag": "max_lag",
    "ChannelSpec.via": "vllm-servers.via",
    "ChannelSpec.connection": "InferenceProvider.auth",
    "TrainerSpec.kind": "TrainerProvider.kind",
    "TrainerSpec.channel": "trainer.channel",
    "TrainerSpec.start": "start",
    "TrainerSpec.bookmark": "bookmark",
    "TrainerSpec.colocated": "TrainerProvider.colocate_with",
    "TrainerSpec.settings": "trainer.rank",
    "EvalsSpec.suite": "evals.suite",
    "EvalsSpec.every": "evals.every",
    "EvalsSpec.episodes": "evals.episodes",
    "GatewaySpec.url": "GatewaySection.url",
    "GatewaySpec.listen": "GatewaySection.listen",
    "GatewaySpec.keys": "GatewaySection.keys",
    "GatewaySpec.lifetime": "GatewaySection.lifetime",
}


def test_every_field_of_a_profile_has_a_place() -> None:
    from rollout_train import profile

    described = {
        f"{each.__name__}.{field.name}"
        for each in (profile.Profile, profile.ChannelSpec, profile.TrainerSpec, profile.EvalsSpec, profile.GatewaySpec)
        for field in dataclasses.fields(each)
    }
    assert described == set(PROFILE_FIELDS)
    owners = (Cluster, GatewaySection, GuardsSection, MonitorSection, RaySection, RunnersSection, InferenceProvider,
              TrainerProvider)  # fmt: skip
    classes = {each.__name__: each for each in owners}
    lora = {each.key for each in settings_of(TRAINER_KINDS["lora"])}
    for field, place in PROFILE_FIELDS.items():
        owner, _, name = place.partition(".")
        if owner in classes:
            assert name in {each.name for each in dataclasses.fields(classes[owner])}, field
        elif owner in INFERENCE_KINDS:
            assert name in INFERENCE_KINDS[owner].fields, field
        else:
            assert key_of(place) is not None or place in lora, field
