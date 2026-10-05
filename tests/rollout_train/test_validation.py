"""Validation: one pure function says everything wrong with a run's settings on a cluster, given what is known of the
environment and the ledger. The acceptance run's settings pass every rule; each rule refuses what it should, with a
reason a person can act on; what only makes a run wait is a note, not a refusal."""

import dataclasses
import tomllib
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import JsonValue

from rollout_train.cluster import Cluster, parsed
from rollout_train.providers import Auth
from rollout_train.run_settings import RunSettings
from rollout_train.validation import (
    RULES,
    CheckpointFacts,
    EnvironmentFacts,
    Finding,
    LedgerFacts,
    Spend,
    SuiteFacts,
    check,
    estimated_spend,
    refusals,
    serves,
    spend_of,
    weights_of,
    with_weights,
)

ROOT = Path(__file__).resolve().parents[2]
GSM8K = "rollout_verifiers.environments:gsm8k"
MINECRAFT = "minecraft_team.environment:environment"
EXTRA = """
[tls]
ca = "~/.config/rollout/ca.pem"
certificate = "~/.config/rollout/gateway.crt"
key = "~/.config/rollout/gateway.key"

[inference.openai]
kind = "api"
endpoint = "rollout_openai:ResponsesEndpoint"
auth = { kind = "vendor", key_env = "OPENAI_API_KEY" }
[inference.openai.models."gpt-5"]
context = 400000
cost = { input = 1.25, output = 10.0 }

[inference.lab]
kind = "vllm-servers"
addresses = ["https://gpu-1.lab.example:8000"]
auth = { kind = "bearer", token_env = "ROLLOUT_ENGINES_TOKEN" }
[inference.lab.models."Qwen/Qwen3-0.6B"]
context = 8192
options = { max_lora_rank = 32 }

[tools.search]
url = "https://search.lab.example"
auth = { kind = "bearer", token_env = "SEARCH_TOKEN" }
"""


def the_cluster() -> Cluster:
    text = (ROOT / "deploy" / "clusters" / "example.toml").read_text() + EXTRA
    return parsed(tomllib.loads(text))


CLUSTER = the_cluster()
ACCEPTANCE: dict[str, JsonValue] = {
    "name": "gsm8k-tinker-4b",
    "environment": GSM8K,
    "trainer.provider": "tinker-lora",
    "channels.policy.provider": "local-vllm",
    "channels.policy.model": "Qwen/Qwen3.5-4B",
    "channels.policy.renderer": "rollout_qwen:qwen35",
    "channels.policy.thinking_tokens": 1024,
    "channels.policy.answer_tokens": 512,
    "trainer.rank": 16,
    "trainer.learning_rate": 1e-4,
    "evals.suite": "math",
    "evals.every": 1,
    "limits.spend": 2,
    "groups": 12,
    "groups_per_step": 4,
}
"""The acceptance run: Tinker trains Qwen3.5-4B, the local vLLM engines serve it through the Tinker to PEFT bridge."""
ENVIRONMENT = EnvironmentFacts(GSM8K, episodes_per_group=4, turns_per_episode=1, prompt_tokens=200)
LEDGER = LedgerFacts(
    suites={"math": SuiteFacts("math", 1, frozenset({GSM8K}))},
    names_taken=frozenset({"team-8"}),
    checkpoints={
        "tinker-run:3": CheckpointFacts("tinker-run:3", formats=frozenset({"tinker"}), model="Qwen/Qwen3.5-4B"),
        "both:2": CheckpointFacts("both:2", formats=frozenset({"tinker", "peft"}), model="Qwen/Qwen3.5-4B"),
        "lora-run:5": CheckpointFacts("lora-run:5", formats=frozenset({"peft"}), model="Qwen/Qwen3.5-4B"),
        "small-lora:1": CheckpointFacts("small-lora:1", formats=frozenset({"peft"}), model="Qwen/Qwen3-0.6B"),
        "full-run:1": CheckpointFacts("full-run:1", formats=frozenset({"full"}), model="Qwen/Qwen3-0.6B"),
        "gone:1": CheckpointFacts("gone:1", released=True, formats=frozenset({"peft"})),
    },
    gpus=1,
    gpus_free=1,
)
LOCAL_LORA: dict[str, JsonValue] = {
    "trainer.provider": "local-lora",
    "channels.policy.model": "Qwen/Qwen3-0.6B",
    "trainer.rank": 16,
    "trainer.learning_rate": 5e-5,
}
"""The acceptance run trained here instead, on a small model."""
FULL: dict[str, JsonValue] = {**LOCAL_LORA, "trainer.provider": "local-full", "trainer.rank": None}


def findings(
    changes: Mapping[str, JsonValue] | None = None,
    *,
    without: tuple[str, ...] = (),
    cluster: Cluster = CLUSTER,
    environment: EnvironmentFacts | None = ENVIRONMENT,
    ledger: LedgerFacts = LEDGER,
) -> list[Finding]:
    values = {**ACCEPTANCE, **(changes or {})}
    for key in without:
        values.pop(key)
    values = {key: value for key, value in values.items() if value is not None or key in (changes or {})}
    if values.get("trainer.rank", 0) is None:
        values.pop("trainer.rank")
    return check(RunSettings(values), cluster, environment, ledger)


def refused(rule: str, findings: list[Finding]) -> list[str]:
    return [each.reason for each in findings if each.rule == rule and each.refuses]


def noted(rule: str, findings: list[Finding]) -> list[str]:
    return [each.reason for each in findings if each.rule == rule and not each.refuses]


def with_provider(cluster: Cluster, name: str, **changes: object) -> Cluster:
    provider = dataclasses.replace(cluster.inference[name], **changes)  # pyright: ignore[reportArgumentType]
    return dataclasses.replace(cluster, inference={**cluster.inference, name: provider})


def with_trainer(cluster: Cluster, name: str, **changes: object) -> Cluster:
    trainer = dataclasses.replace(cluster.trainers[name], **changes)  # pyright: ignore[reportArgumentType]
    return dataclasses.replace(cluster, trainers={**cluster.trainers, name: trainer})


def test_the_acceptance_runs_settings_pass_every_rule() -> None:
    found = findings()
    assert refusals(found) == []
    # Tinker's settings dataclass is read where rollout_tinker is installed; elsewhere its keys wait for the trainer.
    assert all(each.rule == "settings" and "could not be read here" in each.reason for each in found)


def test_the_acceptance_run_trained_here_passes_too() -> None:
    assert findings(LOCAL_LORA) == []


def test_the_rules_are_reported_in_their_order() -> None:
    found = findings({"groups": 0, "name": "a/b", "channels.policy.provider": "openai"})
    order = [each.name for each in RULES]
    assert [each.rule for each in found] == sorted((each.rule for each in found), key=order.index)


# settings


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"groups": 0}, "groups is at least 1, not 0"),
        ({"eval.suite": "math"}, "eval.suite is not a setting a train run takes"),
        ({"colour": "red"}, "colour is not a run setting"),
        ({**LOCAL_LORA, "trainer.rank": "big"}, "trainer.rank is int, not 'big'"),
        ({**LOCAL_LORA, "trainer.wings": 2}, "the local-lora trainer takes no trainer.wings"),
        ({**FULL, "trainer.rank": 8}, "takes no trainer.rank: a full-weight trainer has no adapter"),
        ({"channels.policy.providers": ["local-vllm"]}, "names its providers once: provider or providers"),
        ({"channels.policy.routing": "weighted"}, "shares turns by weight: give each of its providers"),
        ({"channels.policy.weights": {"local-vllm": 1}}, "weights are for routing = weighted"),
        ({"channels.policy.mode": "fixed"}, "policy is the trained channel: it serves what the run trains"),
        ({"channels.rival.provider": "local-vllm", "channels.rival.mode": "follows"}, "follows another channel of"),
        ({"channels.rival.provider": "local-vllm", "channels.rival.lag": 2}, "follows and lag are for mode = follows"),
        ({"slots.agent-1": "nobody"}, "slot agent-1 samples channel nobody, which the settings do not describe"),
        ({"kind": "deploy"}, "kind is one of train, eval, imitate, check"),
    ],
)
def test_settings_refuse(changes: dict[str, JsonValue], reason: str) -> None:
    assert any(reason in each for each in refused("settings", findings(changes))), findings(changes)


def test_settings_refuse_what_a_kind_needs_and_lacks() -> None:
    assert refused("settings", findings(without=("environment",))) == ["a train run needs environment"]
    without_model = refused("settings", findings(without=("channels.policy.model",)))
    assert "the trained channel policy needs a model" in without_model


def test_settings_pass_channels_that_follow_and_attach_to_slots() -> None:
    changes: dict[str, JsonValue] = {
        "channels.rival.provider": "local-vllm", "channels.rival.model": "Qwen/Qwen3.5-4B",
        "channels.rival.mode": "follows", "channels.rival.follows": "policy", "channels.rival.lag": 5,
        "slots.agent-1": "policy", "slots.agent-2": "rival",
    }  # fmt: skip
    facts = dataclasses.replace(ENVIRONMENT, slots=frozenset({"agent-1", "agent-2"}))
    assert refusals(findings(changes, environment=facts)) == []
    facts = dataclasses.replace(ENVIRONMENT, slots=frozenset({"agent-1"}))
    assert refused("settings", findings(changes, environment=facts)) == [
        "the environment's programs have no slot agent-2 (agent-1)"
    ]


JUDGED = dataclasses.replace(
    ENVIRONMENT, slots=frozenset({"policy", "judge"}), untrained=frozenset({"judge"}), judges=frozenset({"judge"})
)
"""An environment whose policy answers and whose judge, not trained, scores the answer."""
JUDGE: dict[str, JsonValue] = {
    "channels.judge.provider": "openai", "channels.judge.model": "gpt-5", "channels.judge.mode": "fixed",
    "slots.judge": "judge",
}  # fmt: skip


def test_a_judge_bound_to_a_fixed_channel_passes() -> None:
    assert refusals(findings(JUDGE, environment=JUDGED)) == []
    following = {
        "channels.judge.provider": "local-vllm", "channels.judge.model": "Qwen/Qwen3.5-4B",
        "channels.judge.mode": "follows", "channels.judge.follows": "policy", "channels.judge.lag": 3,
        "slots.judge": "judge", "self_judging": True,
    }  # fmt: skip
    assert refusals(findings(following, environment=JUDGED)) == []  # (a snapshot of the policy, said so)


def test_a_slot_the_program_declares_must_be_bound() -> None:
    assert refused("settings", findings(environment=JUDGED)) == [
        "slot judge is not trained, so it samples no channel by default: bind it to one (slots.judge)"
    ]


def test_a_judges_channel_needs_a_provider_that_offers_its_model() -> None:
    assert refused("models", findings({**JUDGE, "channels.judge.model": "gpt-9"}, environment=JUDGED)) == [
        "provider openai does not serve gpt-9 (it serves gpt-5)"
    ]
    unprovided = refused("settings", findings({**JUDGE, "channels.judge.provider": None}, environment=JUDGED))
    assert unprovided == ["channel judge, which slot judge samples, needs a provider"]
    unmodelled = refused("settings", findings({**JUDGE, "channels.judge.model": None}, environment=JUDGED))
    assert unmodelled == ["channel judge, which slot judge samples, needs a model"]


@pytest.mark.parametrize(
    "changes",
    [
        {"slots.judge": "policy"},
        {
            "channels.rival.provider": "local-vllm", "channels.rival.model": "Qwen/Qwen3.5-4B",
            "channels.rival.mode": "follows", "channels.rival.follows": "policy", "slots.judge": "rival",
        },
    ],
    ids=["the trained channel", "a channel following it"],
)  # fmt: skip
def test_self_judging_needs_the_explicit_setting(changes: dict[str, JsonValue]) -> None:
    assert refused("settings", findings(changes, environment=JUDGED)) == [
        f"slot judge judges, and channel {changes['slots.judge']} serves the run's own checkpoints: the policy would "
        "judge itself (self_judging allows it)"
    ]
    assert refusals(findings({**changes, "self_judging": True}, environment=JUDGED)) == []


def test_a_following_channel_must_name_an_existing_channel() -> None:
    ghost = {**JUDGE, "channels.judge.mode": "follows", "channels.judge.follows": "ghost"}
    assert refused("settings", findings(ghost, environment=JUDGED)) == [
        "channel judge follows another channel of the run: name it"
    ]


# providers


def test_providers_refuse_what_the_cluster_does_not_offer() -> None:
    assert refused("providers", findings({"trainer.provider": "nowhere"})) == [
        "the cluster offers no trainer nowhere (it offers local-lora, local-full, tinker-lora)"
    ]
    reasons = refused("providers", findings({"channels.policy.provider": "elsewhere"}))
    assert reasons == ["the cluster offers no inference provider elsewhere (it offers local-vllm, tinker, openai, lab)"]
    assert refused("providers", findings()) == []


# auth


def test_auth_none_is_refused_away_from_this_machine() -> None:
    open_lab = with_provider(CLUSTER, "lab", auth=Auth("none"))
    reasons = refused("auth", findings({**FULL, "channels.policy.provider": "lab"}, cluster=open_lab))
    assert reasons == [
        "provider lab is reached with no auth at https://gpu-1.lab.example:8000, which is not this machine: auth none "
        "is only for localhost; say auth = mtls or bearer"
    ]
    assert refused("auth", findings({**LOCAL_LORA, "channels.policy.provider": "lab"})) == []  # (bearer)
    assert refused("auth", findings(LOCAL_LORA)) == []  # (none, on this machine)


# capabilities


def test_a_trained_channel_on_a_text_api_is_refused_with_its_reason() -> None:
    reasons = refused("capabilities", findings({**LOCAL_LORA, "channels.policy.provider": "openai",
                                                "channels.policy.model": "gpt-5"}))  # fmt: skip
    assert reasons == [
        "channel policy is trained, and provider openai (api) returns text, not the sampled token ids and their "
        "logprobs, which the importance weight needs"
    ]


def test_a_trained_channel_needs_honoured_sampling() -> None:
    capabilities = dataclasses.replace(CLUSTER.inference["local-vllm"].capabilities, honours_sampling=False)
    reasons = refused("capabilities", findings(cluster=with_provider(CLUSTER, "local-vllm", capabilities=capabilities)))
    assert reasons == [
        "channel policy is trained, and provider local-vllm (vllm) lacks honoured temperature and top-p: the "
        "importance weight needs the behaviour logprob of each exact sampled token"
    ]


def test_another_channel_may_be_any_provider() -> None:
    judge: dict[str, JsonValue] = {"channels.judge.provider": "openai", "channels.judge.model": "gpt-5"}
    assert refused("capabilities", findings(judge)) == []


# bridge


def test_a_bridge_must_exist_for_the_pair() -> None:
    reasons = refused("bridge", findings({**LOCAL_LORA, "channels.policy.provider": "tinker",
                                          "channels.policy.model": "Qwen/Qwen3.5-4B"}))  # fmt: skip
    assert reasons == [
        "no bridge from the local-lora trainer's peft checkpoints to what provider tinker loads (tinker): Tinker "
        "samples only checkpoints Tinker trained: there is no upload"
    ]
    reasons = refused("bridge", findings({**FULL, "channels.policy.provider": "lab"}))
    assert reasons == [
        "no bridge from the local-full trainer's full checkpoints to what provider lab loads (peft): full weights are "
        "not an adapter: serve them on a provider that reloads full weights"
    ]
    assert refused("bridge", findings({**FULL, "channels.policy.provider": "local-vllm"})) == []


def test_a_bridge_for_an_evals_subject() -> None:
    evaluation: dict[str, JsonValue] = {
        "kind": "eval", "environment": GSM8K, "eval.suite": "math", "start": "lora-run:5",
        "channels.policy.provider": "tinker", "channels.policy.model": "Qwen/Qwen3.5-4B",
    }  # fmt: skip
    found = check(RunSettings(evaluation), CLUSTER, ENVIRONMENT, LEDGER)
    assert refused("bridge", found) == [
        "no bridge from checkpoint lora-run:5's peft checkpoints to what provider tinker loads (tinker): Tinker "
        "samples only checkpoints Tinker trained: there is no upload"
    ]
    on_vllm = check(RunSettings({**evaluation, "start": "tinker-run:3", "channels.policy.provider": "local-vllm"}),
                    CLUSTER, ENVIRONMENT, LEDGER)  # fmt: skip
    assert refusals(on_vllm) == []


# weights


def test_adapters_need_a_provider_that_serves_them() -> None:
    capabilities = dataclasses.replace(CLUSTER.inference["lab"].capabilities, adapters=False)
    cluster = with_provider(CLUSTER, "lab", capabilities=capabilities)
    reasons = refused("weights", findings({**LOCAL_LORA, "channels.policy.provider": "lab"}, cluster=cluster))
    assert reasons == ["provider lab (vllm-servers) serves no adapters, and the run trains a LoRA"]
    assert refused("weights", findings({**LOCAL_LORA, "channels.policy.provider": "lab"})) == []


def test_full_weights_need_a_provider_that_reloads_them() -> None:
    capabilities = dataclasses.replace(CLUSTER.inference["local-vllm"].capabilities, full_reload=False)
    cluster = with_provider(CLUSTER, "local-vllm", capabilities=capabilities)
    reasons = refused("weights", findings(FULL, cluster=cluster))
    assert reasons == [
        "provider local-vllm (vllm) cannot reload full weights in place, and the run trains full weights"
    ]
    assert refused("weights", findings(FULL)) == []


def test_a_run_says_what_it_trains_and_its_trainer_must_make_it() -> None:
    assert refusals(findings({"weights": "lora"})) == []
    assert refused("weights", findings({"weights": "full"})) == [
        "the tinker-lora trainer trains a LoRA only, and the run trains full weights"
    ]
    assert refused("weights", findings({**FULL, "weights": "lora"})) == [
        "the local-full trainer trains full weights only, and the run trains a LoRA"
    ]
    assert refused("settings", findings({"weights": "both"})) == ["weights is one of lora, full, not 'both'"]
    evaluation: dict[str, JsonValue] = {"kind": "eval", "environment": GSM8K, "eval.suite": "math", "weights": "lora"}
    assert refused("settings", check(RunSettings(evaluation), CLUSTER, ENVIRONMENT, LEDGER)) == [
        "weights is not a setting a eval run takes"
    ]


def test_unsaid_a_run_trains_what_its_trainer_makes_and_its_start_records_it() -> None:
    assert weights_of(RunSettings(ACCEPTANCE), CLUSTER) == "lora"
    assert weights_of(RunSettings({**ACCEPTANCE, **FULL}), CLUSTER) == "full"
    assert weights_of(RunSettings({"kind": "train"}), CLUSTER) is None
    said = with_weights(RunSettings({**ACCEPTANCE, **FULL}), CLUSTER)
    assert said["weights"] == "full" and said.split()[0]["weights"] == "full"
    assert with_weights(RunSettings({**ACCEPTANCE, "weights": "full"}), CLUSTER)["weights"] == "full"  # (as said)
    assert "weights" not in with_weights(RunSettings({"kind": "eval"}), CLUSTER).values


def test_full_weights_are_never_served_by_tinkers_sampler() -> None:
    on_tinker = {**FULL, "channels.policy.provider": "tinker", "channels.policy.model": "Qwen/Qwen3.5-4B"}
    assert (
        "provider tinker (tinker) cannot reload full weights in place, and the run trains full weights: Tinker's "
        "sampler serves only checkpoints Tinker trained" in refused("weights", findings(on_tinker))
    )
    assert serves(CLUSTER.inference["tinker"], "lora") is None
    assert (
        serves(CLUSTER.inference["openai"], "lora") == "provider openai (api) serves no adapters, and the run "
        "trains a LoRA"
    )


def test_a_lora_merged_for_its_provider_needs_full_reload() -> None:
    capabilities = dataclasses.replace(CLUSTER.inference["local-vllm"].capabilities, full_reload=False)
    cluster = with_provider(CLUSTER, "local-vllm", capabilities=capabilities)
    merged = {**LOCAL_LORA, "channels.policy.bridge": "merge-quantize"}
    assert (
        "provider local-vllm (vllm) cannot reload full weights in place, and channel policy serves the run's LoRA "
        "merged into them (bridge = merge-quantize)" in refused("weights", findings(merged, cluster=cluster))
    )


# models


def test_models_must_be_offered_and_be_the_one_trained() -> None:
    assert refused("models", findings({"trainer.model": "Qwen/Qwen3-0.6B"})) == [
        "the tinker-lora trainer does not train Qwen/Qwen3-0.6B here (it trains Qwen/Qwen3.5-4B, Qwen/Qwen3.5-9B)",
        "channel policy serves the run's checkpoints, trained over Qwen/Qwen3-0.6B, and Qwen/Qwen3.5-4B is neither "
        "that model nor quantized from it",
    ]
    assert refused(
        "models", findings({"channels.policy.model": "Qwen/Qwen3.5-27B", "trainer.model": "Qwen/Qwen3.5-4B"})
    ) == [
        "provider local-vllm does not serve Qwen/Qwen3.5-27B (it serves Qwen/Qwen3.5-4B, cyankiwi/Qwen3.5-9B-AWQ-4bit, "
        "Qwen/Qwen3-0.6B)"
    ]


def test_an_adapter_over_the_base_serves_on_its_quantized_model() -> None:
    quantized: dict[str, JsonValue] = {"trainer.model": "Qwen/Qwen3.5-9B",
                                       "channels.policy.model": "cyankiwi/Qwen3.5-9B-AWQ-4bit"}  # fmt: skip
    assert refusals(findings(quantized)) == []


def test_a_start_trained_over_another_model_is_refused() -> None:
    reasons = refused("models", findings({**LOCAL_LORA, "start": "lora-run:5"}))
    assert reasons == ["the start lora-run:5 was trained over Qwen/Qwen3.5-4B, and this run trains Qwen/Qwen3-0.6B"]


# rank


def test_the_rank_the_provider_sees_must_fit() -> None:
    assert refused("rank", findings({"trainer.rank": 32})) == [
        "trainer.rank 32 times 3 (the bridge's rank factor for Qwen/Qwen3.5-4B) is 96, above provider local-vllm's "
        "max_lora_rank 64 for Qwen/Qwen3.5-4B"
    ]
    assert refused("rank", findings({"trainer.rank": 21})) == []
    assert refused("rank", findings({**LOCAL_LORA, "trainer.rank": 64})) == [
        "trainer.rank 64 is 64, above provider local-vllm's max_lora_rank 16 for Qwen/Qwen3-0.6B"
    ]


# segment


def test_segments_must_fit_the_trainer_and_the_context() -> None:
    assert refused("segment", findings({"trainer.segment_tokens": 40000})) == [
        "trainer.segment_tokens 40000 is above the 32768 the tinker-lora trainer takes here",
        "segments of up to 40000 tokens are longer than Qwen/Qwen3.5-4B's context on provider local-vllm (8192)",
    ]
    assert refused("segment", findings({"trainer.segment_tokens": 8192})) == []
    assert refused("segment", findings()) == []  # (unsaid: the trainer takes the longest the context leaves)


# start


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"start": "never:1"}, "there is no checkpoint never:1"),
        ({"start": "gone:1"}, "gone:1 was released: its weights were deleted"),
        ({**FULL, "start": "small-lora:1"}, "a full-weight trainer starts from full weights: merge this adapter first "
         "(`rollout merge small-lora:1`)"),
        ({"start": "lora-run:5"}, "Tinker trains only from checkpoints Tinker made, and lora-run:5 is not one: there "
         "is no upload"),
        ({**LOCAL_LORA, "channels.policy.model": "Qwen/Qwen3.5-4B", "trainer.rank": 8, "start": "tinker-run:3"},
         "tinker-run:3 is Tinker's: bridge it to PEFT first, or start a new line"),
    ],
)  # fmt: skip
def test_start_refuses(changes: dict[str, JsonValue], reason: str) -> None:
    assert refused("start", findings(changes)) == [reason]


def test_start_passes_what_the_trainer_starts_from() -> None:
    assert refused("start", findings({"start": "tinker-run:3"})) == []
    assert refused("start", findings({"start": "both:2"})) == []
    assert refused("start", findings({**FULL, "start": "full-run:1"})) == []
    assert refused("start", findings({**LOCAL_LORA, "start": "full-run:1"})) == []  # (an adapter over full weights)


# objective


def test_an_objective_the_trainer_does_not_take_is_refused() -> None:
    capabilities = dataclasses.replace(CLUSTER.trainers["local-lora"].capabilities, families=frozenset({"likelihood"}))
    cluster = with_trainer(CLUSTER, "local-lora", capabilities=capabilities)
    assert refused("objective", findings(LOCAL_LORA, cluster=cluster)) == [
        "the local-lora trainer does not take a policy_gradient objective (it takes likelihood)"
    ]
    assert refused("objective", findings({**LOCAL_LORA, "trainer.objective": "likelihood"}, cluster=cluster)) == []
    assert refused("objective", findings({**LOCAL_LORA, "trainer.ratio": "segment"})) == []


def test_a_component_the_family_does_not_accept_and_a_combination_that_means_nothing_are_refused() -> None:
    found = findings({**LOCAL_LORA, "objective.preset": "dpo", "objective.clip.low": 0.1})
    assert refused("objective", found) == [
        "objective.clip.low is not a component of a preference objective (it is of policy_gradient)"
    ]
    assert [each.key for each in found if each.rule == "objective"] == ["objective.clip.low"]
    assert refused("objective", findings({**LOCAL_LORA, "objective.preset": "reinforce", "objective.clip.kind": "ratio"
                                          })) == ["clip.kind ratio clips a ratio, and ratio is none"]  # fmt: skip
    simpo = {**LOCAL_LORA, "objective.preset": "simpo", "objective.reference": "base"}
    assert refused("objective", findings(simpo)) == ["the margin loss compares likelihoods alone: reference = none"]
    assert refused("settings", findings({**LOCAL_LORA, "objective.preset": "nothing"})) == [
        "objective.preset is one of default, reinforce, rloo, ppo_clip, grpo, dr_grpo, dapo, gspo, cispo, sft, dpo, "
        "ipo, simpo, kto, orpo, on_policy_distillation, distillation, mopd, mopd_top_k, not 'nothing'"
    ]
    assert refused("settings", findings({**LOCAL_LORA, "objective.clip.kind": "tight"})) == [
        "objective.clip.kind is one of none, ratio, weight, dual, not 'tight'"
    ]
    # A KL to the reference reads the base model, which follows from it.
    assert refused("objective", findings({**LOCAL_LORA, "objective.kl.target": "reference",
                                          "objective.kl.coefficient": 0.01})) == []  # fmt: skip


def test_an_importance_correction_needs_behaviour_logprobs_and_a_preference_loss_does_not() -> None:
    api = {**LOCAL_LORA, "channels.policy.provider": "openai", "channels.policy.model": "gpt-5"}
    capabilities = dataclasses.replace(CLUSTER.inference["local-vllm"].capabilities, sampled_logprobs=False)
    without = with_provider(CLUSTER, "local-vllm", capabilities=capabilities)
    assert refused("capabilities", findings(LOCAL_LORA, cluster=without)) == [
        "channel policy is trained, and provider local-vllm (vllm) lacks sampled-token logprobs: the importance weight "
        "needs the behaviour logprob of each exact sampled token"
    ]
    assert refused("capabilities", findings({**LOCAL_LORA, "objective.preset": "reinforce"}, cluster=without)) == []
    assert refused("capabilities", findings({**api, "objective.preset": "reinforce"})) == [
        "channel policy is trained by a policy gradient, and provider openai (api) does not return the exact tokens "
        "it sampled"
    ]
    for preset in ("dpo", "simpo", "kto", "sft"):
        assert refused("capabilities", findings({**api, "objective.preset": preset})) == [], preset


def test_a_k1_kl_penalty_in_the_loss_is_refused_and_one_in_the_reward_is_not() -> None:
    k1: dict[str, JsonValue] = {**LOCAL_LORA, "objective.preset": "grpo", "objective.kl.estimator": "k1"}
    found = findings(k1)
    assert refused("objective", found) == [
        "a k1 KL penalty in the loss does not pull toward its target: its gradient is that of the policy's logprob "
        "alone, which averages to 0 over the policy's own samples, so it adds noise and no pull. kl.estimator = k3 in "
        "the loss, or kl.placement = reward with k1"
    ]
    assert [each.key for each in found if each.rule == "objective"] == ["objective.kl.estimator"]
    assert refused("objective", findings({**k1, "objective.kl.placement": "reward"})) == []


def test_a_reference_the_trainer_cannot_give_is_refused() -> None:
    assert refused("objective", findings({"objective.preset": "dpo"})) == [
        "the tinker-lora trainer gives no reference logprobs (Tinker's SDK offers prompt logprobs from a sampler of "
        "the base model, not yet confirmed by a live test), and the objective reads them (the sigmoid loss)"
    ]
    assert refused("objective", findings({"objective.preset": "grpo"})) == [
        "the tinker-lora trainer gives no reference logprobs (Tinker's SDK offers prompt logprobs from a sampler of "
        "the base model, not yet confirmed by a live test), and the objective reads them (a KL to the reference)"
    ]
    assert refused("objective", findings({"objective.preset": "simpo"})) == []  # (no reference: Tinker takes it)
    assert refused("objective", findings({"objective.preset": "dapo", "objective.entropy.coefficient": 0.01})) == [
        "the tinker-lora trainer gives no entropies, and an entropy bonus reads them"
    ]
    assert refused("objective", findings({**FULL, "objective.preset": "dpo"})) == [
        "the local-full trainer holds a reference only when asked, and the objective reads it (the sigmoid loss): "
        "trainer.frozen_reference = true keeps a frozen copy of the model beside the policy"
    ]
    assert refused("objective", findings({**FULL, "objective.preset": "dpo", "trainer.frozen_reference": True})) == []
    assert refused("objective", findings({**LOCAL_LORA, "objective.preset": "dpo"})) == []  # (the adapter off)
    assert refused("settings", findings({**LOCAL_LORA, "trainer.frozen_reference": True})) == [
        "the local-lora trainer takes no trainer.frozen_reference: an adapter's reference is the model with the "
        "adapter switched off"
    ]


def test_an_imitate_run_trains_a_likelihood_or_a_preference() -> None:
    imitate: dict[str, JsonValue] = {"kind": "imitate", "trainer.provider": "local-lora", "imitation.dataset": "d"}
    assert refused("objective", check(RunSettings({**imitate, "objective.preset": "dapo"}), CLUSTER)) == [
        "an imitate run trains on a dataset's examples, which have no advantages: a likelihood or preference preset, "
        "not dapo"
    ]
    assert refused("objective", check(RunSettings({**imitate, "objective.preset": "orpo"}), CLUSTER)) == []
    assert refused("objective", check(RunSettings(imitate), CLUSTER)) == []  # (none named: sft)


# evals


def test_evals_name_suites_that_exist() -> None:
    assert refused("evals", findings({"evals.suite": "physics"})) == [
        "there is no suite physics: make it with `rollout suite make physics …` (a name never becomes a suite by "
        "itself)"
    ]
    assert refused("evals", findings({"evals.suite": "math@3"})) == ["suite math has versions 1 to 1, not 3"]
    assert refused("evals", findings({"evals.suite": "math@1"})) == []
    elsewhere = LedgerFacts(suites={"math": SuiteFacts("math", 2, frozenset({"far:away"}))})
    assert refused("evals", findings(ledger=elsewhere)) == [
        "suite math plays far:away, which this cluster does not offer"
    ]


# distillation


def teacher(
    provider: str, model: str, renderer: str = "rollout_qwen:qwen35", *, preset: str = "mopd", channel: str = "teacher"
) -> dict[str, JsonValue]:
    """A distillation by `preset` whose every route goes to `channel`, served by `provider`."""
    return {
        "objective.preset": preset, "objective.distillation.teachers": {"*": channel},
        f"channels.{channel}.provider": provider, f"channels.{channel}.model": model,
        f"channels.{channel}.renderer": renderer,
    }  # fmt: skip


def test_a_teacher_needs_the_logprobs_distillation_reads() -> None:
    assert refused("distillation", findings(teacher("tinker", "Qwen/Qwen3.5-9B"))) == [
        "provider tinker's prompt logprobs and top logprobs are declared by its SDK but not yet confirmed by a live "
        "test: it cannot teach until they are"
    ]
    assert refused("distillation", findings(teacher("openai", "gpt-5", "rollout_openai:text"))) == [
        "the teacher's provider openai (api) does not return prompt logprobs, to score the student's tokens",
        "the teacher renders as rollout_openai:text, of another family than the student's rollout_qwen:qwen35: their "
        "tokens do not compare",
    ]
    top = {**teacher("local-vllm", "Qwen/Qwen3.5-4B", preset="mopd_top_k"), "trainer.provider": "local-lora"}
    assert refused("distillation", findings(top)) == [
        "the teacher's provider local-vllm (vllm) returns at most 20 top logprobs a position, and the objective reads "
        "64 (its max_logprobs)"
    ]
    assert refused("distillation", findings({**top, "objective.distillation.top_k": 20})) == []
    assert refused("distillation", findings(teacher("local-vllm", "Qwen/Qwen3.5-4B"))) == []
    assert refused("distillation", findings(teacher("local-vllm", "Qwen/Qwen3.5-4B", "rollout_qwen:qwen3"))) == []
    mixed = {**teacher("local-vllm", "Qwen/Qwen3.5-4B", preset="dapo"), "objective.distillation.coefficient": 0.1}
    assert refused("distillation", findings(mixed)) == [] and refused("objective", findings(mixed)) == []


def test_a_distillation_needs_a_teacher_for_every_route_and_for_the_environment_it_plays() -> None:
    assert refused("distillation", findings({"objective.preset": "mopd"})) == [
        'a distillation needs a teacher channel for each route it plays: objective.distillation.teachers = {"*" = '
        '"teacher"}, say'
    ]
    routes: dict[str, JsonValue] = {"objective.distillation.teachers": {GSM8K: "teacher", "other:env": "elsewhere"}}
    assert refused("distillation", findings({**teacher("local-vllm", "Qwen/Qwen3.5-4B"), **routes})) == [
        "the teacher's channel elsewhere has no provider"
    ]
    away: dict[str, JsonValue] = {"objective.distillation.teachers": {"other:env": "teacher"}}
    assert refused("distillation", findings({**teacher("local-vllm", "Qwen/Qwen3.5-4B"), **away})) == [
        f"no route names {GSM8K}, which the run plays: route it, or every environment (`*`)"
    ]
    rows: dict[str, JsonValue] = {"objective.distillation.teachers": {f"{GSM8K}/hard": "teacher"}}
    said = findings({**teacher("local-vllm", "Qwen/Qwen3.5-4B"), **rows})
    assert refused("distillation", said) == [] and noted("distillation", said) == [
        f"only some rows of {GSM8K} are routed: an episode of another row is not scored, and trains nothing"
    ]


def test_a_teacher_needs_a_trainer_that_scores() -> None:
    capabilities = dataclasses.replace(CLUSTER.trainers["tinker-lora"].capabilities, scores=False)
    cluster = with_trainer(CLUSTER, "tinker-lora", capabilities=capabilities)
    reasons = refused("distillation", findings(teacher("local-vllm", "Qwen/Qwen3.5-4B"), cluster=cluster))
    assert reasons == ["the tinker-lora trainer does not score tokens, which distillation needs"]


def test_tinker_trains_the_policy_gradient_form_of_distillation_and_not_the_top_k_form() -> None:
    assert refused("objective", findings(teacher("local-vllm", "Qwen/Qwen3.5-4B"))) == []
    assert refused("objective", findings(teacher("local-vllm", "Qwen/Qwen3.5-4B", preset="mopd_top_k"))) == [
        "the tinker-lora trainer gives the logprobs of the sampled tokens only, and the top_k form reads the "
        "student's logprobs of the teacher's top-k tokens: distillation.form = policy_gradient"
    ]
    local = {**teacher("local-vllm", "Qwen/Qwen3.5-4B", preset="mopd_top_k"), "trainer.provider": "local-lora"}
    assert refused("objective", findings(local)) == []


# environment


def test_the_environment_must_be_offered_load_and_find_what_it_needs() -> None:
    assert refused("environment", findings({"environment": "nowhere:env"})) == [
        f"this cluster does not offer nowhere:env (it offers {MINECRAFT}, gridworld.environment:environment, {GSM8K})"
    ]
    broken = dataclasses.replace(ENVIRONMENT, loads=False, why="ModuleNotFoundError: verifiers")
    assert refused("environment", findings(environment=broken)) == [
        f"{GSM8K} does not load: ModuleNotFoundError: verifiers"
    ]
    needy = dataclasses.replace(ENVIRONMENT, sandboxes=frozenset({"browser", "minecraft"}),
                                tool_sets=frozenset({"search", "maps"}))  # fmt: skip
    assert refused("environment", findings(environment=needy)) == [
        f"{GSM8K} needs sandboxes of kind browser, and this cluster has no pool of them",
        f"{GSM8K} imports the tool set maps, which this cluster does not serve ([tools])",
    ]


# capacity


def test_more_gpus_than_the_cluster_has_is_refused_and_more_than_are_free_waits() -> None:
    small = dataclasses.replace(LEDGER, gpus=0.5, gpus_free=0.5)
    assert refused("capacity", findings(ledger=small)) == [
        "the run needs 1 GPUs, and the cluster has 0.5: it would never start"
    ]
    busy = dataclasses.replace(LEDGER, gpus_free=0)
    found = findings(ledger=busy)
    assert refused("capacity", found) == [] and noted("capacity", found) == [
        "the run needs 1 GPUs, and 0 are free: it waits"
    ]
    assert refused("capacity", findings(LOCAL_LORA)) == []  # (colocated: the trainer shares the engines' GPU)


# spend


def test_a_spend_limit_below_one_step_is_refused() -> None:
    dear = with_trainer(CLUSTER, "tinker-lora", cost={"train": 100.0}, costs={})
    settings = RunSettings(ACCEPTANCE)
    turns = 4 * 4 * 1  # groups a step, episodes a group, turns an episode
    trained = turns * (1536 + 200)  # every turn's sampled budget and its prompt
    assert estimated_spend(settings, dear, ENVIRONMENT) == pytest.approx(trained * 100 / 1e6)
    assert refused("spend", findings(cluster=dear)) == [
        "one step is estimated at up to $2.78, above limits.spend $2: the run would stop before its first step"
    ]
    assert refused("spend", findings({"limits.spend": 3}, cluster=dear)) == []
    unknown = findings(environment=None)
    assert refused("spend", unknown) == [] and noted("spend", unknown) == [
        "one step's spend cannot be estimated yet (the environment's numbers are not known here): the run still ends "
        "once its spend reaches $2"
    ]


def test_a_trainer_prices_each_model_it_trains_by_its_own_costs() -> None:
    priced = with_trainer(CLUSTER, "tinker-lora", cost={"train": 1.0}, costs={"Qwen/Qwen3.5-4B": {"train": 3.0}})
    settings = RunSettings(ACCEPTANCE)
    trained = 4 * 4 * 1 * (1536 + 200)
    assert estimated_spend(settings, priced, ENVIRONMENT) == pytest.approx(trained * 3.0 / 1e6)
    other = RunSettings({**ACCEPTANCE, "trainer.model": "Qwen/Qwen3.5-9B"})
    assert estimated_spend(other, priced, ENVIRONMENT) == pytest.approx(trained * 1.0 / 1e6)


def test_only_the_metered_parts_are_spent_by_the_step() -> None:
    sampled, prompts = 16 * 1536, 16 * 200
    on_tinker = spend_of(RunSettings({**ACCEPTANCE, "channels.policy.provider": "tinker"}), CLUSTER, ENVIRONMENT)
    assert on_tinker.parts == {
        "tinker-lora": pytest.approx((sampled + prompts) * 0.737 / 1e6),
        "tinker": pytest.approx((sampled * 1.005 + prompts * 0.33) / 1e6),
    }
    assert on_tinker.dollars == pytest.approx(sum(on_tinker.parts.values()))
    acceptance = RunSettings(ACCEPTANCE)
    assert spend_of(acceptance, CLUSTER, ENVIRONMENT).parts.keys() == {"tinker-lora"}  # (local-vllm is placed)
    assert spend_of(RunSettings({**ACCEPTANCE, **LOCAL_LORA}), CLUSTER, ENVIRONMENT) == Spend(0.0)
    turns = dataclasses.replace(ENVIRONMENT, turns_per_episode=None)
    assert spend_of(RunSettings(ACCEPTANCE), CLUSTER, turns) == Spend(
        None, why="the environment does not say how many turns an episode takes"
    )
    hourly = with_trainer(CLUSTER, "tinker-lora", costs={}, cost={"hour": 2.0})
    assert spend_of(RunSettings(ACCEPTANCE), hourly, ENVIRONMENT).why == "the tinker-lora trainer is priced by the hour"
    grouped = dataclasses.replace(ENVIRONMENT, episodes_per_group=None)  # (the objective's group size: 4)
    assert spend_of(acceptance, CLUSTER, grouped) == spend_of(acceptance, CLUSTER, ENVIRONMENT)


def test_a_run_on_metered_parts_with_no_spend_limit_is_told_so() -> None:
    assert noted("spend", findings(without=("limits.spend",))) == [
        "tinker-lora is metered, and no limits.spend bounds what the run spends"
    ]
    on_tinker = findings({"channels.policy.provider": "tinker"}, without=("limits.spend",))
    assert noted("spend", on_tinker) == [
        "tinker, tinker-lora are metered, and no limits.spend bounds what the run spends"
    ]
    assert noted("spend", findings(LOCAL_LORA, without=("limits.spend",))) == []


def test_only_scheduled_parts_ask_for_the_clusters_gpus() -> None:
    lab: dict[str, JsonValue] = {
        **LOCAL_LORA, "channels.policy.provider": "lab", "trainer.provider": "tinker-lora",
        "channels.policy.model": "Qwen/Qwen3-0.6B",
    }  # fmt: skip
    placed = with_provider(CLUSTER, "lab", gpus=2.0)
    assert refused("capacity", findings(lab, cluster=placed)) == [
        "the run needs 2 GPUs, and the cluster has 1: it would never start"
    ]
    metered = with_provider(CLUSTER, "lab", gpus=2.0, allocation="metered")
    assert refused("capacity", findings(lab, cluster=metered)) == []


# name


def test_a_name_must_be_one_and_free() -> None:
    assert refused("name", findings({"name": "a/b"})) == [
        "'a/b' cannot be a name: it must say something, and none of '/', '@', ':' (nor be 'base')"
    ]
    assert refused("name", findings({"name": "team-8"})) == ["another run is called 'team-8'"]
    assert refused("name", findings({"name": "team-9"})) == []
