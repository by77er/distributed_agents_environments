"""The objective, declared: presets and overrides resolved, what follows from what, what each family accepts, how a
trainer's own settings name an objective, and how a run's settings hold and record it."""

import asyncio
from pathlib import Path

import pytest
from pydantic import JsonValue

from rollout_train.ledger import FileLedger
from rollout_train.objectives import (
    COMPONENTS,
    DEFAULT,
    FAMILIES,
    PRESETS,
    Objective,
    component,
    composed,
    from_trainer_settings,
    objective_of,
    problems,
    resolved,
)
from rollout_train.record import STARTS, scope, table, trained_objective
from rollout_train.run_settings import KEYS, RunSettings, flattened, key_of, layered, objective_in


def test_a_preset_and_overrides_resolve_to_a_full_objective() -> None:
    dapo = resolved("dapo", {"kl.target": "reference", "kl.coefficient": 0.01})
    assert (dapo.kl.target, dapo.kl.coefficient, dapo.reference, dapo.preset) == ("reference", 0.01, "base", "dapo")
    assert dapo.clip == PRESETS["dapo"].objective.clip  # (the rest is the preset's)
    assert resolved("dapo", {"kl.target": "old", "kl.coefficient": 0.01}).reference == "none"
    assert resolved("dpo", {"preference.loss": "margin"}).reference == "none"  # (a loss without one reads none)
    assert resolved("simpo", {"preference.loss": "sigmoid"}).reference == "base"
    assert resolved("dpo", {"preference.loss": "odds_ratio"}).preference.length_normalized
    assert resolved("default", {"clip.high": 1}).clip.high == 1.0 and isinstance(
        resolved("default", {"clip.high": 1}).clip.high, float
    )
    with pytest.raises(ValueError, match=r"objective.preset is one of"):
        resolved("ppo")
    with pytest.raises(ValueError, match="no component"):
        resolved("default", {"clip.width": 0.2})
    with pytest.raises(ValueError, match=r"objective.clip.low is at least 0, not -0.1"):
        resolved("default", {"clip.low": -0.1})
    with pytest.raises(ValueError, match="is one of none, untruncated, truncate, mask"):
        resolved("default", {"importance.correction": "clip"})


def test_a_family_accepts_its_components_and_refuses_the_rest() -> None:
    assert FAMILIES == ("policy_gradient", "preference", "likelihood", "distillation")
    accepted = {family: {each.key for each in COMPONENTS if family in each.families} for family in FAMILIES}
    assert "clip.low" in accepted["policy_gradient"] and "clip.low" not in accepted["preference"]
    assert "preference.beta" in accepted["preference"] and "advantage.baseline" in accepted["likelihood"]
    assert accepted["likelihood"] == {"advantage.baseline", "advantage.scale", "advantage.filter",
                                      "advantage.tiebreak", "aggregate", "constant_tokens"}  # fmt: skip
    assert problems(PRESETS["dpo"].objective, {"clip.low": 0.1}) == [
        ("clip.low", "objective.clip.low is not a component of a preference objective (it is of policy_gradient)")
    ]
    assert set(PRESETS["dpo"].objective.components()) == accepted["preference"]
    for preset in PRESETS.values():
        assert problems(preset.objective) == [], preset.name


def test_what_changes_between_steps_is_a_number() -> None:
    changeable = {each.key for each in COMPONENTS if each.changeable}
    assert changeable == {
        "clip.low", "clip.high", "clip.dual", "importance.cap", "importance.floor", "kl.coefficient",
        "entropy.coefficient", "preference.beta", "preference.margin", "preference.desirable",
        "preference.undesirable", "likelihood.coefficient", "distillation.temperature", "distillation.advantage_clip",
        "distillation.beta", "distillation.coefficient",
    }  # fmt: skip
    assert DEFAULT.changed({"clip.high": 0.3}).clip.high == 0.3
    with pytest.raises(ValueError, match=r"objective.ratio cannot change between steps"):
        DEFAULT.changed({"ratio": "segment"})
    with pytest.raises(ValueError, match=r"kl.target is none"):
        DEFAULT.changed({"kl.coefficient": 0.1})


def test_an_objective_records_itself_and_reads_back() -> None:
    made = resolved("grpo", {"kl.coefficient": 0.02})
    said = made.to_json()
    assert said["preset"] == "grpo" and said["family"] == "policy_gradient" and said["kl.coefficient"] == 0.02
    assert "preference.beta" not in said  # (only its family's components)
    assert Objective.from_json(said) == made
    assert objective_of(said) == made  # (a trainer given a recorded objective takes it as it is)
    assert objective_of({"preset": "grpo", "kl": {"coefficient": 0.02}}) == made  # (tables, or dotted keys)


def test_a_trainers_own_settings_name_an_objective() -> None:
    assert from_trainer_settings({"objective": "policy_gradient"}) == ("default", {})
    assert from_trainer_settings({"objective": "likelihood", "ratio": "segment"}) == ("sft", {})
    assert from_trainer_settings({"ratio": "segment", "truncate": None}) == (None, {
        "ratio": "segment", "importance.level": "segment", "aggregate": "segment_mean", "clip.low": 3e-4,
        "clip.high": 4e-4, "importance.correction": "untruncated",
    })  # fmt: skip
    assert from_trainer_settings({"clip_low": 0.1, "truncate": 3}) == (None, {"clip.low": 0.1, "importance.cap": 3.0})
    assert objective_of(None, {"ratio": "segment"}).aggregate == "segment_mean"
    assert objective_of("likelihood").family == "likelihood" and objective_of("dapo").preset == "dapo"
    assert objective_of(DEFAULT, {"clip_high": 0.3}).clip.high == 0.3


def test_a_runs_settings_hold_the_objective_as_its_keys() -> None:
    assert key_of("objective.preset") is not None and key_of("objective.kl.coefficient") is not None
    keys = {each.pattern: each for each in KEYS}
    assert not keys["objective.preset"].changeable and not keys["objective.ratio"].changeable
    assert keys["objective.kl.coefficient"].changeable and keys["objective.kl.coefficient"].default is None
    assert keys["objective.clip.kind"].choices == ("none", "ratio", "weight", "dual")
    # A trainer's own settings that named the objective are its keys, unless those are given.
    settings = RunSettings({"kind": "train", "trainer.objective": "policy_gradient", "trainer.ratio": "segment",
                            "trainer.clip_low": 0.5, "trainer.learning_rate": 1e-4})  # fmt: skip
    assert settings.values == {
        "kind": "train", "trainer.learning_rate": 1e-4, "objective.preset": "default", "objective.ratio": "segment",
        "objective.importance.level": "segment", "objective.aggregate": "segment_mean", "objective.clip.low": 3e-4,
        "objective.clip.high": 4e-4,
    }  # fmt: skip
    given = layered({"trainer.ratio": "segment"}, {"objective.preset": "gspo", "objective.clip.low": 1e-4})
    assert objective_in(given).preset == "gspo" and objective_in(given).clip.low == 1e-4
    assert objective_in(RunSettings({"kind": "train"})) == DEFAULT


def test_a_run_started_again_trains_with_the_objective_its_start_recorded(tmp_path: Path) -> None:
    async def recorded() -> dict[str, JsonValue] | None:
        ledger = FileLedger(tmp_path / "ledger")
        fence = await ledger.take(scope("r"))
        imitation: JsonValue = {
            "run_settings": {"fixed": {"kind": "imitate"}, "objective": PRESETS["sft"].objective.to_json()}
        }
        trained: JsonValue = {
            "run_settings": {"fixed": {"kind": "train"}, "objective": PRESETS["dapo"].objective.to_json()}
        }
        await ledger.append(table("r", STARTS), "1", trained, fence)
        await ledger.append(
            table("r", STARTS), "2", imitation, fence
        )  # (a step on a dataset since: not a training one)
        return await trained_objective(ledger, "r")

    found = asyncio.run(recorded())
    assert found is not None and objective_of(found) == PRESETS["dapo"].objective
    assert component("kl.coefficient") is not None and component("kl") is None


def test_a_distillation_accepts_its_components_and_a_policy_gradient_a_distillation_term() -> None:
    accepted = {each.key for each in COMPONENTS if "distillation" in each.families}
    assert accepted == {
        "importance.correction", "importance.level", "importance.cap", "importance.floor", "kl.target",
        "kl.estimator", "kl.placement", "kl.coefficient", "aggregate", "constant_tokens", "reference",
        "distillation.divergence", "distillation.form", "distillation.top_k", "distillation.temperature",
        "distillation.advantage_clip", "distillation.beta", "distillation.teachers",
    }  # fmt: skip
    mixed = resolved("dapo", {"distillation.coefficient": 0.2, "distillation.form": "top_k", "distillation.top_k": 8})
    assert mixed.distills and mixed.needs_top == 8 and mixed.needs_distribution
    assert not PRESETS["dapo"].objective.distills
    with pytest.raises(ValueError, match=r"distillation\.coefficient is 0"):
        resolved("dapo", {"distillation.top_k": 8})
    with pytest.raises(ValueError, match="not a component of a distillation objective"):
        resolved("mopd", {"distillation.coefficient": 0.5})
    with pytest.raises(ValueError, match="not to or from 0"):
        PRESETS["dapo"].objective.changed({"distillation.coefficient": 0.5})
    assert mixed.changed({"distillation.coefficient": 0.5}).distillation.coefficient == 0.5
    assert resolved("mopd", {"kl.target": "reference"}).reference == "base"  # (the reference follows the KL's target)


@pytest.mark.parametrize(
    ("preset", "overrides", "keys"),
    [
        ("mopd", {"distillation.divergence": "forward_kl"}, {"distillation.divergence"}),
        ("mopd", {"distillation.form": "top_k"}, {"distillation.top_k", "distillation.advantage_clip"}),
        ("mopd", {"distillation.temperature": 2.0}, {"distillation.temperature"}),
        ("mopd_top_k", {"distillation.temperature": 2.0}, {"distillation.temperature"}),
        ("distillation", {"distillation.divergence": "jsd", "distillation.beta": 1.0}, {"distillation.beta"}),
        ("mopd_top_k", {"kl.target": "old", "kl.coefficient": 0.1, "kl.placement": "reward"}, {"kl.placement"}),
    ],
)
def test_a_distillation_that_means_nothing_is_refused(
    preset: str, overrides: dict[str, JsonValue], keys: set[str]
) -> None:
    _, said = composed(preset, overrides)
    assert {key for key, _ in said} == keys, said


def test_teachers_are_a_table_of_channels_by_route_and_record_themselves() -> None:
    made = resolved("mopd", {"distillation.teachers": {"gsm8k:env": "math", "*": "general"}})
    said = made.to_json()
    assert said["distillation.teachers"] == {"gsm8k:env": "math", "*": "general"}
    assert Objective.from_json(said) == made and hash(made) == hash(Objective.from_json(said))
    assert objective_of({"preset": "mopd", "distillation": {"teachers": {"*": "t"}}}).distillation.teachers == {
        "*": "t"
    }
    with pytest.raises(ValueError, match="a table of text by text"):
        resolved("mopd", {"distillation.teachers": {"*": 3}})
    with pytest.raises(ValueError, match="a table of text by text"):
        resolved("mopd", {"distillation.teachers": "teacher"})
    keys = {each.pattern: each for each in KEYS}
    assert keys["objective.distillation.teachers"].types == ("table", "null")
    settings = layered(
        {"objective.preset": "mopd"}, flattened({"objective": {"distillation": {"teachers": {"*": "t"}}}})
    )
    assert objective_in(settings).distillation.teachers == {"*": "t"}
