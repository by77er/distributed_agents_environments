"""The commands that ask for runs: `rollout train`, `eval`, `imitate` and `env check` build a run's settings in layers
(`--preset`, `--settings`, `--set`, their flags), check them against the cluster config (`--check` stops there, saying
each refusal with its setting), and submit the run's job (`--detach` returns once it is submitted) or run it in this
process (`--here`, its launch recorded as a submitted one is); `rollout preset load` saves a directory of presets;
`rollout resume` fails, saying why, where the run's job cannot be made again."""

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rollout_train import jobs as jobs_module
from rollout_train import launching, resuming, submitting
from rollout_train.cli import main
from rollout_train.launches import FAILED, SUBMITTED, Asked, Launch, launches_of, new_launch
from rollout_train.record import RESULTS, recorded_settings, table
from rollout_train.stores import Stores
from rollout_train.submitting import RayJobs
from tests.local_ray import LocalRay
from tests.rollout_train.clusters import POLICY, WORDS, a_cluster
from tests.rollout_train.test_submitting import Jobs


def command(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> int:
    """`rollout ARGUMENTS`: its exit status (0 where it returns)."""
    monkeypatch.setattr("sys.argv", ["rollout", *arguments])
    try:
        main()
    except SystemExit as exit:
        return exit.code if isinstance(exit.code, int) else 1
    return 0


def backend_of(jobs: Jobs) -> Any:
    """What `submitting.backend_of` gives in these tests: the stand-in job server `jobs`."""

    def backend(cluster: Any) -> RayJobs:
        return RayJobs("x", jobs)

    return backend


def a_preset(tmp_path: Path) -> Path:
    """A cluster config under `tmp_path`, and the preset `small` (a training run's settings on it) saved beside its
    ledger; the config's path."""
    cluster = a_cluster(tmp_path)

    async def saved() -> None:
        await Stores.open(cluster).presets.save("small", {**POLICY, "environment": WORDS})

    asyncio.run(saved())
    return tmp_path / "cluster.toml"


def test_a_run_is_checked_against_the_cluster_config_and_its_refusals_said(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("rollout_lora", reason="a LoRA trainer's settings are checked: the gpu extra installs them")
    path = a_preset(tmp_path)
    assert command(monkeypatch, "train", WORDS, "--preset", "small", "--cluster", str(path), "--check") == 0
    said = capsys.readouterr().out
    assert "words: a train run that may be asked for" in said
    refused = ["--set", "trainer.rank=big", "--provider", "elsewhere"]
    assert command(monkeypatch, "train", WORDS, "--preset", "small", "--cluster", str(path), "--check", *refused) == 2
    said = capsys.readouterr().out
    assert "refused: channels.policy.provider: the cluster offers no inference provider elsewhere" in said
    assert "refused: trainer.rank:" in said
    assert command(monkeypatch, "train", WORDS, "--preset", "none-such", "--cluster", str(path), "--check") == 1


def test_a_run_is_submitted_by_its_settings_and_left_to_its_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = a_preset(tmp_path)
    jobs = Jobs()
    monkeypatch.setattr(submitting, "backend_of", backend_of(jobs))
    given = ["--preset", "small", "--groups", "5", "--set", "trainer.learning_rate=3e-5", "--name", "five groups"]
    assert command(monkeypatch, "train", WORDS, *given, "--cluster", str(path), "--detach") == 0
    assert "five groups (run_" in capsys.readouterr().out
    (submitted,) = jobs.submitted
    assert submitted["entrypoint"].startswith("python -m rollout_train.jobs launch_")

    async def asked() -> Any:
        launches = launches_of(Stores.open(a_cluster(tmp_path)).ledger)
        assert launches is not None
        return await launches.all()

    (launch,) = asyncio.run(asked())
    assert launch.state == SUBMITTED and launch.asked.preset == "small@1" and launch.asked.name == "five groups"
    assert launch.asked.settings["groups"] == 5 and launch.asked.settings["trainer.learning_rate"] == 3e-5


def test_a_run_asked_for_here_runs_its_job_in_this_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_ray: LocalRay
) -> None:
    path = a_preset(tmp_path)
    assert command(monkeypatch, "train", WORDS, "--preset", "small", "--cluster", str(path), "--here") == 0
    stores = Stores.open(a_cluster(tmp_path))

    async def recorded() -> tuple[Any, Any]:
        (run,) = await stores.registry.runs()
        return await stores.ledger.read(table(run.id, RESULTS)), await recorded_settings(stores.ledger, run.id)

    results, settings = asyncio.run(recorded())
    assert len(results) == 2 and settings is not None and settings["trainer.provider"] == "steps"


def test_a_run_asked_for_here_is_recorded_with_what_follows_from_its_settings_as_a_submitted_one_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ray

    import rollout_train.recorder.renderers as renderers

    cluster = a_cluster(tmp_path)

    async def saved() -> None:  # (its renderer left to what its model says it is rendered by)
        given = {key: value for key, value in POLICY.items() if key != "channels.policy.renderer"}
        await Stores.open(cluster).presets.save("small", {**given, "environment": WORDS})

    asyncio.run(saved())
    monkeypatch.setattr(renderers, "known", lambda: {"plain": "rollout_train.testing:plain_renderer"})
    monkeypatch.setattr(renderers, "rendered", lambda factory, name: name == "tiny")  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    driven: list[str] = []

    async def drive(launch: str, cluster: Any, stores: Any) -> int:
        driven.append(launch)
        return 0

    monkeypatch.setattr(jobs_module, "driven", drive)
    monkeypatch.setattr(launching, "ray_free", lambda: None)
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    path = tmp_path / "cluster.toml"
    assert command(monkeypatch, "train", WORDS, "--preset", "small", "--cluster", str(path), "--here") == 0

    async def asked() -> list[Launch]:
        launches = launches_of(Stores.open(cluster).ledger)
        assert launches is not None
        return await launches.all()

    (launch,) = asyncio.run(asked())
    assert driven == [launch.id]
    said = launch.asked.settings
    assert said["channels.policy.renderer"] == "rollout_train.testing:plain_renderer" and said["weights"] == "lora"


def test_a_run_resumed_whose_job_cannot_be_made_again_fails_saying_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = a_preset(tmp_path)
    stores = Stores.open(a_cluster(tmp_path))
    run = asyncio.run(stores.registry.create("scout")).id
    refused = "RuntimeError: the job server is not there"
    failed = replace(new_launch(Asked(name="scout", resumes=run), run), state=FAILED, detail=refused)

    async def relaunched(ledger: Any, who: str, **given: Any) -> resuming.Resumed:
        return resuming.Resumed(who, resuming.LAUNCHED, failed)

    monkeypatch.setattr(resuming, "resume", relaunched)
    monkeypatch.setattr("sys.argv", ["rollout", "resume", "scout", "--cluster", str(path)])
    with pytest.raises(SystemExit, match=refused):
        main()
    assert f"scout is launched again: launch {failed.id}" in capsys.readouterr().out


def test_presets_are_loaded_from_a_directory_once_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = a_preset(tmp_path)
    presets = tmp_path / "presets"
    presets.mkdir()
    (presets / "words-local.toml").write_text(f'environment = "{WORDS}"\n"trainer.provider" = "steps"\n')
    assert command(monkeypatch, "preset", "load", str(presets), "--cluster", str(path)) == 0
    assert "saved words-local@1: 2 settings from words-local.toml" in capsys.readouterr().out
    assert command(monkeypatch, "preset", "load", str(presets), "--cluster", str(path)) == 0
    assert "words-local@1: as words-local.toml says" in capsys.readouterr().out
    (presets / "words-local.toml").write_text(f'environment = "{WORDS}"\n"trainer.provider" = "other"\n')
    assert command(monkeypatch, "preset", "load", str(presets), "--cluster", str(path)) == 0
    assert "saved words-local@2" in capsys.readouterr().out


def test_an_environment_is_checked_here_and_groups_asked_for_with_a_models_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = a_preset(tmp_path)
    jobs = Jobs()
    monkeypatch.setattr(submitting, "backend_of", backend_of(jobs))
    assert command(monkeypatch, "env", "check", WORDS) == 0  # (its checks and a scripted episode, nothing asked for)
    assert jobs.submitted == []
    capsys.readouterr()
    asked = ["--preset", "small", "--groups", "2", "--cluster", str(path), "--detach"]
    assert command(monkeypatch, "env", "check", WORDS, *asked) == 0
    assert "check words (run_" in capsys.readouterr().out and len(jobs.submitted) == 1
