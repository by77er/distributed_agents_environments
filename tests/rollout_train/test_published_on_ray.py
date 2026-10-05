"""An environment imported from git, on the session's own Ray: checked in a job in its runtime environment (its source
fetched by Ray, its module imported from it, `rollout` and `rollout-train` the platform's), then trained on by a run a
launcher submits as a job in that runtime environment, whose start records the version."""

import asyncio
from pathlib import Path

import rollout
import rollout_train
from rollout.harness.blobs import FileBlobStore
from rollout_train.database import DatabaseLedger
from rollout_train.launcher import Launcher
from rollout_train.launches import ENDED, FAILED, Asked, launches_of
from rollout_train.presence import presence_of
from rollout_train.published import environment_versions_of
from rollout_train.publishing import Source, checked_on_ray, publish
from rollout_train.record import RESULTS, STARTS, newest_record, table
from rollout_train.registry import registry_of
from tests.local_ray import LocalRay
from tests.rollout_train.sources import repository, tiny

PROFILE = """
directory = "{directory}"

[ledger]
kind = "rollout_train.database:DatabaseLedger"
url = "sqlite:///{database}"

[channels.policy]
model = "a-checkpoint"
renderer = "rollout_train.testing:plain_renderer"
engine = "rollout_train.testing:scripted_engine"
engines = [{{ device = 0 }}]

[trainer]
kind = "words:Steps"
channel = "policy"
colocated = true
segment_tokens = 900
segments_per_step = 3
"""
"""A run on the imported environment: its trainer is the project's own (`words:Steps`), which only its runtime
environment imports."""
DONE = (ENDED, FAILED)


async def test_an_imported_environment_is_checked_and_trained_on_in_its_runtime_environment(
    tmp_path: Path, local_ray: LocalRay
) -> None:
    source = repository(tmp_path / "source", tiny(src=True), under="environments/words")
    database = tmp_path / "ledger.db"
    ledger = DatabaseLedger(f"sqlite:///{database}")
    versions = environment_versions_of(ledger)
    assert versions is not None
    made = await publish(
        Source(str(source), "main", "environments/words"), versions=versions, blobs=FileBlobStore(tmp_path / "blobs"),
        jobs=local_ray.dashboard, scratch=tmp_path / "scratch",
    )  # fmt: skip
    version = made.version
    assert [each["check"] for each in version.check] == ["rows", "description", "starts", "train and eval", "episode"]
    assert all(each["passed"] for each in version.check) and version.description["evals"]
    assert "uv" not in version.runtime_env  # (the platform holds rollout and rollout-train: nothing to build)

    said = await checked_on_ray(local_ray.dashboard, version.entry_point, version.runtime_env)
    assert said["platform"] == {"rollout": rollout.__file__, "rollout_train": rollout_train.__file__}

    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "words.toml").write_text(PROFILE.format(directory=tmp_path / "run", database=database))
    launches, heartbeats = launches_of(ledger), presence_of(ledger)
    assert launches is not None and heartbeats is not None
    launcher = Launcher(
        "launcher/here", launches, heartbeats, profiles, [], tmp_path / "runs", ray=local_ray.dashboard, gpus=0,
        versions=versions, every=0.2,
    )  # fmt: skip
    asked = await launches.ask(
        Asked(
            profile="words", environment=version.reference, name="on words", groups=2, groups_per_step=1,
            settings={"evals.suite": None},
        )
    )  # fmt: skip
    serving = asyncio.create_task(launcher.serve())
    try:
        async with asyncio.timeout(300):
            while (launch := next(each for each in await launches.all() if each.id == asked.id)).state not in DONE:  # noqa: ASYNC110
                await asyncio.sleep(0.5)
    finally:
        serving.cancel()
    assert launch.state == ENDED, launch.detail
    registry = registry_of(ledger)
    assert registry is not None
    run = next(each.id for each in await registry.runs() if each.name == "on words")
    start = newest_record(await ledger.read(table(run, STARTS)))
    assert start["environment"] == version.reference
    assert start["published"]["version"] == version.version and start["published"]["commit"] == version.commit
    assert len(await ledger.read(table(run, RESULTS))) == 2
    ledger.close()
