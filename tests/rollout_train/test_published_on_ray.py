"""An environment imported from git, on the session's own Ray: checked in a job in its runtime environment (its source
fetched by Ray, its module imported from it, `rollout` and `rollout-train` the platform's), then trained on by a run
submitted by its settings as a Ray job in that runtime environment, whose start records the version; and the
workspace's Minecraft team, imported from a repository of its files and checked there."""

import asyncio
from pathlib import Path

import rollout
import rollout_train
from rollout.harness.blobs import FileBlobStore
from rollout_train.cluster import load
from rollout_train.launches import ENDED, FAILED, STOPPED, TRAIN, launch_of, launches_of
from rollout_train.published import environment_versions_of
from rollout_train.publishing import Source, checked_on_ray, publish
from rollout_train.record import RESULTS, STARTS, newest_record, table
from rollout_train.run_settings import RunSettings
from rollout_train.stores import Stores
from rollout_train.submitting import followed, submit
from tests.local_ray import LocalRay
from tests.rollout_train.sources import repository, tiny

MINECRAFT = Path(__file__).resolve().parents[2] / "environments" / "minecraft"
LEFT_OUT = {"node_modules", "__pycache__", ".pytest_cache", ".ruff_cache"}

CLUSTER = """
name = "test"
[ray]
address = "{address}"
jobs = "{jobs}"
[ledger]
url = "sqlite:///{root}/ledger.db"
[blobs]
directory = "{root}/blobs"
[scratch]
directory = "{root}/scratch"
[inference.local]
kind = "vllm"
engine = "rollout_train.testing:scripted_engine"
gpus = 0.5
[inference.local.models.a-checkpoint]
context = 4096
[trainers.words]
kind = "lora"
implementation = "words:Steps"
gpus = 0.5
colocate_with = "local"
models = ["a-checkpoint"]
"""
"""A cluster whose trainer is the imported project's own (`words:Steps`), which only its runtime environment
imports."""
DONE = (ENDED, FAILED, STOPPED)


async def test_an_imported_environment_is_checked_and_trained_on_by_settings_in_its_runtime_environment(
    tmp_path: Path, local_ray: LocalRay
) -> None:
    source = repository(tmp_path / "source", tiny(src=True), under="environments/words")
    path = tmp_path / "cluster.toml"
    path.write_text(CLUSTER.format(root=tmp_path, address=local_ray.address, jobs=local_ray.dashboard))
    cluster = load(path)
    stores = Stores.open(cluster)
    versions = environment_versions_of(stores.ledger)
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

    settings = RunSettings({
        "kind": TRAIN, "name": "on words", "environment": version.reference, "trainer.provider": "words",
        "channels.policy.provider": "local", "channels.policy.model": "a-checkpoint",
        "channels.policy.renderer": "rollout_train.testing:plain_renderer", "groups": 2, "groups_per_step": 1,
        "trainer.segment_tokens": 900, "trainer.segments_per_step": 3,
    })  # fmt: skip
    launch = await submit(settings, cluster, stores.ledger)
    launches = launches_of(stores.ledger)
    assert launches is not None and launch.job
    async with asyncio.timeout(300):
        while (launch := await followed(await launch_of(launches, launch.id), launches, cluster)).state not in DONE:  # noqa: ASYNC110
            await asyncio.sleep(0.5)
    assert launch.state == ENDED, launch.detail
    assert launch.run is not None
    start = newest_record(await stores.ledger.read(table(launch.run, STARTS)))
    assert start["environment"] == version.reference and start["launch"] == launch.id
    assert start["published"]["version"] == version.version and start["published"]["commit"] == version.commit
    assert len(await stores.ledger.read(table(launch.run, RESULTS))) == 2


async def test_the_minecraft_team_is_imported_from_git_and_checked_in_its_runtime_environment(
    tmp_path: Path, local_ray: LocalRay
) -> None:
    files = {
        path.relative_to(MINECRAFT).as_posix(): path.read_text()
        for path in sorted(MINECRAFT.rglob("*"))
        if path.is_file() and not LEFT_OUT & set(path.relative_to(MINECRAFT).parts) and path.suffix != ".pyc"
    }
    source = repository(tmp_path / "source", files, under="environments/minecraft")
    path = tmp_path / "cluster.toml"
    path.write_text(CLUSTER.format(root=tmp_path, address=local_ray.address, jobs=local_ray.dashboard))
    versions = environment_versions_of(Stores.open(load(path)).ledger)
    assert versions is not None
    made = await publish(
        Source(str(source), "main", "environments/minecraft"), versions=versions,
        blobs=FileBlobStore(tmp_path / "blobs"), jobs=local_ray.dashboard, scratch=tmp_path / "scratch",
    )  # fmt: skip
    version = made.version
    assert (version.name, version.entry_point) == ("minecraft-team", "minecraft_team.environment:environment")
    assert all(each["passed"] for each in version.check) and "uv" not in version.runtime_env
    (episode,) = [each for each in version.check if each["check"] == "episode"]
    assert episode["flagged"] and "minecraft sandboxes" in str(episode["said"])  # (its worlds are the cluster's pool)
    assert version.description["sandboxes"] == ["minecraft"]
