"""Importing an environment from git: the clone at a ref, the project and its entry point, the zip (the same bytes for
the same files), the blob and where Ray fetches it, the dependencies the platform lacks, the record of a version (the
same source imported again is the version there), and why an import is refused."""

import asyncio
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

import rollout
from rollout.contracts import BlobReference
from rollout.harness.blobs import Blobs, FileBlobStore
from rollout_train import FileLedger
from rollout_train import publishing as publishing_module
from rollout_train.published import FileEnvironmentVersions, environment_versions_of, is_published
from rollout_train.publishing import (
    MARK,
    Published,
    Refused,
    Source,
    entry_point_of,
    fetched,
    missing,
    packed,
    project_of,
    publish,
    report,
    runtime_env_of,
    stored,
)
from tests.rollout_train.sources import TINY, commit, git, repository, tiny


async def test_a_clone_is_at_the_default_branch_a_branch_a_tag_or_a_commit(tmp_path: Path) -> None:
    source = repository(tmp_path / "source", TINY)
    first = git(source, "rev-parse", "HEAD").strip()
    git(source, "tag", "v1")
    second = commit(source, tiny(version="2"))
    git(source, "checkout", "--quiet", "-b", "older", first)
    third = commit(source, tiny(version="3"))
    git(source, "checkout", "--quiet", "main")
    for ref, expected in ((None, second), ("v1", first), ("older", third), (first, first), (first[:10], first)):
        clone, at = await fetched(str(source), ref, tmp_path / "clones" / str(ref))
        assert at == expected, ref
        assert (clone / "pyproject.toml").is_file()


async def test_a_source_that_does_not_clone_is_refused_saying_why(tmp_path: Path) -> None:
    with pytest.raises(Refused, match="does not clone"):
        await fetched(str(tmp_path / "nothing-here"), None, tmp_path / "clone")
    source = repository(tmp_path / "source", TINY)
    with pytest.raises(Refused, match="no-such-branch"):
        await fetched(str(source), "no-such-branch", tmp_path / "clone")
    with pytest.raises(Refused, match="say the git URL"):
        await fetched("  ", None, tmp_path / "clone")
    with pytest.raises(Refused, match="does not begin with '-'"):
        await fetched("--upload-pack=touch", None, tmp_path / "clone")


def test_a_project_declaring_one_environment_is_imported_by_it() -> None:
    project = {"name": "say-the-word", "entry-points": {"rollout.environments": {"words": "words:environment"}}}
    assert entry_point_of(project, None) == ("words", "words:environment")
    assert entry_point_of(project, "words") == ("words", "words:environment")
    assert entry_point_of(project, "words:environment") == ("words", "words:environment")
    assert entry_point_of(project, "words.other:environment") == ("say-the-word", "words.other:environment")


def test_an_entry_point_not_said_or_not_declared_is_refused() -> None:
    with pytest.raises(Refused, match="declares no environment: say its entry point"):
        entry_point_of({"name": "plain"}, None)
    several = {"name": "two", "entry-points": {"rollout.environments": {"a": "a:one", "b": "b:two"}}}
    with pytest.raises(Refused, match=r"declares 2 environments \(a, b\): say which"):
        entry_point_of(several, None)
    assert entry_point_of(several, "b") == ("b", "b:two")
    with pytest.raises(Refused, match="no environment 'c'"):
        entry_point_of(several, "c")
    with pytest.raises(Refused, match="no entry point"):
        entry_point_of({"name": "plain"}, "not an:entry point")


def test_a_project_is_read_at_its_root_or_under_src_and_refused_without_pyproject(tmp_path: Path) -> None:
    flat = repository(tmp_path / "flat", TINY)
    found = project_of(flat, "", None)
    assert (found.name, found.entry_point, found.root) == ("words", "words:environment", "")
    assert found.dependencies == ("rollout", "rollout-train")
    nested = repository(tmp_path / "nested", tiny(src=True), under="environments/words")
    assert project_of(nested, "environments/words", None).root == "src"
    with pytest.raises(Refused, match=r"has no pyproject\.toml"):
        project_of(nested, "", None)
    with pytest.raises(Refused, match="no directory 'elsewhere'"):
        project_of(nested, "elsewhere", None)
    with pytest.raises(Refused, match="not in the repository"):
        project_of(nested, "../..", None)
    with pytest.raises(Refused, match="module missing is not in the project"):
        project_of(flat, "", "missing:environment")


def test_the_workspaces_gridworld_is_imported_by_the_environment_it_declares() -> None:
    environments = Path(__file__).resolve().parents[2] / "environments"
    found = project_of(environments, "gridworld", None)
    assert (found.name, found.entry_point, found.root) == ("gridworld", "gridworld.environment:environment", "")
    assert missing(found.dependencies) == []  # (it runs in the platform's Python: nothing to build)


def test_the_workspaces_minecraft_team_is_imported_and_checked_without_playing_its_worlds() -> None:
    environments = Path(__file__).resolve().parents[2] / "environments"
    found = project_of(environments, "minecraft", None)
    assert (found.name, found.entry_point, found.root) == (
        "minecraft-team",
        "minecraft_team.environment:environment",
        "",
    )
    assert missing(found.dependencies) == []
    said = report(found.entry_point)
    assert said["loaded"] and all(each["passed"] for each in said["findings"]), said["findings"]
    (episode,) = [each for each in said["findings"] if each["check"] == "episode"]
    assert episode["flagged"] and "minecraft sandboxes" in episode["said"]  # (an import serves no worlds)
    assert said["described"]["sandboxes"] == ["minecraft"] and len(said["described"]["rows"]) == 100


def test_a_project_needing_another_python_is_refused(tmp_path: Path) -> None:
    files = {**TINY, "pyproject.toml": TINY["pyproject.toml"].replace(">=3.13", ">=4")}
    with pytest.raises(Refused, match="requires Python >=4"):
        project_of(repository(tmp_path / "source", files), "", None)


def test_the_same_files_pack_to_the_same_bytes_without_git_caches_or_virtual_environments(tmp_path: Path) -> None:
    one = repository(tmp_path / "one", TINY)
    (one / "__pycache__").mkdir()
    (one / "__pycache__" / "words.cpython-313.pyc").write_bytes(b"compiled")
    (one / ".venv" / "bin").mkdir(parents=True)
    (one / ".venv" / "bin" / "python").write_text("a virtual environment")
    (one / "run.sh").write_text("#!/bin/sh\n")
    os.chmod(one / "run.sh", 0o755)
    two = tmp_path / "two"
    for path, text in {**TINY, "run.sh": "#!/bin/sh\n"}.items():
        (two / path).parent.mkdir(parents=True, exist_ok=True)
        (two / path).write_text(text)
    os.chmod(two / "run.sh", 0o755)
    os.utime(two / "README.md", (1, 1))  # (another time: the same bytes)
    data = packed(one)
    assert data == packed(two)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        assert names == sorted(["README.md", "pyproject.toml", "run.sh", "words/__init__.py"])
        assert archive.getinfo("run.sh").external_attr >> 16 == 0o100755
        assert archive.getinfo("README.md").external_attr >> 16 == 0o100644


def test_the_dependencies_the_platform_holds_are_left_out() -> None:
    assert missing(["rollout", "rollout-train[http]", "ray>=2"]) == []
    assert missing(["rollout>=99", "no-such-distribution-at-all", "rollout"]) == [
        "rollout>=99", "no-such-distribution-at-all",
    ]  # fmt: skip
    assert missing(['no-such-distribution-at-all; python_version < "3"']) == []
    assert missing(["rollout @ https://example.com/rollout.whl"]) == ["rollout @ https://example.com/rollout.whl"]
    with pytest.raises(Refused, match="no requirement"):
        missing(["not a requirement ==="])


def test_a_runtime_environment_holds_the_source_its_layout_and_what_the_platform_lacks() -> None:
    assert runtime_env_of("/blobs/packages/v.zip", "", []) == {"working_dir": "/blobs/packages/v.zip"}
    assert runtime_env_of("s3://b/k.zip", "src", ["tinydep"]) == {
        "working_dir": "s3://b/k.zip", "env_vars": {"PYTHONPATH": "src"}, "uv": {"packages": ["tinydep"]},
    }  # fmt: skip


async def test_the_zip_is_a_blob_and_ray_is_handed_a_zip_it_fetches(blob_store: Blobs) -> None:
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.writestr("pyproject.toml", "[project]\nname = 'x'\n")
    reference, package = await stored(blob_store, zipped.getvalue())
    assert await blob_store.read(reference) == zipped.getvalue()
    assert package.endswith(f"{reference.sha256}.zip")
    if isinstance(blob_store, FileBlobStore):
        assert await asyncio.to_thread(Path(package).read_bytes) == zipped.getvalue()
    else:
        import boto3

        bucket, _, key = package.removeprefix("s3://").partition("/")
        got = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"]).get_object(Bucket=bucket, Key=key)  # pyright: ignore[reportUnknownMemberType]
        assert got["Body"].read() == zipped.getvalue()
        assert await stored(blob_store, zipped.getvalue()) == (reference, package)  # (copied once)


def test_the_check_report_imports_the_entry_point_checks_it_and_plays_a_scripted_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = repository(tmp_path / "source", TINY)
    monkeypatch.syspath_prepend(str(source))  # pyright: ignore[reportUnknownMemberType]
    said = report("words:environment")
    assert said["loaded"] and said["platform"]["rollout"] == rollout.__file__ and said["python"] == sys.executable
    assert [each["check"] for each in said["findings"]] == [
        "rows",
        "description",
        "starts",
        "train and eval",
        "episode",
    ]
    assert all(each["passed"] for each in said["findings"])
    assert said["described"]["rows"] == [{"key": "say-yes", "title": "say yes"}, {"key": "say-no", "title": "say no"}]
    assert said["described"]["evals"] == {"words-eval": ["say-yes", "say-yes", "say-no", "say-no"]}
    assert said["described"]["sandboxes"] == []  # (its program declares none: a profile with no pools plays it)
    failed = report("words:nothing")
    assert not failed["loaded"] and "AttributeError" in failed["error"]


async def published_with(
    tmp_path: Path, source: Source, monkeypatch: pytest.MonkeyPatch, checks: list[dict[str, Any]] | None = None
) -> Published:
    """An import whose check runs here, in this process, rather than on Ray (`checked_on_ray` is replaced)."""
    checked: list[dict[str, Any]] = [] if checks is None else checks

    async def here(jobs: str, entry_point: str, runtime_env: Any, **_: Any) -> dict[str, Any]:
        checked.append({"jobs": jobs, "entry_point": entry_point, "runtime_env": runtime_env})
        return {"loaded": True, "findings": [{"check": "rows", "passed": True, "said": "2 rows"}], "described": {}}

    monkeypatch.setattr(publishing_module, "checked_on_ray", here)
    ledger = FileLedger(tmp_path / "ledger")
    versions = environment_versions_of(ledger)
    assert isinstance(versions, FileEnvironmentVersions)
    blobs = FileBlobStore(tmp_path / "blobs")
    return await publish(source, versions=versions, blobs=blobs, jobs="http://ray:8265", scratch=tmp_path / "scratch")


async def test_the_check_is_handed_the_zip_itself_so_the_cluster_checking_it_holds_no_stores_key(
    tmp_path: Path, blob_store: Blobs, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = repository(tmp_path / "source", tiny(src=True), under="envs/words")
    handed: list[tuple[str, bytes]] = []

    async def here(jobs: str, entry_point: str, runtime_env: Any, **_: Any) -> dict[str, Any]:
        package = Path(str(runtime_env["working_dir"]))
        handed.append((str(package), await asyncio.to_thread(package.read_bytes)))  # (a file here, as the check runs)
        return {"loaded": True, "findings": [], "described": {}}

    monkeypatch.setattr(publishing_module, "checked_on_ray", here)
    versions = environment_versions_of(FileLedger(tmp_path / "ledger"))
    assert versions is not None
    made = await publish(Source(str(source), None, "envs/words"), versions=versions, blobs=blob_store,
                         jobs="http://ray:8265", scratch=tmp_path / "scratch")  # fmt: skip
    ((local, data),) = handed
    stored_at = str(made.version.runtime_env["working_dir"])
    assert data == await blob_store.read(BlobReference.model_validate(made.version.blob))
    if isinstance(blob_store, FileBlobStore):
        assert local == stored_at  # (a store of files' zip is a file here already)
    else:
        assert stored_at.startswith("s3://") and "://" not in local  # (what runs fetch, and what the check is handed)
        assert not await asyncio.to_thread(Path(local).exists)  # (removed once checked)


async def test_an_import_records_a_version_whose_id_is_the_zip_and_the_same_source_is_that_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = repository(tmp_path / "source", tiny(src=True), under="envs/words")
    at = git(source, "rev-parse", "HEAD").strip()
    checks: list[dict[str, Any]] = []
    asked = Source(str(source), "main", "envs/words")
    first = await published_with(tmp_path, asked, monkeypatch, checks)
    version = first.version
    assert not first.existing and is_published(version.reference) and version.reference == f"words@{version.version}"
    assert (version.commit, version.ref, version.subdirectory, version.entry_point) == (
        at,
        "main",
        "envs/words",
        "words:environment",
    )
    assert version.blob["sha256"] == version.version and version.dependencies == ("rollout", "rollout-train")
    package = tmp_path / "blobs" / "packages" / f"{version.version}.zip"
    assert version.runtime_env == {"working_dir": str(package), "env_vars": {"PYTHONPATH": "src"}}
    assert checks == [
        {"jobs": "http://ray:8265", "entry_point": "words:environment", "runtime_env": version.runtime_env}
    ]
    assert version.check == [{"check": "rows", "passed": True, "said": "2 rows"}]
    again = await published_with(tmp_path, Source(f"file://{source}", None, "envs/words/"), monkeypatch, checks)
    assert again.existing and again.version == version and len(checks) == 1  # (not checked again)
    commit(source, {"README.md": "Say it again.\n"}, under="envs/words")
    changed = await published_with(tmp_path, asked, monkeypatch, checks)
    assert not changed.existing and changed.version.version != version.version
    assert not list((tmp_path / "scratch").iterdir())  # (clones are removed once read)


async def test_an_import_is_refused_without_a_clone_a_pyproject_or_an_entry_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(Refused, match="does not clone"):
        await published_with(tmp_path, Source(str(tmp_path / "nothing")), monkeypatch)
    bare = repository(tmp_path / "bare", {"README.md": "no project"})
    with pytest.raises(Refused, match=r"no pyproject\.toml"):
        await published_with(tmp_path, Source(str(bare)), monkeypatch)
    plain = {**TINY, "pyproject.toml": "[project]\nname = 'plain'\nversion = '0'\n"}
    with pytest.raises(Refused, match="declares no environment"):
        await published_with(tmp_path, Source(str(repository(tmp_path / "plain", plain))), monkeypatch)


class Jobs:
    """A stand-in for Ray's job client: each job ends as `status`, with `message` and `logs`."""

    def __init__(self, status: str, logs: str = "", message: str = "") -> None:
        self.status, self.logs, self.message = status, logs, message
        self.submitted: list[dict[str, Any]] = []

    def submit_job(self, **given: Any) -> str:
        self.submitted.append(given)
        return "raysubmit_1"

    def get_job_status(self, job: str) -> Any:
        from ray.job_submission import JobStatus

        return JobStatus(self.status)

    def get_job_info(self, job: str) -> Any:
        return type("Info", (), {"message": self.message})()

    def get_job_logs(self, job: str) -> str:
        return self.logs


UNLOADED = {"loaded": False, "error": "ImportError: no words"}
FAILING = {
    "loaded": True,
    "findings": [{"check": "rows", "passed": False, "said": "it has no rows"}],
    "described": None,
}


@pytest.mark.parametrize(
    ("jobs", "refused"),
    [
        (Jobs("FAILED", message="Failed to set up runtime env: uv pip failed"), "Python environment was not built"),
        (Jobs("SUCCEEDED", logs=MARK + json.dumps(UNLOADED)), "does not load in its runtime environment: ImportError"),
        (Jobs("SUCCEEDED", logs=MARK + json.dumps(FAILING)), "its checks failed: rows: it has no rows"),
        (Jobs("FAILED", logs="Traceback: it broke"), "failed: Traceback: it broke"),
    ],
)
async def test_a_check_on_ray_that_fails_refuses_the_import_saying_why(
    jobs: Jobs, refused: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rollout_train.publishing import checked_on_ray

    def client(address: str) -> Jobs:
        return jobs

    monkeypatch.setattr(publishing_module, "_client", client)
    with pytest.raises(Refused, match=refused):
        await checked_on_ray("http://ray:8265", "words:environment", {"working_dir": "/w.zip"}, every=0.0)
    (submitted,) = jobs.submitted
    assert submitted["entrypoint"] == "python -m rollout_train.publishing check words:environment"
    assert submitted["runtime_env"] == {"working_dir": "/w.zip"}


async def test_two_imports_of_one_source_at_once_record_one_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = repository(tmp_path / "source", TINY)
    both = await asyncio.gather(*(published_with(tmp_path, Source(str(source)), monkeypatch) for _ in range(2)))
    assert both[0].version == both[1].version
    assert len(await FileEnvironmentVersions(tmp_path / "ledger" / "environment_versions").all()) == 1
