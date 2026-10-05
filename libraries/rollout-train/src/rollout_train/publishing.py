# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (Ray is partly untyped.)
"""Importing an environment from git (`publish`): its source fetched at a commit, stored in the blob store, checked in
the Python environment it runs in, and recorded as a version beside the ledger (`rollout_train.published`).

1. **Fetch** (`fetched`): a shallow clone at the ref asked for (a branch, a tag or a commit; none: the default
   branch), and the commit it is. The project is the directory with the `pyproject.toml`: the repository's root, or
   the subdirectory asked for.
2. **Entry point** (`entry_point_of`): `module:name`, as given, or read from the project's entry points in the group
   `rollout.environments` (`GROUP`):

       [project.entry-points."rollout.environments"]
       gridworld = "gridworld.environment:environment"

   A project that declares one environment is imported by it without saying more; of one that declares several, the
   import names one (by its name in the group, or by `module:name`). The environment's name is its name in the group,
   else the project's. Its module is imported from the project's root, or from `src/` for a project that keeps its
   packages there.
3. **Pack** (`packed`): the project's files in one zip, stored uncompressed, sorted, with fixed times and modes, so the
   same files are the same bytes wherever they are packed; without `.git`, `.venv` and caches (`EXCLUDED`) or
   symbolic links. The version's id is the zip's SHA-256.
4. **Store** (`stored`): the zip as a blob, and where Ray fetches it as a runtime environment's `working_dir`. Ray
   tells an archive by its name, so a store in S3 hands Ray a copy of the blob's object named `KEY.zip`
   (`s3://BUCKET/KEY.zip`, which each node reads with the cluster's credentials), and a store of files a `.zip`
   beside its blobs, which the job's submitter uploads to the cluster.
5. **Runtime environment** (`runtime_env_of`): the zip as the `working_dir`; `src` on `PYTHONPATH` for a project laid
   out so; and, under `uv`, the project's dependencies the platform does not hold (`missing`). Ray builds that Python
   environment on each node the first time a job asks for it: a copy of the platform's virtual environment with those
   dependencies installed into it by uv, so `rollout` and `rollout-train` are the platform's own. A project whose
   dependencies the platform holds runs in the platform's Python, with nothing built.
6. **Check** (`checked_on_ray`): a Ray job in that runtime environment imports the entry point, runs the checks
   `rollout env check` runs without a model (`rollout_train.check.checked`, and an episode answered by a scripted
   model, where its program needs no tool set or sandbox), and says what the environment says of itself.
7. **Record**: the version, beside the ledger. Importing the same source again returns the version recorded.

Each step refuses with a reason (`Refused`): a source that does not clone, no `pyproject.toml`, no entry point, a
dependency the platform's Python cannot take, an entry point that does not load in its runtime environment, checks that
fail. `python -m rollout_train.publishing check ENTRY_POINT` is what the check's job runs.
"""

import asyncio
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import tomllib
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import JsonValue

from rollout.contracts import BlobReference
from rollout.harness.blobs import Blobs, FileBlobStore
from rollout_train.published import EnvironmentVersion, EnvironmentVersions

if TYPE_CHECKING:
    from rollout_train.cluster import Cluster

__all__ = [
    "EXCLUDED",
    "GROUP",
    "MARK",
    "Importer",
    "Project",
    "Published",
    "Refused",
    "Source",
    "checked_on_ray",
    "entry_point_of",
    "fetched",
    "missing",
    "packed",
    "project_of",
    "publish",
    "report",
    "runtime_env_of",
    "stored",
]

GROUP = "rollout.environments"
"""The entry-point group a project declares its environments in."""
EXCLUDED = frozenset(
    {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints"}
)
"""Directories never packed."""
LARGEST = 256 * 2**20
"""Bytes a project's zip may hold."""
FETCHING = 300.0
"""Seconds a clone may take."""
CHECKING = 1800.0
"""Seconds the check's job may take, building its Python environment first where it needs one."""
EPISODE = 300.0
"""Seconds the check's scripted episode may take."""
MARK = "rollout-check: "
"""What the check's job begins the line it says what it found with."""
SAID = 4000
"""Characters of a failed job's output kept as why."""
ENTRY = re.compile(r"[A-Za-z_][\w.]*:[A-Za-z_][\w.]*")
TIME = (1980, 1, 1, 0, 0, 0)
"""When every packed file says it was changed: the earliest a zip can say."""


class Refused(ValueError):
    """An import that cannot be made, and why."""


@dataclass(frozen=True)
class Source:
    """What to import: a git URL; the branch, tag or commit (none: the default branch); the project's directory in
    the repository; its entry point (`module:name`, or its name in `GROUP`; none: the one it declares)."""

    url: str
    ref: str | None = None
    subdirectory: str = ""
    entry_point: str | None = None


@dataclass(frozen=True)
class Project:
    """A project as an import reads it: its directory, its environment's name and entry point, where its module is
    imported from (`""` or `"src"`), and its dependencies."""

    directory: Path
    name: str
    entry_point: str
    root: str
    dependencies: tuple[str, ...]


@dataclass(frozen=True)
class Published:
    """What an import made: the version, and whether it was recorded already (the same source imported before)."""

    version: EnvironmentVersion
    existing: bool


@dataclass(frozen=True)
class Importer:
    """Where imports are made: the blob store their zips go to, the Ray cluster's job server their checks run on, and
    the directory clones are read in."""

    blobs: Blobs
    jobs: str
    scratch: Path

    @classmethod
    def of(cls, cluster: "Cluster") -> "Importer":
        """A cluster's: its `[blobs]`, its `[ray] jobs`, and `imports` under its `[scratch]`."""
        from rollout_train.stores import blobs_at, opened

        return cls(opened(blobs_at(cluster)), cluster.ray.jobs, Path(cluster.scratch).expanduser() / "imports")


Said = Callable[[str], None]


def _quiet(stage: str) -> None:
    del stage


async def publish(
    source: Source,
    *,
    versions: EnvironmentVersions,
    blobs: Blobs,
    jobs: str,
    scratch: Path,
    said: Said = _quiet,
    within: float = CHECKING,
) -> Published:
    """Import an environment from git: fetch it, read it, pack it, store it, check it in a Ray job on the cluster
    whose job server is `jobs`, and record it (the module's docstring). `scratch` holds the clone while it is read.
    `said` is told each stage as it begins. Raises `Refused` saying why an import cannot be made."""
    said("fetching")
    await asyncio.to_thread(scratch.mkdir, parents=True, exist_ok=True)
    work = Path(await asyncio.to_thread(tempfile.mkdtemp, dir=scratch, prefix="import-"))
    try:
        clone, commit = await fetched(source.url, source.ref, work / "clone")
        said("reading")
        project = await asyncio.to_thread(project_of, clone, source.subdirectory, source.entry_point)
        needed = await asyncio.to_thread(missing, project.dependencies)
        said("packing")
        data = await asyncio.to_thread(packed, project.directory)
    finally:
        await asyncio.to_thread(shutil.rmtree, work, ignore_errors=True)
    id = hashlib.sha256(data).hexdigest()
    found = await versions.get(id)
    if found is not None:
        return Published(found, existing=True)
    said("storing")
    reference, package = await stored(blobs, data)
    runtime_env = runtime_env_of(package, project.root, needed)
    said("checking")
    report = await checked_on_ray(jobs, project.entry_point, runtime_env, within=within)
    said("recording")
    version = EnvironmentVersion(
        name=project.name, version=id, source=source.url, ref=source.ref or None, commit=commit,
        subdirectory=_relative(source.subdirectory), entry_point=project.entry_point,
        blob=reference.model_dump(mode="json"), runtime_env=runtime_env, dependencies=project.dependencies,
        description=report["described"], check=report["findings"], imported=round(time.time(), 1),
    )  # fmt: skip
    kept = await versions.record(version)
    return Published(kept, existing=kept is not version)


async def fetched(url: str, ref: str | None, into: Path) -> tuple[Path, str]:
    """A shallow clone of `url` at `ref` (a branch or tag; a commit, where the server gives one by its id; none: the
    default branch) in `into`, and the commit it is. Raises `Refused` saying why it does not clone."""
    url, ref = url.strip(), (ref or "").strip() or None
    if not url:
        raise Refused("say the git URL to import from")
    if url.startswith("-") or (ref is not None and ref.startswith("-")):
        raise Refused("a git URL or ref does not begin with '-'")
    attempts: list[list[str]] = [
        ["clone", "--depth", "1", "--quiet", *(["--branch", ref] if ref else []), "--", url, str(into)]
    ]
    if ref is not None:  # (a commit: fetched by its id where the server allows, else the whole history)
        attempts.append(["clone", "--quiet", "--no-checkout", "--", url, str(into)])
    why = ""
    for number, attempt in enumerate(attempts):
        await asyncio.to_thread(shutil.rmtree, into, ignore_errors=True)
        code, said = await _git(attempt)
        if code != 0:
            why = said
            continue
        if number == 1:
            assert ref is not None
            code, said = await _git(["-C", str(into), "checkout", "--quiet", "--detach", ref, "--"])
            if code != 0:
                why = (
                    f"it has no branch, tag or commit {ref!r}"
                    if "did not match" in said or "pathspec" in said
                    else said
                )
                continue
        code, said = await _git(["-C", str(into), "rev-parse", "HEAD"])
        if code == 0:
            return into, said.strip()
        why = said
    raise Refused(f"{url} does not clone{f' at {ref}' if ref else ''}: {why or 'git said nothing'}")


async def _git(arguments: Sequence[str]) -> tuple[int, str]:
    """Run git, never asking for credentials: its exit code and what it said (its output, else its errors)."""
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "true", "SSH_ASKPASS": "true"}
    process = await asyncio.create_subprocess_exec(
        "git", *arguments, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=environment,
        stdin=asyncio.subprocess.DEVNULL,
    )  # fmt: skip
    try:
        output, errors = await asyncio.wait_for(process.communicate(), FETCHING)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        return 1, f"git took more than {FETCHING:.0f} seconds"
    if process.returncode == 0:
        return 0, output.decode(errors="replace")
    lines = [line.strip() for line in errors.decode(errors="replace").splitlines() if line.strip()]
    kept = [
        line.removeprefix("fatal: ").removeprefix("error: ")
        for line in lines
        if not line.startswith(("Cloning", "warning:", "hint:"))
    ]
    return process.returncode or 1, "; ".join(kept[-3:])


def project_of(clone: Path, subdirectory: str, given: str | None) -> Project:
    """The project in a clone's `subdirectory` (empty: its root), with its environment's entry point (`given`, else
    the one it declares: `entry_point_of`). Raises `Refused` for a directory that is not in the clone or holds no
    `pyproject.toml`, a project with no entry point, or an entry point whose module is not in it."""
    relative = _relative(subdirectory)
    directory = (clone / relative).resolve() if relative else clone.resolve()
    if not directory.is_relative_to(clone.resolve()):
        raise Refused(f"the subdirectory {subdirectory!r} is not in the repository")
    if not directory.is_dir():
        raise Refused(f"the repository has no directory {relative!r}")
    where = f"{relative}/pyproject.toml" if relative else "pyproject.toml"
    if not (directory / "pyproject.toml").is_file():
        raise Refused(f"it has no {where}: an environment is imported from a Python project")
    try:
        declared = tomllib.loads((directory / "pyproject.toml").read_text())
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise Refused(f"its {where} does not read: {error}") from None
    project = declared.get("project")
    if not isinstance(project, dict) or not isinstance(cast(dict[str, Any], project).get("name"), str):
        raise Refused(f"its {where} has no [project] with a name")
    said = cast(dict[str, Any], project)
    name, entry = entry_point_of(said, given)
    module = entry.partition(":")[0]
    top = module.split(".")[0]
    roots = [
        root for root in ("", "src") if (directory / root / top).is_dir() or (directory / root / f"{top}.py").is_file()
    ]
    if not roots:
        raise Refused(f"its entry point's module {module} is not in the project: it has neither {top}/ nor src/{top}/")
    dependencies = said.get("dependencies") or []
    if not isinstance(dependencies, list) or not all(isinstance(each, str) for each in cast(list[Any], dependencies)):
        raise Refused(f"its {where} lists its dependencies as something other than requirements")
    requires = said.get("requires-python")
    if isinstance(requires, str):
        _python_fits(requires)
    return Project(directory, name, entry, roots[0], tuple(cast(list[str], dependencies)))


def entry_point_of(project: Mapping[str, Any], given: str | None) -> tuple[str, str]:
    """The environment's name and entry point (`module:name`) of a project (its `[project]` table): `given` (an entry
    point, or its name in `GROUP`), else the one environment the project declares in `GROUP`. The name is the entry
    point's in `GROUP`, else the project's. Raises `Refused` where there is none, or several and none given."""
    groups = project.get("entry-points")
    declared: dict[str, str] = {}
    if isinstance(groups, dict):
        found = cast(dict[str, Any], groups).get(GROUP)
        if isinstance(found, dict):
            declared = {str(key): str(value) for key, value in cast(dict[str, Any], found).items()}
    given = (given or "").strip() or None
    named = str(project["name"])
    if given is not None and ":" not in given:
        if given not in declared:
            listed = f" (it declares {', '.join(sorted(declared))})" if declared else ""
            raise Refused(f'it declares no environment {given!r} in [project.entry-points."{GROUP}"]{listed}')
        name, entry = given, declared[given]
    elif given is not None:
        name = next((key for key, value in declared.items() if value == given), named)
        entry = given
    elif len(declared) == 1:
        ((name, entry),) = declared.items()
    elif not declared:
        raise Refused(
            f"it declares no environment: say its entry point (module:name), or declare one under "
            f'[project.entry-points."{GROUP}"]'
        )
    else:
        raise Refused(f"it declares {len(declared)} environments ({', '.join(sorted(declared))}): say which")
    if not ENTRY.fullmatch(entry):
        raise Refused(f"{entry!r} is no entry point: it should be module:name")
    if not name or any(mark in name for mark in "/@:") or name != name.strip():
        raise Refused(f"{name!r} cannot name an environment: it must say something, and none of '/', '@', ':'")
    return name, entry


def _python_fits(requires: str) -> None:
    from packaging.specifiers import InvalidSpecifier, SpecifierSet

    try:
        fits = SpecifierSet(requires).contains(".".join(map(str, sys.version_info[:3])), prereleases=True)
    except InvalidSpecifier:
        raise Refused(f"its requires-python {requires!r} is no version specifier") from None
    if not fits:
        here = ".".join(map(str, sys.version_info[:3]))
        raise Refused(f"it requires Python {requires}, and the platform runs Python {here}")


def _relative(subdirectory: str) -> str:
    """A subdirectory as a relative path with forward slashes (empty: the root). Raises `Refused` for one that leaves
    the repository."""
    parts = [part for part in subdirectory.strip().replace("\\", "/").split("/") if part not in ("", ".")]
    if subdirectory.strip().startswith("/") or ".." in parts:
        raise Refused(f"the subdirectory {subdirectory!r} is not in the repository")
    return "/".join(parts)


def missing(dependencies: Sequence[str]) -> list[str]:
    """The requirements of `dependencies` this Python does not satisfy: each whose distribution is not installed, is
    installed at a version its specifier leaves out, is asked for at a URL, or lacks what an extra asked for needs.
    Requirements whose markers do not hold here are left out. Raises `Refused` for one that is no requirement."""
    from packaging.requirements import InvalidRequirement, Requirement

    needed: list[str] = []
    for line in dependencies:
        try:
            requirement = Requirement(line)
        except InvalidRequirement as error:
            raise Refused(f"its dependency {line!r} is no requirement: {error}") from None
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        if not _satisfied(requirement):
            needed.append(line)
    return needed


def _satisfied(requirement: Any, seen: frozenset[str] = frozenset()) -> bool:
    """Whether this Python holds a requirement: its distribution at a version its specifier allows, and for each extra
    it asks for, what that extra adds."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    if requirement.url:
        return False
    name = canonicalize_name(requirement.name)
    try:
        installed = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return False
    if not requirement.specifier.contains(installed.version, prereleases=True):
        return False
    for wanted in requirement.extras:
        if f"{name}[{wanted}]" in seen:
            continue
        for line in installed.requires or []:
            needs = Requirement(line)
            if needs.marker is None or not needs.marker.evaluate({"extra": wanted}):
                continue
            if needs.marker.evaluate({"extra": ""}):  # (needed without the extra too: not what the extra adds)
                continue
            if not _satisfied(needs, seen | {f"{name}[{wanted}]"}):
                return False
    return True


def packed(directory: Path) -> bytes:
    """A project's files as a zip: every regular file under `directory` but those in `EXCLUDED` directories and
    compiled bytecode, by its path relative to `directory`, stored uncompressed in sorted order with fixed times and
    modes (executable or not). Raises `Refused` for a zip larger than `LARGEST`."""
    files: list[tuple[str, Path]] = []
    for root, directories, names in os.walk(directory):
        directories[:] = sorted(
            each for each in directories if each not in EXCLUDED and not (Path(root) / each).is_symlink()
        )
        for name in names:
            path = Path(root) / name
            if path.is_symlink() or not path.is_file() or name.endswith((".pyc", ".pyo")):
                continue
            files.append((path.relative_to(directory).as_posix(), path))
    files.sort()
    total = sum(path.stat().st_size for _, path in files)
    if total > LARGEST:
        raise Refused(f"its files hold {total / 2**20:.0f} MiB: an import holds {LARGEST // 2**20} MiB at most")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name, path in files:
            entry = zipfile.ZipInfo(name, date_time=TIME)
            executable = path.stat().st_mode & stat.S_IXUSR
            entry.external_attr = (0o100755 if executable else 0o100644) << 16
            entry.create_system = 3  # (Unix: the modes are read as Unix's)
            archive.writestr(entry, path.read_bytes())
    return buffer.getvalue()


async def stored(blobs: Blobs, data: bytes) -> tuple[BlobReference, str]:
    """Store a project's zip, and say where Ray fetches it as a runtime environment's `working_dir`: for a store that
    names copies of its blobs (`with_extension`: in S3, `s3://BUCKET/KEY.zip`), that; for a store of files, a `.zip`
    beside its blobs (`packages/VERSION.zip`), which whoever submits the job uploads. Raises `Refused` for a store Ray
    cannot be handed a blob of."""
    reference = await blobs.put(data, "application/zip")
    extended = getattr(blobs, "with_extension", None)
    if extended is not None:
        return reference, str(await extended(reference, ".zip"))
    if isinstance(blobs, FileBlobStore):
        target = blobs.directory / "packages" / f"{reference.sha256}.zip"
        if not await blobs.link(reference, target):
            raise Refused(f"the blob store at {blobs.directory} lost the zip it was just given")
        return reference, str(target)
    raise Refused(f"Ray cannot fetch a blob of the store {type(blobs).__name__}: it reads files and S3")


def runtime_env_of(package: str, root: str, dependencies: Sequence[str]) -> dict[str, JsonValue]:
    """The Ray runtime environment a version's code runs in: `package` (the zip, where Ray fetches it) as its
    `working_dir`; `root` (`src`, for a project that keeps its packages there) on `PYTHONPATH`, relative to that
    directory, which every process of the job starts in; and `dependencies` (those the platform does not hold) for uv
    to install over a copy of the platform's Python. Ray names that copy by the dependencies, so versions that need the
    same ones share it."""
    found: dict[str, JsonValue] = {"working_dir": package}
    if root:
        found["env_vars"] = {"PYTHONPATH": root}
    if dependencies:
        found["uv"] = {"packages": list(dependencies)}
    return found


def _client(jobs: str) -> Any:
    from rollout_train.ray_cluster import prepare

    prepare()
    from ray.job_submission import JobSubmissionClient

    return JobSubmissionClient(jobs)


async def checked_on_ray(
    jobs: str, entry_point: str, runtime_env: Mapping[str, JsonValue], *, within: float = CHECKING, every: float = 1.0
) -> dict[str, Any]:
    """Check an environment in a Ray job in its runtime environment (`report`, run by `python -m
    rollout_train.publishing check`), on the cluster whose job server is `jobs`: what it found (`findings`), what the
    environment says of itself (`described`), and the job's Python (`python`, and where it imported `rollout` and
    `rollout_train` from: `platform`). Raises `Refused` where the job's Python environment is not built, the entry
    point does not load, a check fails, or the job does not end within `within` seconds."""
    client = await asyncio.to_thread(_client, jobs)
    entrypoint = f"python -m rollout_train.publishing check {entry_point}"
    job: str = await asyncio.to_thread(
        client.submit_job, entrypoint=entrypoint, runtime_env=dict(runtime_env), entrypoint_num_cpus=1,
        metadata={"kind": "environment-check", "entry_point": entry_point},
    )  # fmt: skip
    deadline = time.monotonic() + within
    while not (status := await asyncio.to_thread(client.get_job_status, job)).is_terminal():
        if time.monotonic() > deadline:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(client.stop_job, job)
            raise Refused(f"its check (Ray job {job}) had not ended after {within / 60:.0f} minutes")
        await asyncio.sleep(every)
    info = await asyncio.to_thread(client.get_job_info, job)
    output: str = await asyncio.to_thread(client.get_job_logs, job)
    lines = [line[len(MARK) :] for line in output.splitlines() if line.startswith(MARK)]
    if not lines:
        message = str(info.message or "")
        if "runtime env" in message.lower() or "runtime_env" in message:
            raise Refused(f"its Python environment was not built (Ray job {job}): {_tail(message)}")
        raise Refused(f"its check (Ray job {job}) {status.value.lower()}: {_tail(output or message)}")
    report: dict[str, Any] = json.loads(lines[-1])
    if not report.get("loaded"):
        raise Refused(f"its entry point {entry_point} does not load in its runtime environment: {report.get('error')}")
    failed = [each for each in report["findings"] if not each["passed"]]
    if failed:
        raise Refused("its checks failed: " + "; ".join(f"{each['check']}: {each['said']}" for each in failed))
    return report


def _tail(text: str) -> str:
    return text.strip()[-SAID:]


def report(entry_point: str) -> dict[str, Any]:
    """What the check's job says of an environment, imported here by its entry point: whether it loaded (and why not),
    the findings of `rollout_train.check.checked` and of an episode answered by a scripted model (where its program
    imports no tool set and declares no sandbox, which an import cannot serve), what it says of itself, and this
    Python and where it imported `rollout` and `rollout_train` from."""
    import random

    import rollout
    import rollout_train
    from rollout.harness.runner import instantiate, with_row
    from rollout.names import named
    from rollout_train.check import Finding, checked, scripted
    from rollout_train.monitor.environments import described

    platform = {"rollout": rollout.__file__, "rollout_train": rollout_train.__file__}
    here: dict[str, Any] = {"python": sys.executable, "platform": platform}
    try:
        environment = named(entry_point)
    except Exception as error:  # (whatever importing it raises is why it does not load)
        return {**here, "loaded": False, "error": f"{type(error).__name__}: {error}"}
    findings = checked(environment)
    if all(each.passed for each in findings):
        try:
            rows = list(environment.rows())
            program = instantiate(with_row(environment.program, environment.start(rows[0], random.Random(0))))
            imports, kinds = list(program.imports()), sorted({spec.kind for spec in program.sandboxes().values()})
        except Exception as error:
            findings.append(Finding("episode", False, f"its program cannot be made: {type(error).__name__}: {error}"))
        else:
            if imports or kinds:
                needs = ", ".join(
                    [*(f"the tool set {each}" for each in imports), *(f"{kind} sandboxes" for kind in kinds)]
                )
                findings.append(Finding("episode", True, f"not played here: its program needs {needs}", flagged=True))
            else:
                findings.append(asyncio.run(scripted(environment, within=EPISODE)))
    passed = all(each.passed for each in findings)
    return {
        **here,
        "loaded": True,
        "findings": [
            {"check": each.check, "passed": each.passed, "said": each.said, "flagged": each.flagged}
            for each in findings
        ],
        "described": described(environment) if passed else None,
    }


def main(arguments: Sequence[str] | None = None) -> None:
    given = list(sys.argv[1:] if arguments is None else arguments)
    if len(given) != 2 or given[0] != "check":
        raise SystemExit("python -m rollout_train.publishing check ENTRY_POINT")
    print(MARK + json.dumps(report(given[1])), flush=True)


if __name__ == "__main__":
    main()
