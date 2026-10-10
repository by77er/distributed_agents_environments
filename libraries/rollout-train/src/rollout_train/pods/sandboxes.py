"""A pod's sandbox host: the sandboxes of whatever kinds the run that holds the pod asks for, on the CPUs and memory its
engine and trainer leave, each kind's pool in a process, a Python environment and a user of its own (docs/research/
run-placement.md, "Sandbox pools on a host pod"). The pod's image holds no environment's code.

**What it serves** is its lease's: `settings["sandboxes"]`, a source for each kind (`rollout_train.pods.sources`: the
provider, its settings, its projects' zips in the pods' blob store, pins of everything else, a Python version, and
what a pool is sized by), which the run's driver gives when it takes the pod. A source that is not well formed is left
out. The host reads its lease every 15 seconds.

**Each kind's Python** is made from its source once (`Venvs`): its zips fetched and unpacked, then a virtual environment
made by uv with a uv-managed Python of the source's version (the pod's own Python is the vLLM image's), the projects
installed editable (`--no-sources`: a workspace's sources do not apply to a project shipped alone) and the pins as they
are (`--no-deps`: nothing is resolved, so nothing is fetched by a name alone). It is kept on the volume under the
source's digest (its zips' contents, its pins and its Python), made under a file lock, found again by later runs and
restarts, and deleted once no kind has used it for 14 days, with the zips no kept environment uses. Making one may
download a Python (from python-build-standalone's GitHub releases) and wheels (from PyPI) the first time: a failure
there is waited out and tried again, never given up on.

**Each kind's pool** runs in a process of its own (`python -P -m rollout.harness.pool_server`, nothing of its working
directory on its path), in that Python, as a user of its own (`ROLLOUT_SANDBOX_USERS`: one per kind, assigned once and
kept), on a Unix socket the host makes, owned by root and readable by no one else, and hands it: no other process of the
pod reaches it, another kind's included. The process dies with the host (`PR_SET_PDEATHSIG`), is ended with its process
group, and one a host before this one left is ended when it starts (its pid kept under `pids/`). One that ends is
started again after 5 seconds, then 10, 20, up to 5 minutes; one that ends within a minute five times running is given
up on until its source changes, the run changes, or 15 minutes pass. One kind failing (its code does not import, its
settings are wrong) leaves the others served. The host says each kind's state in a log line whenever one changes.

**What a kind's process may touch.** Its environment is made, not inherited: `PATH`, `HOME` and `TMPDIR` in its own
directory (`ROLLOUT_SANDBOX_DIRECTORY/kinds/KIND`, its own and no other user's: its leases, logs and caches, kept across
restarts), a locale. It cannot read the pod's secrets: the host, the follower, the training service and the container's
first process run as root, so their environments (`/proc/PID/environ`: the ledger token, the store's keys, the
certificate's token) are not its to read, and the certificates' directory is root's alone. The environments and zips it
runs from are root's, and read-only to it. What it can still reach is what any process of the pod can: vLLM
(`127.0.0.1:8000`) and the training service (`127.0.0.1:8001`) on the loopback interface, which take requests without
a key; the code a run gives a pod must be code the cluster trusts to run beside its trainer.

**Routing.** The pod's proxy sends `/v1/sandboxes/KIND/...` here as `/KIND/...`; the host passes `GET operations` and
`capacity` and `POST acquire`, `release` and `call` on to the kind's process, and answers 503 (`PoolUnavailable`, never
"full") while that process is not up.

**Whose leases.** An acquire for a key of another run than the host serves makes it read its lease again first; the
kinds' processes admit only the served run's keys. When the lease names another run, each process forgets the other
runs' leases. When its `renewed` is older than 5 minutes (`HOLD_AFTER`: the run's driver renews it every 30 seconds,
but may stall), the processes take no new keys and go on serving those they hold. When the lease names no run, is idle,
or its `renewed` is older than 30 minutes (`END_AFTER`: the driver is gone), each marks every lease lost, so their keys
get `SandboxLost` (their episodes are played again, never in a fresh sandbox) until their run releases them.

**How many.** Each kind's pool holds as many sandboxes as the pod's spare CPUs and memory hold (`sized`): the pod's
vCPUs less `ROLLOUT_SANDBOX_RESERVED_CPUS` (4), at the source's `cpus` each, and no more than its memory less
`ROLLOUT_SANDBOX_RESERVED_GIB` (64) at its `memory_gib` each, nor than its `size`. Where a pod serves several kinds,
those that say only a `size` take it first, and those with a `share` take their part of what is left. The pod's vCPUs
and memory are the lesser of what its lease records from RunPod and its container's limits (its cgroup), each where
known, read again at each look; a pool whose size changes is told so (or, where its provider's size is fixed, started
again once it holds nothing), and one that is down starts at the new size.

    python -m rollout_train.pods.sandboxes

- `ROLLOUT_SANDBOX_ADDRESS`: where the host serves (default `127.0.0.1:8710`);
- `ROLLOUT_SANDBOX_DIRECTORY`: its state (default `/workspace/sandboxes`);
- `ROLLOUT_SANDBOX_RESERVED_CPUS`, `ROLLOUT_SANDBOX_RESERVED_GIB`: what is kept for the pod's other processes;
- `ROLLOUT_SANDBOX_USERS`: the users kinds' processes run as, one per kind (default `sandbox1` to `sandbox8`, which the
  image makes); a host that does not run as root runs them as itself;

and `ROLLOUT_POD_NAME`, `ROLLOUT_LEDGER` and `ROLLOUT_BLOBS` (`rollout_train.pods.environment`).
"""

import asyncio
import contextlib
import fcntl
import json
import logging
import math
import os
import pwd
import shutil
import signal
import socket
import subprocess
import time
import zipfile
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx

from rollout.contracts import BlobReference
from rollout.harness.blobs import Blobs
from rollout.harness.pool_server import of_run
from rollout_train.ledger import opened
from rollout_train.pods.environment import listening, location, required, served, stores
from rollout_train.pods.leases import IDLE, PodLease, PodLeases, pod_leases_of
from rollout_train.pods.sources import SandboxSource, sources_in

if TYPE_CHECKING:
    from starlette.applications import Starlette

__all__ = ["Account", "Machine", "SandboxHost", "Venvs", "Worker", "machine_of", "sized", "worker_environment"]

log = logging.getLogger(__name__)

ADDRESS = "127.0.0.1:8710"
DIRECTORY = "/workspace/sandboxes"
USERS = tuple(f"sandbox{number}" for number in range(1, 9))
"""The users kinds' processes run as, one per kind, which the host image makes."""
RESERVED_CPUS = 4.0
"""vCPUs kept for vLLM, the trainer, the follower and Envoy."""
RESERVED_GIB = 64.0
"""GiB of memory kept for them (the trainer and vLLM each load the model through the machine's memory)."""
EVERY = 15.0
"""Seconds between looks at the pod's lease."""
FIRST_WAIT, LONGEST_WAIT = 5.0, 300.0
"""Seconds before a kind's process is started again (or its Python made again): the first time, and at most."""
QUICK_END = 60.0
"""A process that ends within this many seconds of its start ended quickly."""
GIVE_UP = 5
"""Quick ends running after which a kind is given up on (until its source or run changes, or `RETRY_FAILED`)."""
RETRY_FAILED = 900.0
"""Seconds after which a kind given up on is tried again."""
HOLD_AFTER = 300.0
"""Seconds since the lease's renewal after which no new sandbox is made."""
END_AFTER = 1800.0
"""Seconds since the lease's renewal after which the run's driver is taken to be gone, and its leases lost."""
KEEP_DAYS = 14.0
"""Days a Python environment (or a zip) no kind uses is kept."""
COLLECT_EVERY = 86400.0
ROUTES = {"operations": "GET", "capacity": "GET", "acquire": "POST", "release": "POST", "call": "POST"}
"""What the host passes on to a kind's process, and by which method."""
FORWARDING = httpx.Timeout(600.0, connect=5.0)
GIB = 2**30


@dataclass(frozen=True)
class Machine:
    """What the pod has: vCPUs and memory in GiB."""

    cpus: float
    memory_gib: float


def sized(machine: Machine, sources: Mapping[str, SandboxSource], *, cpus: float = RESERVED_CPUS,
          memory_gib: float = RESERVED_GIB) -> dict[str, int]:  # fmt: skip
    """How many sandboxes each kind's pool holds (the module's docstring): those that say only a size first, then those
    with a share, of what those left, then the rest, of what is left; each in order of its kind's name."""
    left_cpus, left_gib = max(machine.cpus - cpus, 0.0), max(machine.memory_gib - memory_gib, 0.0)
    sizes: dict[str, int] = {}

    def fits(source: SandboxSource, budget_cpus: float, budget_gib: float) -> int:
        count = max(0, min(math.floor(budget_cpus / source.cpus + 1e-9), math.floor(budget_gib / source.memory_gib)))
        return min(count, source.size) if source.size is not None else count

    def take(kind: str, count: int) -> None:
        nonlocal left_cpus, left_gib
        sizes[kind] = count
        left_cpus = max(left_cpus - count * sources[kind].cpus, 0.0)
        left_gib = max(left_gib - count * sources[kind].memory_gib, 0.0)

    sized_only = sorted(kind for kind, each in sources.items() if each.size is not None and each.share is None)
    shared = sorted(kind for kind, each in sources.items() if each.share is not None)
    rest = sorted(set(sources) - set(sized_only) - set(shared))
    for kind in sized_only:
        take(kind, fits(sources[kind], left_cpus, left_gib))
    spare_cpus, spare_gib = left_cpus, left_gib
    for kind in shared:
        share = cast(float, sources[kind].share)
        take(kind, fits(sources[kind], min(spare_cpus * share, left_cpus), min(spare_gib * share, left_gib)))
    for kind in rest:
        take(kind, fits(sources[kind], left_cpus, left_gib))
    return {kind: sizes[kind] for kind in sorted(sizes)}


def machine_of(lease: PodLease | None, cgroup: Path = Path("/sys/fs/cgroup")) -> Machine:
    """The pod's vCPUs and memory: the lesser of what its lease records from RunPod and its container's limits, each
    where known; where neither is, the machine's."""
    limited_cpus, limited_gib = _limits(cgroup)
    leased_cpus = float(lease.vcpus) if lease is not None and lease.vcpus else None
    leased_gib = lease.memory_gb * 1e9 / GIB if lease is not None and lease.memory_gb else None
    known_cpus = [each for each in (leased_cpus, limited_cpus) if each]
    known_gib = [each for each in (leased_gib, limited_gib) if each]
    return Machine(min(known_cpus) if known_cpus else float(_machine_cpus()),
                   min(known_gib) if known_gib else _machine_gib())  # fmt: skip


def _limits(cgroup: Path) -> tuple[float | None, float | None]:
    """The container's CPU and memory limits (in vCPUs and GiB), as its cgroup (v2, else v1) says; none where unset."""

    def read(path: Path) -> str | None:
        try:
            return path.read_text().strip()
        except OSError:
            return None

    cpus = gib = None
    if (quota := read(cgroup / "cpu.max")) is not None:
        allowed, _, period = quota.partition(" ")
        if allowed != "max" and period:
            cpus = int(allowed) / int(period)
    elif (allowed := read(cgroup / "cpu" / "cpu.cfs_quota_us")) is not None and int(allowed) > 0:
        period = read(cgroup / "cpu" / "cpu.cfs_period_us")
        cpus = int(allowed) / int(period or 100000)
    memory = read(cgroup / "memory.max") or read(cgroup / "memory" / "memory.limit_in_bytes")
    if memory is not None and memory != "max" and int(memory) < 2**60:  # (v1 says "unlimited" as a huge number)
        gib = int(memory) / GIB
    return cpus, gib


def _machine_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # (not on Linux)
        return os.cpu_count() or 1


def _machine_gib() -> float:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIB


def worker_environment(home: Path, temporary: Path, passed: Mapping[str, str] | None = None) -> dict[str, str]:
    """What a kind's process (and uv, making a Python) is given: a path, `home`, `temporary`, a locale, and `passed`;
    nothing of the pod's own environment."""
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(home), "TMPDIR": str(temporary), "LANG": "C.UTF-8",
        "PYTHONUNBUFFERED": "1", **(passed or {}),
    }  # fmt: skip


@contextlib.contextmanager
def _locked(path: Path) -> Generator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


Run = Callable[[Sequence[str], Mapping[str, str]], None]
"""Runs a command with an environment, raising where it fails."""


def _run(command: Sequence[str], environment: Mapping[str, str]) -> None:
    done = subprocess.run(list(command), env=dict(environment), capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(command[:3])} failed: {(done.stderr or done.stdout)[-2000:]}")


class Venvs:
    """The kinds' Python environments under `root` (`venvs/DIGEST`), made by uv from their sources' zips (fetched from
    `blobs` into `projects/SHA256`), each once, under a file lock, and kept while used."""

    def __init__(
        self,
        root: Path,
        blobs: Blobs | None,
        *,
        run: Run = _run,
        passed: Mapping[str, str] | None = None,
        python_preference: str = "only-managed",
        uv: str = "uv",
    ) -> None:
        self.root = root
        self.uv = uv
        self.blobs = blobs
        self.run = run
        self.passed = dict(passed or {})
        self.python_preference = python_preference

    async def resolved(self, source: SandboxSource) -> Path:
        """The Python of `source`'s environment, made where it is not yet."""
        source.checked()
        directory = self.root / "venvs" / source.digest
        python = directory / "bin" / "python"
        if not (directory / ".ready").exists():
            projects = [await self._project(each.sha256, each.blob) for each in source.projects]
            await asyncio.to_thread(self._made, source, directory, projects)
        (directory / ".used").touch()
        return python

    async def _project(self, sha256: str, blob: Mapping[str, Any]) -> Path:
        """A project's zip, fetched and unpacked once (`projects/SHA256`)."""
        directory = self.root / "projects" / sha256
        if (directory / ".ready").exists():
            return directory
        if self.blobs is None:
            raise RuntimeError("this pod reads no blob store: it cannot fetch a source's code")
        data = await self.blobs.read(BlobReference.model_validate(dict(blob)))
        await asyncio.to_thread(self._unpacked, directory, data)
        return directory

    def _unpacked(self, directory: Path, data: bytes) -> None:
        import io

        with _locked(directory.with_name(f"{directory.name}.lock")):
            if (directory / ".ready").exists():
                return
            shutil.rmtree(directory, ignore_errors=True)
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for entry in archive.infolist():  # (paths inside the directory only, modes as packed)
                    target = (directory / entry.filename).resolve()
                    if not target.is_relative_to(directory.resolve()):
                        raise RuntimeError(f"a project's zip names {entry.filename}, outside it")
                    archive.extract(entry, directory)
                    mode = entry.external_attr >> 16
                    if mode:
                        target.chmod(mode & 0o755)
            (directory / ".ready").write_text("")

    def _made(self, source: SandboxSource, directory: Path, projects: Sequence[Path]) -> None:
        with _locked(directory.with_name(f"{directory.name}.lock")):
            if (directory / ".ready").exists():
                return
            shutil.rmtree(directory, ignore_errors=True)
            (self.root / "uv" / "tmp").mkdir(parents=True, exist_ok=True)
            environment = {
                **worker_environment(self.root / "uv", self.root / "uv" / "tmp"),
                "UV_CACHE_DIR": str(self.root / "uv" / "cache"),
                "UV_PYTHON_INSTALL_DIR": str(self.root / "uv" / "python"), **self.passed,
            }  # fmt: skip
            self.run([self.uv, "venv", "--python", source.python, "--python-preference", self.python_preference,
                      str(directory)], environment)  # fmt: skip
            pins = directory / "pins.txt"
            pins.write_text("".join(f"{each}\n" for each in source.pins))
            editable: list[str] = []
            for project, path in zip(source.projects, projects, strict=True):
                extras = f"[{','.join(project.extras)}]" if project.extras else ""
                editable += ["--editable", f"{path}{extras}"]
            self.run([self.uv, "pip", "install", "--python", str(directory / "bin" / "python"), "--no-sources",
                      "--no-deps", *editable, "--requirement", str(pins)], environment)  # fmt: skip
            (directory / "projects.txt").write_text("".join(f"{each.sha256}\n" for each in source.projects))
            (directory / ".ready").write_text(source.digest)

    def collect(self, used: set[str], *, days: float = KEEP_DAYS) -> list[str]:
        """Delete the environments no kind in `used` (by digest) has used for `days`, and the zips no environment kept
        uses and none has used for as long; each under its lock. Returns what was deleted, by name."""
        old = time.time() - days * 86400
        deleted: list[str] = []
        kept: set[str] = set()
        for directory in sorted((self.root / "venvs").glob("*")):
            if not directory.is_dir():
                continue
            marker = directory / ".used"
            stamp = marker.stat().st_mtime if marker.exists() else directory.stat().st_mtime
            if directory.name in used or stamp > old:
                with contextlib.suppress(OSError):
                    kept |= set((directory / "projects.txt").read_text().split())
                continue
            with _locked(directory.with_name(f"{directory.name}.lock")):
                shutil.rmtree(directory, ignore_errors=True)
            deleted.append(f"venvs/{directory.name}")
        for directory in sorted((self.root / "projects").glob("*")):
            if not directory.is_dir() or directory.name in kept or directory.stat().st_mtime > old:
                continue
            with _locked(directory.with_name(f"{directory.name}.lock")):
                shutil.rmtree(directory, ignore_errors=True)
            deleted.append(f"projects/{directory.name}")
        return deleted


@dataclass(frozen=True)
class Account:
    """A user a kind's process runs as."""

    uid: int
    gid: int


def accounts_of(names: Sequence[str]) -> list[Account]:
    """The users named that this machine has."""
    found: list[Account] = []
    for name in names:
        try:
            entry = pwd.getpwnam(name)
        except KeyError:
            continue
        found.append(Account(entry.pw_uid, entry.pw_gid))
    return found


def _dying_with_parent() -> None:
    """In a child before it runs its program: end it when its parent ends (`PR_SET_PDEATHSIG`)."""
    import ctypes

    with contextlib.suppress(Exception):
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)


def _listening(path: Path) -> socket.socket:
    """A Unix socket listening at `path`, readable and writable by its owner alone."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.unlink(missing_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    path.chmod(0o600)
    listener.listen(256)
    return listener


@dataclass
class Worker:
    """A kind's pool, in a process of its own: its Python made (a failure waited out, with a wait that doubles), its
    process started, watched, and started again after it ends, until it ends quickly `GIVE_UP` times running."""

    kind: str
    source: SandboxSource
    size: int
    directory: Path
    """Its own: its state, home and temporary files."""
    socket_path: Path
    pid_path: Path
    name: str
    run: str | None
    admitting: bool = True
    account: Account | None = None
    passed: dict[str, str] = field(default_factory=dict[str, str])
    state: str = "starting"
    why: str = ""
    process: asyncio.subprocess.Process | None = None
    task: asyncio.Task[None] | None = None
    quick_ends: int = 0
    failed_at: float = 0.0
    client: httpx.AsyncClient | None = None
    _listener: socket.socket | None = None

    @property
    def answering(self) -> bool:
        return self.state == "running" and self.client is not None

    def said(self) -> str:
        detail = f" ({self.size})" if self.state == "running" else f": {self.why}" if self.why else ""
        return f"{self.kind} {self.state}{detail}"

    async def supervise(self, venvs: Venvs, changed: Callable[[], None]) -> None:
        """Make its Python and run its process, again after each end, until cancelled or given up on."""
        building = running = FIRST_WAIT
        while True:
            try:
                python = await venvs.resolved(self.source)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # (a fetch or a download that failed: waited out, never given up on)
                self.state, self.why = "waiting", f"its Python was not made: {type(error).__name__}: {error}"[:500]
                changed()
                await asyncio.sleep(building)
                building = min(building * 2, LONGEST_WAIT)
                continue
            building = FIRST_WAIT
            began = time.monotonic()
            try:
                await self._started(python)
                self.state, self.why = "running", ""
                changed()
                assert self.process is not None
                status = await self.process.wait()
                self.why = f"its process ended (status {status})"
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.why = f"{type(error).__name__}: {error}"[:500]
            await self._ended()
            if time.monotonic() - began < QUICK_END:
                self.quick_ends += 1
            else:
                self.quick_ends, running = 1, FIRST_WAIT
            if self.quick_ends >= GIVE_UP:
                self.state, self.failed_at = "failed", time.monotonic()
                changed()
                return
            self.state = "waiting"
            changed()
            await asyncio.sleep(running)
            running = min(running * 2, LONGEST_WAIT)
            self.state = "starting"

    def _own(self) -> tuple[Path, Path]:
        """Its directory, home and temporary directory: made, and its user's alone."""
        home, temporary = self.directory / "home", self.directory / "tmp"
        for each in (self.directory, home, temporary):
            each.mkdir(parents=True, exist_ok=True)
            each.chmod(0o700)
            if self.account is not None:
                os.chown(each, self.account.uid, self.account.gid)
        return home, temporary

    async def _started(self, python: Path) -> None:
        home, temporary = self._own()
        self._listener = _listening(self.socket_path)
        config = self.directory / "worker.json"
        written = self.run
        config.write_text(json.dumps({
            "kind": self.kind, "provider": self.source.provider, "settings": dict(self.source.settings),
            "size": self.size, "directory": str(self.directory), "name": self.name, "fd": self._listener.fileno(),
            "run": written,
        }))  # fmt: skip
        config.chmod(0o600)
        options: dict[str, Any] = {}
        if self.account is not None:
            os.chown(config, self.account.uid, self.account.gid)
            options = {"user": self.account.uid, "group": self.account.gid}
            if os.geteuid() == 0:  # (none of root's groups either)
                options["extra_groups"] = []
        self.process = await asyncio.create_subprocess_exec(
            str(python), "-P", "-m", "rollout.harness.pool_server", str(config),
            env=worker_environment(home, temporary, self.passed), cwd=str(home), start_new_session=True,
            pass_fds=(self._listener.fileno(),), preexec_fn=_dying_with_parent, **options,
        )  # fmt: skip
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_path.write_text(str(self.process.pid))
        self.client = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(self.socket_path)),
                                        base_url="http://pool", timeout=FORWARDING)  # fmt: skip
        deadline = time.monotonic() + 120.0
        while True:
            if self.process.returncode is not None:
                raise RuntimeError(f"its process ended while starting (status {self.process.returncode})")
            with contextlib.suppress(httpx.HTTPError):
                if (await self.client.get("/capacity", timeout=2.0)).status_code == 200:
                    break
            if time.monotonic() > deadline:
                raise RuntimeError("its process did not answer within 120 seconds")
            await asyncio.sleep(0.2)
        if (self.run, self.admitting) != (written, True):  # (the run changed while it started)
            await self._told()

    async def _ended(self) -> None:
        """Stop its process, if it runs: its process group asked to end, then ended."""
        process, self.process = self.process, None
        client, self.client = self.client, None
        if client is not None:
            await client.aclose()
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), 30.0)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
        if process is not None:  # (what it left in its group)
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
        self.pid_path.unlink(missing_ok=True)
        if self._listener is not None:
            self._listener.close()
            self._listener = None

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self._ended()
        self.state = "stopped"

    async def held(self) -> int | None:
        """How many sandboxes its pool holds now (none: it does not answer)."""
        if not self.answering or self.client is None:
            return None
        try:
            return int((await self.client.get("/capacity", timeout=5.0)).json()["leased"])
        except (httpx.HTTPError, ValueError, KeyError):
            return None

    async def follow(self, run: str | None, admitting: bool) -> None:
        """Tell its pool the run it serves now, and whether it takes new keys (told when it starts, if it is down)."""
        self.run, self.admitting = run, admitting
        if self.answering:
            await self._told()

    async def _told(self) -> None:
        assert self.client is not None
        answer = await self.client.post("/_follow", json={"run": self.run, "admitting": self.admitting})
        answer.raise_for_status()

    async def resize(self, size: int) -> bool:
        """Tell its pool to hold at most `size`; whether it could (a pool that is down starts at that size)."""
        if not self.answering or self.client is None:
            self.size = size
            return True
        answer = await self.client.post("/_resize", json={"size": size})
        if answer.status_code != 200:
            return False
        self.size = size
        return True


class SandboxHost:
    """The pools of pod `name`, as its lease in `leases` says (the module's docstring), their state under `root`, their
    processes run as `accounts` (none: as the host's own user)."""

    def __init__(
        self,
        name: str,
        leases: PodLeases | None,
        venvs: Venvs,
        root: Path,
        *,
        reserved_cpus: float = RESERVED_CPUS,
        reserved_gib: float = RESERVED_GIB,
        cgroup: Path = Path("/sys/fs/cgroup"),
        passed: Mapping[str, str] | None = None,
        accounts: Sequence[Account] | None = None,
        sockets: Path | None = None,
    ) -> None:
        self.name = name
        self.leases = leases
        self.venvs = venvs
        self.root = root
        self.reserved = (reserved_cpus, reserved_gib)
        self.cgroup = cgroup
        self.passed = dict(passed or {})
        self.accounts = list(accounts) if accounts is not None else None
        self.sockets = sockets or root / "sockets"
        self.run: str | None = None
        self.admitting = True
        self.workers: dict[str, Worker] = {}
        self._following = asyncio.Lock()
        self._said = ""
        self._collected = 0.0

    def reap(self) -> list[int]:
        """End the kinds' processes a host before this one left (their pids under `pids/`). Returns their pids."""
        ended: list[int] = []
        for path in sorted((self.root / "pids").glob("*.pid")):
            with contextlib.suppress(ValueError, OSError):
                pid = int(path.read_text())
                if "rollout.harness.pool_server" in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace"):
                    os.killpg(pid, signal.SIGKILL)
                    ended.append(pid)
            path.unlink(missing_ok=True)
        return ended

    async def followed(self) -> None:
        """Read the pod's lease, and make the kinds' processes what it says: started, resized, started again with
        another source, stopped, told another run (or to take no new keys)."""
        async with self._following:
            lease = await self.leases.get(self.name) if self.leases is not None else None
            run, admitting = await self._served(lease)
            held = lease is not None and lease.run is not None and lease.state != IDLE
            wanted = sources_in(lease.settings) if lease is not None and held else {}
            sizes = sized(machine_of(lease, self.cgroup), wanted, cpus=self.reserved[0], memory_gib=self.reserved[1])
            changed = (run, admitting) != (self.run, self.admitting)
            for kind in [each for each in self.workers if each not in wanted or sizes.get(each, 0) < 1]:
                await self.workers.pop(kind).stop()
            for kind, source in sorted(wanted.items()):
                size = sizes.get(kind, 0)
                worker = self.workers.get(kind)
                if size < 1:
                    continue
                if worker is not None and worker.source.digest == source.digest:
                    retry = worker.state == "failed" and (
                        run != self.run or time.monotonic() - worker.failed_at > RETRY_FAILED
                    )
                    if not retry and (worker.size == size or await worker.resize(size)):
                        continue
                    if not retry and await worker.held() != 0:
                        continue  # (a size its provider will not take while it holds sandboxes)
                if worker is not None:
                    await self.workers.pop(kind).stop()
                self._start(kind, source, size, run, admitting)
            self.run, self.admitting = run, admitting
            if changed:
                for worker in self.workers.values():
                    try:
                        await worker.follow(run, admitting)
                    except httpx.HTTPError as error:
                        log.warning("telling the %s pool it serves %s failed: %s", worker.kind, run, error)
            self._say()

    async def _served(self, lease: PodLease | None) -> tuple[str | None, bool]:
        """The run the pod serves, and whether new sandboxes are made for it: its lease's run, while the lease is held;
        none once it is renewed no more for `END_AFTER`, and no new ones once it is renewed no more for `HOLD_AFTER`."""
        if lease is None or lease.run is None or lease.state == IDLE:
            return None, True
        age = 0.0
        if lease.renewed is not None and self.leases is not None:
            try:
                now = await self.leases.now()
            except Exception:  # (the store's clock is not answering: this one's)
                now = time.time()
            age = now - lease.renewed
        if age > END_AFTER:
            return None, True
        return lease.run, age <= HOLD_AFTER

    def _account(self, kind: str) -> Account | None:
        """The user `kind`'s process runs as: assigned once (kept in `accounts.json`), one per kind."""
        if self.accounts is None:
            return None
        path = self.root / "accounts.json"
        assigned: dict[str, int] = json.loads(path.read_text()) if path.exists() else {}
        by_uid = {each.uid: each for each in self.accounts}
        if kind in assigned and assigned[kind] in by_uid:
            return by_uid[assigned[kind]]
        free = [each for each in self.accounts if each.uid not in assigned.values()]
        if not free:
            raise RuntimeError(f"no user is left for the kind {kind}: each kind runs as one of its own")
        assigned[kind] = free[0].uid
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(assigned))
        path.chmod(0o600)
        directory = self.root / "kinds" / kind
        if directory.exists():  # (a kind's directory another user held: given to its own)
            for each in [directory, *directory.rglob("*")]:
                with contextlib.suppress(OSError):
                    os.lchown(each, free[0].uid, free[0].gid)
        return free[0]

    def _start(self, kind: str, source: SandboxSource, size: int, run: str | None, admitting: bool) -> None:
        worker = Worker(kind, source, size, self.root / "kinds" / kind, self.sockets / f"{kind}.sock",
                        self.root / "pids" / f"{kind}.pid", f"{kind}@{self.name}", run, admitting,
                        passed=dict(self.passed))  # fmt: skip
        try:
            worker.account = self._account(kind)
        except RuntimeError as error:
            worker.state, worker.why, worker.failed_at = "failed", str(error), time.monotonic()
            self.workers[kind] = worker
            return
        worker.task = asyncio.ensure_future(worker.supervise(self.venvs, self._say))
        self.workers[kind] = worker

    def _say(self) -> None:
        said = "; ".join(worker.said() for _, worker in sorted(self.workers.items())) or "none"
        served = self.run or "no run"
        if not self.admitting:
            served += ", taking no new sandboxes (its lease is not renewed)"
        if f"{served}: {said}" != self._said:
            self._said = f"{served}: {said}"
            log.info("sandboxes on %s for %s", self.name, self._said)

    async def serve(self, every: float = EVERY) -> None:
        while True:
            try:
                await self.followed()
                if time.monotonic() - self._collected > COLLECT_EVERY or not self._collected:
                    self._collected = time.monotonic()
                    used = {worker.source.digest for worker in self.workers.values()}
                    if deleted := await asyncio.to_thread(self.venvs.collect, used):
                        log.info("deleted what no kind has used for %g days: %s", KEEP_DAYS, ", ".join(deleted))
            except Exception:  # (looked at again next time)
                log.exception("following pod %s's lease failed; trying again", self.name)
            await asyncio.sleep(every)

    async def stop(self) -> None:
        for worker in list(self.workers.values()):
            await worker.stop()
        self.workers.clear()

    def app(self) -> "Starlette":
        """`/KIND/ROUTE`, passed on to the kind's process (`ROUTES`)."""
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, Response
        from starlette.routing import Route

        async def forwarded(request: Request) -> Response:
            kind, route = request.path_params["kind"], request.path_params["route"]
            if ROUTES.get(route) != request.method:
                return JSONResponse({"error": f"no {request.method} /{kind}/{route}"}, status_code=404)
            worker = self.workers.get(kind)
            if worker is None or not worker.answering or worker.client is None:
                said = worker.said() if worker is not None else f"{kind} is not served here now"
                return JSONResponse({"error": f"the {kind} pool is not up: {said}"}, status_code=503)
            body = await request.body()
            if route == "acquire":
                key = str(cast(dict[str, Any], json.loads(body or b"{}")).get("key", ""))
                if not of_run(key, self.run):
                    await self.followed()  # (another run may have taken the pod since the last look)
                if not of_run(key, self.run):
                    return JSONResponse({"error": f"{key} is not of the run this pod serves"}, status_code=409)
            try:
                answer = await worker.client.request(request.method, f"/{route}", content=body,
                                                     headers={"content-type": "application/json"})  # fmt: skip
            except httpx.TransportError as error:
                return JSONResponse({"error": f"the {kind} pool did not answer: {type(error).__name__}"},
                                    status_code=503)  # fmt: skip
            return Response(answer.content, status_code=answer.status_code, media_type="application/json")

        return Starlette(routes=[Route("/{kind}/{route}", forwarded, methods=["GET", "POST"])])


def _undumpable() -> None:
    """This process's memory and environment are root's alone, even to its own user's processes (`PR_SET_DUMPABLE`)."""
    import ctypes

    with contextlib.suppress(Exception):
        ctypes.CDLL(None).prctl(4, 0)


async def main(environ: Mapping[str, str]) -> None:
    """Serve what the pod's lease asks for until cancelled; the kinds' processes stopped on the way out."""
    os.umask(0o022)  # (environments and zips: readable by the kinds' users, written by root alone)
    _undumpable()
    name = required(environ, "ROLLOUT_POD_NAME")
    ledger, blobs = stores(environ) if environ.get("ROLLOUT_BLOBS") else (opened(location(environ, "ROLLOUT_LEDGER")),
                                                                          None)  # fmt: skip
    root = Path(environ.get("ROLLOUT_SANDBOX_DIRECTORY") or DIRECTORY)
    root.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 (before it serves)
    root.chmod(0o755)  # noqa: ASYNC240
    names = [each for each in (environ.get("ROLLOUT_SANDBOX_USERS") or ",".join(USERS)).split(",") if each]
    accounts = accounts_of(names) if os.geteuid() == 0 else None
    if accounts is None:
        log.warning("the sandbox host does not run as root: the kinds' processes run as its own user")
    host = SandboxHost(
        name, pod_leases_of(ledger), Venvs(root, blobs), root, accounts=accounts,
        reserved_cpus=float(environ.get("ROLLOUT_SANDBOX_RESERVED_CPUS") or RESERVED_CPUS),
        reserved_gib=float(environ.get("ROLLOUT_SANDBOX_RESERVED_GIB") or RESERVED_GIB),
    )  # fmt: skip
    if ended := host.reap():
        log.info("ended the processes a host before this one left: %s", ", ".join(map(str, ended)))
    address, port = listening(environ, "ROLLOUT_SANDBOX_ADDRESS", ADDRESS)
    try:
        await asyncio.gather(host.serve(), served(host.app(), address, port))
    finally:
        await host.stop()


async def _until_stopped(environ: Mapping[str, str], run: Callable[[Mapping[str, str]], Awaitable[None]]) -> None:
    """`run`, until the process is told to stop (`SIGTERM`, `SIGINT`)."""
    task = asyncio.current_task()
    assert task is not None
    loop = asyncio.get_running_loop()
    for each in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(each, task.cancel)
    with contextlib.suppress(asyncio.CancelledError):
        await run(environ)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(_until_stopped(os.environ, main))
