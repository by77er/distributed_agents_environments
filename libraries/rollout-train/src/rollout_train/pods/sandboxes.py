"""A pod's sandbox host: the sandboxes of whatever kinds the run that holds the pod asks for, on the CPUs and memory its
engine and trainer leave, each kind's pool in a process and a Python environment of its own (docs/research/
run-placement.md, "Sandbox pools on a host pod"). The pod's image holds no environment's code.

**What it serves** is its lease's: `settings["sandboxes"]`, a source for each kind (`rollout_train.pods.sources`: the
provider, its settings, its projects' zips in the pods' blob store, pins of what they need, a Python version, and
what a pool is sized by), which the run's driver gives when it takes the pod. The host reads its lease every 15 seconds.

**Each kind's Python** is made from its source once (`Venvs`): its zips fetched and unpacked, then a virtual environment
made by uv with a uv-managed Python of the source's version (the pod's own Python is the vLLM image's), the projects
installed editable, constrained to the source's pins. It is kept on the volume under the source's digest (its zips'
contents, its pins and its Python), made under a file lock, and found again by later runs and restarts.

**Each kind's pool** runs in a process of its own (`python -m rollout.harness.pool_server`), in that Python, on a
loopback port of its own, its state in `ROLLOUT_SANDBOX_DIRECTORY/kinds/KIND` (its leases, kept across restarts). A
process that ends is started again after 5 seconds, then 10, 20, up to 5 minutes; one that ends within a minute five
times running is given up on until its source changes; and one kind's failing (its code does not import, its settings
are wrong) leaves the others served. The host says each kind's state in a log line whenever one changes.

**Its environment** is made, not inherited: `PATH`, `HOME` (on the volume, `ROLLOUT_SANDBOX_DIRECTORY/home`, so every
provider's caches, `~/.cache`, are kept across restarts), `TMPDIR`, a locale. Nothing of the pod's (its ledger token,
the store's keys, the certificate's token) reaches environment code.

**Routing.** The pod's proxy sends `/v1/sandboxes/KIND/...` here as `/KIND/...`; the host passes `GET operations` and
`capacity` and `POST acquire`, `release` and `call` on to the kind's process, and answers 503 (`PoolUnavailable`, never
"full") while that process is not up.

**Whose leases.** An acquire for a key of another run than the host serves makes it read its lease again first; the
kinds' processes admit only the served run's keys. When the lease names another run, each process forgets the other
runs' leases; when it names none, or its `renewed` is older than `STALE` (the run's driver is gone: its keeper would
renew it), each marks every lease lost, so their keys get `SandboxLost` (their episodes are played again, never in a
fresh sandbox) until their run releases them.

**How many.** Each kind's pool holds as many sandboxes as the pod's spare CPUs and memory hold (`sized`): the pod's
vCPUs less `ROLLOUT_SANDBOX_RESERVED_CPUS` (4), at the source's `cpus` each, and no more than its memory less
`ROLLOUT_SANDBOX_RESERVED_GIB` (64) at its `memory_gib` each, nor than its `size`. Where a pod serves several kinds
each takes its `share` of what is spare, or what the kinds before it left, up to its `size`. The pod's vCPUs and
memory are the lesser of what its lease records from RunPod and its container's limits (its cgroup), each where known;
read again at each look, a pool whose size changes is started again once it holds nothing.

    python -m rollout_train.pods.sandboxes

- `ROLLOUT_SANDBOX_ADDRESS`: where the host serves (default `127.0.0.1:8710`);
- `ROLLOUT_SANDBOX_DIRECTORY`: its state (default `/workspace/sandboxes`);
- `ROLLOUT_SANDBOX_RESERVED_CPUS`, `ROLLOUT_SANDBOX_RESERVED_GIB`: what is kept for the pod's other processes;

and `ROLLOUT_POD_NAME`, `ROLLOUT_LEDGER` and `ROLLOUT_BLOBS` (`rollout_train.pods.environment`).
"""

import asyncio
import contextlib
import fcntl
import json
import logging
import math
import os
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
from rollout_train.pods.leasing import STALE
from rollout_train.pods.sources import SandboxSource, sources_in

if TYPE_CHECKING:
    from starlette.applications import Starlette

__all__ = ["Machine", "SandboxHost", "Venvs", "Worker", "machine_of", "sized", "worker_environment"]

log = logging.getLogger(__name__)

ADDRESS = "127.0.0.1:8710"
DIRECTORY = "/workspace/sandboxes"
RESERVED_CPUS = 4.0
"""vCPUs kept for vLLM, the trainer, the follower and Envoy."""
RESERVED_GIB = 64.0
"""GiB of memory kept for them (the trainer and vLLM each load the model through the machine's memory)."""
EVERY = 15.0
"""Seconds between looks at the pod's lease."""
FIRST_WAIT, LONGEST_WAIT = 5.0, 300.0
"""Seconds before a kind's process is started again: the first time, and at most (doubling between)."""
QUICK_END = 60.0
"""A process that ends within this many seconds of its start ended quickly."""
GIVE_UP = 5
"""Quick ends running after which a kind is given up on until its source changes."""
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
    """How many sandboxes each kind's pool holds (the module's docstring)."""
    spare_cpus, spare_gib = max(machine.cpus - cpus, 0.0), max(machine.memory_gib - memory_gib, 0.0)
    left_cpus, left_gib = spare_cpus, spare_gib
    sizes: dict[str, int] = {}
    for kind in sorted(sources):
        source = sources[kind]
        budget_cpus, budget_gib = (
            (spare_cpus * source.share, spare_gib * source.share) if source.share is not None else (left_cpus, left_gib)
        )
        count = max(0, min(math.floor(budget_cpus / source.cpus + 1e-9), math.floor(budget_gib / source.memory_gib)))
        if source.size is not None:
            count = min(count, source.size)
        sizes[kind] = count
        left_cpus, left_gib = max(left_cpus - count * source.cpus, 0.0), max(left_gib - count * source.memory_gib, 0.0)
    return sizes


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


def worker_environment(root: Path, passed: Mapping[str, str] | None = None) -> dict[str, str]:
    """What a kind's process (and uv, making its Python) is given: a path, a home and a temporary directory on the
    volume, a locale, and `passed`; nothing of the pod's own environment."""
    for each in ("home", "tmp"):
        (root / each).mkdir(parents=True, exist_ok=True)
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(root / "home"), "TMPDIR": str(root / "tmp"),
        "LANG": "C.UTF-8", "PYTHONUNBUFFERED": "1", **(passed or {}),
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
    `blobs` into `projects/SHA256`), each once, under a file lock, and kept."""

    def __init__(self, root: Path, blobs: Blobs | None, *, run: Run = _run, passed: Mapping[str, str] | None = None):
        self.root = root
        self.blobs = blobs
        self.run = run
        self.passed = dict(passed or {})

    async def resolved(self, source: SandboxSource) -> Path:
        """The Python of `source`'s environment, made where it is not yet."""
        directory = self.root / "venvs" / source.digest
        python = directory / "bin" / "python"
        if (directory / ".ready").exists():
            return python
        projects = [await self._project(each.sha256, each.blob) for each in source.projects]
        await asyncio.to_thread(self._made, source, directory, projects)
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
                        target.chmod(mode & 0o777)
            (directory / ".ready").write_text("")

    def _made(self, source: SandboxSource, directory: Path, projects: Sequence[Path]) -> None:
        with _locked(directory.with_name(f"{directory.name}.lock")):
            if (directory / ".ready").exists():
                return
            shutil.rmtree(directory, ignore_errors=True)
            environment = {
                **worker_environment(self.root), "UV_CACHE_DIR": str(self.root / "uv" / "cache"),
                "UV_PYTHON_INSTALL_DIR": str(self.root / "uv" / "python"), **self.passed,
            }  # fmt: skip
            self.run(["uv", "venv", "--python", source.python, "--python-preference", "only-managed",
                      str(directory)], environment)  # fmt: skip
            constraints = directory / "constraints.txt"
            constraints.write_text("".join(f"{each}\n" for each in source.constraints))
            editable: list[str] = []
            for project, path in zip(source.projects, projects, strict=True):
                extras = f"[{','.join(project.extras)}]" if project.extras else ""
                editable += ["--editable", f"{path}{extras}"]
            self.run(["uv", "pip", "install", "--python", str(directory / "bin" / "python"), *editable,
                      "--constraints", str(constraints)], environment)  # fmt: skip
            (directory / ".ready").write_text(source.digest)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@dataclass
class Worker:
    """A kind's pool, in a process of its own: started, watched, and started again after it ends, with a wait that
    doubles, until it ends quickly `GIVE_UP` times running."""

    kind: str
    source: SandboxSource
    size: int
    directory: Path
    name: str
    run: str | None
    state: str = "starting"
    why: str = ""
    port: int | None = None
    process: asyncio.subprocess.Process | None = None
    task: asyncio.Task[None] | None = None
    quick_ends: int = 0
    environment: dict[str, str] = field(default_factory=dict[str, str])

    @property
    def answering(self) -> bool:
        return self.state == "running" and self.port is not None

    def said(self) -> str:
        detail = f" ({self.size})" if self.state == "running" else f": {self.why}" if self.why else ""
        return f"{self.kind} {self.state}{detail}"

    async def supervise(self, venvs: Venvs, changed: Callable[[], None]) -> None:
        """Make its Python and run its process, again after each end, until cancelled or given up on."""
        wait = FIRST_WAIT
        while True:
            began = time.monotonic()
            try:
                python = await venvs.resolved(self.source)
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
                self.quick_ends, wait = 1, FIRST_WAIT
            if self.quick_ends >= GIVE_UP:
                self.state = "failed"
                changed()
                return
            self.state = "waiting"
            changed()
            await asyncio.sleep(wait)
            wait = min(wait * 2, LONGEST_WAIT)
            self.state = "starting"

    async def _started(self, python: Path) -> None:
        self.port = _free_port()
        self.directory.mkdir(parents=True, exist_ok=True)
        config = self.directory / "worker.json"
        config.write_text(json.dumps({
            "kind": self.kind, "provider": self.source.provider, "settings": dict(self.source.settings),
            "size": self.size, "directory": str(self.directory), "name": self.name, "host": "127.0.0.1",
            "port": self.port, "run": self.run,
        }))  # fmt: skip
        config.chmod(0o600)
        self.process = await asyncio.create_subprocess_exec(
            str(python), "-m", "rollout.harness.pool_server", str(config), env=self.environment,
            cwd=str(self.directory), start_new_session=True,
        )  # fmt: skip
        deadline = time.monotonic() + 120.0
        async with httpx.AsyncClient(timeout=2.0) as client:
            while True:
                if self.process.returncode is not None:
                    raise RuntimeError(f"its process ended while starting (status {self.process.returncode})")
                with contextlib.suppress(httpx.HTTPError):
                    if (await client.get(f"http://127.0.0.1:{self.port}/capacity")).status_code == 200:
                        return
                if time.monotonic() > deadline:
                    raise RuntimeError("its process did not answer within 120 seconds")
                await asyncio.sleep(0.2)

    async def _ended(self) -> None:
        """Stop its process, if it runs: asked to end, then ended."""
        process, self.process = self.process, None
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), 30.0)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self._ended()
        self.state = "stopped"

    async def held(self) -> int | None:
        """How many sandboxes its pool holds now (none: it does not answer)."""
        if not self.answering:
            return None
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                return int((await client.get(f"http://127.0.0.1:{self.port}/capacity")).json()["leased"])
        except (httpx.HTTPError, ValueError, KeyError):
            return None

    async def follow(self, run: str | None) -> None:
        """Tell its pool the run it serves now."""
        self.run = run
        if not self.answering:
            return  # (it is told when it starts)
        async with httpx.AsyncClient(timeout=FORWARDING) as client:
            (await client.post(f"http://127.0.0.1:{self.port}/_follow", json={"run": run})).raise_for_status()


class SandboxHost:
    """The pools of pod `name`, as its lease in `leases` says (the module's docstring), their state under `root`."""

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
        stale: float = STALE,
    ) -> None:
        self.name = name
        self.leases = leases
        self.venvs = venvs
        self.root = root
        self.reserved = (reserved_cpus, reserved_gib)
        self.cgroup = cgroup
        self.passed = dict(passed or {})
        self.stale = stale
        self.run: str | None = None
        self.workers: dict[str, Worker] = {}
        self._following = asyncio.Lock()
        self._said = ""

    async def followed(self) -> None:
        """Read the pod's lease, and make the kinds' processes what it says: started, started again with another
        source or size, stopped, told another run."""
        async with self._following:
            lease = await self.leases.get(self.name) if self.leases is not None else None
            run = await self._served(lease)
            held = lease is not None and lease.run is not None and lease.state != IDLE
            wanted = sources_in(lease.settings) if lease is not None and held else {}
            sizes = sized(machine_of(lease, self.cgroup), wanted, cpus=self.reserved[0], memory_gib=self.reserved[1])
            for kind in [each for each in self.workers if each not in wanted or sizes.get(each, 0) < 1]:
                await self.workers.pop(kind).stop()
            for kind, source in sorted(wanted.items()):
                size = sizes.get(kind, 0)
                worker = self.workers.get(kind)
                if size < 1:
                    continue
                if worker is not None and worker.source.digest == source.digest and worker.size == size:
                    continue
                if worker is not None and worker.source.digest == source.digest and await worker.held() != 0:
                    continue  # (another size, once it holds nothing)
                if worker is not None:
                    await self.workers.pop(kind).stop()
                self._start(kind, source, size, run)
            if run != self.run:
                self.run = run
                for worker in self.workers.values():
                    try:
                        await worker.follow(run)
                    except httpx.HTTPError as error:
                        log.warning("telling the %s pool it serves %s failed: %s", worker.kind, run, error)
            self._say()

    async def _served(self, lease: PodLease | None) -> str | None:
        """The run the pod serves: its lease's, while the lease is held and renewed within `stale` seconds."""
        if lease is None or lease.run is None or lease.state == IDLE:
            return None
        if lease.renewed is not None and self.leases is not None:
            try:
                now = await self.leases.now()
            except Exception:  # (the store's clock is not answering: this one's)
                now = time.time()
            if now - lease.renewed > self.stale:
                return None
        return lease.run

    def _start(self, kind: str, source: SandboxSource, size: int, run: str | None) -> None:
        worker = Worker(kind, source, size, self.root / "kinds" / kind, f"{kind}@{self.name}", run,
                        environment=worker_environment(self.root, self.passed))  # fmt: skip
        worker.task = asyncio.ensure_future(worker.supervise(self.venvs, self._say))
        self.workers[kind] = worker

    def _say(self) -> None:
        said = "; ".join(worker.said() for _, worker in sorted(self.workers.items())) or "none"
        if said != self._said:
            self._said = said
            log.info("sandboxes on %s for %s: %s", self.name, self.run or "no run", said)

    async def serve(self, every: float = EVERY) -> None:
        while True:
            try:
                await self.followed()
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

        client = httpx.AsyncClient(timeout=FORWARDING)

        async def forwarded(request: Request) -> Response:
            kind, route = request.path_params["kind"], request.path_params["route"]
            if ROUTES.get(route) != request.method:
                return JSONResponse({"error": f"no {request.method} /{kind}/{route}"}, status_code=404)
            worker = self.workers.get(kind)
            if worker is None or not worker.answering:
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
                answer = await client.request(request.method, f"http://127.0.0.1:{worker.port}/{route}", content=body,
                                              headers={"content-type": "application/json"})  # fmt: skip
            except httpx.TransportError as error:
                return JSONResponse({"error": f"the {kind} pool did not answer: {type(error).__name__}"},
                                    status_code=503)  # fmt: skip
            return Response(answer.content, status_code=answer.status_code, media_type="application/json")

        return Starlette(routes=[Route("/{kind}/{route}", forwarded, methods=["GET", "POST"])])


async def main(environ: Mapping[str, str]) -> None:
    """Serve what the pod's lease asks for until cancelled; the kinds' processes stopped on the way out."""
    name = required(environ, "ROLLOUT_POD_NAME")
    ledger, blobs = stores(environ) if environ.get("ROLLOUT_BLOBS") else (opened(location(environ, "ROLLOUT_LEDGER")),
                                                                          None)  # fmt: skip
    root = Path(environ.get("ROLLOUT_SANDBOX_DIRECTORY") or DIRECTORY)
    host = SandboxHost(
        name, pod_leases_of(ledger), Venvs(root, blobs), root,
        reserved_cpus=float(environ.get("ROLLOUT_SANDBOX_RESERVED_CPUS") or RESERVED_CPUS),
        reserved_gib=float(environ.get("ROLLOUT_SANDBOX_RESERVED_GIB") or RESERVED_GIB),
    )  # fmt: skip
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
