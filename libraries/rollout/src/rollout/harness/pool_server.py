"""A sandbox pool in a process of its own, for one run at a time: what a machine that serves sandboxes for runs (a GPU
pod's sandbox host, `rollout_train.pods.sandboxes`) runs for each kind, in the kind's own Python environment.

    python -m rollout.harness.pool_server CONFIG

`CONFIG` is a JSON file: the kind, the provider (`module:name`, called as `provider(directory, size=SIZE,
**settings)`), its settings and size, the directory its state goes in, the pool's name, where to serve
(`host`, `port`), and the run it serves (`run`; none: no run). The process serves the pool over HTTP
(`rollout.harness.remote.serve_pool`) and one route more, for whoever started it:

    POST /_follow             {"run": RUN or null}: the run it serves now

- **Whose keys.** A key is admitted only while it is of the run it serves (`of_run`): its first part is the run's id,
  or the id of one of the run's evals (`RUN-...`). Any other gets `LeaseRefused`. A sandbox made for a key whose run
  stopped being served while it was made is deleted, and the key refused.
- **Another run, or none.** Told another run, it marks every live lease not of that run lost (deleting its sandbox)
  and forgets the leases of other runs; told none (the run's driver is gone, or the machine released), it marks every
  live lease lost and forgets nothing. A lost lease's key gets `SandboxLost` from then on, so its episode is played
  again, never in a fresh sandbox under the same key; a lease ends only when its run releases it.
- **What a restart keeps.** Its leases are in `sandboxes.json` in its directory. Started again, it finds them and marks
  lost those whose sandboxes are gone. Stopped, it deletes its sandboxes and keeps its leases.
- It sweeps every 15 seconds: leases past their time limit end, leases whose sandboxes are gone are marked lost, and
  sandboxes no lease names are deleted.
"""

import asyncio
import contextlib
import fcntl
import json
import logging
import os
import signal
import sys
from collections.abc import Callable, Generator, Mapping
from pathlib import Path
from typing import Any

from rollout.harness.sandboxes import Lease, LeaseRefused, Provider, SandboxPool, SandboxSpec
from rollout.names import named

__all__ = ["FollowingPool", "JsonLeases", "of_run", "serve"]

log = logging.getLogger(__name__)

SWEEP = 15.0
"""Seconds between sweeps."""


def of_run(key: str, run: str | None) -> bool:
    """Whether a lease's key is of `run`: its first part is the run's id, or one of its evals' (`RUN-...`)."""
    if run is None:
        return False
    first = key.split("/", 1)[0]
    return first == run or first.startswith(f"{run}-")


class JsonLeases:
    """`Leases` in a JSON file, written whole under a lock file beside it (one process writes it at a time)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    async def get(self, key: str) -> Lease | None:
        return (await asyncio.to_thread(self._locked_read)).get(key)

    async def put(self, lease: Lease) -> None:
        await asyncio.to_thread(self._change, lambda leases: {**leases, lease.key: lease})

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._change, lambda leases: {k: v for k, v in leases.items() if k != key})

    async def all(self) -> list[Lease]:
        return list((await asyncio.to_thread(self._locked_read)).values())

    @contextlib.contextmanager
    def _locked(self) -> Generator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _locked_read(self) -> dict[str, Lease]:
        with self._locked():
            return self._read()

    def _change(self, change: Callable[[dict[str, Lease]], dict[str, Lease]]) -> None:
        with self._locked():
            leases = change(self._read())
            staged = self.path.with_suffix(".staged")
            staged.write_text(json.dumps([lease.model_dump(mode="json") for lease in leases.values()]))
            staged.replace(self.path)

    def _read(self) -> dict[str, Lease]:
        if not self.path.exists():
            return {}
        return {each["key"]: Lease.model_validate(each) for each in json.loads(self.path.read_text())}


class FollowingPool(SandboxPool):
    """A `SandboxPool` that serves one run at a time (`run`): it admits only that run's keys, and `follow` moves it to
    another run, or none."""

    def __init__(self, provider: Provider, *, name: str, leases: JsonLeases, run: str | None) -> None:
        super().__init__(provider, name=name, leases=leases, admits=self._admitted)
        self.run = run

    async def _admitted(self, key: str) -> bool:
        return of_run(key, self.run)

    async def acquire(self, spec: SandboxSpec, key: str, environment: Mapping[str, str] | None = None) -> Lease:
        lease = await super().acquire(spec, key, environment)
        if not of_run(key, self.run):  # (another run, or none, since it was made: it is not that run's to keep)
            await self.release(key)
            raise LeaseRefused(f"{key} may hold no sandbox: its run is no longer the one served here")
        return lease

    async def follow(self, run: str | None) -> list[str]:
        """Serve `run` (none: no run). Every live lease not of it is marked lost; with a run, the leases of other runs
        are forgotten (their keys are refused from now on). Returns the keys marked lost or forgotten."""
        self.run = run
        gone: list[str] = []
        for lease in await self.held():
            if of_run(lease.key, run):
                continue
            if run is not None:
                await self.release(lease.key)
            elif not lease.lost:
                await self.lose(lease.key)
            else:
                continue
            gone.append(lease.key)
        return gone


async def serve(config: Mapping[str, Any]) -> None:
    """Serve the pool `config` describes until cancelled; its sandboxes are deleted on the way out, its leases kept."""
    import uvicorn
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    from rollout.harness.remote import serve_pool

    directory = Path(str(config["directory"]))
    directory.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 (before it serves)
    made: Provider = named(str(config["provider"]))(directory, size=int(config["size"]), **dict(config["settings"]))
    pool = FollowingPool(made, name=str(config["name"]), leases=JsonLeases(directory / "sandboxes.json"),
                         run=config.get("run"))  # fmt: skip
    await pool.sweep()  # (leases a process before this one left: their sandboxes are gone with it)
    if pool.run is not None:
        await pool.follow(pool.run)  # (and those of other runs are forgotten)

    async def follow(request: Request) -> Response:
        said = await request.json()
        run = said.get("run")
        gone = await pool.follow(str(run) if run else None)
        if gone:
            log.info("serving %s now: %d leases lost or ended", run or "no run", len(gone))
        return JSONResponse({"run": pool.run, "gone": gone})

    async def sweeping() -> None:
        while True:
            await asyncio.sleep(SWEEP)
            try:
                await pool.sweep()
            except Exception:  # (looked at again next time)
                log.exception("sweeping the pool %s failed", pool.name)

    app = serve_pool(pool, [Route("/_follow", follow, methods=["POST"])])
    server = uvicorn.Server(uvicorn.Config(app, host=str(config.get("host", "127.0.0.1")), port=int(config["port"]),
                                           log_level="warning", lifespan="off"))  # fmt: skip
    task = asyncio.ensure_future(sweeping())
    try:
        await server.serve()
    finally:
        task.cancel()
        closing = getattr(made, "close", None)
        if closing is not None:
            with contextlib.suppress(Exception):
                await closing()


async def _until_stopped(config: Mapping[str, Any]) -> None:
    current = asyncio.current_task()
    assert current is not None
    loop = asyncio.get_running_loop()
    for each in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(each, current.cancel)
    with contextlib.suppress(asyncio.CancelledError):
        await serve(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=f"[{os.getpid()}] %(levelname)s %(name)s: %(message)s")
    asyncio.run(_until_stopped(json.loads(Path(sys.argv[1]).read_text())))
