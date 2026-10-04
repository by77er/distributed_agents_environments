"""A launcher: it starts the runs asked for (`rollout_train.launches`) that it can run.

`rollout launcher --ledger WHERE --profiles DIRECTORY --environment module:name … --runs DIRECTORY` beats like a runner
(`rollout_train.presence`), saying what it offers: each profile it can run (every `*.toml` under `--profiles` that loads
and names a trainer), with what its trainer makes (`lora` or `full` weights) and the settings a launch may change and
their values in the profile; the environments; and whether it has room. It claims the oldest launch asked for one of its
profiles, whose environments it offers each of, while it plays fewer than `--at-once`, starts `rollout train` for it in
a directory of its own under `--runs` (`NAME-ID`), each setting the launch changes as `--set KEY=VALUE` (no evals, said
so, as `evals.suite=""`, so that the profile's `[evals]` is not used), and notes how it goes. A launch asked to stop is
sent an interrupt: the run stops as it does on Ctrl-C, at a group boundary of the ledger.

Without `--ray`, it starts each run as a process of its own, on its own machine. With `--ray ADDRESS` (a Ray
cluster's job server), it submits each run as a Ray job asking for `--gpus` accelerators: Ray places it on a node
with room and supervises it, and the launcher follows the job until it ends (started again, it follows its jobs
again). `rollout launcher … --ray ADDRESS --as-job` submits the launcher itself as a Ray job.
"""

import asyncio
import contextlib
import json
import os
import re
import shlex
import signal
import socket
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rollout_train.launches import (
    ASKED,
    CLAIMED,
    ENDED,
    EVAL,
    FAILED,
    OPEN,
    RUNNING,
    STOPPED,
    STOPPING,
    Launch,
    Launches,
)
from rollout_train.machine import alive, measured
from rollout_train.presence import Presence
from rollout_train.ray_cluster import prepare
from rollout_train.settings import EVALS_SUITE

LAUNCHER = "launcher"
"""What a launcher's heartbeat says it is (`about["kind"]`)."""
OUTPUT = "train.log"
"""Where a launched run's output goes, in its directory."""
TAIL = 2000
"""Characters of a failed run's output kept as why it failed."""


def offered(directory: Path) -> list[dict[str, Any]]:
    """The profiles under `directory` that a launch can name: each by its name (its file's, without `.toml`), with
    its path, what its trainer makes (`weights`: `lora`, `full`, or None where its trainer cannot be read here) and the
    settings a launch may change, with their values in the profile."""
    from rollout_train.profile import Profile

    found: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.toml")):
        try:
            profile = Profile.load(path)
        except Exception:  # (a file that is not a profile, or not one this machine can load)
            continue
        if profile.trainer is None:
            continue
        settings: dict[str, Any] = {f"trainer.{key}": value for key, value in profile.trainer.settings.items()}
        settings |= {
            "trainer.start": profile.trainer.start,
            "trainer.bookmark": profile.trainer.bookmark,
            "episodes_at_once": profile.episodes_at_once,
        }
        for channel, spec in profile.channels.items():
            settings |= {f"channels.{channel}.thinking_tokens": spec.thinking_tokens}
            settings |= {f"channels.{channel}.answer_tokens": spec.answer_tokens}
        evals = profile.evals
        settings |= {"evals.suite": evals.suite if evals else None, "evals.every": evals.every if evals else None}
        settings |= {"evals.episodes": evals.episodes if evals else None}
        model = profile.channels[profile.trainer.channel].model
        found.append(
            {
                "profile": path.stem,
                "path": str(path),
                "model": model,
                "weights": _weights(profile.trainer.kind),
                "settings": settings,
            }
        )
    return found


def _weights(trainer: str) -> str | None:
    """What a trainer (`module:name`) makes, as it says (`Trainer.weights`), without making one."""
    from rollout.names import named

    try:
        made = getattr(named(trainer), "weights", None)
    except Exception:  # (a trainer this machine cannot import)
        return None
    return made if isinstance(made, str) else None


def slug(name: str) -> str:
    """A name as a directory's: letters, digits and dashes."""
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower() or "run"


@dataclass
class Launcher:
    name: str
    launches: Launches
    presence: Presence
    profiles: Path
    environments: Sequence[str]
    runs: Path
    at_once: int = 1
    ray: str | None = None
    """A Ray cluster's job server (`http://127.0.0.1:8265`): each run is then a Ray job, placed and supervised by Ray,
    asking for `gpus` accelerators; without, a process of this launcher's."""
    gpus: float = 1.0
    every: float = 2.0
    beating: float = 15.0
    _playing: dict[str, asyncio.subprocess.Process] = field(default_factory=dict[str, asyncio.subprocess.Process])
    _jobs: dict[str, str] = field(default_factory=dict[str, str])
    """The Ray jobs playing launches, by launch."""
    _offered: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    _watching: set[asyncio.Task[None]] = field(default_factory=set[asyncio.Task[None]])

    async def serve(self) -> None:
        """Beat, claim, start and watch runs until cancelled; the runs it started go on."""
        self._offered = await asyncio.to_thread(offered, self.profiles)
        await self._adopt()
        await self._beat()
        beating = asyncio.create_task(self._beats())
        try:
            while True:
                await self._step()
                await asyncio.sleep(self.every)
        finally:
            beating.cancel()

    async def _step(self) -> None:
        launches = await self.launches.all()
        for launch in launches:
            if launch.id in self._playing and launch.state == STOPPING:
                with contextlib.suppress(ProcessLookupError):
                    self._playing[launch.id].send_signal(signal.SIGINT)
            if launch.id in self._jobs and launch.state == STOPPING:
                with contextlib.suppress(Exception):  # (a job that already ended)
                    await asyncio.to_thread(self._client().stop_job, self._jobs[launch.id])
        mine = {each["profile"] for each in self._offered}
        asked = sorted(
            (each for each in launches if each.state == ASKED and each.asked.profile in mine and self._plays(each)),
            key=lambda each: each.at,
        )
        for launch in asked[: max(0, self.at_once - len(self._playing) - len(self._jobs))]:
            claimed = await self.launches.claim(launch.id, self.name)
            if claimed is not None:
                await self._start(claimed)

    def _plays(self, launch: Launch) -> bool:
        """Whether it offers every environment a launch plays (one that names no environments offers any)."""
        return not self.environments or launch.asked.plays() <= set(self.environments)

    async def _start(self, launch: Launch) -> None:
        asked = launch.asked
        profile = next(each for each in self._offered if each["profile"] == asked.profile)
        directory = self.runs / f"{slug(asked.name)}-{launch.id[-6:].lower()}"
        settings: dict[str, Any] = dict(asked.settings)
        if asked.kind != EVAL:  # (an eval's checkpoint is what plays, not where training starts)
            settings |= {"trainer.start": asked.start} if asked.start else {}
            settings |= {"trainer.bookmark": asked.bookmark} if asked.bookmark else {}
        if asked.kind != EVAL and EVALS_SUITE in settings and not settings[EVALS_SUITE]:
            settings[EVALS_SUITE] = ""  # (no evals, said so: the profile's `[evals]` is not used)
        changed = [
            argument for key, value in settings.items() if value is not None
            for argument in ("--set", f"{key}={json.dumps(value)}")
        ]  # fmt: skip
        if asked.kind == EVAL:
            command = [
                sys.executable, "-m", "rollout_train.cli", "eval", profile["path"], str(asked.suite),
                "--directory", str(directory), "--name", asked.name,
                *(["--episodes", str(asked.episodes)] if asked.episodes else []),
                *(["--environment", asked.environment] if asked.environment else []),
                *(["--checkpoint", asked.start] if asked.start else []), *changed,
            ]  # fmt: skip
        else:
            command = [
                sys.executable, "-m", "rollout_train.cli", "train", profile["path"], asked.environment,
                "--directory", str(directory), "--name", asked.name, "--groups", str(asked.groups),
                "--groups-per-step", str(asked.groups_per_step), "--seed", str(asked.seed), *changed,
            ]  # fmt: skip
        if self.ray is not None:
            await self._submit(launch, command, directory)
            return
        try:
            await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
            output = await asyncio.to_thread((directory / OUTPUT).open, "ab")
            process = await asyncio.create_subprocess_exec(
                *command, stdout=output, stderr=asyncio.subprocess.STDOUT, start_new_session=True
            )
        except Exception as error:  # a run that cannot start is a failed launch
            await self.launches.note(
                launch.id, state=FAILED, directory=str(directory), detail=f"{type(error).__name__}: {error}"
            )
            return
        self._playing[launch.id] = process
        where: dict[str, Any] = {"directory": str(directory), "pid": process.pid}
        noted = await self.launches.note(launch.id, expect=(CLAIMED,), state=RUNNING, **where)
        if noted.state != RUNNING:  # asked to stop while it started: it stays so, and the next step signals it
            await self.launches.note(launch.id, expect=OPEN, **where)
        watching = asyncio.create_task(self._watch(launch.id, process, directory))
        self._watching.add(watching)
        watching.add_done_callback(self._watching.discard)

    async def _watch(self, id: str, process: asyncio.subprocess.Process, directory: Path) -> None:
        code = await process.wait()
        self._playing.pop(id, None)
        stopped = await self.launches.note(id, expect=(STOPPING,), state=STOPPED, detail=f"stopped (exit {code})")
        if stopped.state == STOPPED:
            return
        if code == 0:
            await self.launches.note(id, state=ENDED, detail="ended")
        else:
            tail = await asyncio.to_thread(_tail, directory / OUTPUT)
            await self.launches.note(id, state=FAILED, detail=f"exit {code}: {tail}")

    def _client(self) -> Any:
        prepare()
        from ray.job_submission import JobSubmissionClient

        assert self.ray is not None
        return JobSubmissionClient(self.ray)

    async def _submit(self, launch: Launch, command: list[str], directory: Path) -> None:
        """Submit a launch's run as a Ray job: it waits in Ray's queue until a node has `gpus` free, and runs there
        from this launcher's working directory (every node sees the same checkout and run directories)."""
        entrypoint = f"cd {shlex.quote(os.getcwd())} && exec {shlex.join(command)}"
        try:
            await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
            job: str = await asyncio.to_thread(
                self._client().submit_job,
                entrypoint=entrypoint,
                submission_id=f"run-{launch.id}",
                entrypoint_num_gpus=self.gpus,
                entrypoint_num_cpus=1,
                metadata={"kind": launch.asked.kind, "launch": launch.id, "name": launch.asked.name},
            )
        except Exception as error:  # a run Ray refuses is a failed launch
            detail = f"{type(error).__name__}: {error}"
            await self.launches.note(launch.id, state=FAILED, directory=str(directory), detail=detail)
            return
        await self.launches.note(launch.id, directory=str(directory), job=job)
        self._follow(launch.id, job, directory)

    def _follow(self, id: str, job: str, directory: Path) -> None:
        self._jobs[id] = job
        following = asyncio.create_task(self._following(id, job, directory))
        self._watching.add(following)
        following.add_done_callback(self._watching.discard)

    async def _following(self, id: str, job: str, directory: Path) -> None:
        """Note how a Ray job goes until it ends; its output is then written to the run's directory."""
        from ray.job_submission import JobStatus

        client, running = self._client(), False
        while True:
            status = await asyncio.to_thread(client.get_job_status, job)
            if status == JobStatus.RUNNING and not running:
                running = True
                await self.launches.note(id, expect=(CLAIMED,), state=RUNNING)  # (not over a stop asked for meanwhile)
            if status.is_terminal():
                break
            await asyncio.sleep(self.every)
        self._jobs.pop(id, None)
        output: str = await asyncio.to_thread(client.get_job_logs, job)
        await asyncio.to_thread((directory / OUTPUT).write_text, output)
        stopped = f"stopped (Ray job {job})"
        if (await self.launches.note(id, expect=(STOPPING,), state=STOPPED, detail=stopped)).state == STOPPED:
            return
        if status == JobStatus.STOPPED:
            await self.launches.note(id, state=STOPPED, detail=stopped)
        elif status == JobStatus.SUCCEEDED:
            await self.launches.note(id, state=ENDED, detail="ended")
        else:
            await self.launches.note(id, state=FAILED, detail=f"Ray job {job} {status.value}: {output[-TAIL:].strip()}")

    async def _adopt(self) -> None:
        """Follow again the Ray jobs this launcher submitted before it was started again (a process it started cannot
        be waited on by another: its launch is noted when its process is gone)."""
        for launch in await self.launches.all():
            if launch.launcher != self.name or launch.state not in (CLAIMED, RUNNING, STOPPING):
                continue
            if launch.job and self.ray is not None:
                self._follow(launch.id, launch.job, Path(launch.directory or self.runs))
            elif launch.pid and not alive(launch.pid):
                await self.launches.note(
                    launch.id, expect=(CLAIMED, RUNNING, STOPPING), state=ENDED,
                    detail="its end was not seen: the launcher was restarted",
                )  # fmt: skip

    async def _beats(self) -> None:
        while True:
            await asyncio.sleep(self.beating)
            with contextlib.suppress(Exception):
                await self._beat()

    async def _beat(self) -> None:
        self._offered = await asyncio.to_thread(offered, self.profiles)  # (a profile added or changed is offered)
        about: dict[str, Any] = {
            "kind": LAUNCHER,
            "host": socket.gethostname(),
            "machine": await asyncio.to_thread(measured, self.runs),
            "profiles": self._offered,
            "environments": list(self.environments),
            "at_once": self.at_once,
            "playing": len(self._playing) + len(self._jobs),
            "backend": "ray" if self.ray else "process",
            "ray": self.ray,
            "pid": os.getpid(),
        }
        await self.presence.beat(self.name, about)


def _tail(path: Path) -> str:
    try:
        return path.read_text(errors="replace")[-TAIL:].strip()
    except OSError:
        return ""


def name_of(host: str | None = None) -> str:
    """A launcher's name: `launcher/HOST`."""
    return f"{LAUNCHER}/{host or socket.gethostname()}"
