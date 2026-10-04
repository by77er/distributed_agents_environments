"""A launcher: on a training machine, it starts the runs asked for (`rollout_train.launches`) that it can run.

`rollout launcher --ledger WHERE --profiles DIRECTORY --catalog module:name … --runs DIRECTORY` beats like a runner
(`rollout_train.presence`), saying what it offers: each profile it can run (every `*.toml` under `--profiles` that
loads and names a trainer), with the settings a launch may change and their values in the profile; the catalogs; and
whether it has room. It claims the oldest launch asked for one of its profiles while it plays fewer than `--at-once`,
starts `rollout train` for it in a directory of its own under `--runs` (`NAME-ID`), and notes how it goes. A launch
asked to stop is sent an interrupt: the run stops as it does on Ctrl-C, at a group boundary of the ledger.
"""

import asyncio
import contextlib
import json
import os
import re
import signal
import socket
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rollout_train.launches import ASKED, ENDED, FAILED, RUNNING, STOPPED, STOPPING, Launch, Launches
from rollout_train.machine import measured
from rollout_train.presence import Presence

LAUNCHER = "launcher"
"""What a launcher's heartbeat says it is (`about["kind"]`)."""
OUTPUT = "train.log"
"""Where a launched run's output goes, in its directory."""
TAIL = 2000
"""Characters of a failed run's output kept as why it failed."""


def offered(directory: Path) -> list[dict[str, Any]]:
    """The profiles under `directory` that a launch can name: each by its name (its file's, without `.toml`), with
    its path and the settings a launch may change, with their values in the profile."""
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
        model = profile.channels[profile.trainer.channel].model
        found.append({"profile": path.stem, "path": str(path), "model": model, "settings": settings})
    return found


def slug(name: str) -> str:
    """A name as a directory's: letters, digits and dashes."""
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower() or "run"


@dataclass
class Launcher:
    name: str
    launches: Launches
    presence: Presence
    profiles: Path
    catalogs: Sequence[str]
    runs: Path
    at_once: int = 1
    every: float = 2.0
    beating: float = 15.0
    _playing: dict[str, asyncio.subprocess.Process] = field(default_factory=dict[str, asyncio.subprocess.Process])
    _offered: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    _watching: set[asyncio.Task[None]] = field(default_factory=set[asyncio.Task[None]])

    async def serve(self) -> None:
        """Beat, claim, start and watch runs until cancelled; the runs it started go on."""
        self._offered = await asyncio.to_thread(offered, self.profiles)
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
        mine = {each["profile"] for each in self._offered}
        asked = sorted(
            (each for each in launches if each.state == ASKED and each.asked.profile in mine), key=lambda each: each.at
        )
        for launch in asked[: max(0, self.at_once - len(self._playing))]:
            claimed = await self.launches.claim(launch.id, self.name)
            if claimed is not None:
                await self._start(claimed)

    async def _start(self, launch: Launch) -> None:
        asked = launch.asked
        profile = next(each for each in self._offered if each["profile"] == asked.profile)
        directory = self.runs / f"{slug(asked.name)}-{launch.id[-6:].lower()}"
        settings: dict[str, Any] = dict(asked.settings)
        settings |= {"trainer.start": asked.start} if asked.start else {}
        settings |= {"trainer.bookmark": asked.bookmark} if asked.bookmark else {}
        command = [
            sys.executable, "-m", "rollout_train.cli", "train", profile["path"], asked.catalog,
            "--directory", str(directory), "--name", asked.name, "--groups", str(asked.groups),
            "--groups-per-step", str(asked.groups_per_step), "--seed", str(asked.seed),
            *(argument for key, value in settings.items() if value is not None
              for argument in ("--set", f"{key}={json.dumps(value)}")),
        ]  # fmt: skip
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
        await self.launches.note(launch.id, state=RUNNING, directory=str(directory), pid=process.pid)
        watching = asyncio.create_task(self._watch(launch.id, process, directory))
        self._watching.add(watching)
        watching.add_done_callback(self._watching.discard)

    async def _watch(self, id: str, process: asyncio.subprocess.Process, directory: Path) -> None:
        code = await process.wait()
        self._playing.pop(id, None)
        now = next((each for each in await self.launches.all() if each.id == id), None)
        if now is not None and now.state == STOPPING:
            await self.launches.note(id, state=STOPPED, detail=f"stopped (exit {code})")
        elif code == 0:
            await self.launches.note(id, state=ENDED, detail="ended")
        else:
            tail = await asyncio.to_thread(_tail, directory / OUTPUT)
            await self.launches.note(id, state=FAILED, detail=f"exit {code}: {tail}")

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
            "catalogs": list(self.catalogs),
            "at_once": self.at_once,
            "playing": len(self._playing),
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
