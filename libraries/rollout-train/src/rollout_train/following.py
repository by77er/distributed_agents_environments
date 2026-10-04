"""Following: keeping a process's channels serving what a run says they should, wherever the process runs.

The training loop writes down what each of its run's channels should serve (`rollout_train.serving`). A `Follower`
reads it every few seconds and, when a channel of its process serves something older, reads the checkpoint's files from
the blob store (hard links, where the store is files on the same disk: `Checkpoints.files`) and publishes them on the
channel's engines. An adapter over a full checkpoint has that checkpoint's weights loaded first. The process may be a
GPU box serving engines and nothing else (`rollout engines`), or a runner whose engines are clients of a service, such
as a sampler that is switched to the checkpoint named.

It beats like a runner (`rollout_train.presence`): its host, the run it follows, its machine, and for each channel what
it serves and how fast, with each replica's address where its engines are served over HTTP
(`rollout_train.inference.remote.serve_engines`). Runners find a run's replicas from these beats.

A replica behind what it should serve keeps serving what it has while it loads the newest: every sample is stamped
with the version that made it, and a runner gives it no turn once it is more than `max_lag` checkpoints behind
(`rollout_train.inference.remote.RemoteChannel`).
"""

import asyncio
import contextlib
import shutil
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from rollout_train.checkpoints import Checkpoints
from rollout_train.inference import Channel
from rollout_train.inference.remote import ENGINES, replica_id
from rollout_train.presence import Presence
from rollout_train.serving import Serving, wanted
from rollout_train.trainer import WEIGHTS


class Follower:
    """Keeps `channels` serving what `run` says each should (by the channel's name within the run), checking every
    `every` seconds, with each checkpoint's files under `directory` while they are served. With `presence`, it beats as
    `name` every `beating` seconds, and at once when a channel serves something new: what `about` says of the machine,
    and what each channel serves; where `serve_engines` serves them, under the server name `server`, at `address` as
    others reach it, each replica's id and that address."""

    def __init__(
        self,
        name: str,
        checkpoints: Checkpoints,
        run: str,
        channels: Mapping[str, Channel],
        directory: Path,
        *,
        address: str | None = None,
        server: str | None = None,
        presence: Presence | None = None,
        about: Callable[[], Mapping[str, JsonValue]] | None = None,
        every: float = 2.0,
        beating: float = 15.0,
    ) -> None:
        self.name = name
        self.checkpoints = checkpoints
        self.run = run
        self.channels = dict(channels)
        self.directory = directory
        self.address = address
        self.server = server or name
        self.presence = presence
        self.about = about
        self.every = every
        self.beating = beating
        self.errors: dict[str, str] = {}
        """Why a channel could not be given what it should serve, by channel, until it is."""

    async def serve(self) -> None:
        """Follow until cancelled."""
        beaten = -float("inf")
        while True:
            changed = await self.follow()
            if self.presence is not None and (changed or time.monotonic() - beaten >= self.beating):
                with contextlib.suppress(Exception):  # (a beat missed is said at the next)
                    await self.beat()
                beaten = time.monotonic()
            await asyncio.sleep(self.every)

    async def follow(self) -> bool:
        """Give every channel what the run says it should serve, if it serves something older; whether any changed."""
        changed = False
        for name, channel in self.channels.items():
            try:
                said = await wanted(self.checkpoints.ledger, self.run, name)
                if said is not None and await self._served(channel, said):
                    changed = True
                self.errors.pop(name, None)
            except Exception as error:  # (tried again at the next look; the beat says why)
                self.errors[name] = f"{type(error).__name__}: {error}"
        return changed

    async def beat(self) -> None:
        assert self.presence is not None
        said: Mapping[str, JsonValue] = await asyncio.to_thread(self.about) if self.about is not None else {}
        await self.presence.beat(self.name, {**said, "kind": ENGINES, "follows": self.run, "channels": self.served()})

    def served(self) -> list[JsonValue]:
        """What each channel serves, how fast since the last call, and each of its replicas with its id and address."""
        listed: list[JsonValue] = []
        for name, channel in self.channels.items():
            replicas: list[JsonValue] = [
                {
                    "replica": replica_id(self.server, name, index),
                    "address": self.address,
                    "adapter": channel.adapter,
                    "serving": channel.serving,
                    "version": channel.version,
                }
                for index in range(len(channel.engines))
            ]
            entry: dict[str, JsonValue] = {"channel": name, "adapter": channel.serving, "version": channel.version}
            entry |= {**channel.take(), "replicas": replicas}
            if name in self.errors:
                entry["error"] = self.errors[name]
            listed.append(entry)
        return listed

    async def _served(self, channel: Channel, said: Serving) -> bool:
        """Publish what `said` names on a channel serving something older; whether it did."""
        if said.checkpoint is None or said.files is None or said.checkpoint == channel.serving:
            return False
        if said.depth < channel.version:
            return False  # (a channel does not go back)
        if said.over is not None and channel.held != said.over:  # the full weights the adapter is trained over first
            over = await self.checkpoints.checkpoint(said.over)
            if over.weights is None:
                raise ValueError(f"{over.id} was released: the adapter {said.checkpoint} cannot be served over it")
            held = await self.checkpoints.files(over.weights, self.directory / over.id / WEIGHTS)
            await channel.publish(over.id, str(held), over.depth, full=True)
        place = self.directory / said.checkpoint / ("resharded" if said.layout else WEIGHTS)
        files = await self.checkpoints.files(said.files, place)
        await channel.publish(said.checkpoint, str(files), said.depth, full=said.kind == "full")
        keep = {channel.held, *_loaded(channel)}
        for old in await asyncio.to_thread(lambda: list(self.directory.iterdir())):
            if old.name not in keep:
                await asyncio.to_thread(shutil.rmtree, old, ignore_errors=True)
        return True


def _loaded(channel: Channel) -> list[str]:
    loaded: Any = channel.state()["loaded"]
    return list(loaded)
