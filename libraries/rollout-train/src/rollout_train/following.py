"""Following: keeping a process's engines serving what runs say their channels should, wherever the process runs.

The training loop writes down what each of its run's channels should serve (`rollout_train.serving`). A `Follower`
reads it every few seconds for each run's channel it serves and, when one serves something older, reads the
checkpoint's files from the blob store (hard links, where the store is files on the same disk: `Checkpoints.files`)
and publishes them on that channel's engines, named by the checkpoint's id. An adapter over a full checkpoint has that
checkpoint's weights loaded first. The process may be an engine host (`rollout_train.inference.hosts.EngineHost`, its
engines in its care), a process beside vLLM servers on their machine (over `RemoteEngine`), or a runner whose engines
are in its own process or are clients of a service, such as a sampler that is switched to the checkpoint named.

What it serves is a set of runs' channels, `(run, channel)`, which may change while it runs (`bindings`): runs bound
to a shared pool join and leave it without the follower starting again. Each is a `Channel` of its own over engines
that may be shared, so adapters of several runs sit side by side on one engine. Each channel keeps the adapters a turn
may still sample from loaded: the one served and those before it, as many as its run's serving record says (`max_lag
+ 1`), following the record as it changes. Full weights are loaded replica by replica: a follower that is replica `i`
of several loads a channel's new full checkpoint only once the replicas before it beat that they hold it (or have
stopped beating), so the others serve the checkpoint before meanwhile.

It beats like a runner (`rollout_train.presence`): its host, the runs it follows, its machine, its replica, and for
each channel what it serves and how fast, and each engine's address and every adapter it holds, with its run, channel,
checkpoint and depth. Where an engine can say what it holds (a vLLM server's `/v1/models`), the follower holds its
view to that before each look: an adapter the server lost (it started again) is loaded again, and one the follower
does not know of (loaded before the follower started again) is removed, unless a channel should serve it.
"""

import asyncio
import contextlib
import shutil
import time
from collections.abc import Awaitable, Callable, Collection, Mapping
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from rollout_train.checkpoints import Checkpoints
from rollout_train.inference import Channel
from rollout_train.inference.channel import MAX_LAG
from rollout_train.inference.remote import ENGINES
from rollout_train.presence import Presence, alive
from rollout_train.serving import Serving, qualified, wanted
from rollout_train.trainer import WEIGHTS

Binding = tuple[str, str]
"""A run's channel a follower serves: the run's id, and the channel's name within the run."""
BINDINGS = "bindings"
"""Where `Follower.errors` says why what it serves, or what its engines hold, could not be asked."""


class Follower:
    """Keeps runs' channels serving what each run says they should, checking every `every` seconds, with each
    checkpoint's files under `directory` while they are served.

    Given `run` and `channels`, it serves those channels of that one run (by their names within it). Given `bindings`
    (what it serves now, asked at each look) and `opened` (the channel for a binding new to it), it serves whatever
    `bindings` says, and a binding it no longer says has its adapters removed. `replica` is this follower's index among
    the replicas of its channels and how many there are. With `presence`, it beats as `name` every `beating` seconds,
    and at once when a channel serves something new: what `about` says of the machine, and what each channel serves."""

    def __init__(
        self,
        name: str,
        checkpoints: Checkpoints,
        run: str | None,
        channels: Mapping[str, Channel],
        directory: Path,
        *,
        bindings: Callable[[], Awaitable[Collection[Binding]]] | None = None,
        opened: Callable[[str, str], Channel] | None = None,
        replica: tuple[int, int] = (0, 1),
        presence: Presence | None = None,
        about: Callable[[], Mapping[str, JsonValue]] | None = None,
        every: float = 2.0,
        beating: float = 15.0,
    ) -> None:
        if bindings is not None and opened is None:
            raise ValueError("a follower told its bindings is told how to open a channel for each")
        self.name = name
        self.checkpoints = checkpoints
        self.run = run
        self.bound: dict[Binding, Channel] = {(run, each): channel for each, channel in channels.items() if run}
        self.bindings = bindings
        self.opened = opened
        self.replica = replica
        self.directory = directory
        self.presence = presence
        self.about = about
        self.every = every
        self.beating = beating
        self.errors: dict[str, str] = {}
        """Why a channel could not be given what it should serve, by `RUN/CHANNEL`, until it is (and under `BINDINGS`,
        why what it serves could not be asked)."""

    @property
    def channels(self) -> dict[str, Channel]:
        """The channels it serves, by `RUN/CHANNEL`."""
        return {qualified(run, name): channel for (run, name), channel in self.bound.items()}

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
        """Give every channel what its run says it should serve, if it serves something older; whether any changed."""
        try:
            await self._rebound()
            await self._reconciled()
            self.errors.pop(BINDINGS, None)
        except Exception as error:  # (it serves what it served, and looks again at the next look)
            self.errors[BINDINGS] = f"{type(error).__name__}: {error}"
        changed = False
        for (run, name), channel in list(self.bound.items()):
            key = qualified(run, name)
            try:
                said = await wanted(self.checkpoints.ledger, run, name)
                if said is not None:
                    await channel.keeping((MAX_LAG if said.max_lag is None else said.max_lag) + 1)
                    if await self._served(run, name, channel, said):
                        changed = True
                self.errors.pop(key, None)
            except Exception as error:  # (tried again at the next look; the beat says why)
                self.errors[key] = f"{type(error).__name__}: {error}"
        if changed:
            await self._tidied()
        return changed

    async def beat(self) -> None:
        assert self.presence is not None
        said: Mapping[str, JsonValue] = await asyncio.to_thread(self.about) if self.about is not None else {}
        runs = sorted({run for run, _ in self.bound})
        follows: JsonValue = self.run or (runs[0] if len(runs) == 1 else None)
        listed: list[JsonValue] = [*runs]
        about: dict[str, JsonValue] = {"kind": ENGINES, "follows": follows, "runs": listed, "replica": self.replica[0]}
        if BINDINGS in self.errors:
            about["error"] = self.errors[BINDINGS]
        await self.presence.beat(self.name, {**said, **about, "channels": self.served()})

    def served(self) -> list[JsonValue]:
        """What each channel serves, how fast since the last call, and each of its engines: its address (where it is a
        server elsewhere), what it serves, and every adapter it holds for any channel."""
        held = self._held()
        listed: list[JsonValue] = []
        for (run, name), channel in self.bound.items():
            engines: list[JsonValue] = [
                {
                    "address": getattr(engine, "address", None), "serving": channel.serving, "version": channel.version,
                    "adapters": held.get(id(engine), []),
                }
                for engine in channel.engines
            ]  # fmt: skip
            entry: dict[str, JsonValue] = {"run": run, "channel": name, "adapter": channel.serving}
            entry |= {"version": channel.version, **channel.take(), "engines": engines}
            if (key := qualified(run, name)) in self.errors:
                entry["error"] = self.errors[key]
            listed.append(entry)
        return listed

    def _held(self) -> dict[int, list[JsonValue]]:
        """Every adapter (and full checkpoint) each engine holds, by the engine's identity: its run, channel,
        checkpoint and depth."""
        held: dict[int, list[JsonValue]] = {}
        for (run, name), channel in self.bound.items():
            for engine in channel.engines:
                for checkpoint, depth, full in channel.adapters():
                    each: JsonValue = {"run": run, "channel": name, "checkpoint": checkpoint, "depth": depth}
                    held.setdefault(id(engine), []).append({**each, "full": True} if full else each)
        return held

    async def _rebound(self) -> None:
        """Serve what `bindings` says now: a channel for each binding new to it, and none for one it no longer says."""
        if self.bindings is None or self.opened is None:
            return
        now = set(await self.bindings())
        for gone in [each for each in self.bound if each not in now]:
            await self.bound.pop(gone).dropped()
            self.errors.pop(qualified(*gone), None)
        for new in sorted(now - set(self.bound)):
            self.bound[new] = self.opened(*new)

    async def _reconciled(self) -> None:
        """Hold what each channel takes to be loaded to what its engines say they hold, where they can say (a server's
        `models`): adapters an engine lost are loaded again, and adapters no channel knows of are removed, unless a
        channel should serve one now."""
        engines: dict[int, Any] = {}
        for channel in self.bound.values():
            for engine in channel.engines:
                if callable(getattr(engine, "models", None)):
                    engines[id(engine)] = engine
        for key, engine in engines.items():
            try:
                listing: dict[str, Any] = await engine.models()
            except Exception:  # (an engine that does not answer is looked at again at the next look)
                continue
            cards = {name: cast(dict[str, Any], card) for name, card in listing.items() if isinstance(card, dict)}
            has = {name for name, card in cards.items() if card.get("parent")}  # (an adapter names its model)
            sharing = [channel for channel in self.bound.values() if any(id(each) == key for each in channel.engines)]
            known: set[str] = set()
            for channel in sharing:
                channel.forget([name for name in channel.loaded if name not in has])
                known |= set(channel.loaded)
            wanted_now: set[str] = set()
            for run, name in [each for each, channel in self.bound.items() if channel in sharing]:
                said = await wanted(self.checkpoints.ledger, run, name)
                if said is not None and said.checkpoint is not None:
                    wanted_now.add(said.checkpoint)
            for stray in sorted(has - known - wanted_now):
                with contextlib.suppress(Exception):
                    await engine.remove_adapter(stray)

    async def _served(self, run: str, name: str, channel: Channel, said: Serving) -> bool:
        """Publish what `said` names on a channel serving something older; whether it did."""
        if said.checkpoint is None or said.files is None or said.checkpoint == channel.serving:
            return False
        if said.depth < channel.version:
            return False  # (a channel does not go back)
        full = said.kind == "full" or (said.over is not None and channel.held != said.over)
        if full and not await self._turn(run, name, said.depth):
            return False  # (a replica before this one is loading it: the next look tries again)
        if said.over is not None and channel.held != said.over:  # the full weights the adapter is trained over first
            over = await self.checkpoints.checkpoint(said.over)
            if over.weights is None:
                raise ValueError(f"{over.id} was released: the adapter {said.checkpoint} cannot be served over it")
            held = await self.checkpoints.files(over.weights, self.directory / over.id / WEIGHTS)
            await channel.publish(over.id, str(held), over.depth, full=True)
        place = self.directory / said.checkpoint / ("resharded" if said.layout else WEIGHTS)
        files = await self.checkpoints.files(said.files, place)
        await channel.publish(said.checkpoint, str(files), said.depth, full=said.kind == "full")
        return True

    async def _turn(self, run: str, name: str, depth: int) -> bool:
        """Whether this replica may load a channel's full checkpoint of `depth` now: every replica before it that still
        beats holds it already (replica by replica, the others serving the checkpoint before meanwhile)."""
        index, count = self.replica
        if index == 0 or count <= 1 or self.presence is None:
            return True
        for beat in await self.presence.beats():
            about = beat.about
            before = isinstance(replica := about.get("replica"), int) and replica < index
            if beat.runner == self.name or not alive(beat) or about.get("kind") != ENGINES or not before:
                continue
            listed: list[Any] = cast(list[Any], about.get("channels") or [])
            for entry in (cast(dict[str, Any], each) for each in listed if isinstance(each, dict)):
                if entry.get("run") == run and entry.get("channel") == name and int(entry.get("version") or 0) < depth:
                    return False
        return True

    async def _tidied(self) -> None:
        """Remove the files of checkpoints no channel holds any more."""
        keep = {held for channel in self.bound.values() for held in (channel.held, *channel.loaded) if held}
        for old in await asyncio.to_thread(lambda: list(self.directory.iterdir())):
            if old.name not in keep:
                await asyncio.to_thread(shutil.rmtree, old, ignore_errors=True)
