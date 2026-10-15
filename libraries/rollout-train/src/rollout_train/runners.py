"""Episode runners as the cluster config builds them: the pools a runner leases its episodes' sandboxes from
(`pools_of`) and the gateway it serves on its machine for the harnesses it starts (`listening`, `serve`), which a run's
driver builds its runner with (`rollout_train.jobs`); and a runner of its own, which plays a run's episodes beside its
driver's runner, on any machine that reaches the cluster's stores and the run's channels.

    rollout runner --cluster --run RUN          (or python -m rollout_train.runners --cluster --run RUN)

A runner of its own (`runner_of`, `played`) opens the cluster's stores, samples through a gateway in its own process
over the channels each run's start names (`rollout_train.gateway.ChannelDirectory.of`), served on `127.0.0.1` for the
harnesses it starts, and claims, plays and records episodes under a name of its own (`runner/HOST/PID` unless named;
runners' names fence each other, so each is unique), beating while it serves. Its sandboxes come from pools made as
the run's driver makes them, named apart from the driver's (`KIND@RUN/NAME`), so that neither's keeper takes the
other's fence or sweeps the other's leases.

**Which runs.** The runs it is given (`--run`, each by its name or id) and their evals', as their drivers name them
(`Played`); never every run of the ledger. A pool made here is a run's (its pods, its directory, its name), and a
runner holds one pool of each kind, so two runs that each need a pool of their own of one kind need a runner each; and
a run's channels, environment and blob store are the run's. At start it says, for each run it is given, whether it
plays it, and why not: no run of that name; no start that records its settings yet (its channels are built from them);
a channel its driver alone reaches (`unreachable`: engine hosts in the driver's Ray cluster, Tinker's engines in the
driver's process), or a hosted API's while the run has a `limits.spend` (its driver's gateway counts what the run
spends from the turns it records itself); an environment that does not load here; a pool of its own of a kind
another run played here has one of, or a blob store other than theirs. It plays none of those, and nothing where it
can play none.

**How much.** A training run keeps `episodes_at_once` episodes' work asked for, in whole groups, and its driver's
runner has as many places: a runner of its own plays what that one leaves, the rest of the groups asked for and what
the driver's runner has no room for in its pools or memory. A run meant for more runners is asked for with a larger
`episodes_at_once` (fixed once the run starts). An eval or a check asks for all its episodes at once.

It serves until it is interrupted, terminated or hung up: what it plays then is cut short, noted, and played again by
any runner with room. Its runner, its gateway's server or a pool's keeper ending with an error ends it, logged, with
status 1.
"""

import argparse
import asyncio
import contextlib
import logging
import os
import re
import secrets
import socket
import sys
from collections.abc import Callable, Collection, Coroutine, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from rollout.harness.remote import RemotePool
from rollout.harness.sandboxes import MemoryLeases, Pool, PoolBinding, SandboxPool
from rollout.local import LocalRunner
from rollout.names import named
from rollout_train.cluster import Cluster, SandboxesSection
from rollout_train.gateway import ChannelDirectory, Gateway, GatewayEndpoints, Keyring, TurnStore
from rollout_train.gateway.directory import started_settings
from rollout_train.inference.api import ApiChannel
from rollout_train.ledger import Ledger
from rollout_train.machine import measured
from rollout_train.pods.leasing import pods_store, serving
from rollout_train.presence import presence_of
from rollout_train.rollouts.scheduler import EpisodeRunner
from rollout_train.run_settings import RunSettings
from rollout_train.sandboxes import ClusterPools, admits, keep, leases_of
from rollout_train.stores import Stores

__all__ = [
    "RUNNER",
    "Considered",
    "Played",
    "background",
    "considered",
    "listening",
    "played",
    "pools_of",
    "runner_directory",
    "runner_of",
    "serve",
    "unreachable",
    "until_stopped",
]

logger = logging.getLogger(__name__)

RUNNER = "runner"
"""What a runner of its own is named after, unless named: `runner/HOST/PID`."""
EPISODES = ("train", "eval", "check")
"""The kinds of run that play episodes."""


def background(
    stack: contextlib.AsyncExitStack,
    work: Coroutine[Any, Any, None],
    failed: Callable[[BaseException], None] | None = None,
) -> None:
    """Run `work` until the stack is closed; where it ends with an error before then, tell `failed` (`Run.fail`)."""
    task = asyncio.ensure_future(work)

    def ended(done: "asyncio.Future[None]") -> None:
        if failed is not None and not done.cancelled() and (error := done.exception()) is not None:
            failed(error)

    task.add_done_callback(ended)

    async def stop() -> None:
        task.remove_done_callback(ended)  # (what it raises as it is stopped fails nothing)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    stack.push_async_callback(stop)


def listening() -> socket.socket:
    """A socket bound to a free port on this node: bound before the gateway serves on it, so that no other process
    takes the port in between."""
    bound = socket.socket()
    bound.bind(("127.0.0.1", 0))
    return bound


async def serve(gateway: Gateway, bound: socket.socket) -> None:
    """A runner's gateway over HTTP on this node, for harnesses, on a socket bound for it (`listening`)."""
    import uvicorn

    from rollout_train.gateway import create_app

    server = uvicorn.Server(uvicorn.Config(create_app(gateway), log_level="warning", lifespan="off"))
    try:
        await server.serve(sockets=[bound])
    finally:  # (cancelled, it stops listening as it does when it shuts down)
        for each in getattr(server, "servers", ()):
            each.close()
        bound.close()


def pools_of(
    cluster: Cluster,
    ledger: Ledger,
    run: str,
    kinds: Collection[str],
    directory: Path,
    stack: contextlib.AsyncExitStack,
    failed: Callable[[BaseException], None],
    *,
    held: Collection[str] = (),
    apart: str | None = None,
    unserved: Callable[[str, Pool], Coroutine[Any, Any, None]] | None = None,
) -> tuple[dict[str, Pool], dict[str, PoolBinding]]:
    """The pools a runner of `run` leases the sandboxes of `kinds` from (those its environment's programs declare), by
    kind, and how the run binds each kind, from the cluster's `[sandboxes]`: one served from the run's pods (`on_pods`,
    where it holds a pod of a provider in `held` that serves the kind) is reached on the pods that serve it, the pool at
    the section's `url` behind them for when they are full (where several serve it, those as one); one several pools
    serve elsewhere (`pools`) is reached on each, through one pool here; one served elsewhere (`url`) is bound to its
    URL, and is no pool here; any other is made here (its provider made with `directory`, `size` and its settings), its
    leases kept beside the ledger.

    A pool here is named `KIND@RUN` (`KIND@RUN/APART` with `apart`: a runner of its own names its pools apart from the
    run's driver's, so that neither's keeper takes the other's fence or sweeps the other's leases), keeps where its
    leases are in `directory`, is swept by a keeper of its own, and is closed with `stack` (after a runner registered
    later: its runs release theirs first). A keeper that ends with an error tells `failed`. `unserved` is given each
    kind its pods serve with no pool of the cluster's behind them, with its pool, to watch (a run's driver ends the run
    once no pod of its has served the kind for long)."""
    from rollout_train.pods.pools import FilePlacements, PodPools
    from rollout_train.pods.routing import LeasedPools

    beats = presence_of(ledger)
    named_as = f"{run}/{apart}" if apart else run

    def spread(kind: str, section: SandboxesSection, name: str) -> Pool:
        """The pools that serve `kind` elsewhere (the section's `urls`), as one pool named `name` (`PodPools` over
        `ClusterPools`): a new lease on the one with the most room whose keeper beats, each lease's calls on the one
        that holds it, where each is kept in `directory`."""
        discover = ClusterPools(section.urls, beats)
        stack.push_async_callback(discover.aclose)
        placements = FilePlacements(directory / "pools" / f"{kind}.json")
        pool = PodPools(kind, discover, name=name, placements=placements, admits=admits(ledger, beats))
        stack.push_async_callback(pool.close)  # (after the runner: its runs release theirs first)
        background(stack, keep(pool, ledger, beats), failed)
        return pool

    def on_pods(kind: str, section: SandboxesSection, name: str) -> Pool:
        """The pools of `kind` the run's pods serve, as one named `name`, the cluster's behind them."""
        providers = serving(cluster, kind)
        auth = cluster.inference[providers[0]].auth
        discover = LeasedPools(ledger, run, kind, providers, auth, cluster.tls)
        stack.push_async_callback(discover.aclose)
        fallback: Pool | None = None
        if section.pools > 1:  # (named apart from the pool of its pods: each keeper takes its own pool's fence)
            fallback = spread(kind, section, f"{name}/cluster")
        elif section.url is not None:
            fallback = RemotePool(section.url)
            stack.push_async_callback(fallback.aclose)
        placements = FilePlacements(directory / "pods" / f"{kind}.json")
        pool = PodPools(
            kind, discover, name=name, placements=placements, admits=admits(ledger, beats), fallback=fallback
        )
        stack.push_async_callback(pool.close)  # (after the runner: its runs release theirs first)
        background(stack, keep(pool, ledger, beats), failed)
        if fallback is None and unserved is not None:  # (no pool of the cluster's to fall back on)
            background(stack, unserved(kind, pool), failed)
        return pool

    pools: dict[str, Pool] = {}
    bindings: dict[str, PoolBinding] = {}
    for kind in sorted(kinds):
        section = cluster.sandboxes.get(kind)
        if section is None:
            continue  # (validation refused a run whose environment needs it)
        if section.on_pods is not None and any(each in held for each in serving(cluster, kind)):
            pools[kind] = on_pods(kind, section, f"{kind}@{named_as}")
        elif section.pools > 1:
            pools[kind] = spread(kind, section, f"{kind}@{named_as}")
        elif section.url is not None:
            bindings[kind] = PoolBinding(url=section.url)
            continue
        else:
            assert section.provider is not None
            provider = named(section.provider)(directory, size=section.size, **dict(section.settings))
            made = SandboxPool(
                provider, name=f"{kind}@{named_as}", leases=leases_of(ledger) or MemoryLeases(),
                admits=admits(ledger, beats),
            )  # fmt: skip
            stack.push_async_callback(made.close)  # (after the runner: its runs release theirs first)
            background(stack, keep(made, ledger, beats), failed)
            pools[kind] = made
        bindings[kind] = PoolBinding(local=kind)
    return pools, bindings


def unreachable(cluster: Cluster, settings: RunSettings) -> list[str]:
    """Why a runner of its own cannot sample a run's channels, one reason for each channel it cannot (none: it can): a
    channel on a provider this cluster config lacks; on a `vllm` provider, whose engines are engine hosts in the run's
    driver's Ray cluster; on Tinker, whose engines are in its driver's process; on a hosted API while the run has a
    `limits.spend`, which its driver's gateway counts from the turns it records itself. A channel no slot samples (no
    provider, model or renderer) needs nothing."""
    why: list[str] = []
    for channel in settings.channels:
        providers = settings.providers(channel)
        if missing := [each for each in providers if each not in cluster.inference]:
            why.append(f"its channel {channel} is on {', '.join(missing)}, which this cluster config lacks")
            continue
        kinds = {cluster.inference[each].kind for each in providers}
        model, renderer = settings.get(f"channels.{channel}.model"), settings.get(f"channels.{channel}.renderer")
        if not providers or model is None or (renderer is None and "api" not in kinds):
            continue  # (a channel no slot samples)
        if "vllm" in kinds:
            why.append(f"its channel {channel} is served by engine hosts in its driver's Ray cluster")
        elif "tinker" in kinds:
            why.append(f"its channel {channel} is sampled by Tinker's engines in its driver's process")
        elif "api" in kinds and settings["limits.spend"] is not None:
            why.append(
                f"its channel {channel} samples a hosted API, and its limits.spend is counted by its driver's "
                "gateway, from the turns that gateway records"
            )
    return why


class Played(Collection[str]):
    """The runs a runner of its own plays: those it is given, and their evals', as their drivers name them
    (`rollout_train.jobs.Run.eval_run`): a training run's scheduled evals (`RUN-eval-STEP`) and their parts
    (`RUN-eval-STEP-PART`), and an eval's parts (`RUN-PART`). It lists those it is given, and holds their evals' too."""

    def __init__(self, runs: Collection[str]) -> None:
        self.runs = sorted(runs)
        either = "|".join(re.escape(each) for each in self.runs)
        self._played = re.compile(rf"(?:{either})(?:-eval-\d+)?(?:-\d+)?")

    def __contains__(self, run: object) -> bool:
        return isinstance(run, str) and bool(self.runs) and self._played.fullmatch(run) is not None

    def __iter__(self) -> Iterator[str]:
        return iter(self.runs)

    def __len__(self) -> int:
        return len(self.runs)


@dataclass(frozen=True)
class Considered:
    """A run a runner of its own is given, as it finds it at start: its id, the settings its start records, the kinds
    of sandbox its environment declares, and why it cannot play it here (none: it can)."""

    run: str
    settings: RunSettings | None = None
    kinds: frozenset[str] = frozenset()
    why: tuple[str, ...] = ()


async def considered(cluster: Cluster, ledger: Ledger, who: str) -> Considered:
    """A run by its name or id, as a runner of its own finds it: its settings as its newest start records them, its
    channels each one this process reaches (`unreachable`), and its environment imported here as its driver imports it
    (a published one from its version), for the kinds of sandbox it declares."""
    from rollout_train.launching import declared
    from rollout_train.published import loaded
    from rollout_train.registry import registry_of, run_id

    registry = registry_of(ledger)
    try:
        run = await run_id(registry, who) if registry is not None else who
    except KeyError:
        return Considered(who, why=(f"there is no run {who}",))
    settings = await started_settings(ledger, run)
    if settings is None:
        return Considered(run, why=("no start of it records its settings yet, which its channels are built from",))
    if settings.kind not in EPISODES:
        return Considered(run, settings, why=(f"a {settings.kind} run plays no episodes",))
    if why := unreachable(cluster, settings):
        return Considered(run, settings, why=tuple(why))
    environment = settings["environment"]
    if not isinstance(environment, str):
        return Considered(run, settings)
    try:
        found, _ = await loaded(environment, ledger)
        kinds, _ = await asyncio.to_thread(declared, found)
    except Exception as error:  # (its code is not here: none of its episodes can be played here)
        return Considered(run, settings, why=(f"its environment {environment} does not load here: {error}",))
    return Considered(run, settings, kinds)


def runner_directory(cluster: Cluster, name: str) -> Path:
    """Where a runner of its own keeps its files on its machine (`[scratch]/runners/NAME`): for each run it plays, what
    the providers of its pools keep, and where their leases are."""
    return Path(cluster.scratch).expanduser() / "runners" / name


async def _held(ledger: Ledger, run: str) -> set[str]:
    """The providers of the pods a run holds now, as their leases beside the ledger say."""
    from rollout_train.pods.leases import IDLE, pod_leases_of

    leases = pod_leases_of(ledger)
    if leases is None:
        return set()
    return {each.provider for each in await leases.all() if each.run == run and each.state != IDLE}


async def runner_of(
    cluster: Cluster,
    stores: Stores,
    runs: Sequence[str],
    stack: contextlib.AsyncExitStack,
    failed: Callable[[BaseException], None],
    *,
    name: str | None = None,
    places: int | None = None,
    directory: ChannelDirectory | None = None,
    told: Callable[[str], None] = print,
) -> EpisodeRunner:
    """A runner of its own, serving in the background until `stack` is closed: it plays those of `runs` (each by its
    name or id, `considered`) it can, and their evals' (`Played`), as `name` (by default `runner/HOST/PID`), at most
    `places` episodes at once (by default the most `episodes_at_once` of those runs). It samples through a gateway in
    this process over `directory` (by default `ChannelDirectory.of` the cluster), served on this machine for
    harnesses, records in the blob store the runs' blobs go to, and leases sandboxes from pools of its own
    (`pools_of`, named apart from the runs' drivers', kept under `[scratch]/runners/NAME`). `told` is given a line for
    each of `runs`: whether it plays it, and why not. What it runs beside (its runner, its gateway's server, a pool's
    keeper) ending with an error tells `failed`. Raises `ValueError` where it can play none of `runs`."""
    name = name or f"{RUNNER}/{socket.gethostname()}/{os.getpid()}"
    ledger = stores.ledger
    root = runner_directory(cluster, name)
    pools: dict[str, Pool] = {}
    whose: dict[str, str] = {}
    """The run each pool here is made for, by kind."""
    writes: dict[str, str | None] = {}
    """The blob store each run played here writes to (`[stores.NAME]`; none: the default)."""
    played: list[Considered] = []
    for who in runs:
        found = await considered(cluster, ledger, who)
        its = contextlib.AsyncExitStack()
        made: dict[str, Pool] = {}
        bindings: dict[str, PoolBinding] = {}
        why = list(found.why)
        if not why and found.settings is not None:
            store = pods_store(found.settings, cluster)
            if others := sorted(each for each, written in writes.items() if written != store):
                why.append(f"its blobs go to another store than those of {', '.join(others)}: a runner for each")
        if not why:
            await asyncio.to_thread((root / found.run).mkdir, parents=True, exist_ok=True)
            try:
                made, bindings = pools_of(cluster, ledger, found.run, found.kinds, root / found.run, its, failed,
                                          held=await _held(ledger, found.run), apart=name)  # fmt: skip
            except Exception as error:  # (a provider that cannot be made here)
                why.append(f"its pools cannot be made here: {error}")
            if shared := sorted(set(made) & set(pools)):
                others = ", ".join(sorted({whose[kind] for kind in shared}))
                why.append(f"its {', '.join(shared)} sandboxes come from a pool of its own, and so do those of "
                           f"{others}: a runner for each")  # fmt: skip
        if why:
            await its.aclose()
            told(f"{who}: not played here: {'; '.join(why)}")
            continue
        assert found.settings is not None
        stack.push_async_callback(its.aclose)
        pools |= made
        whose |= dict.fromkeys(made, who)
        writes[who] = pods_store(found.settings, cluster)
        played.append(found)
        told(f"{who}: played here, as {found.run}{_sandboxes(made, bindings)}")
        if found.settings.kind == "train":
            at_once = _at_once(found)
            told(f"{who}: its loop keeps {at_once} episodes' work asked for (episodes_at_once, fixed once it started), "
                 f"and its driver's runner has {at_once} places: this runner plays what that one leaves (a run meant "
                 "for more runners is asked for with a larger episodes_at_once)")  # fmt: skip
    if not played:
        raise ValueError(f"it can play none of {', '.join(runs)}")
    store = next(iter(writes.values()))
    blobs = stores.writing_to(cluster, store).blobs if store is not None else stores.blobs
    channels = directory or ChannelDirectory.of(cluster, ledger)
    stack.callback(channels.close)
    keyring = Keyring.parse([("run", secrets.token_urlsafe(32))])  # (keys only this process mints and takes)
    gateway = Gateway(TurnStore(ledger, blobs), keyring, directory=channels)
    bound = listening()
    recorder = GatewayEndpoints.of(gateway, f"http://127.0.0.1:{bound.getsockname()[1]}")
    background(stack, serve(gateway, bound), failed)
    runner = LocalRunner(gateway=recorder, tool_sets={}, pools=pools, blobs=blobs)
    ids = [each.run for each in played]
    episodes = EpisodeRunner(
        name, ledger, runner, recorder, blobs, places=places or max(_at_once(each) for each in played), pools=pools,
        runs=Played(ids), guard=_guard(cluster.guards.runs_gib), presence=presence_of(ledger),
        about=lambda: _about(gateway, ids, root),
    )  # fmt: skip
    await episodes.prepare()
    await runner.launch()
    stack.push_async_callback(runner.close)
    background(stack, episodes.serve(), failed)
    return episodes


def _at_once(found: Considered) -> int:
    """How many episodes' work a run keeps asked for (`episodes_at_once`)."""
    return int(cast(int, found.settings["episodes_at_once"])) if found.settings is not None else 1


def _sandboxes(made: dict[str, Pool], bound: dict[str, PoolBinding]) -> str:
    """What a run played here leases its sandboxes from, said."""
    said = [f"{kind} from {getattr(pool, 'name', kind)}" for kind, pool in sorted(made.items())]
    said += [f"{kind} from {binding.url}" for kind, binding in sorted(bound.items()) if binding.url is not None]
    return f"; its sandboxes: {', '.join(said)}" if said else ""


def _guard(gib: float) -> Callable[[], None] | None:
    """What holds claims back while this machine has less than `gib` of memory free (`[guards] runs_gib`)."""
    from rollout_train.jobs import require_memory

    return (lambda: require_memory(gib, "to run more episodes")) if gib else None


def _about(gateway: Gateway, runs: Sequence[str], directory: Path) -> dict[str, JsonValue]:
    """What a runner of its own says in each beat (called in a thread: it measures): its host, its directory and its
    machine, the runs it plays (`runs`: never `run`, by which a run's driver's runner says its run is alive), and each
    channel its gateway samples (`RUN/NAME`), with what passed through it since the beat before."""
    channels: list[JsonValue] = []
    built = gateway.directory.channels() if gateway.directory is not None else {}
    for name, channel in built.items():
        if isinstance(channel, ApiChannel):
            channels.append({"channel": name, "provider": channel.provider.name, "model": channel.model,
                             **channel.take()})  # fmt: skip
        else:
            servers: list[JsonValue] = list(channel.servers())
            channels.append({"channel": name, **channel.take(), "servers": servers})
    return {
        "host": socket.gethostname(), "directory": str(directory), "machine": measured(directory),
        "runs": list[JsonValue](runs), "channels": channels,
    }  # fmt: skip


async def played(
    cluster: Cluster,
    runs: Sequence[str],
    *,
    name: str | None = None,
    places: int | None = None,
    stores: Stores | None = None,
    directory: ChannelDirectory | None = None,
) -> int:
    """Play `runs` with a runner of its own (`runner_of`, over `stores` and `directory`: by default the cluster's) until
    cancelled, which closes it: what it plays then is cut short, noted, and played again. 1 once it, its gateway's
    server or a pool's keeper ends with an error, which is logged. Raises `ValueError` where it can play none of
    `runs`."""
    stores = stores or Stores.open(cluster)
    failing: asyncio.Future[BaseException] = asyncio.get_running_loop().create_future()

    def fail(error: BaseException) -> None:
        if not failing.done():
            failing.set_result(error)

    async with contextlib.AsyncExitStack() as stack:
        runner = await runner_of(cluster, stores, runs, stack, fail, name=name, places=places, directory=directory)
        print(f"{runner.name} plays at most {runner.places} episodes at once", flush=True)
        error = await failing
    logger.error("the runner %s stopped: %s", runner.name, error, exc_info=error)
    return 1


def until_stopped(cluster: Cluster, given: argparse.Namespace) -> int:
    """Play what `given` says (`rollout_train.cli.runner_options`) until interrupted, terminated or hung up; the exit
    status. Exits saying why where it can play none of the runs."""
    from rollout_train.cli import until_signalled

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    work = played(cluster, given.run, name=given.name, places=given.places)
    try:
        return asyncio.run(until_signalled(work))
    except ValueError as error:
        raise SystemExit(str(error)) from None


def main(arguments: Sequence[str] | None = None) -> None:
    """`python -m rollout_train.runners --cluster --run RUN`: a runner of its own."""
    from rollout_train.cli import runner_options
    from rollout_train.cluster import ClusterError, located

    parser = argparse.ArgumentParser(
        prog="python -m rollout_train.runners", description="A runner of its own: it plays runs' episodes."
    )
    runner_options(parser)
    given = parser.parse_args(arguments)
    try:
        cluster = located(given.cluster or None)
    except ClusterError as error:
        raise SystemExit(str(error)) from None
    sys.exit(until_stopped(cluster, given))


if __name__ == "__main__":
    main()
