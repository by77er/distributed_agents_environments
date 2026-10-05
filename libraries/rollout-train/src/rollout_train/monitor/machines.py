"""The machines that run things, and what each holds and how full it is: the monitor's Machines page.

Every process that beats (`rollout_train.presence`) says what it is in its beat's `kind`: an episode runner says
nothing (its places, how many it plays and how full its pools are), a run's driver that waits for what it asked Ray
for `run` (what it waits for: its runner beats under the same name once it plays), an engine host `engines` (the run it
follows, and what each channel's engines serve), a pool served from a machine of its own `pool` (how full it is) and a
gateway `gateway` (the channels it samples). `machines` puts those beside what the ledger says of them: the episodes
each runner's claims hold, each pool's leases in the `sandboxes` table, and what each followed run's channels should
serve. A process is alive while its newest beat is younger than `STALE` seconds by the store's clock
(`rollout_train.presence.alive`); a pool within a runner is alive while its runner is.
"""

from collections.abc import Collection, Mapping
from typing import Any, cast

from pydantic import JsonValue

from rollout.harness.sandboxes import Lease
from rollout_train.gateway.beats import GATEWAY
from rollout_train.inference.remote import ENGINES
from rollout_train.presence import Beat, alive
from rollout_train.rollouts.scheduler import Claims
from rollout_train.sandboxes import POOL

RUNNER = "runner"
"""The kind of an episode runner, whose beat says none."""
WAITING = "run"
"""The kind of a run's driver that waits for what it asked Ray for."""
THROUGHPUT = ("tokens_per_second", "mean_concurrency")
"""What the page shows of what a channel's beat says passed through it since the beat before."""


def machines(
    beats: list[Beat],
    *,
    now: float,
    claims: Mapping[str, Claims] | None = None,
    fences: Mapping[str, int] | None = None,
    leases: Collection[Lease] = (),
    serving: Mapping[str, Mapping[str, JsonValue]] | None = None,
) -> dict[str, Any]:
    """Every machine that beats and every role on it, as the beats and the ledger say: `claims` by run (with `fences`,
    which say whether each holds), the pools' `leases`, and each followed run's `serving` table.
    A beat's time (`at`) is given by `now`'s clock (the monitor's): `now` less its age by the store's clock."""
    by_name = {beat.runner: beat for beat in beats}
    playing = _playing(claims or {}, fences or {}, by_name)
    runners: list[dict[str, Any]] = []
    pools: dict[str, dict[str, Any]] = {}
    engines: list[dict[str, Any]] = []
    gateways: list[dict[str, Any]] = []
    for beat in beats:
        kind = kind_of(beat)
        said = beat.about
        shown = _shown(beat, now)
        if kind == RUNNER:
            places, played = _count(said.get("places")), _count(said.get("playing"))
            held: Any = said.get("pools") or {}
            runners.append(
                shown
                | {"run": said.get("run"), "places": places, "playing": played, "free": max(places - played, 0)}
                | {"claims": playing.get(beat.runner, []), "pools": [f"{name}@{beat.runner}" for name in held]}
                | {"channels": _channels(said.get("channels"))}
            )
            for name, capacity in dict(held).items():
                pools[f"{name}@{beat.runner}"] = _pool(shown, f"{name}@{beat.runner}", name, capacity, beat.runner)
        elif kind == WAITING:
            runners.append(
                shown
                | {"run": said.get("run"), "places": 0, "playing": 0, "free": 0, "claims": [], "pools": []}
                | {"channels": [], "waiting": said.get("waiting") or []}
            )
        elif kind == POOL:
            name = str(said.get("pool") or beat.runner)
            pools[name] = _pool(shown, name, str(said.get("sandboxes") or name), said, None)
        elif kind == ENGINES:
            follows = str(said.get("follows") or "")
            wanted = _wanted((serving or {}).get(follows, {}))
            served = [_served(each, wanted) for each in _list(said.get("channels"))]
            engines.append(shown | {"follows": follows or None, "channels": served})
        elif kind == GATEWAY:
            gateways.append(shown | {"listen": said.get("listen"), "channels": _channels(said.get("channels"))})
    for lease in leases:
        if lease.pool not in pools:  # (a pool that does not beat: a runner's that closed and kept its leases, say)
            pools[lease.pool] = {"name": lease.pool, "kind": lease.kind, "host": None, "runner": None, "alive": False}
            pools[lease.pool] |= {"at": None, "size": None, "leased": None, "free": None}
        pools[lease.pool].setdefault("leases", []).append(_lease(lease, claims or {}, fences or {}, by_name))
    listed = [pool | {"leases": sorted(pool.get("leases", []), key=_newest)} for pool in pools.values()]
    roles: dict[str, list[dict[str, Any]]] = {
        "runners": runners, "pools": listed, "engines": engines, "gateways": gateways,
    }  # fmt: skip
    for found in roles.values():
        found.sort(key=_order)
    return {"now": round(now, 1), "hosts": _hosts(beats, roles, now), **roles}


def kind_of(beat: Beat) -> str:
    """What a beat says its process is (`RUNNER` where it says nothing)."""
    return str(beat.about.get("kind") or RUNNER)


def host_of(beat: Beat) -> str:
    """The machine a beat is from: the host it names, else its name."""
    return str(beat.about.get("host") or beat.runner)


def _shown(beat: Beat, now: float) -> dict[str, Any]:
    return {"name": beat.runner, "host": host_of(beat), "alive": alive(beat), "at": round(now - beat.age)}


def _hosts(beats: list[Beat], roles: Mapping[str, list[dict[str, Any]]], now: float) -> list[dict[str, Any]]:
    """Each machine: alive while any of its processes is, when it last beat, its newest measurements and their history
    (from the newest beat that measures it), and its roles (`{"kind", "name", "alive"}`)."""
    found: dict[str, dict[str, Any]] = {}
    for beat in sorted(beats, key=lambda each: each.age):  # (newest first: its measurements are the machine's)
        host = found.setdefault(host_of(beat), {"host": host_of(beat), "alive": False, "at": None, "machine": None})
        host["alive"] = host["alive"] or alive(beat)
        host["at"] = host["at"] if host["at"] is not None else round(now - beat.age)
        if host["machine"] is None and beat.about.get("machine"):
            host["machine"] = beat.about["machine"]
            measured = [point for point in beat.history if point.get("machine")]
            host["history"] = [{"at": point["at"], "machine": point["machine"]} for point in measured]
    for host in found.values():
        host.setdefault("history", [])
        host["roles"] = []
    for kind, listed in roles.items():
        for role in listed:
            if role["host"] in found:
                found[role["host"]]["roles"].append({"kind": kind, "name": role["name"], "alive": role["alive"]})
    return sorted(found.values(), key=lambda host: (not host["alive"], host["host"]))


def _playing(
    claims: Mapping[str, Claims], fences: Mapping[str, int], beats: Mapping[str, Beat]
) -> dict[str, list[dict[str, Any]]]:
    """The episodes each runner's claims hold, by runner: each one's run, group, episode, attempt, when it was claimed
    and the run id it plays under."""
    found: dict[str, list[dict[str, Any]]] = {}
    for run, each in claims.items():
        for key in sorted(each.holding(fences, beats)):
            made = each.made[key]
            group, episode, attempt = key.split("/")
            found.setdefault(str(made["runner"]), []).append(
                {"run": run, "group": int(group), "episode": int(episode), "attempt": int(attempt)}
                | {"at": made.get("at"), "run_id": made.get("run_id")}
            )
    for listed in found.values():
        listed.sort(key=lambda claim: (claim["run"], claim["group"], claim["episode"]))
    return found


def _pool(shown: Mapping[str, Any], name: str, kind: str, capacity: Any, runner: str | None) -> dict[str, Any]:
    size, leased = _count(capacity.get("size")), _count(capacity.get("leased"))
    return {**shown, "name": name, "kind": kind, "runner": runner, "size": size, "leased": leased} | {
        "free": max(size - leased, 0)
    }


def _lease(
    lease: Lease, claims: Mapping[str, Claims], fences: Mapping[str, int], beats: Mapping[str, Beat]
) -> dict[str, Any]:
    """A lease: its key, the sandbox's name, the run's episode it was acquired for (where its key names one) and its
    run id, whether that claim holds, when it was made, how long it may last, and whether its sandbox is lost."""
    shown: dict[str, Any] = {"key": lease.key, "kind": lease.kind, "at": lease.at, "seconds": lease.seconds}
    shown |= {"lost": lease.lost, "sandbox": lease.key.rsplit("/", 1)[-1], "run": None, "holds": None}
    parts = lease.key.split("/")
    if len(parts) >= 5 and all(part.isdigit() for part in parts[1:4]):
        run, group, episode, attempt = parts[0], *(int(part) for part in parts[1:4])
        shown |= {"run": run, "group": group, "episode": episode, "attempt": attempt}
        found = claims.get(run)
        claim = f"{group}/{episode}/{attempt}"
        if found is not None and claim in found.made:
            shown["run_id"] = found.made[claim].get("run_id")
            shown["holds"] = f"{group}/{episode}" not in found.done and found.holds(claim, fences, beats)
    return shown


def _wanted(table: Mapping[str, JsonValue]) -> dict[str, dict[str, Any]]:
    """What each channel of a run should serve, by channel: the checkpoint of its record of the greatest depth."""
    found: dict[str, dict[str, Any]] = {}
    for record in table.values():
        if isinstance(record, dict) and isinstance(record.get("channel"), str):
            name, depth = str(record["channel"]), _count(record.get("depth"))
            if name not in found or depth >= found[name]["version"]:
                found[name] = {"checkpoint": record.get("checkpoint"), "version": depth}
    return found


def _served(channel: Mapping[str, Any], wanted: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """An engine host's channel: what it serves, what it should, how far behind that each engine is, how fast."""
    should = wanted.get(str(channel.get("channel")))
    target = should["version"] if should is not None else None

    def behind(version: Any) -> int | None:
        return max(target - version, 0) if target is not None and isinstance(version, int) else None

    engines = [
        {"address": each.get("address"), "serving": each.get("serving"), "version": each.get("version")}
        | {"behind": behind(each.get("version"))}
        for each in _list(channel.get("engines"))
    ]
    return {
        **_channel(channel),
        "wanted": should["checkpoint"] if should is not None else None,
        "behind": behind(channel.get("version")),
        "error": channel.get("error"),
        "engines": engines,
    }


def _channels(listed: Any) -> list[dict[str, Any]]:
    return [_channel(each) for each in _list(listed)]


def _channel(channel: Mapping[str, Any]) -> dict[str, Any]:
    """A channel as a beat says it: its name, what it serves, how fast, and (routed) its servers."""
    shown = {"channel": channel.get("channel"), "serving": channel.get("adapter"), "version": channel.get("version")}
    shown |= {key: channel.get(key) for key in THROUGHPUT}
    if "servers" in channel:
        shown["servers"] = channel["servers"]
    return shown


def _list(value: Any) -> list[Mapping[str, Any]]:
    listed = cast(list[Any], value) if isinstance(value, list) else []
    return [each for each in listed if isinstance(each, dict)]


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _order(role: Mapping[str, Any]) -> tuple[bool, str]:
    return (not role["alive"], str(role["name"]))


def _newest(lease: Mapping[str, Any]) -> tuple[bool, float]:
    return (bool(lease["lost"]), -float(lease["at"] or 0.0))
