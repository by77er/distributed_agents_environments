"""`HttpLedger`: the ledger and the stores beside it, reached through the ledger service.

It is a `Ledger`, and has beside it what a database ledger has (`registry`, `launches`, `presence`,
`desired_settings`, `sandboxes`, `presets`, `environment_versions`), each a client of the service. Every role can use
it by its URL and a token: in a cluster config, `[ledger] url = "https://…"` with `token_env` or `token_file`; for a
pod, `ROLLOUT_LEDGER` naming `rollout_train.ledger_service:HttpLedger` with its `url` and `token_env`.

What the client does, as the service's guarantees ask (docs/research/ledger-guarantees.md#10-the-http-ledger-service):

- every operation is sent with a request id of its own, and sent again with the same id while the service does not
  answer or answers that it is unavailable (502, 503, 504), for `deadline` seconds at most (well under the
  heartbeats' `STALE`), each attempt for `attempt` seconds; then `LedgerUnreachable`. A timeout is never read as a
  refusal;
- `Fenced` is raised as itself, never retried; so are a name taken, a row missing (`KeyError`), a request refused
  (`ValueError`), a token that may not (`Forbidden`) and a compare-and-set that lost (`Conflict`);
- an append retried after a lost answer, which finds the key holding the very record it sent, wrote it.
"""

import asyncio
import json
import os
import time
import uuid
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, cast

import httpx
from pydantic import JsonValue

from rollout.harness.sandboxes import Lease
from rollout_train.launches import Asked, Launch
from rollout_train.ledger import Appended, Fence
from rollout_train.ledger_service.wire import ERRORS, decoded, encoded
from rollout_train.presence import Beat
from rollout_train.presets import Preset
from rollout_train.published import EnvironmentVersion
from rollout_train.registry import Bookmark, Entry, Named, SuiteName
from rollout_train.settings import Desired

__all__ = ["ATTEMPT", "DEADLINE", "HttpLedger", "LedgerUnreachable"]

DEADLINE = 30.0
"""Seconds an operation is tried for before it is given up (`LedgerUnreachable`)."""
ATTEMPT = 10.0
"""Seconds one attempt waits for an answer."""
RETRIED = (502, 503, 504)


class LedgerUnreachable(Exception):
    """The ledger service did not answer within the deadline: what was asked may or may not have been done."""


class HttpLedger:
    """The ledger at the service `url`, with the token read (at each request, never kept) from the environment
    variable `token_env` or the file `token_file`, or given as `token`."""

    def __init__(
        self,
        url: str,
        *,
        token_env: str | None = None,
        token_file: str | None = None,
        token: str | None = None,
        client: httpx.AsyncClient | None = None,
        deadline: float = DEADLINE,
        attempt: float = ATTEMPT,
    ) -> None:
        self.url = url.rstrip("/")
        self.token_env, self.token_file = token_env, token_file
        self._token = token
        self._http = client or httpx.AsyncClient(timeout=attempt)
        self._owned = client is None
        self.deadline = deadline
        self.attempt = attempt

    def __repr__(self) -> str:
        return f"HttpLedger({self.url!r})"

    def token(self) -> str | None:
        if self._token is not None:
            return self._token
        if self.token_env is not None:
            return os.environ.get(self.token_env) or None
        if self.token_file is not None:
            path = Path(self.token_file).expanduser()
            return (path.read_text().strip() or None) if path.exists() else None
        return None

    async def call(self, operation: str, args: Mapping[str, Any] | None = None) -> tuple[Any, bool]:
        """Do `operation` (`STORE/OPERATION`) with `args`: its result, and whether an attempt before the one answered
        may have been done (its answer lost)."""
        body = {"args": json.loads(json.dumps(args or {})), "request": uuid.uuid4().hex}
        token = self.token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        given_up = time.monotonic() + self.deadline
        retried = False
        while True:
            try:
                response = await self._http.post(f"{self.url}/v1/{operation}", json=body, headers=headers,
                                                 timeout=self.attempt)  # fmt: skip
            except httpx.TransportError as error:
                why = f"{type(error).__name__}"
            else:
                if response.status_code not in RETRIED:
                    return _answered(operation, response), retried
                why = f"{response.status_code}"
            if time.monotonic() >= given_up:
                raise LedgerUnreachable(f"the ledger service at {self.url} did not answer {operation} ({why})")
            retried = True
            await asyncio.sleep(min(1.0, max(0.0, given_up - time.monotonic())))

    async def result(self, operation: str, args: Mapping[str, Any] | None = None) -> Any:
        return (await self.call(operation, args))[0]

    async def take(self, scope: str) -> Fence:
        return Fence(scope, int(await self.result("ledger/take", {"scope": scope})))

    async def append(self, table: str, key: str, record: JsonValue, fence: Fence) -> bool:
        return (await self.append_returning(table, key, record, fence)).wrote

    async def append_returning(self, table: str, key: str, record: JsonValue, fence: Fence) -> Appended:
        sent = json.loads(json.dumps(record))
        args = {"table": table, "key": key, "record": sent, "fence": {"scope": fence.scope, "number": fence.number}}
        said, retried = await self.call("ledger/append", args)
        wrote = bool(said["wrote"]) or (retried and said["record"] == sent)  # (its own, whose answer was lost)
        return Appended(wrote, said["record"])

    async def read(self, table: str) -> dict[str, JsonValue]:
        return dict(await self.result("ledger/read", {"table": table}))

    async def tables(self) -> list[str]:
        return list(await self.result("ledger/tables"))

    async def read_all(self, *, leaving_out: str | None = None) -> dict[str, dict[str, JsonValue]]:
        return dict(await self.result("ledger/read_all", {"leaving_out": leaving_out}))

    async def fences(self) -> dict[str, int]:
        return {str(scope): int(number) for scope, number in (await self.result("ledger/fences")).items()}

    async def now(self) -> float:
        """The service's clock, in seconds since the epoch."""
        return float(await self.result("clock/now"))

    @property
    def registry(self) -> "HttpRegistry":
        return HttpRegistry(self)

    @property
    def launches(self) -> "HttpLaunches":
        return HttpLaunches(self)

    @property
    def presence(self) -> "HttpPresence":
        return HttpPresence(self)

    @property
    def desired_settings(self) -> "HttpDesiredSettings":
        return HttpDesiredSettings(self)

    @property
    def sandboxes(self) -> "HttpLeases":
        return HttpLeases(self)

    @property
    def presets(self) -> "HttpPresets":
        return HttpPresets(self)

    @property
    def environment_versions(self) -> "HttpEnvironmentVersions":
        return HttpEnvironmentVersions(self)

    async def aclose(self) -> None:
        if self._owned:
            await self._http.aclose()

    def close(self) -> None:
        from rollout_train.inference.remote import _closing  # pyright: ignore[reportPrivateUsage]

        if self._owned:
            _closing(self._http)


def _answered(operation: str, response: httpx.Response) -> Any:
    try:
        said: Any = response.json()
    except ValueError:
        said = {}
    if response.status_code < 400 and isinstance(said, dict) and "result" in said:
        return cast(dict[str, Any], said)["result"]
    found = cast(dict[str, Any], said) if isinstance(said, dict) else {}
    message = str(found.get("message") or response.text[:300] or response.reason_phrase)
    kind = ERRORS.get(str(found.get("error")))
    if kind is not None:
        raise kind(message)
    raise RuntimeError(f"the ledger service refused {operation}: {response.status_code} {message}")


class HttpRegistry:
    """`Registry` (`rollout_train.registry`) through the service."""

    def __init__(self, ledger: HttpLedger) -> None:
        self._ledger = ledger

    async def runs(self) -> list[Entry]:
        return [decoded(Entry, each) for each in await self._ledger.result("registry/runs")]

    async def create(self, name: str, id: str | None = None) -> Entry:
        return decoded(Entry, await self._ledger.result("registry/create", {"name": name, "id": id}))

    async def rename(self, who: str, name: str) -> Entry:
        return decoded(Entry, await self._ledger.result("registry/rename", {"who": who, "name": name}))

    async def bookmarks(self) -> list[Bookmark]:
        return [decoded(Bookmark, each) for each in await self._ledger.result("registry/bookmarks")]

    async def bookmark(self, name: str, checkpoint: str) -> Bookmark:
        said = await self._ledger.result("registry/bookmark", {"name": name, "checkpoint": checkpoint})
        return decoded(Bookmark, said)

    async def unbookmark(self, name: str) -> None:
        await self._ledger.result("registry/unbookmark", {"name": name})

    async def datasets(self) -> list[Named]:
        return [decoded(Named, each) for each in await self._ledger.result("registry/datasets")]

    async def name_dataset(self, name: str, dataset: str) -> Named:
        return decoded(Named, await self._ledger.result("registry/name_dataset", {"name": name, "dataset": dataset}))

    async def suites(self) -> list[SuiteName]:
        return [decoded(SuiteName, each) for each in await self._ledger.result("registry/suites")]

    async def point_suite(self, name: str, version: str, *, forward: bool = False) -> SuiteName:
        said = await self._ledger.result("registry/point_suite", {"name": name, "version": version, "forward": forward})
        return decoded(SuiteName, said)


class HttpLaunches:
    """`Launches` (`rollout_train.launches`) through the service."""

    def __init__(self, ledger: HttpLedger) -> None:
        self._ledger = ledger

    async def ask(self, asked: Asked, run: str | None = None) -> Launch:
        return decoded(Launch, await self._ledger.result("launches/ask", {"asked": encoded(asked), "run": run}))

    async def all(self) -> list[Launch]:
        return [decoded(Launch, each) for each in await self._ledger.result("launches/all")]

    async def note(self, id: str, *, expect: Collection[str] | None = None, **changes: Any) -> Launch:
        args = {"id": id, "expect": list(expect) if expect is not None else None, "changes": changes}
        return decoded(Launch, await self._ledger.result("launches/note", args))


class HttpPresence:
    """`Presence` (`rollout_train.presence`) through the service: beats are stamped and aged by its database's clock."""

    def __init__(self, ledger: HttpLedger) -> None:
        self._ledger = ledger

    async def beat(self, runner: str, about: Mapping[str, JsonValue]) -> None:
        await self._ledger.result("presence/beat", {"runner": runner, "about": dict(about)})

    async def beats(self) -> list[Beat]:
        return [decoded(Beat, each) for each in await self._ledger.result("presence/beats")]


class HttpDesiredSettings:
    """`DesiredSettings` (`rollout_train.settings`) through the service."""

    def __init__(self, ledger: HttpLedger) -> None:
        self._ledger = ledger

    async def desired(self, run: str) -> Desired | None:
        said = await self._ledger.result("settings/desired", {"run": run})
        return decoded(Desired, said) if said is not None else None

    async def want(self, run: str, settings: Mapping[str, JsonValue]) -> Desired:
        return decoded(Desired, await self._ledger.result("settings/want", {"run": run, "settings": dict(settings)}))


class HttpLeases:
    """`Leases` (`rollout.harness.sandboxes`) through the service."""

    def __init__(self, ledger: HttpLedger) -> None:
        self._ledger = ledger

    async def get(self, key: str) -> Lease | None:
        said = await self._ledger.result("sandboxes/get", {"key": key})
        return decoded(Lease, said) if said is not None else None

    async def put(self, lease: Lease) -> None:
        await self._ledger.result("sandboxes/put", {"lease": encoded(lease)})

    async def delete(self, key: str) -> None:
        await self._ledger.result("sandboxes/delete", {"key": key})

    async def all(self) -> list[Lease]:
        return [decoded(Lease, each) for each in await self._ledger.result("sandboxes/all")]


class HttpPresets:
    """`Presets` (`rollout_train.presets`) through the service."""

    def __init__(self, ledger: HttpLedger) -> None:
        self._ledger = ledger

    async def all(self) -> list[Preset]:
        return [decoded(Preset, each) for each in await self._ledger.result("presets/all")]

    async def versions(self, name: str) -> list[Preset]:
        return [decoded(Preset, each) for each in await self._ledger.result("presets/versions", {"name": name})]

    async def get(self, reference: str) -> Preset | None:
        said = await self._ledger.result("presets/get", {"reference": reference})
        return decoded(Preset, said) if said is not None else None

    async def save(self, name: str, settings: Mapping[str, JsonValue], note: str = "") -> Preset:
        args = {"name": name, "settings": dict(settings), "note": note}
        return decoded(Preset, await self._ledger.result("presets/save", args))

    async def delete(self, name: str) -> Preset:
        return decoded(Preset, await self._ledger.result("presets/delete", {"name": name}))


class HttpEnvironmentVersions:
    """`EnvironmentVersions` (`rollout_train.published`) through the service."""

    def __init__(self, ledger: HttpLedger) -> None:
        self._ledger = ledger

    async def all(self) -> list[EnvironmentVersion]:
        return [decoded(EnvironmentVersion, each) for each in await self._ledger.result("environments/all")]

    async def get(self, reference: str) -> EnvironmentVersion | None:
        said = await self._ledger.result("environments/get", {"reference": reference})
        return decoded(EnvironmentVersion, said) if said is not None else None

    async def record(self, version: EnvironmentVersion) -> EnvironmentVersion:
        said = await self._ledger.result("environments/record", {"version": version.to_json()})
        return decoded(EnvironmentVersion, said)
