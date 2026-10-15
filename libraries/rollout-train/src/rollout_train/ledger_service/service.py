"""The ledger service: the ledger and the stores beside it, over HTTP, for whoever holds a token
(docs/research/ledger-guarantees.md#10-the-http-ledger-service).

`app(ledger, secret)` serves a ledger in this process (a `DatabaseLedger` in a deployment) to `HttpLedger` clients. Each
operation is one call of the store's own method, so each keeps what the store keeps: a take and an append are one
transaction each, the fence is checked when the append is applied, a beat is stamped by the database's clock.
What HTTP adds is answered here:

- **Retries.** A take that comes with a request id is answered with the number its first attempt took, kept in the
  same transaction (`DatabaseLedger.take`); so is any operation that changes something retried with the same id
  while this replica holds its answer (the newest `KEPT` answers; a read's is not kept, as it is read again), and one
  retried while its first attempt is still being done waits for that attempt's answer. An append's answer says what
  the table holds under the key, so a client whose first attempt's answer was lost tells its own record from
  another's (`HttpLedger.append`); a run created and a launch asked for come with the id their client chose, so one
  created again is found as its own.
- **Errors.** `Fenced` is its own error (409, `fenced`), never retried; so are a name taken, a row missing, a request
  that cannot be read, a token that may not, and a compare-and-set that lost (`wire.ERRORS`). A database that could
  not do an operation for a moment is `unavailable` (503), which the client asks again: a transaction it rolled back
  (a deadlock, a serialization failure, a lock or a connection not to be had: `ROLLED_BACK`), and a connection lost
  while a transaction may have committed, for an operation that may be done again (`repeatable`). Anything else is
  the service's own failure (500, `failed`).
- **JSON.** Answers are written as the stores write records (`wire.body_of`): a NaN a record holds goes as it is.
- **Tokens.** The platform's token may do everything; a pod's may do what `rollout_train.ledger_service.scopes` says,
  which `allowed` checks before the store is asked, and `_read` and `_tables` narrow what it reads.

    rollout ledger serve --cluster [--listen 0.0.0.0:8840]

serves the cluster config's ledger (`[ledger] url`, a database) with its token (`[ledger] token_env` or `token_file`).
"""

import asyncio
import functools
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

from pydantic import JsonValue

from rollout.harness.sandboxes import Lease
from rollout_train.checkpoints import CHECKPOINTS, COMPLETED, RELEASED
from rollout_train.launches import Asked, launches_of
from rollout_train.ledger import Fence, Ledger
from rollout_train.ledger_service.scopes import Forbidden, Scope, scope_of
from rollout_train.ledger_service.wire import STATUS, body_of, code_of, decoded, encoded, message_of
from rollout_train.presence import presence_of
from rollout_train.presets import presets_of
from rollout_train.published import EnvironmentVersion, environment_versions_of
from rollout_train.record import STARTS, mapping, table
from rollout_train.registry import registry_of
from rollout_train.sandboxes import leases_of
from rollout_train.serving import SERVING
from rollout_train.settings import desired_settings_of

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from rollout_train.cluster import Cluster

__all__ = [
    "KEPT",
    "READ_ONLY",
    "REPEATED",
    "ROLLED_BACK",
    "Handler",
    "app",
    "for_cluster",
    "held_by_run",
    "operations",
    "repeatable",
    "unavailable",
]

KEPT = 4096
"""Answers kept for operations retried with the same request id, newest first."""
READ_ONLY = frozenset({
    "ledger/read", "ledger/tables", "ledger/read_all", "ledger/fences", "clock/now", "registry/runs",
    "registry/bookmarks", "registry/datasets", "registry/suites", "launches/all", "presence/beats", "settings/desired",
    "sandboxes/get", "sandboxes/all", "presets/all", "presets/versions", "presets/get", "environments/all",
    "environments/get", "pods/all", "pods/get", "pods/times",
})  # fmt: skip
"""The operations that change nothing: their answers are not kept (a retry reads again), however large."""
REPEATED = READ_ONLY | frozenset({
    "ledger/append", "launches/note", "presence/beat", "settings/want", "sandboxes/put", "sandboxes/delete",
    "registry/bookmark", "registry/name_dataset", "registry/point_suite", "environments/record", "pods/put",
    "pods/delete", "pods/charge",
})  # fmt: skip
"""The operations that may be done again after an attempt that may have been done: reads; an append (its client
tells its own record from another's); a launch's change, which compares and sets; writes that come to the same done
twice; and pods' leases, whose client reads one back where its compare-and-set lost to an attempt of its own
(`HttpPodLeases`)."""
ROLLED_BACK = frozenset({"40001", "40P01", "55P03", "53300", "57P03", "57014"})
"""Postgres's errors (SQLSTATE) that say nothing was done: a serialization failure, a deadlock, a lock not to be had,
too many connections, a server that cannot take one now, a statement cancelled (its transaction rolled back)."""

READS = ("ledger/read", "ledger/tables")
"""What a pod's token is honoured for only while its pod is held by its run."""

type Handler = Callable[[Ledger, Mapping[str, Any], Scope], Awaitable[Any]]


def _store[T](found: T | None, name: str) -> T:
    if found is None:
        raise KeyError(f"this ledger keeps no {name}")
    return found


async def _take(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> JsonValue:
    request = args.get("request")
    if request is not None and hasattr(ledger, "database"):  # (a database ledger keeps takes by request id)
        fence = await cast(Any, ledger).take(str(args["scope"]), request=str(request))
    else:
        fence = await ledger.take(str(args["scope"]))
    return fence.number


async def _append(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> JsonValue:
    fence = Fence(str(args["fence"]["scope"]), int(args["fence"]["number"]))
    appended = await ledger.append_returning(str(args["table"]), str(args["key"]), args["record"], fence)
    return {"wrote": appended.wrote, "record": appended.record}


async def _read(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> JsonValue:
    name = str(args["table"])
    if scope.platform:
        return cast(JsonValue, await ledger.read(name))
    assert scope.run is not None
    if name in (table(scope.run, SERVING), table(scope.run, STARTS)):
        return cast(JsonValue, await ledger.read(name))
    if name in (CHECKPOINTS, RELEASED, COMPLETED):
        readable = await _checkpoints_of(ledger, scope.run)
        return {key: value for key, value in (await ledger.read(name)).items() if key in readable}
    raise Forbidden(f"a pod's token reads its run's serving records, starts and checkpoints, not {name}")


async def _tables(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> JsonValue:
    names = await ledger.tables()
    if scope.platform:
        return cast(JsonValue, names)
    assert scope.run is not None
    readable = {table(scope.run, SERVING), table(scope.run, STARTS), CHECKPOINTS, RELEASED, COMPLETED}
    return [name for name in names if name in readable]


async def _checkpoints_of(ledger: Ledger, run: str) -> set[str]:
    """The checkpoints a pod serving `run` may read: those the run made, those its serving records name (and the full
    checkpoints they are served over), those its channels are fixed on, and every checkpoint each of those was trained
    over."""
    records = await ledger.read(CHECKPOINTS)
    found = {key for key, record in records.items() if mapping(record).get("run") == run}
    for record in (await ledger.read(table(run, SERVING))).values():
        said = mapping(record)
        found |= {str(said[key]) for key in ("checkpoint", "over") if isinstance(said.get(key), str)}
    for record in (await ledger.read(table(run, STARTS))).values():
        settings = mapping(record).get("run_settings")
        for part in mapping(settings).values():
            for key, value in mapping(part).items():
                if key.startswith("channels.") and key.endswith(".checkpoint") and isinstance(value, str):
                    found.add(value)
        start = mapping(settings).get("fixed")
        if isinstance(said_start := mapping(start).get("start"), str):
            found.add(said_start)
    pending = list(found)
    while pending:  # (each one's line: the checkpoints it was trained over)
        base = mapping(records.get(pending.pop())).get("base")
        if isinstance(base, str) and base in records and base not in found:
            found.add(base)
            pending.append(base)
    return found


async def _read_all(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> JsonValue:
    leaving_out = args.get("leaving_out")
    return cast(JsonValue, await ledger.read_all(leaving_out=str(leaving_out) if leaving_out is not None else None))


async def _fences(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> JsonValue:
    return cast(JsonValue, await ledger.fences())


async def _now(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> JsonValue:
    return time.time()


def _registry(method: str, *names: str) -> Handler:
    async def handled(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> Any:
        store = _store(registry_of(ledger), "registry")
        return await getattr(store, method)(*(args.get(name) for name in names))

    return handled


async def _point_suite(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> Any:
    store = _store(registry_of(ledger), "registry")
    return await store.point_suite(str(args["name"]), str(args["version"]), forward=bool(args.get("forward")))


async def _ask(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> Any:
    store = _store(launches_of(ledger), "launches")
    id = args.get("id")
    return await store.ask(Asked(**args["asked"]), args.get("run"), str(id) if id is not None else None)


async def _launches(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> Any:
    return await _store(launches_of(ledger), "launches").all()


async def _note(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> Any:
    store = _store(launches_of(ledger), "launches")
    expect = args.get("expect")
    return await store.note(str(args["id"]), expect=list(expect) if expect is not None else None,
                            **dict(args.get("changes") or {}))  # fmt: skip


async def _beat(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> Any:
    await _store(presence_of(ledger), "beats").beat(str(args["runner"]), dict(args["about"]))


async def _beats(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> Any:
    return await _store(presence_of(ledger), "beats").beats()


async def _desired(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> Any:
    return await _store(desired_settings_of(ledger), "desired settings").desired(str(args["run"]))


async def _want(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> Any:
    store = _store(desired_settings_of(ledger), "desired settings")
    return await store.want(str(args["run"]), dict(args["settings"]))


def _sandboxes(method: str) -> Handler:
    async def handled(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> Any:
        store = _store(leases_of(ledger), "sandbox leases")
        if method == "put":
            return await store.put(Lease.model_validate(args["lease"]))
        if method == "all":
            return await store.all()
        return await getattr(store, method)(str(args["key"]))

    return handled


def _presets(method: str) -> Handler:
    async def handled(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> Any:
        store = _store(presets_of(ledger), "presets")
        if method == "all":
            return await store.all()
        if method == "save":
            return await store.save(str(args["name"]), dict(args["settings"]), str(args.get("note") or ""))
        return await getattr(store, method)(str(args["name"] if "name" in args else args["reference"]))

    return handled


def _versions(method: str) -> Handler:
    async def handled(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> Any:
        store = _store(environment_versions_of(ledger), "environment versions")
        if method == "all":
            return await store.all()
        if method == "record":
            return await store.record(decoded(EnvironmentVersion, args["version"]))
        return await store.get(str(args["reference"]))

    return handled


def _pods(method: str) -> Handler:
    async def handled(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> Any:
        from rollout_train.pods.leases import PodLease, PodTime, pod_leases_of

        store = _store(pod_leases_of(ledger), "pods' leases")
        if method == "all":
            return [each.to_json() for each in await store.all()]
        if method == "get":
            found = await store.get(str(args["pod"]))
            return found.to_json() if found is not None else None
        if method == "put":
            expect = args.get("expect")
            made = await store.put(
                PodLease.from_json(args["lease"]), expect=int(expect) if expect is not None else None
            )
            return made.to_json()
        if method == "delete":
            return await store.delete(str(args["pod"]), expect=int(args["expect"]))
        if method == "times":
            run = args.get("run")
            return [each.to_json() for each in await store.times(str(run) if run is not None else None)]
        return await store.charge(PodTime.from_json(args["entry"]))

    return handled


async def _own_lease(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> None:
    if args.get("pod") != scope.pod:
        raise Forbidden(f"a pod's token reads its pod's ({scope.pod}) lease alone")


async def held_by_run(ledger: Ledger, scope: Scope) -> bool:
    """Whether a pod's token is honoured now: its pod's lease names its run."""
    from rollout_train.pods.leases import pod_leases_of

    store = pod_leases_of(ledger)
    if store is None:
        return False
    lease = await store.get(str(scope.pod))
    return lease is not None and lease.run == scope.run


async def _own_beat(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> None:
    if args.get("runner") != scope.pod:
        raise Forbidden(f"a pod's token beats as its pod ({scope.pod}) alone")


async def _any(ledger: Ledger, args: Mapping[str, Any], scope: Scope) -> None:
    return None


def repeatable(name: str, args: Mapping[str, Any]) -> bool:
    """Whether the operation `name` with `args` may be done again where an attempt at it may have been done: those of
    `REPEATED`, a take with a request id (answered with the number its first attempt took), and a run created or a
    launch asked for with the id its client chose (found as its own)."""
    if name == "ledger/take":
        return args.get("request") is not None
    if name in ("registry/create", "launches/ask"):
        return args.get("id") is not None
    return name in REPEATED


def unavailable(error: BaseException, *, repeated: bool) -> bool:
    """Whether `error` is a database's that could not do an operation for a moment, which is asked again: a
    transaction it rolled back (`ROLLED_BACK`), a connection its pool did not give in time, SQLite's lock not had in
    time; and, where `repeated` (the operation may be done again), a connection lost, which may have committed what
    it was doing."""
    import sqlite3

    import sqlalchemy.exc  # (here: a service that serves no database needs none)

    if isinstance(error, sqlalchemy.exc.TimeoutError):
        return True
    if not isinstance(error, sqlalchemy.exc.DBAPIError):
        return False
    cause = error.orig
    if isinstance(cause, sqlite3.Error):  # (one machine's file: busy or locked, the transaction was rolled back)
        return getattr(cause, "sqlite_errorname", "").startswith(("SQLITE_BUSY", "SQLITE_LOCKED")) or (
            "database is locked" in str(cause)
        )
    state = str(getattr(cause, "sqlstate", None) or getattr(cause, "pgcode", None) or "")
    if state in ROLLED_BACK:
        return True
    lost = error.connection_invalidated or state.startswith("08") or state in ("57P01", "57P02")
    return repeated and (lost or (isinstance(error, sqlalchemy.exc.OperationalError) and not state))


def operations() -> dict[str, tuple[Handler, Handler | None]]:
    """Every operation, by `STORE/OPERATION`: what does it, and what a pod's token is checked by before it is done
    (none: a pod's token may not)."""
    return {
        "ledger/take": (_take, None),
        "ledger/append": (_append, None),
        "ledger/read": (_read, _any),
        "ledger/tables": (_tables, _any),
        "ledger/read_all": (_read_all, None),
        "ledger/fences": (_fences, None),
        "clock/now": (_now, _any),
        "registry/runs": (_registry("runs"), None),
        "registry/create": (_registry("create", "name", "id"), None),
        "registry/rename": (_registry("rename", "who", "name"), None),
        "registry/bookmarks": (_registry("bookmarks"), None),
        "registry/bookmark": (_registry("bookmark", "name", "checkpoint"), None),
        "registry/unbookmark": (_registry("unbookmark", "name"), None),
        "registry/datasets": (_registry("datasets"), None),
        "registry/name_dataset": (_registry("name_dataset", "name", "dataset"), None),
        "registry/suites": (_registry("suites"), None),
        "registry/point_suite": (_point_suite, None),
        "launches/ask": (_ask, None),
        "launches/all": (_launches, None),
        "launches/note": (_note, None),
        "presence/beat": (_beat, _own_beat),
        "presence/beats": (_beats, None),
        "settings/desired": (_desired, None),
        "settings/want": (_want, None),
        "sandboxes/get": (_sandboxes("get"), None),
        "sandboxes/put": (_sandboxes("put"), None),
        "sandboxes/delete": (_sandboxes("delete"), None),
        "sandboxes/all": (_sandboxes("all"), None),
        "presets/all": (_presets("all"), None),
        "presets/versions": (_presets("versions"), None),
        "presets/get": (_presets("get"), None),
        "presets/save": (_presets("save"), None),
        "presets/delete": (_presets("delete"), None),
        "environments/all": (_versions("all"), None),
        "environments/get": (_versions("get"), None),
        "environments/record": (_versions("record"), None),
        "pods/all": (_pods("all"), None),
        "pods/get": (_pods("get"), _own_lease),
        "pods/put": (_pods("put"), None),
        "pods/delete": (_pods("delete"), None),
        "pods/times": (_pods("times"), None),
        "pods/charge": (_pods("charge"), None),
    }


def app(
    ledger: Ledger,
    secret: Callable[[], str | None],
    *,
    more: Mapping[str, tuple[Handler, Handler | None]] | None = None,
    honoured: Callable[[Ledger, Scope], Awaitable[bool]] | None = None,
) -> "Starlette":
    """The service over `ledger`, its tokens checked against the platform's token, which `secret` reads (at each
    request: a token rotated where it is kept is honoured at once). `more` adds operations. `honoured` says whether a
    pod's token is honoured now (whether the pod's lease names its run)."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import Route

    table_of = {**operations(), **(more or {})}
    answers: OrderedDict[tuple[str, str, str], Response] = OrderedDict()
    doing: dict[tuple[str, str, str], asyncio.Task[Response]] = {}

    def answer(said: Mapping[str, Any], status: int = 200) -> Response:
        return Response(body_of(said), status_code=status, media_type="application/json")

    def refused(code: str, message: str) -> Response:
        return answer({"error": code, "message": message[:2000]}, STATUS.get(code, 500))

    async def done(name: str, arguments: Mapping[str, Any], scope: Scope) -> Response:
        """`name` done with `arguments` for `scope`, as its answer."""
        handler, check = table_of[name]
        try:
            if not scope.platform:
                if check is None:
                    raise Forbidden(f"a pod's token may not {name}")
                await check(ledger, arguments, scope)
                if name in READS and honoured is not None and not await honoured(ledger, scope):
                    raise Forbidden(f"pod {scope.pod} is not held by run {scope.run}")
            result = await handler(ledger, arguments, scope)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if (code := code_of(error)) is not None:
                return refused(code, message_of(error))
            failure = f"{type(error).__name__}: {error}"
            if unavailable(error, repeated=repeatable(name, arguments)):
                return refused("unavailable", failure)
            return answer({"error": "failed", "message": failure[:2000]}, 500)
        return answer({"result": encoded(result)})

    def kept(remembered: tuple[str, str, str], finished: asyncio.Task[Response]) -> None:
        """Keep the answer of an operation done (an error's is not: asked again, it is done again)."""
        del doing[remembered]
        if finished.cancelled() or finished.exception() is not None or finished.result().status_code != 200:
            return
        answers[remembered] = finished.result()
        while len(answers) > KEPT:
            answers.popitem(last=False)

    async def call(request: Request) -> Response:
        said = request.headers.get("authorization", "")
        token = said.removeprefix("Bearer ").strip() if said.startswith("Bearer ") else None
        known = secret()
        if not known:
            return answer({"error": "unconfigured", "message": "the service has no token"}, 503)
        scope = scope_of(token, known)
        if scope is None:
            return answer({"error": "unauthorized", "message": "no token, or one not of this ledger"}, 401)
        name = f"{request.path_params['store']}/{request.path_params['operation']}"
        if name not in table_of:
            return refused("missing", f"there is no operation {name}")
        try:
            body: Any = await request.json()
        except ValueError:
            return refused("invalid", "the request is not JSON")
        given: dict[str, Any] = cast(dict[str, Any], body) if isinstance(body, dict) else {}
        args: Any = given.get("args")
        request_id: Any = given.get("request")
        if not isinstance(args, dict):
            return refused("invalid", "the request has no args")
        arguments = cast(dict[str, Any], args)
        if name in READ_ONLY or not isinstance(request_id, str):
            return await done(name, arguments, scope)
        remembered = (scope.key, name, request_id)
        if remembered in answers:
            return answers[remembered]
        if remembered not in doing:  # (one retried while its first attempt is being done waits for that one's answer)
            if name == "ledger/take":
                arguments = {**arguments, "request": request_id}
            doing[remembered] = asyncio.ensure_future(done(name, arguments, scope))
            doing[remembered].add_done_callback(functools.partial(kept, remembered))
        return await asyncio.shield(doing[remembered])

    async def healthz(request: Request) -> Response:
        return answer({"healthy": True})

    return Starlette(
        routes=[
            Route("/v1/{store}/{operation}", call, methods=["POST"]),
            Route("/healthz", healthz, methods=["GET"]),
            Route("/readyz", healthz, methods=["GET"]),
        ]
    )


def for_cluster(cluster: "Cluster") -> "Starlette":
    """The service over a cluster config's ledger (`[ledger] url`, which must be a database's here), with its token
    (`[ledger] token_env` or `token_file`, read at each request). Raises `ClusterError` where either is missing."""
    from rollout_train.cluster import ClusterError
    from rollout_train.database import DatabaseLedger
    from rollout_train.stores import DATABASES, ledger_url

    url = ledger_url(cluster)
    if not url.startswith(DATABASES):
        raise ClusterError("the ledger service serves a database: [ledger] url is the database's where it runs")
    token = cluster.ledger.token
    if token is None:
        raise ClusterError("the ledger service checks tokens against the platform's: name it in [ledger] token_env")
    return app(DatabaseLedger(url), token.resolve, honoured=held_by_run)
