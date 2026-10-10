"""Tool sets and sandbox pools over HTTP: an environment's own infrastructure, served where it runs.

Whoever builds an environment decides where its worlds live: in the process that runs episodes, or on machines of
their own. `serve(tool_set)` and `serve_pool(pool)` are the second, and a binding's `ToolBinding(url=...)` or
`PoolBinding(url=...)` is all a run needs to reach them; the program calls `run.tools` and `run.sandbox(name)` the
same way either way.

A tool set:

    GET  /specifications      {"specifications", "deduplicates"}: the tools, and whether the tool set deduplicates
    POST /call                {"name", "arguments", "effect_id", "arguments_digest"} → a `ToolResult`

A pool (`rollout.harness.sandboxes`):

    GET  /operations          {"operations", "deduplicates"}: what can be done to its sandboxes
    GET  /capacity            a `Capacity`
    POST /acquire             {"spec", "key", "environment"} → a `Lease`; 503 with `"full": true` when the pool is
                              full, 409 when the key may hold no lease (`LeaseRefused`), 410 when its sandbox is gone
                              (`SandboxLost`)
    POST /release             {"key"}
    POST /call                {"key", "name", "arguments", "effect_id", "arguments_digest"} → a `ToolResult`; 410 for a
                              key with no live sandbox (`SandboxLost`)

A 503 that does not say the pool is full (a proxy in front of a pool that is starting, say) is `PoolUnavailable`.

A call that raises is a platform failure, as in process: the service answers 500 with the error, and the client
raises it.
"""

from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any, cast

import httpx
from pydantic import JsonValue

from rollout.contracts import ToolResult, ToolSpecification
from rollout.harness.imports import ToolSet, deduplicates
from rollout.harness.sandboxes import (
    Capacity,
    Lease,
    LeaseRefused,
    NoCapacity,
    Pool,
    PoolUnavailable,
    SandboxLost,
    SandboxSpec,
)

QUICK = httpx.Timeout(10.0, connect=5.0)
"""How long asking a pool how full it is, or what it offers, may take."""


def _failed(error: Exception) -> Any:
    """The answer to a call that raised: 500, with the error."""
    from starlette.responses import JSONResponse

    return JSONResponse({"error": f"{type(error).__name__}: {error}"}, status_code=500)


class _Described:
    """A client of a service at `url` (which may end in a path: every route is under it) that says once what it offers
    (`GET /LISTED`: the list under that key, and whether it deduplicates), and answers 500 with the error of a call that
    raised. A client given reaches it as that client says (mutual TLS, say); `describe` asks over it."""

    listed: str
    """What it says it offers, and where (`specifications`, `operations`)."""
    what: str
    """The service, in words."""

    def __init__(
        self,
        url: str,
        client: httpx.AsyncClient | None,
        offered: Sequence[ToolSpecification] | None,
        deduplicating: bool,
        timeout: float,
    ) -> None:
        self._url = url.rstrip("/")
        self._http = client or httpx.AsyncClient(timeout=timeout)
        self._described = (list(offered), deduplicating) if offered is not None else None

    @property
    def url(self) -> str:
        return self._url

    @property
    def deduplicates(self) -> bool:
        return self._describe()[1]

    async def describe(self) -> None:
        """Ask what it offers over its own client, once: after this, `deduplicates` and what it offers are known."""
        if self._described is None:
            self._described = self._offered(self._checked(await self._http.get(self._at(self.listed), timeout=QUICK)))

    async def aclose(self) -> None:
        """Close its client."""
        await self._http.aclose()

    def _describe(self) -> tuple[list[ToolSpecification], bool]:
        if self._described is None:  # (asked once, before any call: a plain request, as runs are being set up)
            response = httpx.get(f"{self._url}/{self.listed}", timeout=30)
            response.raise_for_status()
            self._described = self._offered(response.json())
        return self._described

    def _offered(self, described: Any) -> tuple[list[ToolSpecification], bool]:
        offered = [ToolSpecification.model_validate(entry) for entry in described[self.listed]]
        return offered, bool(described["deduplicates"])

    def _at(self, route: str) -> str:
        return f"{self._url}/{route}"

    def _checked(self, response: httpx.Response) -> Any:
        if response.status_code == 500:
            raise RuntimeError(response.json().get("error", f"the {self.what} failed"))
        response.raise_for_status()
        return response.json()


def serve(tool_set: ToolSet) -> Any:
    """A Starlette application serving `tool_set` (needs the `http` extra's starlette)."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    async def specifications(request: Request) -> Response:
        tools = [s.model_dump(mode="json", exclude_none=True) for s in tool_set.specifications()]
        return JSONResponse({"specifications": tools, "deduplicates": deduplicates(tool_set)})

    async def call(request: Request) -> Response:
        body = await request.json()
        try:
            result = await tool_set.call(
                body["name"],
                body["arguments"],
                effect_id=body["effect_id"],
                arguments_digest=body["arguments_digest"],
            )
        except Exception as error:
            return _failed(error)
        return JSONResponse(result.model_dump(mode="json", exclude_none=True))

    return Starlette(routes=[Route("/specifications", specifications), Route("/call", call, methods=["POST"])])


class RemoteToolSet(_Described):
    """A `DeduplicatingToolSet` served at `url`. It deduplicates if the tool set behind it does: the effect's id goes
    with every call."""

    listed, what = "specifications", "tool set"

    def __init__(
        self,
        url: str,
        *,
        client: httpx.AsyncClient | None = None,
        specifications: Sequence[ToolSpecification] | None = None,
        deduplicating: bool = False,
        timeout: float = 600.0,
    ) -> None:
        """`specifications` and `deduplicating`: the tools and whether the tool set deduplicates, if the caller
        already knows (the tool set is asked otherwise)."""
        super().__init__(url, client, specifications, deduplicating, timeout)

    def specifications(self) -> Sequence[ToolSpecification]:
        return self._describe()[0]

    async def call(
        self, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        body = {
            "name": name,
            "arguments": dict(arguments),
            "effect_id": effect_id,
            "arguments_digest": arguments_digest,
        }
        return ToolResult.model_validate(self._checked(await self._http.post(self._at("call"), json=body)))


@cache
def remote_tool_set(url: str) -> RemoteToolSet:
    """The client for a tool set's URL (one per URL in a process)."""
    return RemoteToolSet(url)


def serve_pool(pool: Pool, extra: Sequence[Any] = ()) -> Any:
    """A Starlette application serving `pool` (needs the `http` extra's starlette), with `extra` routes of the
    caller's."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    async def operations(request: Request) -> Response:
        listed = [each.model_dump(mode="json", exclude_none=True) for each in pool.operations()]
        return JSONResponse({"operations": listed, "deduplicates": deduplicates(pool)})

    async def capacity(request: Request) -> Response:
        return JSONResponse((await pool.capacity()).model_dump(mode="json"))

    async def acquire(request: Request) -> Response:
        body = await request.json()
        try:
            lease = await pool.acquire(SandboxSpec.model_validate(body["spec"]), body["key"], body.get("environment"))
        except PoolUnavailable as error:
            return JSONResponse({"error": str(error)}, status_code=503)
        except NoCapacity as error:
            return JSONResponse({"error": str(error), "full": True}, status_code=503)
        except LeaseRefused as error:
            return JSONResponse({"error": str(error)}, status_code=409)
        except SandboxLost as error:
            return JSONResponse({"error": str(error)}, status_code=410)
        except Exception as error:
            return _failed(error)
        return JSONResponse(lease.model_dump(mode="json"))

    async def release(request: Request) -> Response:
        try:
            await pool.release((await request.json())["key"])
        except Exception as error:
            return _failed(error)
        return JSONResponse({})

    async def call(request: Request) -> Response:
        body = await request.json()
        try:
            result = await pool.call(
                body["key"],
                body["name"],
                body["arguments"],
                effect_id=body["effect_id"],
                arguments_digest=body["arguments_digest"],
            )
        except SandboxLost as error:
            return JSONResponse({"error": str(error)}, status_code=410)
        except Exception as error:
            return _failed(error)
        return JSONResponse(result.model_dump(mode="json", exclude_none=True))

    return Starlette(
        routes=[
            Route("/operations", operations),
            Route("/capacity", capacity),
            Route("/acquire", acquire, methods=["POST"]),
            Route("/release", release, methods=["POST"]),
            Route("/call", call, methods=["POST"]),
            *extra,
        ]
    )


class RemotePool(_Described):
    """A `Pool` served at `url`."""

    listed, what = "operations", "pool"

    def __init__(
        self,
        url: str,
        *,
        client: httpx.AsyncClient | None = None,
        operations: Sequence[ToolSpecification] | None = None,
        deduplicating: bool = False,
        timeout: float = 600.0,
    ) -> None:
        """`operations` and `deduplicating`: what can be done to its sandboxes and whether it deduplicates, if the
        caller already knows (the pool is asked otherwise)."""
        super().__init__(url, client, operations, deduplicating, timeout)

    def operations(self) -> Sequence[ToolSpecification]:
        return self._describe()[0]

    async def acquire(self, spec: SandboxSpec, key: str, environment: Mapping[str, str] | None = None) -> Lease:
        body = {"spec": spec.model_dump(mode="json"), "key": key, "environment": dict(environment or {})}
        response = await self._http.post(self._at("acquire"), json=body)
        if response.status_code == 503:
            said = _said(response)
            refused = NoCapacity if said.get("full") is True else PoolUnavailable
            raise refused(str(said.get("error") or f"the pool at {self._url} does not answer now"))
        refused = {409: LeaseRefused, 410: SandboxLost}.get(response.status_code)
        if refused is not None:
            raise refused(_said(response).get("error", f"the pool answered {response.status_code}"))
        return Lease.model_validate(self._checked(response))

    async def release(self, key: str) -> None:
        self._checked(await self._http.post(self._at("release"), json={"key": key}))

    async def capacity(self) -> Capacity:
        return Capacity.model_validate(self._checked(await self._http.get(self._at("capacity"), timeout=QUICK)))

    async def call(
        self, key: str, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        body = {
            "key": key,
            "name": name,
            "arguments": dict(arguments),
            "effect_id": effect_id,
            "arguments_digest": arguments_digest,
        }
        response = await self._http.post(self._at("call"), json=body)
        if response.status_code == 410:
            raise SandboxLost(_said(response).get("error", f"the pool holds no live sandbox of {key}"))
        return ToolResult.model_validate(self._checked(response))


def _said(response: httpx.Response) -> dict[str, Any]:
    """A refusal's JSON object (empty where it is not one: a proxy's own answer, say)."""
    try:
        said = response.json()
    except ValueError:
        return {}
    return cast(dict[str, Any], said) if isinstance(said, dict) else {}


@cache
def remote_pool(url: str) -> RemotePool:
    """The client for a pool's URL (one per URL in a process)."""
    return RemotePool(url)
