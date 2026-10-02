"""Tool sets over HTTP: an environment's own infrastructure, served where it runs and imported by name.

Whoever builds an environment decides where its worlds live: in the process that runs episodes, or on machines of
their own. `serve(tool_set)` is the second, and a binding's `ToolBinding(url=...)` is all a run needs to reach it;
the program calls `run.tools` the same way either way.

    GET  /specifications      the tools
    POST /call                {"name", "arguments", "effect_id", "arguments_digest"} → a `ToolResult`

A call that raises is a platform failure, as in process: the service answers 500 with the error, and the client
raises it.
"""

from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

import httpx
from pydantic import JsonValue

from rollout.core.contracts import ToolResult, ToolSpecification
from rollout.core.harness.imports import ToolSet


def serve(tool_set: ToolSet) -> Any:
    """A Starlette application serving `tool_set` (needs the `monitor` extra's starlette)."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    async def specifications(request: Request) -> Response:
        return JSONResponse([s.model_dump(mode="json", exclude_none=True) for s in tool_set.specifications()])

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
            return JSONResponse({"error": f"{type(error).__name__}: {error}"}, status_code=500)
        return JSONResponse(result.model_dump(mode="json", exclude_none=True))

    return Starlette(routes=[Route("/specifications", specifications), Route("/call", call, methods=["POST"])])


class RemoteToolSet:
    """A `ToolSet` served at `url`. It deduplicates if the tool set behind it does: the effect's id goes with every
    call."""

    def __init__(
        self,
        url: str,
        *,
        client: httpx.AsyncClient | None = None,
        specifications: Sequence[ToolSpecification] | None = None,
        timeout: float = 600.0,
    ) -> None:
        """`specifications`: the tools, if the caller already has them (they are asked for otherwise)."""
        self._url = url
        self._http = client or httpx.AsyncClient(base_url=url, timeout=timeout)
        self._specifications = list(specifications) if specifications is not None else None

    def specifications(self) -> Sequence[ToolSpecification]:
        if self._specifications is None:  # (asked once, before any call: a plain request, as runs are being set up)
            response = httpx.get(f"{self._url}/specifications", timeout=30)
            response.raise_for_status()
            self._specifications = [ToolSpecification.model_validate(entry) for entry in response.json()]
        return self._specifications

    async def call(
        self, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        body = {
            "name": name,
            "arguments": dict(arguments),
            "effect_id": effect_id,
            "arguments_digest": arguments_digest,
        }
        response = await self._http.post("/call", json=body)
        if response.status_code == 500:
            raise RuntimeError(response.json().get("error", "the tool set failed"))
        response.raise_for_status()
        return ToolResult.model_validate(response.json())


@cache
def remote_tool_set(url: str) -> RemoteToolSet:
    """The client for a tool set's URL (one per URL in a process)."""
    return RemoteToolSet(url)
