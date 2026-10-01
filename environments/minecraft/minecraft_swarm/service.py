"""The Minecraft world service: temporary servers and ground-truth rewards over HTTP.

It fronts `MinecraftWorlds` so rollout workers elsewhere can use worlds without running Paper or the bots:

    POST   /episodes                          {"task", "world_seed", "layout_seed"} → {"episode", "task", ...}
    GET    /episodes/{episode}/agents/{agent} what the agent perceives
    POST   /episodes/{episode}/agents/{agent} {"action": {...}} → {"started": bool}
    POST   /episodes/{episode}/window         run game time while actions happen
    GET    /episodes/{episode}/score          the team's diamonds and what happened (ground truth)
    DELETE /episodes/{episode}                stop the episode's server
    GET    /tasks                             the task catalog

`RemoteMinecraftTools` is the `minecraft` tool set as a client of the service.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from pydantic import JsonValue
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from minecraft_swarm.worlds import MinecraftTools, MinecraftWorlds
from rollout.core.contracts import Text, ToolResult, ToolSpecification


def create_app(worlds: MinecraftWorlds) -> Starlette:
    async def begin(request: Request) -> Response:
        body = await request.json()
        try:
            started = await worlds.begin(str(body["task"]), int(body["world_seed"]), int(body["layout_seed"]))
        except KeyError as error:
            return JSONResponse({"error": f"unknown {error}"}, status_code=400)
        return JSONResponse(started, status_code=201)

    async def observe(request: Request) -> Response:
        return await _guarded(lambda: worlds.observe(request.path_params["episode"], request.path_params["agent"]))

    async def act(request: Request) -> Response:
        body = await request.json()
        episode, agent = request.path_params["episode"], request.path_params["agent"]

        async def start() -> dict[str, Any]:
            return {"started": await worlds.act(episode, agent, body.get("action") or {})}

        return await _guarded(start)

    async def window(request: Request) -> Response:
        return await _guarded(lambda: worlds.window(request.path_params["episode"]))

    async def score(request: Request) -> Response:
        return await _guarded(lambda: worlds.score(request.path_params["episode"]))

    async def end(request: Request) -> Response:
        await worlds.end(request.path_params["episode"])
        return JSONResponse({"ended": True})

    async def tasks(request: Request) -> Response:
        return JSONResponse({"tasks": [task.model_dump(mode="json") for task in worlds.tasks.values()]})

    return Starlette(
        routes=[
            Route("/episodes", begin, methods=["POST"]),
            Route("/episodes/{episode}/agents/{agent}", observe, methods=["GET"]),
            Route("/episodes/{episode}/agents/{agent}", act, methods=["POST"]),
            Route("/episodes/{episode}/window", window, methods=["POST"]),
            Route("/episodes/{episode}/score", score, methods=["GET"]),
            Route("/episodes/{episode}", end, methods=["DELETE"]),
            Route("/tasks", tasks, methods=["GET"]),
        ],
        lifespan=lambda app: _Lifespan(worlds),
    )


class _Lifespan:
    """Stops every episode's server when the service stops."""

    def __init__(self, worlds: MinecraftWorlds) -> None:
        self._worlds = worlds

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exception: object) -> None:
        await self._worlds.close()


async def _guarded(work: Any) -> Response:
    try:
        return JSONResponse(await work())
    except KeyError as error:
        return JSONResponse({"error": str(error)}, status_code=404)


class RemoteMinecraftTools:
    """The `minecraft` tool set over HTTP: the same operations as `MinecraftTools`, served by the world service."""

    def __init__(self, url: str, *, client: httpx.AsyncClient | None = None) -> None:
        self.url = url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=300)
        self._specifications = MinecraftTools(MinecraftWorlds()).specifications()

    def specifications(self) -> Sequence[ToolSpecification]:
        return self._specifications

    async def call(
        self, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        episode = str(arguments.get("episode", ""))
        match name:
            case "begin":
                response = await self._client.post(f"{self.url}/episodes", json=dict(arguments))
            case "observe":
                response = await self._client.get(f"{self.url}/episodes/{episode}/agents/{arguments['agent']}")
            case "act":
                response = await self._client.post(
                    f"{self.url}/episodes/{episode}/agents/{arguments['agent']}", json={"action": arguments["action"]}
                )
            case "window":
                response = await self._client.post(f"{self.url}/episodes/{episode}/window")
            case "score":
                response = await self._client.get(f"{self.url}/episodes/{episode}/score")
            case "end":
                response = await self._client.delete(f"{self.url}/episodes/{episode}")
            case _:
                return ToolResult(content=[Text(text=f"unknown operation {name}")], is_error=True)
        value: JsonValue = response.json()
        if response.status_code >= 400:
            return ToolResult(content=[Text(text=json.dumps(value))], is_error=True)
        return ToolResult(content=[Text(text=json.dumps(value))], structured=value)
