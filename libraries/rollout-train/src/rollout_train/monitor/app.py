"""The monitor's web page over a run's directory: `rollout monitor DIRECTORY [--port 8765]`."""

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from rollout_train.layout import FEED
from rollout_train.monitor.feed import FeedReader
from rollout_train.monitor.system import System

PAGE = Path(__file__).with_name("page.html")
SCRIPT = Path(__file__).with_name("monitor.js")


def create_app(directory: Path) -> Starlette:
    """Serves the page and what it asks for, over a run's directory (`rollout_train.layout`):

    - `/api/system`: where the run stands (`System.snapshot`);
    - `/api/groups/{run}/{number}`: one group, its episodes, its step and its outcome (`System.group`);
    - `/api/episodes/{run_id}?after=N`: one episode's lines from index N on (its rollouts, one per model slot), and
      what it reported (`System.episode`);
    - `/api/runs`: every run in the feed, summarised.

    It only reads the directory; the runs' own process writes it."""
    reader = FeedReader(directory / FEED)
    system = System(directory, reader)

    async def page(request: Request) -> Response:
        return HTMLResponse(await asyncio.to_thread(PAGE.read_text))

    async def script(request: Request) -> Response:
        return Response(await asyncio.to_thread(SCRIPT.read_text), media_type="text/javascript")

    async def state(request: Request) -> Response:
        return JSONResponse(await system.snapshot())

    async def group(request: Request) -> Response:
        found = await system.group(request.path_params["run"], int(request.path_params["number"]))
        return JSONResponse(found) if found is not None else JSONResponse({"error": "no such group"}, status_code=404)

    async def episode(request: Request) -> Response:
        after = int(request.query_params.get("after", "0"))
        return JSONResponse(await system.episode(request.path_params["run_id"], after))

    async def runs(request: Request) -> Response:
        return JSONResponse(await asyncio.to_thread(reader.runs))

    @contextlib.asynccontextmanager
    async def measuring(app: Starlette) -> AsyncGenerator[None]:
        watch = asyncio.create_task(system.machine.watch())
        try:
            yield
        finally:
            watch.cancel()

    routes = [
        Route("/", page),
        Route("/monitor.js", script),
        Route("/api/system", state),
        Route("/api/groups/{run}/{number:int}", group),
        Route("/api/episodes/{run_id}", episode),
        Route("/api/runs", runs),
    ]
    return Starlette(routes=routes, lifespan=measuring)
