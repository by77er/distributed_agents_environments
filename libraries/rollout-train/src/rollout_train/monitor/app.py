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


def create_app(directory: Path) -> Starlette:
    """Serves the page and what it asks for, over a run's directory (`rollout_train.layout`):

    - `/api/system`: where the run stands (`System.snapshot`);
    - `/api/runs`: every run in the feed, summarised;
    - `/api/runs/{run_id}?after=N`: a run's feed lines from index N on;
    - `/api/job?after=N`: what the rollout job did (its groups, updates and engines).

    It only reads the directory; the runs' own process writes it."""
    reader = FeedReader(directory / FEED)
    system = System(directory, reader)

    async def page(request: Request) -> Response:
        return HTMLResponse(await asyncio.to_thread(PAGE.read_text))

    async def state(request: Request) -> Response:
        return JSONResponse(await system.snapshot())

    async def runs(request: Request) -> Response:
        return JSONResponse(await asyncio.to_thread(reader.runs))

    async def run(request: Request) -> Response:
        run_id = request.path_params["run_id"]
        after = int(request.query_params.get("after", "0"))
        return JSONResponse({"run_id": run_id, "lines": await asyncio.to_thread(reader.lines, run_id, after)})

    async def job(request: Request) -> Response:
        after = int(request.query_params.get("after", "0"))
        return JSONResponse({"lines": await asyncio.to_thread(reader.job, after)})

    @contextlib.asynccontextmanager
    async def measuring(app: Starlette) -> AsyncGenerator[None]:
        watch = asyncio.create_task(system.machine.watch())
        try:
            yield
        finally:
            watch.cancel()

    routes = [
        Route("/", page),
        Route("/api/system", state),
        Route("/api/runs", runs),
        Route("/api/runs/{run_id}", run),
        Route("/api/job", job),
    ]
    return Starlette(routes=routes, lifespan=measuring)
