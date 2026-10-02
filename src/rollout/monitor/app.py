"""The monitor's web page over a feed directory: `rollout monitor DIRECTORY [--port 8765]`."""

from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from rollout.monitor.feed import FeedReader

PAGE = Path(__file__).with_name("page.html")


def create_app(directory: Path) -> Starlette:
    """Serves the page and what it asks for: `/api/runs` (every run, summarised), `/api/runs/{run_id}?after=N` (a
    run's feed lines from index N on) and `/api/job?after=N` (what the rollout job did: its groups, updates and
    engines). It only reads the directory; the runs' own process writes it."""
    reader = FeedReader(directory)

    async def page(request: Request) -> Response:
        return HTMLResponse(PAGE.read_text())

    async def runs(request: Request) -> Response:
        return JSONResponse(reader.runs())

    async def run(request: Request) -> Response:
        run_id = request.path_params["run_id"]
        after = int(request.query_params.get("after", "0"))
        return JSONResponse({"run_id": run_id, "lines": reader.lines(run_id, after)})

    async def job(request: Request) -> Response:
        return JSONResponse({"lines": reader.job(int(request.query_params.get("after", "0")))})

    routes = [Route("/", page), Route("/api/runs", runs), Route("/api/runs/{run_id}", run), Route("/api/job", job)]
    return Starlette(routes=routes)
