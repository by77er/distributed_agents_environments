"""The monitor's web page over a feed directory: `python -m rollout.monitor DIRECTORY [--port 8765]`."""

import argparse
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from rollout.monitor.feed import FeedReader

PAGE = Path(__file__).with_name("page.html")


def create_app(directory: Path) -> Starlette:
    """Serves the page and what it asks for: `/api/runs` (every run, summarised) and `/api/runs/{run_id}?after=N` (a
    run's feed lines from index N on). It only reads the directory; the runs' own process writes it."""
    reader = FeedReader(directory)

    async def page(request: Request) -> Response:
        return HTMLResponse(PAGE.read_text())

    async def runs(request: Request) -> Response:
        return JSONResponse(reader.runs())

    async def run(request: Request) -> Response:
        run_id = request.path_params["run_id"]
        after = int(request.query_params.get("after", "0"))
        return JSONResponse({"run_id": run_id, "lines": reader.lines(run_id, after)})

    return Starlette(routes=[Route("/", page), Route("/api/runs", runs), Route("/api/runs/{run_id}", run)])


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="rollout-monitor", description="Watch runs as they happen.")
    parser.add_argument("directory", type=Path, help="the feed directory a RunFeed writes")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    arguments = parser.parse_args()
    uvicorn.run(create_app(arguments.directory), host=arguments.host, port=arguments.port, log_level="warning")
