"""The monitor's web page over a ledger and every run in it: `rollout monitor WHERE [--port 8765]`."""

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from rollout_train.layout import LEDGER
from rollout_train.ledger import FENCES, LOCATION, FileLedger
from rollout_train.monitor.system import RELAYED, System

PAGE = Path(__file__).with_name("page.html")
SCRIPT = Path(__file__).with_name("monitor.js")


def watched(where: str | Path) -> System:
    """What a monitor over `where` reads: a database's URL (`sqlite:///…`, `postgresql://…`) or a ledger's directory of
    files, every run in it; or a run's directory (`rollout_train.layout`), its ledger and every run that shares it,
    with the directory's own logs and feed."""
    if "://" in str(where):
        from rollout_train.database import DatabaseLedger

        return System(ledger=DatabaseLedger(str(where)))
    path = Path(where).expanduser()
    if (path / FENCES).exists() and not (path / LOCATION).exists() and not (path / LEDGER).is_dir():
        return System(ledger=FileLedger(path))
    return System(path)


def create_app(where: str | Path) -> Starlette:
    """Serves the page and what it asks for, over a ledger or a run's directory (`watched`):

    - `/api/system`: where every run stands (`System.snapshot`);
    - `/api/groups/{run}/{number}`: one group, its episodes, its step and its outcome (`System.group`);
    - `/api/episodes/{run_id}?after=N`: one episode's lines from index N on (its rollouts, one per model slot), and
      what it reported (`System.episode`);
    - `/api/policies?sample=1`: the policies as a graph, with the trainers, the inference workers and evaluations
      (`System.lineage`; `sample` adds the fixture of the tables proposed for them);
    - `/api/statistics`: every run of the ledger in figures, the engines' throughput and the machine
      (`System.statistics`);
    - `/api/runs`: every episode in the runs' feeds, summarised.

    A run whose directory is on another machine has its episodes asked of the monitor its start names
    (`System._source`); what one monitor asks another, the other answers from its own machine (`RELAYED`). It only
    reads; the runs' own processes write."""
    system = watched(where)

    async def page(request: Request) -> Response:
        return HTMLResponse(await asyncio.to_thread(PAGE.read_text))

    async def script(request: Request) -> Response:
        return Response(await asyncio.to_thread(SCRIPT.read_text), media_type="text/javascript")

    async def state(request: Request) -> Response:
        return JSONResponse(await system.snapshot(relayed=RELAYED in request.headers))

    async def group(request: Request) -> Response:
        relayed = RELAYED in request.headers
        found = await system.group(request.path_params["run"], int(request.path_params["number"]), relayed)
        return JSONResponse(found) if found is not None else JSONResponse({"error": "no such group"}, status_code=404)

    async def episode(request: Request) -> Response:
        after = int(request.query_params.get("after", "0"))
        return JSONResponse(await system.episode(request.path_params["run_id"], after, RELAYED in request.headers))

    async def policies(request: Request) -> Response:
        return JSONResponse(await system.lineage(sample=request.query_params.get("sample") in ("1", "true")))

    async def figures(request: Request) -> Response:
        return JSONResponse(await system.statistics())

    async def runs(request: Request) -> Response:
        return JSONResponse(await asyncio.to_thread(system.feeds, RELAYED in request.headers))

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
        Route("/api/policies", policies),
        Route("/api/statistics", figures),
        Route("/api/runs", runs),
    ]
    return Starlette(routes=routes, lifespan=measuring)
