"""The monitor's web page over a ledger and every run in it: `rollout monitor WHERE [--port 8765]`."""

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from rollout_train.layout import LEDGER
from rollout_train.ledger import FENCES, LOCATION, FileLedger
from rollout_train.monitor.stream import BEAT, MISSING, Hub, Reading
from rollout_train.monitor.system import RELAYED, System
from rollout_train.registry import Taken

STATIC = Path(__file__).with_name("static")
"""The page, built from libraries/rollout-train/web (`npm run build` there writes it here)."""
KEEPALIVE = 15.0
"""Seconds between the stream's keep-alive comments while nothing changes."""


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


def create_app(where: str | Path, *, beat: float = BEAT) -> Starlette:
    """Serves the page and what it asks for, over a ledger or a run's directory (`watched`):

    - `/`: the page (`STATIC`), and `/assets/...` its scripts and styles;
    - `/api/system`: where every run stands (`System.snapshot`);
    - `/api/machines`: every machine that beats and the roles on it, what each holds and how full it is
      (`System.machines`);
    - `/api/evals`: the suites, their versions and the evals that played them (`System.evals`); `POST
      /api/suites/{name}` makes a suite or its next version, which its name then points to (`System.save_suite`), and
      `/api/environments/{name}` says what the suites' forms need of an environment (`System.environment`);
    - `/api/launches`: the runs asked for and the launchers alive (GET); `POST` asks for a run or an eval
      (`System.launch`), `POST /api/launches/{id}/stop` stops one;
    - `/api/groups/{run}/{number}`: one group, its episodes, its step and its outcome (`System.group`);
    - `/api/episodes/{run_id}?after=N`: one episode's lines from index N on (its rollouts, one per model slot), and
      what it reported (`System.episode`);
    - `/api/checkpoints?sample=1`: the checkpoints as a graph from their base models, with the trainers, the inference
      workers and evaluations (`System.lineage`; `sample` adds the fixture of the tables proposed for them);
    - `/api/checkpoints/{id}/evals`: every eval a checkpoint has had (`System.checkpoint_evals`), and
      `/api/checkpoints/{id}/path` its line from the base model with each point's scores (`System.path`);
    - `/api/runs/{run}/settings`: a training run's settings, fixed and changeable, and what is wanted of them
      (`System.settings`); `POST` (`{"settings": {KEY: VALUE}}`) wants changeable ones from its next step on
      (`System.want`);
    - `/api/statistics`: every run of the ledger in figures and the engines' throughput (`System.statistics`);
    - `/api/runs`: every episode in the runs' feeds, summarised;
    - `/api/stream?topic=...`: server-sent events, a `version` event (`{"topic", "version"}`) for each topic at once
      and then each time it changes (`rollout_train.monitor.stream`);
    - `POST /api/rename` (`{"id", "name"}`): calls a run by a new name (`System.rename`);
    - `POST /api/bookmarks` (`{"name", "checkpoint"}`): makes a bookmark name a checkpoint, or moves it there
      (`System.bookmark`); `DELETE /api/bookmarks/{name}` takes it away (`System.unbookmark`). Each answers 200 with
      what the registry now says, 409 for a name that cannot be one, 404 when what it names is not there.

    Each JSON answer carries its topic's version as its ETag: a request that names it (`If-None-Match`) is answered
    304, with nothing. A run whose directory is on another machine has its episodes asked of the monitor its start
    names (`System._source`); what one monitor asks another, the other answers from its own machine (`RELAYED`),
    directly and in full. It reads, and writes names (a run's, bookmarks, suites') and suites' versions; the runs' own
    processes write the rest."""
    system = watched(where)
    hub = Hub(system, beat)

    def answered(request: Request, reading: Reading) -> Response:
        if reading.version == MISSING:
            return JSONResponse({"error": "no such thing"}, status_code=404)
        etag = f'W/"{reading.version}"'
        headers = {"ETag": etag, "Cache-Control": "no-cache"}
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        return Response(reading.body, media_type="application/json", headers=headers)

    async def page(request: Request) -> Response:
        index = STATIC / "index.html"
        if not index.exists():
            return HTMLResponse(UNBUILT, status_code=503)
        return HTMLResponse(await asyncio.to_thread(index.read_text), headers={"Cache-Control": "no-cache"})

    async def icon(request: Request) -> Response:
        found = STATIC / "favicon.svg"
        if not found.exists():
            return Response(status_code=404)
        body = await asyncio.to_thread(found.read_bytes)
        return Response(body, media_type="image/svg+xml", headers={"Cache-Control": "max-age=86400"})

    async def state(request: Request) -> Response:
        if RELAYED in request.headers:
            return JSONResponse(await system.snapshot(relayed=True))
        return answered(request, await hub.read("system"))

    async def machines(request: Request) -> Response:
        return answered(request, await hub.read("machines"))

    async def group(request: Request) -> Response:
        run, number = request.path_params["run"], int(request.path_params["number"])
        if RELAYED in request.headers:
            found = await system.group(run, number, relayed=True)
            if found is None:
                return JSONResponse({"error": "no such group"}, status_code=404)
            return JSONResponse(found)
        return answered(request, await hub.read(f"group/{run}/{number}"))

    async def episode(request: Request) -> Response:
        after = int(request.query_params.get("after", "0"))
        return JSONResponse(await system.episode(request.path_params["run_id"], after, RELAYED in request.headers))

    async def checkpoints(request: Request) -> Response:
        sample = request.query_params.get("sample") in ("1", "true")
        return answered(request, await hub.read("checkpoints/sample" if sample else "checkpoints"))

    async def checkpoint_evals(request: Request) -> Response:
        return answered(request, await hub.read(f"checkpoint-evals/{request.path_params['id']}"))

    async def path(request: Request) -> Response:
        return answered(request, await hub.read(f"path/{request.path_params['id']}"))

    async def settings(request: Request) -> Response:
        run = request.path_params["run"]
        if request.method == "GET":
            return answered(request, await hub.read(f"settings/{run}"))
        try:
            body: Any = await request.json()
        except ValueError:
            body = None
        wanted: Any = cast(dict[str, Any], body).get("settings") if isinstance(body, dict) else None
        if not isinstance(wanted, dict):
            return JSONResponse({"error": 'say the settings wanted, as {"settings": {KEY: VALUE}}'}, status_code=400)

        async def change() -> Any:
            return {"desired": asdict(await system.want(run, cast(dict[str, Any], wanted)))}

        return await written(change)

    async def figures(request: Request) -> Response:
        return answered(request, await hub.read("statistics"))

    async def runs(request: Request) -> Response:
        if RELAYED in request.headers:
            return JSONResponse(await asyncio.to_thread(system.feeds, True))
        return answered(request, await hub.read("feeds"))

    async def stream(request: Request) -> Response:
        topics = request.query_params.getlist("topic") or ["system"]

        async def events() -> AsyncGenerator[str]:
            yield f"retry: 2000\nevent: hello\ndata: {json.dumps({'beat': hub.beat})}\n\n"
            versions = hub.watch(topics)
            waiting: asyncio.Future[tuple[str, str]] | None = None
            try:
                while True:
                    waiting = waiting or asyncio.ensure_future(anext(versions))
                    done, _ = await asyncio.wait({waiting}, timeout=KEEPALIVE)
                    if not done:
                        yield ": still here\n\n"
                        continue
                    topic, version = waiting.result()
                    waiting = None
                    yield f"event: version\ndata: {json.dumps({'topic': topic, 'version': version})}\n\n"
            finally:
                if waiting is not None:
                    waiting.cancel()
                    with contextlib.suppress(BaseException):
                        await waiting
                await versions.aclose()

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(events(), media_type="text/event-stream", headers=headers)

    async def written(change: Callable[[], Awaitable[Any]]) -> Response:
        """Make one change to the registry, and have every page read what it shows afresh."""
        try:
            done = await change()
        except Taken as error:
            return JSONResponse({"error": str(error)}, status_code=409)
        except KeyError as error:
            return JSONResponse({"error": str(error.args[0]) if error.args else "no such thing"}, status_code=404)
        hub.forget()
        return JSONResponse(done)

    async def asked(request: Request, *keys: str) -> list[str] | None:
        try:
            body: Any = await request.json()
            return [str(body[key]) for key in keys]
        except (ValueError, KeyError, TypeError):
            return None

    async def rename(request: Request) -> Response:
        said = await asked(request, "id", "name")
        if said is None:
            return JSONResponse({"error": "say the run's id and its new name"}, status_code=400)

        async def change() -> Any:
            return {"entry": asdict(await system.rename(*said))}

        return await written(change)

    async def bookmark(request: Request) -> Response:
        said = await asked(request, "name", "checkpoint")
        if said is None:
            return JSONResponse({"error": "say the bookmark's name and the checkpoint"}, status_code=400)

        async def change() -> Any:
            return {"bookmark": asdict(await system.bookmark(*said))}

        return await written(change)

    async def evals(request: Request) -> Response:
        return answered(request, await hub.read("evals"))

    async def suite(request: Request) -> Response:
        name = request.path_params["name"]
        try:
            body: Any = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return JSONResponse({"error": "say the suite, as JSON"}, status_code=400)

        async def change() -> Any:
            made = await system.save_suite(name, cast(dict[str, Any], body))
            return {"suite": made.name, "version": made.id, "number": made.number}

        return await written(change)

    async def environment(request: Request) -> Response:
        try:
            return JSONResponse(await system.environment(request.path_params["name"]))
        except KeyError as error:
            return JSONResponse({"error": str(error.args[0])}, status_code=404)

    async def launches(request: Request) -> Response:
        if request.method == "GET":
            return answered(request, await hub.read("launches"))
        try:
            body: Any = await request.json()
        except ValueError:
            return JSONResponse({"error": "say the run to launch, as JSON"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "say the run to launch, as JSON"}, status_code=400)

        async def change() -> Any:
            return {"launch": asdict(await system.launch(cast(dict[str, Any], body)))}

        return await written(change)

    async def stop(request: Request) -> Response:
        id = request.path_params["id"]

        async def change() -> Any:
            return {"launch": asdict(await system.stop(id))}

        return await written(change)

    async def unbookmark(request: Request) -> Response:
        name = request.path_params["name"]

        async def change() -> Any:
            await system.unbookmark(name)
            return {"unbookmarked": name}

        return await written(change)

    @contextlib.asynccontextmanager
    async def measuring(app: Starlette) -> AsyncGenerator[None]:
        tasks = [asyncio.create_task(hub.run())]
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()

    routes = [
        Route("/", page),
        Route("/favicon.svg", icon),
        Route("/api/system", state),
        Route("/api/machines", machines),
        Route("/api/launches", launches, methods=["GET", "POST"]),
        Route("/api/evals", evals),
        Route("/api/suites/{name}", suite, methods=["POST"]),
        Route("/api/environments/{name}", environment),
        Route("/api/launches/{id}/stop", stop, methods=["POST"]),
        Route("/api/groups/{run}/{number:int}", group),
        Route("/api/episodes/{run_id}", episode),
        Route("/api/checkpoints", checkpoints),
        Route("/api/checkpoints/{id}/evals", checkpoint_evals),
        Route("/api/checkpoints/{id}/path", path),
        Route("/api/runs/{run}/settings", settings, methods=["GET", "POST"]),
        Route("/api/statistics", figures),
        Route("/api/runs", runs),
        Route("/api/stream", stream),
        Route("/api/rename", rename, methods=["POST"]),
        Route("/api/bookmarks", bookmark, methods=["POST"]),
        Route("/api/bookmarks/{name}", unbookmark, methods=["DELETE"]),
        Mount("/assets", app=StaticFiles(directory=STATIC / "assets", check_dir=False), name="assets"),
    ]
    middleware = [Middleware(GZipMiddleware, minimum_size=1024)]
    return Starlette(routes=routes, middleware=middleware, lifespan=measuring)


UNBUILT = """<!doctype html><title>Rollout</title>
<p>The monitor's page is not built: run <code>npm ci &amp;&amp; npm run build</code> in
libraries/rollout-train/web.</p>"""
