"""The monitor's web page over a ledger and every run in it: `rollout monitor WHERE [--cluster] [--port 8765]`."""

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from rollout_train.launching import Refused as LaunchRefused
from rollout_train.layout import LEDGER
from rollout_train.ledger import FENCES, LOCATION, FileLedger
from rollout_train.monitor.stream import BEAT, MISSING, Hub, Reading
from rollout_train.monitor.system import RELAYED, System
from rollout_train.publishing import Importer, Refused
from rollout_train.registry import Taken

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rollout_train.cluster import Cluster
    from rollout_train.submitting import Backend

STATIC = Path(__file__).with_name("static")
"""The page, built from libraries/rollout-train/web (`npm run build` there writes it here)."""
KEEPALIVE = 15.0
"""Seconds between the stream's keep-alive comments while nothing changes."""


def watched(
    where: str | Path,
    importer: Importer | None = None,
    cluster: "Cluster | None" = None,
    backends: "Mapping[str, Backend] | None" = None,
) -> System:
    """What a monitor over `where` reads: a database's URL (`sqlite:///…`, `postgresql://…`), the ledger service's
    (`https://…`, with the cluster config's token), or a ledger's directory of files, every run in it; or a run's
    directory (`rollout_train.layout`), its ledger and every run that shares it, with the directory's own logs and
    feed. `importer`: where environments imported from git go. `cluster`: the
    cluster config runs are asked for on (none: this monitor asks for none), and `backends` where their jobs go in
    place of its own (a test's)."""
    if "://" in str(where):
        from rollout_train.stores import opened_ledger

        ledger = opened_ledger(str(where), cluster.ledger.token if cluster is not None else None)
        return System(ledger=ledger, importer=importer, cluster=cluster, backends=backends)
    path = Path(where).expanduser()
    if (path / FENCES).exists() and not (path / LOCATION).exists() and not (path / LEDGER).is_dir():
        return System(ledger=FileLedger(path), importer=importer, cluster=cluster, backends=backends)
    return System(path, importer=importer, cluster=cluster, backends=backends)


def create_app(
    where: str | Path,
    *,
    beat: float = BEAT,
    importer: Importer | None = None,
    cluster: "Cluster | None" = None,
    backends: "Mapping[str, Backend] | None" = None,
) -> Starlette:
    """Serves the page and what it asks for, over a ledger or a run's directory (`watched`):

    - `/`: the page (`STATIC`), and `/assets/...` its scripts and styles;
    - `/api/system`: where every run stands (`System.snapshot`);
    - `/api/machines`: every machine that beats and the roles on it, what each holds and how full it is
      (`System.machines`);
    - `/api/queue`: how the runs share what the cluster gives them: its capacity and what is used, the runs admitted
      and what each holds, and the runs that wait, in the queue's order, with why (`System.queue`);
    - `/api/evals`: the suites, their versions and the evals that played them (`System.evals`); `POST
      /api/suites/{name}` makes a suite or its next version, which its name then points to (`System.save_suite`);
    - `/api/evals/subjects`: every subject (a checkpoint or a base model) that has had an eval
      (`System.eval_subjects`); `/api/evals/checkpoint/{id}` and `/api/evals/model/{name}` a subject's history, every
      eval it has had (`System.history`);
    - `/api/environments`: every environment the system knows of, with the versions seen, whether the cluster
      offers it, its training runs and suites, and when a run last started on it (`System.environments`); and
      `/api/environments/{name}` one environment's page: its rows, eval data and curriculum where it loads here, what
      was played of each row, its runs, suites, evals and newest check (`System.environment`; 404 for one neither
      known nor loading);
    - `/api/environments/versions`: the published environments' versions (`System.environment_versions`), and
      `/api/environments/versions/{version}` one, by its id or `NAME@VERSION` (`System.environment_version`); `POST
      /api/environments/import` (`{"url", "ref", "subdirectory", "entry_point"}`) imports one from git, the monitor's
      importer (`importer`, from its cluster config) making it, and answers with the version once it is recorded, or
      422 with why it was refused (`System.import_environment`); `/api/environments/imports` the imports this monitor
      made, each with its stage (`System.imports`);
    - `/api/offers`: what a run can be asked for on the monitor's cluster: environments, trainers, inference
      providers and their models, the pairs that bridge, sandbox pools, presets and free GPUs (`System.offers`);
    - `/api/launches`: the runs asked for, each with its job and state (GET); `POST` (`{"kind", "name",
      "environment", "settings", "preset"}`) asks for a run and starts its job (`System.launch`), answering 422 with
      the refusals (each with the setting it is about) where its settings are refused; `POST /api/launches/check`
      says the refusals and notes without asking, with what the run trains, one step's estimated spend and its
      environment's slots (`System.check`); `POST /api/launches/{id}/stop` stops one;
    - `/api/presets`: every preset's newest version (`System.presets`); `/api/presets/{name}` one preset's versions
      (`System.preset`); `POST /api/presets/{name}` (`{"settings", "note"}`) saves its next version
      (`System.save_preset`), and `DELETE` deletes it (`System.delete_preset`);
    - `/api/groups/{run}/{number}`: one group, its episodes, its step and its outcome (`System.group`);
    - `/api/episodes/{run_id}?after=N`: one episode's lines from index N on (its rollouts, one per model slot), and
      what it reported (`System.episode`);
    - `/api/checkpoints`: the checkpoints as a graph from their base models, with the runs' trainers, their engines
      and evaluations (`System.lineage`);
    - `/api/checkpoints/{id}/evals`: every eval a checkpoint has had (`System.checkpoint_evals`), and
      `/api/checkpoints/{id}/path` its line from the base model with each point's scores (`System.path`);
    - `/api/runs/{run}/settings`: a training run's settings, fixed and changeable, and what is wanted of them
      (`System.settings`); `POST` (`{"settings": {KEY: VALUE}}`) wants changeable ones from its next step on
      (`System.want`);
    - `POST /api/runs/{run}/pause`: pauses a run (`System.pause`); `POST /api/runs/{run}/resume` resumes it, in place
      while its process beats, else by a launch of its recorded settings (`{"preset"}` for a run whose start has none:
      `System.resume`);
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
    system = watched(where, importer, cluster, backends)
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

    async def queue(request: Request) -> Response:
        return answered(request, await hub.read("queue"))

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
        return answered(request, await hub.read("checkpoints"))

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
        except LaunchRefused as error:
            refusals = [asdict(each) for each in error.refusals]
            notes = [asdict(each) for each in error.findings if not each.refuses]
            return JSONResponse({"error": str(error), "refusals": refusals, "notes": notes}, status_code=422)
        except Refused as error:
            return JSONResponse({"error": str(error)}, status_code=422)
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

    async def eval_subjects(request: Request) -> Response:
        return answered(request, await hub.read("eval-subjects"))

    async def history(request: Request) -> Response:
        kind, reference = request.path_params["kind"], request.path_params["reference"]
        return answered(request, await hub.read(f"history/{kind}/{reference}"))

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

    async def environments(request: Request) -> Response:
        return answered(request, await hub.read("environments"))

    async def environment(request: Request) -> Response:
        return answered(request, await hub.read(f"environment/{request.path_params['name']}"))

    async def environment_versions(request: Request) -> Response:
        return answered(request, await hub.read("environment-versions"))

    async def environment_version(request: Request) -> Response:
        return answered(request, await hub.read(f"environment-version/{request.path_params['version']}"))

    async def imports(request: Request) -> Response:
        return answered(request, await hub.read("imports"))

    async def import_environment(request: Request) -> Response:
        try:
            body: Any = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return JSONResponse({"error": "say what to import, as JSON: url, ref, subdirectory, entry_point"}, 400)

        async def change() -> Any:
            return await system.import_environment(cast(dict[str, Any], body))

        hub.forget()  # (its stages are read as it goes)
        return await written(change)

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
            return await system.launch(cast(dict[str, Any], body))

        return await written(change)

    async def check(request: Request) -> Response:
        try:
            body: Any = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return JSONResponse({"error": "say the run to check, as JSON"}, status_code=400)
        try:
            return JSONResponse(await system.check(cast(dict[str, Any], body)))
        except Taken as error:
            return JSONResponse({"error": str(error)}, status_code=409)
        except KeyError as error:
            return JSONResponse({"error": str(error.args[0]) if error.args else "no such thing"}, status_code=404)

    async def offered(request: Request) -> Response:
        return answered(request, await hub.read("offers"))

    async def presets(request: Request) -> Response:
        return answered(request, await hub.read("presets"))

    async def preset(request: Request) -> Response:
        name = request.path_params["name"]
        if request.method == "GET":
            return answered(request, await hub.read(f"preset/{name}"))
        if request.method == "DELETE":

            async def deleted() -> Any:
                return await system.delete_preset(name)

            return await written(deleted)
        try:
            body: Any = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return JSONResponse({"error": "say the preset's settings, as JSON"}, status_code=400)

        async def saved() -> Any:
            return await system.save_preset(name, cast(dict[str, Any], body))

        return await written(saved)

    async def stop(request: Request) -> Response:
        id = request.path_params["id"]

        async def change() -> Any:
            return {"launch": asdict(await system.stop(id))}

        return await written(change)

    async def pause(request: Request) -> Response:
        run = request.path_params["run"]

        async def change() -> Any:
            return {"desired": asdict(await system.pause(run))}

        return await written(change)

    async def resume(request: Request) -> Response:
        run = request.path_params["run"]
        try:
            body: Any = await request.json()
        except ValueError:
            body = None
        said = cast(dict[str, Any], body) if isinstance(body, dict) else {}
        preset = str(said["preset"]) if said.get("preset") else None

        async def change() -> Any:
            return {"resumed": asdict(await system.resume(run, preset))}

        return await written(change)

    async def unbookmark(request: Request) -> Response:
        name = request.path_params["name"]

        async def change() -> Any:
            await system.unbookmark(name)
            return {"unbookmarked": name}

        return await written(change)

    @contextlib.asynccontextmanager
    async def measuring(app: Starlette) -> AsyncGenerator[None]:
        reading = asyncio.create_task(hub.run())
        try:
            yield
        finally:
            reading.cancel()

    routes = [
        Route("/", page),
        Route("/favicon.svg", icon),
        Route("/api/system", state),
        Route("/api/machines", machines),
        Route("/api/queue", queue),
        Route("/api/offers", offered),
        Route("/api/launches", launches, methods=["GET", "POST"]),
        Route("/api/launches/check", check, methods=["POST"]),
        Route("/api/presets", presets),
        Route("/api/presets/{name}", preset, methods=["GET", "POST", "DELETE"]),
        Route("/api/evals", evals),
        Route("/api/evals/subjects", eval_subjects),
        Route("/api/evals/{kind:str}/{reference:path}", history),
        Route("/api/suites/{name}", suite, methods=["POST"]),
        Route("/api/environments", environments),
        Route("/api/environments/import", import_environment, methods=["POST"]),
        Route("/api/environments/imports", imports),
        Route("/api/environments/versions", environment_versions),
        Route("/api/environments/versions/{version}", environment_version),
        Route("/api/environments/{name}", environment),
        Route("/api/launches/{id}/stop", stop, methods=["POST"]),
        Route("/api/groups/{run}/{number:int}", group),
        Route("/api/episodes/{run_id}", episode),
        Route("/api/checkpoints", checkpoints),
        Route("/api/checkpoints/{id}/evals", checkpoint_evals),
        Route("/api/checkpoints/{id}/path", path),
        Route("/api/runs/{run}/settings", settings, methods=["GET", "POST"]),
        Route("/api/runs/{run}/pause", pause, methods=["POST"]),
        Route("/api/runs/{run}/resume", resume, methods=["POST"]),
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
