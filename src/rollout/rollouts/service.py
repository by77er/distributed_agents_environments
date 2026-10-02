"""Rollout jobs for a caller on another machine: the same `Jobs`, `Job` and `Ticket`, over HTTP.

`create_app(jobs)` serves jobs that run where the runner and the recorder are; `RolloutClient(url)` is what a
trainer elsewhere holds. Nothing in a training loop written against `Jobs` says which one it has.

    POST /jobs                                   start a job                    → {"job": id}
    POST /jobs/{job}/runs                        queue runs of a row            → {"ticket": id}
    GET  /jobs/{job}/tickets/{ticket}?wait=S     its episodes, once all ended   → {"done", "episodes", "refused"}
    GET  /jobs/{job}/episodes?cursor=N&wait=S    episodes after a cursor        → {"episodes", "closed"}
    POST /jobs/{job}/acknowledge | publish | notes
    GET  /jobs/{job}/status

Reads wait up to `wait` seconds for news and then answer with what there is, so a caller polls without spinning.
"""

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import asdict
from typing import Any

import httpx
from pydantic import JsonValue
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from rollout.core.harness.runner import ProgramReference, RunBinding
from rollout.rollouts.episodes import Episode
from rollout.rollouts.jobs import Refused, RolloutJob, RolloutJobs, Status

WAIT_SECONDS = 20.0


def create_app(jobs: RolloutJobs) -> Starlette:
    def job_of(request: Request) -> RolloutJob:
        return jobs.job(request.path_params["job"])

    async def start(request: Request) -> Response:
        body = await request.json()
        job = await jobs.start(
            program=ProgramReference.model_validate(body["program"]),
            binding=RunBinding.model_validate(body["binding"]),
            in_flight=int(body["in_flight"]),
            name=str(body.get("name", "")),
        )
        return JSONResponse({"job": job.id})

    async def run(request: Request) -> Response:
        body = await request.json()
        ticket = await job_of(request).run(body["parameters"], labels=body.get("labels"), count=int(body["count"]))
        return JSONResponse({"ticket": ticket.id})

    async def ticket(request: Request) -> Response:
        held = job_of(request).ticket(request.path_params["ticket"])
        if not await held.ready(float(request.query_params.get("wait", WAIT_SECONDS))):
            return JSONResponse({"done": False, "episodes": [], "refused": None})
        return JSONResponse({"done": True, "episodes": [e.to_json() for e in held.ended], "refused": held.refused})

    async def episodes(request: Request) -> Response:
        cursor = int(request.query_params.get("cursor", "0"))
        stream = job_of(request).episodes(cursor)
        ready: list[dict[str, Any]] = []
        closed = False
        try:
            first = await asyncio.wait_for(anext(stream), float(request.query_params.get("wait", WAIT_SECONDS)))
            ready = [first.to_json()] + [e.to_json() for e in job_of(request).after(first.cursor)]
        except TimeoutError:
            pass
        except StopAsyncIteration:
            closed = True
        return JSONResponse({"episodes": ready, "closed": closed})

    async def acknowledge(request: Request) -> Response:
        await job_of(request).acknowledge(int((await request.json())["cursor"]))
        return JSONResponse({})

    async def publish(request: Request) -> Response:
        body = await request.json()
        version = await job_of(request).publish(body["channel"], body["adapter"], body["path"])
        return JSONResponse({"version": version})

    async def note(request: Request) -> Response:
        body = await request.json()
        await job_of(request).note(body["kind"], body["payload"])
        return JSONResponse({})

    async def status(request: Request) -> Response:
        return JSONResponse(asdict(await job_of(request).status()))

    return Starlette(
        routes=[
            Route("/jobs", start, methods=["POST"]),
            Route("/jobs/{job}/runs", run, methods=["POST"]),
            Route("/jobs/{job}/tickets/{ticket}", ticket),
            Route("/jobs/{job}/episodes", episodes),
            Route("/jobs/{job}/acknowledge", acknowledge, methods=["POST"]),
            Route("/jobs/{job}/publish", publish, methods=["POST"]),
            Route("/jobs/{job}/notes", note, methods=["POST"]),
            Route("/jobs/{job}/status", status),
        ]
    )


class RolloutClient:
    """`Jobs`, served at `url`."""

    def __init__(self, url: str, *, client: httpx.AsyncClient | None = None) -> None:
        self._http = client or httpx.AsyncClient(base_url=url, timeout=WAIT_SECONDS + 30)

    async def start(
        self, *, program: ProgramReference, binding: RunBinding, in_flight: int, name: str = ""
    ) -> "RemoteJob":
        body = {
            "program": program.model_dump(mode="json"),
            "binding": binding.model_dump(mode="json"),
            "in_flight": in_flight,
            "name": name,
        }
        return RemoteJob((await _post(self._http, "/jobs", body))["job"], self._http)

    async def close(self) -> None:
        await self._http.aclose()


class RemoteJob:
    def __init__(self, job_id: str, http: httpx.AsyncClient) -> None:
        self.id = job_id
        self._http = http

    async def run(
        self, parameters: JsonValue, *, labels: Mapping[str, str] | None = None, count: int = 1
    ) -> "RemoteTicket":
        body = {"parameters": parameters, "labels": dict(labels or {}), "count": count}
        return RemoteTicket((await _post(self._http, f"/jobs/{self.id}/runs", body))["ticket"], self.id, self._http)

    async def episodes(self, cursor: int = 0) -> AsyncIterator[Episode]:
        while True:
            answer = await _get(self._http, f"/jobs/{self.id}/episodes", cursor=cursor)
            for data in answer["episodes"]:
                episode = Episode.from_json(data)
                cursor = episode.cursor
                yield episode
            if answer["closed"]:
                return

    async def acknowledge(self, cursor: int) -> None:
        await _post(self._http, f"/jobs/{self.id}/acknowledge", {"cursor": cursor})

    async def publish(self, channel: str, adapter: str, path: str) -> int:
        body = {"channel": channel, "adapter": adapter, "path": path}
        return int((await _post(self._http, f"/jobs/{self.id}/publish", body))["version"])

    async def note(self, kind: str, payload: Mapping[str, JsonValue]) -> None:
        await _post(self._http, f"/jobs/{self.id}/notes", {"kind": kind, "payload": dict(payload)})

    async def status(self) -> Status:
        return Status(**await _get(self._http, f"/jobs/{self.id}/status"))


class RemoteTicket:
    def __init__(self, ticket_id: str, job: str, http: httpx.AsyncClient) -> None:
        self.id = ticket_id
        self._job = job
        self._http = http

    async def episodes(self) -> list[Episode]:
        while True:
            answer = await _get(self._http, f"/jobs/{self._job}/tickets/{self.id}")
            if answer["done"]:
                if answer["refused"] is not None:
                    raise Refused(answer["refused"])
                return [Episode.from_json(data) for data in answer["episodes"]]


async def _post(http: httpx.AsyncClient, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
    response = await http.post(path, json=dict(body))
    response.raise_for_status()
    return response.json()


async def _get(http: httpx.AsyncClient, path: str, **query: Any) -> dict[str, Any]:
    response = await http.get(path, params=query)
    response.raise_for_status()
    return response.json()
