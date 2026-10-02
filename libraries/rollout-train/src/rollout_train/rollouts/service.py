"""Rollout jobs for a caller on another machine: the same `Jobs`, `Job` and `Ticket`, over HTTP.

`create_app(jobs)` serves jobs that run where the runner and the recorder are; `RolloutClient(url, blobs)` is what
a trainer elsewhere holds. Nothing in a training loop written against `Jobs` says which one it has. Episodes cross
as their records; their traces are read from the blob store both sides share.

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

from rollout.harness.blobs import Blobs
from rollout.harness.runner import ProgramReference, RunBinding
from rollout_train.rollouts.episodes import Episode, Record, loaded
from rollout_train.rollouts.jobs import Refused, RolloutJob, RolloutJobs, Status

WAIT_SECONDS = 20.0


def create_app(jobs: RolloutJobs) -> Starlette:
    if jobs.blobs is None:
        raise ValueError("jobs served over HTTP keep their episodes in a blob store: give RolloutJobs a log")

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
        ticket = await job_of(request).run(
            body["parameters"], labels=body.get("labels"), count=int(body["count"]), key=str(body.get("key", ""))
        )
        return JSONResponse({"ticket": ticket.id})

    async def ticket(request: Request) -> Response:
        held = job_of(request).ticket(request.path_params["ticket"])
        if not await held.ready(float(request.query_params.get("wait", WAIT_SECONDS))):
            return JSONResponse({"done": False, "episodes": [], "refused": None})
        records = [record.to_json() for record in held.ended]
        return JSONResponse({"done": True, "episodes": records, "refused": held.refused})

    async def episodes(request: Request) -> Response:
        job, cursor = job_of(request), int(request.query_params.get("cursor", "0"))
        closed = not await job.news(cursor, float(request.query_params.get("wait", WAIT_SECONDS)))
        return JSONResponse({"episodes": [record.to_json() for record in job.after(cursor)], "closed": closed})

    async def acknowledge(request: Request) -> Response:
        await job_of(request).acknowledge(int((await request.json())["cursor"]))
        return JSONResponse({})

    async def publish(request: Request) -> Response:
        body = await request.json()
        version = await job_of(request).publish(body["channel"], body["adapter"], body["path"], body.get("version"))
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
    """`Jobs`, served at `url`. `blobs` is the store the served jobs keep their episodes in."""

    def __init__(self, url: str, blobs: Blobs, *, client: httpx.AsyncClient | None = None) -> None:
        self._http = client or httpx.AsyncClient(base_url=url, timeout=WAIT_SECONDS + 30)
        self._blobs = blobs

    async def start(
        self, *, program: ProgramReference, binding: RunBinding, in_flight: int, name: str = ""
    ) -> "RemoteJob":
        body = {
            "program": program.model_dump(mode="json"),
            "binding": binding.model_dump(mode="json"),
            "in_flight": in_flight,
            "name": name,
        }
        return RemoteJob((await _post(self._http, "/jobs", body))["job"], self._http, self._blobs)

    async def close(self) -> None:
        await self._http.aclose()


class RemoteJob:
    def __init__(self, job_id: str, http: httpx.AsyncClient, blobs: Blobs) -> None:
        self.id = job_id
        self._http = http
        self._blobs = blobs

    async def run(
        self, parameters: JsonValue, *, labels: Mapping[str, str] | None = None, count: int = 1, key: str = ""
    ) -> "RemoteTicket":
        body = {"parameters": parameters, "labels": dict(labels or {}), "count": count, "key": key}
        ticket = (await _post(self._http, f"/jobs/{self.id}/runs", body))["ticket"]
        return RemoteTicket(ticket, self.id, self._http, self._blobs)

    async def episodes(self, cursor: int = 0) -> AsyncIterator[Episode]:
        while True:
            answer = await _get(self._http, f"/jobs/{self.id}/episodes", cursor=cursor)
            for data in answer["episodes"]:
                episode = await loaded(Record.from_json(data), self._blobs)
                cursor = episode.cursor
                yield episode
            if answer["closed"]:
                return

    async def acknowledge(self, cursor: int) -> None:
        await _post(self._http, f"/jobs/{self.id}/acknowledge", {"cursor": cursor})

    async def publish(self, channel: str, adapter: str, path: str, version: int | None = None) -> int:
        body = {"channel": channel, "adapter": adapter, "path": path, "version": version}
        return int((await _post(self._http, f"/jobs/{self.id}/publish", body))["version"])

    async def note(self, kind: str, payload: Mapping[str, JsonValue]) -> None:
        await _post(self._http, f"/jobs/{self.id}/notes", {"kind": kind, "payload": dict(payload)})

    async def status(self) -> Status:
        return Status(**await _get(self._http, f"/jobs/{self.id}/status"))


class RemoteTicket:
    def __init__(self, ticket_id: str, job: str, http: httpx.AsyncClient, blobs: Blobs) -> None:
        self.id = ticket_id
        self._job = job
        self._http = http
        self._blobs = blobs

    async def episodes(self) -> list[Episode]:
        while True:
            answer = await _get(self._http, f"/jobs/{self._job}/tickets/{self.id}")
            if answer["done"]:
                if answer["refused"] is not None:
                    raise Refused(answer["refused"])
                records = [Record.from_json(data) for data in answer["episodes"]]
                return list(await asyncio.gather(*(loaded(record, self._blobs) for record in records)))


async def _post(http: httpx.AsyncClient, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
    response = await http.post(path, json=dict(body))
    response.raise_for_status()
    return response.json()


async def _get(http: httpx.AsyncClient, path: str, **query: Any) -> dict[str, Any]:
    response = await http.get(path, params=query)
    response.raise_for_status()
    return response.json()
