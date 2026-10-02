"""The agent sessions HTTP API, for the `agents` CLI and anything else.

GET  /sessions                           every session, with status
POST /sessions                           {"name", "instructions"}: start a session
GET  /sessions/{name}/activity           messages, replies and tool calls, in order
GET  /sessions/{name}/stream             the same, as server-sent events, following new activity
POST /sessions/{name}/messages           {"text", "urgent"?}: message a session
POST /sessions/{name}/stop               stop a session and destroy its environment
GET  /board?channel=&status=             posts, newest first
POST /board                              {"channel", "title", "body", "kind"?}: post as the operator
GET  /inbox                              messages sessions sent to the operator
"""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import asdict

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from agent_sessions.service import SessionsService


def create_app(service: SessionsService) -> Starlette:
    async def sessions(request: Request) -> Response:
        return JSONResponse({"sessions": [asdict(session) for session in service.sessions()]})

    async def create(request: Request) -> Response:
        body = await request.json()
        try:
            await service.create(str(body["name"]), str(body["instructions"]))
        except (KeyError, ValueError) as error:
            return JSONResponse({"error": str(error)}, status_code=400)
        return JSONResponse({"created": body["name"]}, status_code=201)

    async def activity(request: Request) -> Response:
        return JSONResponse({"activity": service.activity(request.path_params["name"])})

    async def stream(request: Request) -> Response:
        name = request.path_params["name"]

        async def follow() -> AsyncIterator[str]:
            seen = 0
            while True:
                entries = service.activity(name)
                for entry in entries[seen:]:
                    yield f"data: {json.dumps(entry)}\n\n"
                seen = len(entries)
                await asyncio.sleep(0.5)

        return StreamingResponse(follow(), media_type="text/event-stream")

    async def message(request: Request) -> Response:
        body = await request.json()
        try:
            message_id = await service.send(
                request.path_params["name"], str(body["text"]), urgent=bool(body.get("urgent", False))
            )
        except KeyError:
            return JSONResponse({"error": "no such session"}, status_code=404)
        return JSONResponse({"message_id": message_id})

    async def stop(request: Request) -> Response:
        return JSONResponse({"stopped": await service.stop(request.path_params["name"])})

    async def board(request: Request) -> Response:
        posts = service.board(request.query_params.get("channel"), request.query_params.get("status"))
        return JSONResponse({"posts": [asdict(post) for post in posts]})

    async def post(request: Request) -> Response:
        body = await request.json()
        text = service.post(str(body["channel"]), str(body["title"]), str(body["body"]), str(body.get("kind", "note")))
        return JSONResponse({"result": text})

    async def inbox(request: Request) -> Response:
        return JSONResponse({"messages": service.inbox()})

    return Starlette(
        lifespan=lambda app: _Lifespan(service),
        routes=[
            Route("/sessions", sessions),
            Route("/sessions", create, methods=["POST"]),
            Route("/sessions/{name}/activity", activity),
            Route("/sessions/{name}/stream", stream),
            Route("/sessions/{name}/messages", message, methods=["POST"]),
            Route("/sessions/{name}/stop", stop, methods=["POST"]),
            Route("/board", board),
            Route("/board", post, methods=["POST"]),
            Route("/inbox", inbox),
        ],
    )


class _Lifespan:
    def __init__(self, service: SessionsService) -> None:
        self._service = service

    async def __aenter__(self) -> None:
        await self._service.start()

    async def __aexit__(self, *exception: object) -> None:
        await self._service.close()
