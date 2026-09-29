"""The project assistant's HTTP API.

POST /conversations/{key}/messages   {"text", "priority"?, "idempotency_key"?, "wait"?, "timeout_seconds"?}
GET  /conversations/{key}/transcript
GET  /conversations/{key}/events?from_seq=0     server-sent events of the live run
POST /conversations/{key}/cancel
GET  /health
"""

import asyncio
from dataclasses import asdict

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from project_assistant.service import AssistantService
from rollout.core.harness import Priority


def create_app(service: AssistantService) -> Starlette:
    async def send_message(request: Request) -> Response:
        conversation = request.path_params["key"]
        body = await request.json()
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return JSONResponse({"error": "text is required"}, status_code=400)
        try:
            priority = Priority(body.get("priority", "normal"))
        except ValueError:
            return JSONResponse({"error": "priority must be low, normal or high"}, status_code=400)
        message_id, run = await service.send(
            conversation, text, priority=priority, idempotency_key=body.get("idempotency_key")
        )
        accepted = {"conversation": conversation, "message_id": message_id, "run_id": run.run_id}
        if not body.get("wait", True):
            return JSONResponse(accepted, status_code=202)
        try:
            async with asyncio.timeout(float(body.get("timeout_seconds", 300))):
                reply = await service.reply_to(run, message_id)
        except TimeoutError:
            return JSONResponse({**accepted, "error": "no reply before the timeout"}, status_code=504)
        if reply is None:
            outcome = run.outcome
            detail = outcome.detail if outcome else None
            return JSONResponse(
                {**accepted, "error": "the run ended without replying", "detail": detail}, status_code=500
            )
        return JSONResponse({**accepted, "reply": reply})

    async def transcript(request: Request) -> Response:
        entries = service.transcript(request.path_params["key"])
        return JSONResponse({"messages": [asdict(entry) for entry in entries]})

    async def events(request: Request) -> Response:
        conversation = request.path_params["key"]
        from_seq = int(request.query_params.get("from_seq", "0"))

        async def stream():
            async for event in service.events(conversation, from_seq=from_seq):
                yield f"event: {event.type.value}\ndata: {event.model_dump_json()}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    async def cancel(request: Request) -> Response:
        cancelled = await service.cancel(request.path_params["key"])
        return JSONResponse({"cancelled": cancelled})

    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok", "deployment": service.deployment_name})

    return Starlette(
        routes=[
            Route("/conversations/{key}/messages", send_message, methods=["POST"]),
            Route("/conversations/{key}/transcript", transcript),
            Route("/conversations/{key}/events", events),
            Route("/conversations/{key}/cancel", cancel, methods=["POST"]),
            Route("/health", health),
        ]
    )
