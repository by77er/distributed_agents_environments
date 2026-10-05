"""A fake of RunPod's pods API, served in this process: the surface `rollout_runpod.RunPod` uses (create, list, get,
start, stop, delete), pods kept in memory, nothing rented and nothing spent.

Like RunPod's front, it refuses a request whose User-Agent it does not accept (403: Python's default one, or none) and
one without the key (401). A pod is created `RUNNING` with a public address and its price (or, with `addressed` false,
with none until a test says it with `address`, as RunPod says a pod's address only once it runs); `gpu` is the GPU type
it is given (the first asked for). `created` is called with each pod made (and its body), so a test can start a process
that stands in for what runs on it; `deleted` with each pod's id as it is deleted.
"""

import itertools
from collections.abc import Callable
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

KEY = "rpa_TESTKEY0123456789abcdefABCDEF"
ADDRESS = "203.0.113.7"


class FakeRunPod:
    """Pods in memory, behind RunPod's REST API at `/v1`."""

    def __init__(
        self,
        *,
        key: str = KEY,
        cost: float = 1.89,
        created: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
        deleted: Callable[[str], None] | None = None,
        addressed: bool = True,
    ) -> None:
        self.key = key
        self.addressed = addressed
        self.cost = cost
        self.pods: dict[str, dict[str, Any]] = {}
        self.asked: list[tuple[str, str, Any]] = []
        self.agents: list[str] = []
        self.created = created
        self.deleted = deleted
        self._ids = itertools.count(1)
        self._ports = itertools.count(40123)
        self.app = Starlette(
            routes=[
                Route("/v1/pods", self._collection, methods=["GET", "POST"]),
                Route("/v1/pods/{id}", self._one, methods=["GET", "DELETE"]),
                Route("/v1/pods/{id}/{verb}", self._action, methods=["POST"]),
            ]
        )

    def _refused(self, request: Request) -> Response | None:
        agent = request.headers.get("user-agent", "")
        self.agents.append(agent)
        if not agent.startswith("rollout/"):
            return JSONResponse({"error": "error code: 1010"}, status_code=403)
        if request.headers.get("authorization") != f"Bearer {self.key}":
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return None

    async def _collection(self, request: Request) -> Response:
        if (refusal := self._refused(request)) is not None:
            return refusal
        if request.method == "GET":
            self.asked.append(("GET", "/pods", dict(request.query_params)))
            name = request.query_params.get("name")
            return JSONResponse([pod for pod in self.pods.values() if name is None or pod["name"] == name])
        body = await request.json()
        self.asked.append(("POST", "/pods", body))
        id = f"pod{next(self._ids):011d}"
        gpus: list[str] = body.get("gpuTypeIds") or ["NVIDIA GeForce RTX 4090"]
        self.pods[id] = {
            "id": id, "name": body["name"], "image": body["imageName"], "desiredStatus": "RUNNING",
            "publicIp": ADDRESS if self.addressed else "",
            "portMappings": {"8443": next(self._ports)} if self.addressed else {}, "costPerHr": self.cost,
            "env": body["env"], "gpu": {"id": gpus[0], "count": body.get("gpuCount", 1)},
            "cloudType": body.get("cloudType"), "dataCenterIds": body.get("dataCenterIds"),
        }  # fmt: skip
        if self.created is not None:
            self.created(self.pods[id], body)
        return JSONResponse(self.pods[id], status_code=201)

    async def _one(self, request: Request) -> Response:
        if (refusal := self._refused(request)) is not None:
            return refusal
        id = request.path_params["id"]
        self.asked.append((request.method, f"/pods/{id}", None))
        if id not in self.pods:
            return JSONResponse({"error": "pod not found"}, status_code=400)
        if request.method == "DELETE":
            del self.pods[id]
            if self.deleted is not None:
                self.deleted(id)
            return Response(status_code=204)
        return JSONResponse(self.pods[id])

    async def _action(self, request: Request) -> Response:
        if (refusal := self._refused(request)) is not None:
            return refusal
        id, verb = request.path_params["id"], request.path_params["verb"]
        self.asked.append(("POST", f"/pods/{id}/{verb}", None))
        if id not in self.pods:
            return JSONResponse({"error": "pod not found"}, status_code=400)
        self.pods[id]["desiredStatus"] = {"start": "RUNNING", "stop": "EXITED"}[verb]
        if verb == "stop":
            self.pods[id]["publicIp"], self.pods[id]["portMappings"] = "", {}
        return JSONResponse(self.pods[id])

    def address(self, id: str, url: str) -> None:
        """Say that pod `id` is reached at `url` (`https://IP:PORT`): its public IP, and the public port its 8443/tcp
        is mapped to."""
        from urllib.parse import urlsplit

        parts = urlsplit(url)
        self.pods[id]["publicIp"], self.pods[id]["portMappings"] = parts.hostname, {"8443": parts.port}

    def client(self) -> Any:
        """A `RunPod` client of this fake, in this process (no socket)."""
        import httpx

        from rollout_runpod import RunPod

        transport = httpx.ASGITransport(app=self.app)
        return RunPod(url="http://runpod.test/v1", client=httpx.AsyncClient(transport=transport))

    def created_bodies(self) -> list[dict[str, Any]]:
        return [body for method, path, body in self.asked if method == "POST" and path == "/pods"]
