"""What runs beside a stock vLLM server on an inference pod: a follower that keeps the server serving what one run's
channel should, beats, and says whether the pod is ready.

The pod's vLLM server listens on the pod's loopback interface, with adapters loaded and unloaded while it runs
(`VLLM_ALLOW_RUNTIME_LORA_UPDATING`). `InferencePod` is the follower every engine host runs (`rollout_train.following`),
over one channel whose one engine is that server: it reads what the run says the channel should serve, fetches the
checkpoint's files from the blob store into the pod's volume, and loads them as an adapter named by the checkpoint's id.
After each look it asks the server what it has: the pod is ready once the server answers and has what the channel should
serve now (its base model, while the run says nothing else). Its beats say, beside what every follower says, the pod's
name, identity, public address, readiness and certificate serial (`rollout_train.pods.identity`).

`/healthz` and `/readyz` are served on the pod's loopback interface (by default `127.0.0.1:8081`), for the pod's own
checks; the proxy in front of the pod passes neither on.

    python -m rollout_train.pods.inference

- `ROLLOUT_RUN`: the run whose channel the pod serves, by id.
- `ROLLOUT_CHANNEL`: the channel, by its name within the run (default `policy`).
- `ROLLOUT_MODEL`: the model the vLLM server was started with, by the name it serves it under.
- `ROLLOUT_VLLM`: the server's address (default `http://127.0.0.1:8000`).
- `ROLLOUT_CHECKPOINTS`: where checkpoints' files are kept while they are served (default `/workspace/checkpoints`).
- `ROLLOUT_HEALTH`: where `/healthz` and `/readyz` are served (default `127.0.0.1:8081`);

and those every pod reads (`rollout_train.pods.environment`).
"""

import asyncio
import os
import socket
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import JsonValue

from rollout_train.checkpoints import Checkpoints
from rollout_train.following import Follower
from rollout_train.inference import Channel, RemoteEngine
from rollout_train.pods.environment import listening, public_address, required, serial, serial_file, served, stores
from rollout_train.pods.identity import POD, pod_identity
from rollout_train.presence import Presence, presence_of
from rollout_train.serving import qualified, wanted

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from rollout_train.recorder import Renderer

INFERENCE = "inference"
"""The role an inference pod says in its beats."""
VLLM = "http://127.0.0.1:8000"
STALE = 60.0
"""Seconds without a look after which the pod is not healthy."""


class InferencePod(Follower):
    """Keeps the vLLM server at `vllm` (serving `model`) serving what `run` says its channel `channel` should, with
    the checkpoints' files under `directory`, as pod `name` reached at `address`. `ready` says whether the server has
    what the channel should serve now, and `why` what it lacks when it does not."""

    def __init__(
        self,
        name: str,
        checkpoints: Checkpoints,
        run: str,
        channel: str,
        model: str,
        directory: Path,
        *,
        vllm: str = VLLM,
        address: str | None = None,
        presence: Presence | None = None,
        serial_file: Path | None = None,
        every: float = 2.0,
        beating: float = 15.0,
    ) -> None:
        self.engine = RemoteEngine(model, address=vllm)
        # (the follower publishes to this channel, and nothing samples through it here: it needs no renderer)
        served = Channel(channel, [self.engine], cast("Renderer", None))
        super().__init__(name, checkpoints, run, {channel: served}, directory, presence=presence, about=self.about_pod,
                         every=every, beating=beating)  # fmt: skip
        self.channel = served
        self.followed = run
        self.model = model
        self.identity = pod_identity(name)
        self.address = address
        self.serial_file = serial_file
        self.ready = False
        self.why = "not looked at yet"
        self.looked: float | None = None
        """When it last looked (`time.monotonic()`), whether or not it found the pod ready."""
        self.started = time.monotonic()

    async def follow(self) -> bool:
        changed = await super().follow()
        await self.look()
        return changed

    async def look(self) -> None:
        """Whether the server answers and has what the channel should serve now."""
        try:
            has = await self.engine.models()
            said = await wanted(self.checkpoints.ledger, self.followed, self.channel.name)
        except Exception as error:  # (looked at again at the next look)
            self.ready, self.why = False, f"{type(error).__name__}: {error}"
        else:
            name = said.checkpoint if said is not None and said.checkpoint is not None else self.model
            self.ready = name in has and (name == self.model or name == self.channel.serving)
            error = self.errors.get(qualified(self.followed, self.channel.name))
            self.why = "" if self.ready else error or f"{name} is not served yet"
        self.looked = time.monotonic()

    def healthy(self, stale: float = STALE) -> bool:
        """Whether it looked within `stale` seconds (or started within them)."""
        return time.monotonic() - (self.looked or self.started) <= stale

    def about_pod(self) -> Mapping[str, JsonValue]:
        """What its beats say of the pod, beside what every follower says."""
        pod: dict[str, JsonValue] = {
            "name": self.name, "identity": self.identity, "address": self.address, "role": INFERENCE,
            "ready": self.ready, "why": self.why, "model": self.model, "serial": serial(self.serial_file),
        }  # fmt: skip
        return {"host": socket.gethostname(), POD: pod}


def health(pod: InferencePod, *, stale: float = STALE) -> "Starlette":
    """`/healthz`: 200 while the pod looks at what it should serve; `/readyz`: 200 once it serves it."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def healthz(request: Request) -> JSONResponse:
        healthy = pod.healthy(stale)
        return JSONResponse({"healthy": healthy}, status_code=200 if healthy else 503)

    async def readyz(request: Request) -> JSONResponse:
        said: dict[str, Any] = {"ready": pod.ready, "serving": pod.channel.serving, "why": pod.why}
        return JSONResponse(said, status_code=200 if pod.ready else 503)

    return Starlette(routes=[Route("/healthz", healthz), Route("/readyz", readyz)])


async def main(environ: Mapping[str, str]) -> None:
    """Follow, beat and serve the pod's health, as the environment says, until cancelled."""
    ledger, blobs = stores(environ)
    pod = InferencePod(
        required(environ, "ROLLOUT_POD_NAME"), Checkpoints(ledger, blobs), required(environ, "ROLLOUT_RUN"),
        environ.get("ROLLOUT_CHANNEL", "policy"), required(environ, "ROLLOUT_MODEL"),
        Path(environ.get("ROLLOUT_CHECKPOINTS", "/workspace/checkpoints")), vllm=environ.get("ROLLOUT_VLLM", VLLM),
        address=public_address(environ), presence=presence_of(ledger), serial_file=serial_file(environ),
    )  # fmt: skip
    host, port = listening(environ, "ROLLOUT_HEALTH", "127.0.0.1:8081")
    await asyncio.gather(pod.serve(), served(health(pod), host, port))


if __name__ == "__main__":
    asyncio.run(main(os.environ))
