"""What runs beside a stock vLLM server on an inference pod: a follower that keeps the server serving what the run
that holds the pod says its channel should, beats, and says whether the pod is ready.

The pod's vLLM server listens on the pod's loopback interface, with adapters loaded and unloaded while it runs
(`VLLM_ALLOW_RUNTIME_LORA_UPDATING`). `InferencePod` is the follower every engine host runs (`rollout_train.following`),
over one channel whose one engine is that server: it reads what the run says the channel should serve, fetches the
checkpoint's files from the blob store into the pod's volume, and loads them as an adapter named by the checkpoint's id.
After each look it asks the server what it has: the pod is ready once the server answers and has what the channel should
serve now (its base model, while the run says nothing else). Its beats say, beside what every follower says, the pod's
name, identity, readiness, certificate serial, and the run and channel it serves (`rollout_train.pods.identity`); where
the pod is reached is its lease's, from RunPod's API.

Which run and channel it serves is its lease's (`rollout_train.pods.leases`): it reads its lease at each look, and
serves the channel of the run that holds the pod, with the ledger token the lease gives for that run. A pod released by
one run and taken by another drops the first's adapters and follows the second's; a pod no run holds serves nothing,
and says so. Given `ROLLOUT_RUN`, it serves that run's channel instead, whatever its lease says.

`/healthz` and `/readyz` are served on the pod's loopback interface (by default `127.0.0.1:8081`), for the pod's own
checks; the proxy in front of the pod passes neither on.

    python -m rollout_train.pods.inference

- `ROLLOUT_RUN`: the run whose channel the pod serves, by id (by default the run its lease names).
- `ROLLOUT_CHANNEL`: the channel, by its name within the run (default `policy`; with no `ROLLOUT_RUN`, its lease's).
- `ROLLOUT_MODEL`: the model the vLLM server was started with, by the name it serves it under.
- `ROLLOUT_VLLM`: the server's address (default `http://127.0.0.1:8000`).
- `ROLLOUT_CHECKPOINTS`: where checkpoints' files are kept while they are served (default `/workspace/checkpoints`).
- `ROLLOUT_HEALTH`: where `/healthz` and `/readyz` are served (default `127.0.0.1:8081`);
- `ROLLOUT_TRAINER_URL`: on a host pod, the training service beside vLLM (`http://127.0.0.1:8001`): the pod is ready
  only once it holds the trainer of the run that holds the pod too;

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
from rollout_train.following import Binding, Follower
from rollout_train.inference import Channel, RemoteEngine
from rollout_train.pods.environment import listening, required, serial, serial_file, served, stores
from rollout_train.pods.identity import POD, pod_identity
from rollout_train.pods.leases import IDLE, pod_leases_of
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
    """Keeps the vLLM server at `vllm` (serving `model`) serving what `run` says its channel `channel` should (none: the
    run and channel its lease names now), with the checkpoints' files under `directory`, as pod `name`. `ready` says
    whether the server has what the channel should serve now, and `why` what it lacks when it does not."""

    def __init__(
        self,
        name: str,
        checkpoints: Checkpoints,
        run: str | None,
        channel: str | None,
        model: str,
        directory: Path,
        *,
        vllm: str = VLLM,
        presence: Presence | None = None,
        serial_file: Path | None = None,
        every: float = 2.0,
        beating: float = 15.0,
        trainer: str | None = None,
        role: str = INFERENCE,
    ) -> None:
        self.engine = RemoteEngine(model, address=vllm)
        self.trainer = trainer
        """On a host pod, the training service beside the server, which must hold the run's trainer too."""
        self.role = role
        self.leases = pod_leases_of(checkpoints.ledger) if run is None else None
        if run is not None:
            served = self._opened(run, channel or "policy")
            super().__init__(name, checkpoints, run, {served.name: served}, directory, presence=presence,
                             about=self.about_pod, every=every, beating=beating)  # fmt: skip
        else:
            super().__init__(name, checkpoints, None, {}, directory, bindings=self._leased, opened=self._opened,
                             presence=presence, about=self.about_pod, every=every, beating=beating)  # fmt: skip
        self.held: Binding | None = (run, channel or "policy") if run is not None else None
        """The run and channel it serves now."""
        self.model = model
        self.identity = pod_identity(name)
        self.serial_file = serial_file
        self.ready = False
        self.why = "not looked at yet"
        self.looked: float | None = None
        """When it last looked (`time.monotonic()`), whether or not it found the pod ready."""
        self.started = time.monotonic()

    def _opened(self, run: str, channel: str) -> Channel:
        # (the follower publishes to this channel, and nothing samples through it here: it needs no renderer)
        return Channel(channel, [self.engine], cast("Renderer", None))

    async def _leased(self) -> list[Binding]:
        """The run and channel its lease names now (none while no run holds it), its ledger token switched to the
        one the lease gives for that run."""
        assert self.leases is not None
        lease = await self.leases.get(self.name)
        if lease is None or lease.run is None or lease.channel is None or lease.state == IDLE:
            self.held = None
            return []
        if lease.token and callable(use := getattr(self.checkpoints.ledger, "use", None)):
            use(lease.token)
        self.held = (lease.run, lease.channel)
        return [self.held]

    @property
    def channel(self) -> Channel | None:
        """The channel it serves now."""
        return self.bound.get(self.held) if self.held is not None else None

    async def follow(self) -> bool:
        changed = await super().follow()
        await self.look()
        return changed

    async def look(self) -> None:
        """Whether the server answers and has what the channel should serve now."""
        channel = self.channel
        if self.held is None or channel is None:
            self.ready, self.why = False, self.errors.get("bindings") or "no run holds it"
            self.looked = time.monotonic()
            return
        run, name = self.held
        try:
            has = await self.engine.models()
            said = await wanted(self.checkpoints.ledger, run, name)
        except Exception as error:  # (looked at again at the next look)
            self.ready, self.why = False, f"{type(error).__name__}: {error}"
        else:
            serves = said.checkpoint if said is not None and said.checkpoint is not None else self.model
            self.ready = serves in has and (serves == self.model or serves == channel.serving)
            error = self.errors.get(qualified(run, name))
            self.why = "" if self.ready else error or f"{serves} is not served yet"
            if self.ready and self.trainer is not None and (held := await self._trained()) != run:
                self.ready, self.why = False, f"the training service holds the trainer of {held or 'no run'} yet"
        self.looked = time.monotonic()

    async def _trained(self) -> str | None:
        """The run whose trainer the training service beside the server holds (none: none, or it does not answer)."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                said: Any = (await client.get(f"{self.trainer}/v1/trainer")).json()
        except (httpx.HTTPError, ValueError):
            return None
        run: Any = cast(dict[str, Any], said).get("run") if isinstance(said, dict) else None
        return str(run) if run else None

    def healthy(self, stale: float = STALE) -> bool:
        """Whether it looked within `stale` seconds (or started within them)."""
        return time.monotonic() - (self.looked or self.started) <= stale

    def about_pod(self) -> Mapping[str, JsonValue]:
        """What its beats say of the pod, beside what every follower says."""
        run, channel = self.held if self.held is not None else (None, None)
        pod: dict[str, JsonValue] = {
            "name": self.name, "identity": self.identity, "role": self.role,
            "ready": self.ready, "why": self.why, "model": self.model, "serial": serial(self.serial_file),
            "run": run, "channel": channel,
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
        serving = pod.channel.serving if pod.channel is not None else None
        said: dict[str, Any] = {"ready": pod.ready, "serving": serving, "why": pod.why}
        return JSONResponse(said, status_code=200 if pod.ready else 503)

    return Starlette(routes=[Route("/healthz", healthz), Route("/readyz", readyz)])


async def main(environ: Mapping[str, str]) -> None:
    """Follow, beat and serve the pod's health, as the environment says, until cancelled."""
    ledger, blobs = stores(environ)
    pod = InferencePod(
        required(environ, "ROLLOUT_POD_NAME"), Checkpoints(ledger, blobs), environ.get("ROLLOUT_RUN") or None,
        environ.get("ROLLOUT_CHANNEL") or None, required(environ, "ROLLOUT_MODEL"),
        Path(environ.get("ROLLOUT_CHECKPOINTS", "/workspace/checkpoints")), vllm=environ.get("ROLLOUT_VLLM", VLLM),
        presence=presence_of(ledger), serial_file=serial_file(environ),
        trainer=environ.get("ROLLOUT_TRAINER_URL") or None, role=environ.get("ROLLOUT_ROLE") or INFERENCE,
    )  # fmt: skip
    host, port = listening(environ, "ROLLOUT_HEALTH", "127.0.0.1:8081")
    await asyncio.gather(pod.serve(), served(health(pod), host, port))


if __name__ == "__main__":
    asyncio.run(main(os.environ))
