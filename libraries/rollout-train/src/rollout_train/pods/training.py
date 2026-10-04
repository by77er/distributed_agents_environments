"""The training service on a training pod: a `Trainer` taking one step at a time, asked for over HTTP.

A step is asked for by the checkpoint it makes (`into`, the checkpoint's id), with its seed, its batch (a blob: the
weighted segments, `batch_bytes`), its parent's files (manifests of blobs: the weights, and the trainer's state if it
left any), and the settings the trainer takes between steps (`rollout_train.trainer.Changeable`). The service fetches
the batch and the parent's files from the blob store, runs the step, keeps the new weights and state in the blob store,
and answers with their manifests and the step's metrics. A step runs in a fresh process when the trainer runs each step
so (`rollout_lora`'s trainers do).

A step is idempotent by `into`: asked again while it runs, it is the same step; asked again after it was made, the
answer is the one it made (kept on the pod's disk), or, where the ledger already has the checkpoint, the checkpoint's
own files. A step that failed is taken again when it is asked for again. One step runs at a time: another asked for
meanwhile is refused (409), and the asker tries it again later.

- `POST /v1/steps` with a `StepAsked`: 202 and `{"state": "running"}`; 200 and the step's state if it was made; 409
  while another step runs; 400 for a request it cannot read.
- `GET /v1/steps/INTO`: the step's state (`StepState`: `running`, `made` with what it made, or `failed` with why); 404
  for a step it never heard of.
- `GET /v1/trainer`: what the trainer is: its kind, model, `weights`, `budget`, and the settings it takes between steps.
- `GET /healthz`, `GET /readyz`: 200 while the service answers (for the pod's own checks: the proxy passes neither on).

    python -m rollout_train.pods.training

- `ROLLOUT_TRAINER`: the trainer, by `module:name` (`rollout_lora:LoraTrainer`, `rollout_lora:FullTrainer`).
- `ROLLOUT_TRAINER_MODEL`: the model it trains.
- `ROLLOUT_TRAINER_SETTINGS`: its settings, as a JSON object (default `{}`).
- `ROLLOUT_WORK`: where steps' files and the answers of the steps made are kept (default `/workspace/rollout`).
- `ROLLOUT_LISTEN`: where the service listens (default `127.0.0.1:8001`);

and those every pod reads (`rollout_train.pods.environment`). It beats every 15 seconds with the pod's name, identity,
address and whether a step is running.
"""

import asyncio
import contextlib
import dataclasses
import json
import os
import re
import shutil
import socket
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import JsonValue, TypeAdapter, ValidationError

from rollout.contracts import BlobReference
from rollout.names import named
from rollout_train.checkpoints import Checkpoints, Manifest, kept
from rollout_train.pods.environment import listening, public_address, required, serial, serial_file, served, stores
from rollout_train.pods.identity import POD, pod_identity
from rollout_train.presence import beating, presence_of
from rollout_train.trainer import STATE, WEIGHTS, Changeable, Files, Trainer, Weighted

if TYPE_CHECKING:
    from starlette.applications import Starlette

TRAINER = "trainer"
"""The role a training pod says in its beats."""
RUNNING, MADE, FAILED = "running", "made", "failed"
"""What a step's state says."""
CHECKPOINT_ID = re.compile(r"[A-Za-z0-9_@-]{1,128}")
"""What `into` may be: a checkpoint's id (and so a directory's name)."""
MAX_BATCH = 512 * 2**20
"""The largest batch blob a step reads, in bytes."""


@dataclass(frozen=True)
class Parent:
    """A step's parent's files, as blobs: the weights, and what the trainer left for itself beside them."""

    weights: Manifest
    state: Manifest | None = None


@dataclass(frozen=True)
class StepAsked:
    """A step, as it is asked for."""

    into: str
    """The checkpoint it makes, by id: what the step is known by."""
    seed: int
    batch: BlobReference
    """The weighted segments it trains on (`batch_bytes`)."""
    parent: Parent | None = None
    """None: from the base model."""
    settings: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """The settings the trainer takes between steps, as they are for this step."""


@dataclass(frozen=True)
class StepMade:
    """What a step made: its metrics, and the new weights and state, as blobs."""

    into: str
    metrics: Mapping[str, float]
    weights: Manifest
    state: Manifest | None = None


@dataclass(frozen=True)
class StepState:
    state: str
    """`running`, `made` or `failed`."""
    made: StepMade | None = None
    error: str | None = None


_ASKED = TypeAdapter(StepAsked)
_MADE = TypeAdapter(StepMade)
_STATE = TypeAdapter(StepState)
_BATCH = TypeAdapter(list[Weighted])


def batch_bytes(batch: Sequence[Weighted]) -> bytes:
    """A batch as one blob's bytes: JSON, NaN kept as NaN (a forced token's logprob)."""
    return json.dumps([dataclasses.asdict(weighted) for weighted in batch], separators=(",", ":")).encode()


def batch_of(data: bytes) -> list[Weighted]:
    return _BATCH.validate_python(json.loads(data))


def asked_json(asked: StepAsked) -> Any:
    return _ASKED.dump_python(asked, mode="json")


def state_of(said: Any) -> StepState:
    return _STATE.validate_python(said)


class Busy(Exception):
    """Another step is running."""


class TrainerService:
    """`trainer`'s steps (a trainer of `model`), asked for by the checkpoint each makes, with their files in
    `checkpoints`' blob store and their work under `directory`."""

    def __init__(
        self, trainer: Trainer, checkpoints: Checkpoints, directory: Path, *, model: str | None = None
    ) -> None:
        self.trainer = trainer
        self.model = model
        self.checkpoints = checkpoints
        self.directory = directory
        self.states: dict[str, StepState] = {}
        """The steps asked for since the service started, by `into` (steps made before are on disk)."""
        self._running: tuple[str, asyncio.Task[None]] | None = None

    @property
    def running(self) -> str | None:
        """The step running now, by `into`."""
        return self._running[0] if self._running is not None and not self._running[1].done() else None

    async def ask(self, asked: StepAsked) -> StepState:
        """Start the step (unless it runs or was made); its state now. `Busy` while another runs."""
        if not CHECKPOINT_ID.fullmatch(asked.into):
            raise ValueError(f"{asked.into!r} is not a checkpoint's id")
        if (state := await self.state(asked.into)) is not None and state.state != FAILED:
            return state
        if (running := self.running) is not None:
            raise Busy(f"step {running} is running")
        self.states[asked.into] = StepState(RUNNING)
        self._running = (asked.into, asyncio.create_task(self._step(asked)))
        return self.states[asked.into]

    async def state(self, into: str) -> StepState | None:
        """A step's state: as it is here, as the answer kept on disk says, or as the ledger's checkpoint says; None
        for a step never heard of."""
        if not CHECKPOINT_ID.fullmatch(into):
            return None
        if into in self.states:
            return self.states[into]
        with contextlib.suppress(OSError, ValueError, ValidationError):
            said = await asyncio.to_thread(self._answer(into).read_bytes)
            return StepState(MADE, _MADE.validate_json(said))
        try:
            checkpoint = await self.checkpoints.checkpoint(into)
        except KeyError:
            return None
        if checkpoint.weights is None:
            return StepState(FAILED, error=f"{into} was made and its files released")
        return StepState(MADE, StepMade(into, dict(checkpoint.metrics), checkpoint.weights, checkpoint.state))

    def describe(self) -> dict[str, JsonValue]:
        trainer = self.trainer
        changeable = dict(trainer.changeable) if isinstance(trainer, Changeable) else {}
        return {
            "kind": f"{type(trainer).__module__}:{type(trainer).__name__}",
            "model": self.model,
            "weights": self.trainer.weights,
            "budget": dataclasses.asdict(self.trainer.budget),
            "changeable": changeable,
            "running": self.running,
        }

    async def _step(self, asked: StepAsked) -> None:
        work = self.directory / "steps" / asked.into
        try:
            await asyncio.to_thread(shutil.rmtree, work, ignore_errors=True)  # (what a step that died left)
            data = await self.checkpoints.blobs.read(asked.batch)
            if len(data) > MAX_BATCH:
                raise ValueError(f"the batch is {len(data)} bytes, more than {MAX_BATCH}")
            batch = batch_of(data)
            parent = None
            if asked.parent is not None:
                weights = await self.checkpoints.files(asked.parent.weights, work / "parent" / WEIGHTS)
                state = asked.parent.state
                parent = Files(weights, await self.checkpoints.files(state, work / "parent" / STATE) if state else None)
            if asked.settings and isinstance(self.trainer, Changeable):
                self.trainer.change(asked.settings)
            into = work / "made" / asked.into  # (named by the checkpoint, as the loop names it: a trainer may read it)
            await asyncio.to_thread(into.mkdir, parents=True, exist_ok=True)
            step = await self.trainer.step(batch, seed=asked.seed, parent=parent, into=into)
            weights = await kept(into / WEIGHTS, self.checkpoints.blobs)
            state = await kept(into / STATE, self.checkpoints.blobs) if (into / STATE).exists() else None
            made = StepMade(asked.into, {name: float(value) for name, value in step.metrics.items()}, weights, state)
            answer = self._answer(asked.into)
            await asyncio.to_thread(answer.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(_written, answer, _MADE.dump_json(made))
            self.states[asked.into] = StepState(MADE, made)
        except Exception as error:  # (a step that failed: taken again when it is asked for again)
            self.states[asked.into] = StepState(FAILED, error=f"{type(error).__name__}: {error}"[-2000:])
        finally:
            await asyncio.to_thread(shutil.rmtree, work, ignore_errors=True)

    def _answer(self, into: str) -> Path:
        return self.directory / "made" / f"{into}.json"


def app(service: TrainerService) -> "Starlette":
    """The service's HTTP API."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    def said(state: StepState, status: int = 200) -> JSONResponse:
        return JSONResponse(_STATE.dump_python(state, mode="json", exclude_none=True), status_code=status)

    async def steps(request: Request) -> JSONResponse:
        try:
            asked = _ASKED.validate_json(await request.body())
            state = await service.ask(asked)
        except (ValidationError, ValueError) as error:
            return JSONResponse({"error": str(error)[:2000]}, status_code=400)
        except Busy as error:
            return JSONResponse({"error": str(error), "running": service.running}, status_code=409)
        return said(state, 202 if state.state == RUNNING else 200)

    async def step(request: Request) -> JSONResponse:
        state = await service.state(request.path_params["into"])
        if state is None:
            return JSONResponse({"error": "no such step"}, status_code=404)
        return said(state)

    async def trainer(request: Request) -> JSONResponse:
        return JSONResponse(service.describe())

    async def alive(request: Request) -> JSONResponse:
        return JSONResponse({"healthy": True, "running": service.running})

    return Starlette(
        routes=[
            Route("/v1/steps", steps, methods=["POST"]),
            Route("/v1/steps/{into}", step, methods=["GET"]),
            Route("/v1/trainer", trainer, methods=["GET"]),
            Route("/healthz", alive, methods=["GET"]),
            Route("/readyz", alive, methods=["GET"]),
        ]
    )


def _written(path: Path, data: bytes) -> None:
    staging = path.with_suffix(".writing")
    staging.write_bytes(data)
    os.replace(staging, path)


async def main(environ: Mapping[str, str]) -> None:
    """Serve the trainer the environment names, and beat, until cancelled."""
    ledger, blobs = stores(environ)
    settings: Any = json.loads(environ.get("ROLLOUT_TRAINER_SETTINGS") or "{}")
    model = required(environ, "ROLLOUT_TRAINER_MODEL")
    trainer: Trainer = named(required(environ, "ROLLOUT_TRAINER"))(model, **settings)
    work = Path(environ.get("ROLLOUT_WORK", "/workspace/rollout"))
    service = TrainerService(trainer, Checkpoints(ledger, blobs), work, model=model)
    name = required(environ, "ROLLOUT_POD_NAME")
    address, serials = public_address(environ), serial_file(environ)
    identity = pod_identity(name)

    def about() -> Mapping[str, JsonValue]:
        pod: dict[str, JsonValue] = {"name": name, "identity": identity, "address": address, "role": TRAINER,
                                     "ready": True, "running": service.running, "serial": serial(serials)}  # fmt: skip
        return {"host": socket.gethostname(), "kind": TRAINER, POD: pod}

    host, port = listening(environ, "ROLLOUT_LISTEN", "127.0.0.1:8001")
    waits = [served(app(service), host, port)]
    if (presence := presence_of(ledger)) is not None:
        waits.append(beating(presence, name, about))
    await asyncio.gather(*waits)


if __name__ == "__main__":
    asyncio.run(main(os.environ))
