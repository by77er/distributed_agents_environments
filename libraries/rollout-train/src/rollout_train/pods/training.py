"""The training service on a training pod: a `Trainer` taking one step at a time, asked for over HTTP.

A step is asked for by the checkpoint it makes (`into`, the checkpoint's id), with its seed, its batch (a blob: the
weighted segments, `batch_bytes`), its parent (its id, and manifests of blobs: the weights, and the trainer's state if
it left any), and the settings the trainer takes between steps (`rollout_train.trainer.Changeable`). The service
fetches the batch and the parent's files from the blob store, runs the step, keeps the new weights in the blob store,
and answers with their manifest, the state's, and the step's metrics (with how long the step took on the pod,
`step_seconds`, and keeping the weights, `upload_adapter_seconds`).

**The weights first, the state after.** A trainer that keeps a step's full state itself after the step returns
(`rollout_train.trainer.Keeps`, told the pod's blob store, `ROLLOUT_BLOBS`) has the step answered as soon as the weights
and what it wrote of the state are kept: the answer says the state is incomplete (`StepMade.complete` false), and the
loop serves the weights at once. Once the trainer has kept the rest, the answer is the whole state's (`complete`); if
keeping it failed, it says why (`state_failed`), loudly in the service's log too, and stays incomplete. A step whose
parent's state this service is still keeping, and which its trainer does not hold, waits for it.

A step is idempotent by `into`: asked again while it runs, it is the same step; asked again after it was made, the
answer is the one it made (kept on the pod's disk), or, where the ledger already has the checkpoint, the checkpoint's
own files. A step that failed is taken again when it is asked for again. One step runs at a time: another asked for
meanwhile is refused (409), and the asker tries it again later.

A trainer that keeps its processes, policy and optimizer between steps (`rollout_train.trainer.Resident`:
`rollout_lora`'s trainers do, but beside the pod's sleeping vLLM, where each step's processes end after it): where a
step's parent is what it holds (the parent's state names it, `HELD`), the service fetches that one small file of the
parent's and none of the rest. When the run that holds the pod changes, or its lease is released, the trainer is
closed, which ends its processes and frees the GPUs.

- `POST /v1/steps` with a `StepAsked`: 202 and `{"state": "running"}`; 200 and the step's state if it was made; 409
  while another step runs; 400 for a request it cannot read.
- `GET /v1/steps/INTO`: the step's state (`StepState`: `running`, `made` with what it made, or `failed` with why); 404
  for a step it never heard of.
- `GET /v1/trainer`: what the trainer is: its kind, model, `weights`, `budget`, and the settings it takes between steps.
- `GET /healthz`, `GET /readyz`: 200 while the service answers (for the pod's own checks: the proxy passes neither on).

    python -m rollout_train.pods.training

- `ROLLOUT_TRAINER`: the trainer, by `module:name` (`rollout_lora:LoraTrainer`, `rollout_lora:FullTrainer`).
- `ROLLOUT_TRAINER_MODEL`: the model it trains.
- `ROLLOUT_TRAINER_SETTINGS`: its settings, as a JSON object (default `{}`), until a run that holds the pod says its
  own.
- `ROLLOUT_TRAINER_GPUS`: the GPUs it steps on (`gpus`, which `rollout_lora`'s trainers take: a process per GPU under
  torchrun, the policy sharded over them on more than one); by default those the pod has.
- `ROLLOUT_SLEEP_VLLM`: on a pod that serves too (`runpod-host`), `1` to have the pod's vLLM sleep while a step is
  taken (`rollout_train.colocated`), and the trainer told it is `colocated`.
- `ROLLOUT_WORK`: where steps' files and the answers of the steps made are kept (default `/workspace/rollout`).
- `ROLLOUT_LISTEN`: where the service listens (default `127.0.0.1:8001`);

and those every pod reads (`rollout_train.pods.environment`). It beats every 15 seconds with the pod's name, identity,
the run it takes steps for, whether it is ready for that run, and whether a step is running.

Which run it takes steps for is its lease's (`rollout_train.pods.leases`): when a run takes the pod, the service makes
its trainer anew with the run's settings (the trainer's implementation, model and settings the lease says), and reads
the ledger with the token the lease gives for that run, once no step runs. It is then ready for that run.
"""

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import re
import shutil
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import JsonValue, TypeAdapter, ValidationError

from rollout.contracts import BlobReference
from rollout.names import named
from rollout_train.checkpoints import Checkpoints, Manifest, kept
from rollout_train.pods.environment import listening, location, required, serial, serial_file, served, stores
from rollout_train.pods.identity import POD, pod_identity
from rollout_train.presence import beating, presence_of
from rollout_train.trainer import (
    HELD,
    STATE,
    WEIGHTS,
    Changeable,
    Distilled,
    Files,
    Item,
    Keeps,
    Labelled,
    Pair,
    Resident,
    Trainer,
    Weighted,
)

if TYPE_CHECKING:
    from starlette.applications import Starlette

logger = logging.getLogger(__name__)

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
    """A step's parent's files, as blobs: the weights, and what the trainer left for itself beside them; and the
    parent's id (the step that made it, here or elsewhere)."""

    weights: Manifest
    state: Manifest | None = None
    id: str | None = None


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
    complete: bool = True
    """Whether `state` is the whole state: false while the trainer keeps the rest (`state` then holds what it wrote
    before the step returned), and for good once keeping it failed (`state_failed`)."""
    state_failed: str | None = None
    """Why the rest of the state was not kept, if it was not."""


@dataclass(frozen=True)
class StepState:
    state: str
    """`running`, `made` or `failed`."""
    made: StepMade | None = None
    error: str | None = None


_ASKED = TypeAdapter(StepAsked)
_MADE = TypeAdapter(StepMade)
_STATE = TypeAdapter(StepState)
_BATCH = TypeAdapter(list[Distilled | Weighted | Pair | Labelled])


def batch_bytes(batch: Sequence[Item]) -> bytes:
    """A batch as one blob's bytes: JSON, NaN kept as NaN (a forced token's logprob)."""
    return json.dumps([dataclasses.asdict(item) for item in batch], separators=(",", ":")).encode()


def batch_of(data: bytes) -> list[Item]:
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
        self.keeping: dict[str, asyncio.Task[None]] = {}
        """The steps whose state the trainer is still keeping, by `into`."""
        self._running: tuple[str, asyncio.Task[None]] | None = None
        self.run: str | None = None
        """The run whose trainer it holds (as its lease said), if it follows a lease."""

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
            said = _MADE.validate_json(await asyncio.to_thread(self._answer(into).read_bytes))
            if not said.complete and said.state_failed is None and into not in self.keeping:
                said = dataclasses.replace(said, state_failed="the service started again while it kept the state")
            return StepState(MADE, said)
        try:
            checkpoint = await self.checkpoints.checkpoint(into)
        except KeyError:
            return None
        if checkpoint.weights is None:
            return StepState(FAILED, error=f"{into} was made and its files released")
        failed = None if checkpoint.state_complete else "its state was not kept, and this service is not keeping it"
        made = StepMade(into, dict(checkpoint.metrics), checkpoint.weights, checkpoint.state,
                        checkpoint.state_complete, failed)  # fmt: skip
        return StepState(MADE, made)

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
            "run": self.run,
        }

    async def _step(self, asked: StepAsked) -> None:
        work = self.directory / "steps" / asked.into
        trainer = self.trainer  # (the one that takes the step keeps its state, whatever the service holds by then)
        keeping = False
        try:
            await asyncio.to_thread(shutil.rmtree, work, ignore_errors=True)  # (what a step that died left)
            data = await self.checkpoints.blobs.read(asked.batch)
            if len(data) > MAX_BATCH:
                raise ValueError(f"the batch is {len(data)} bytes, more than {MAX_BATCH}")
            batch = batch_of(data)
            parent = await self._parent(asked, work / "parent")
            if asked.settings and isinstance(trainer, Changeable):
                trainer.change(asked.settings)
            into = work / "made" / asked.into  # (named by the checkpoint, as the loop names it: a trainer may read it)
            await asyncio.to_thread(into.mkdir, parents=True, exist_ok=True)
            began = time.monotonic()
            step = await trainer.step(batch, seed=asked.seed, parent=parent, into=into)
            stepped, began = time.monotonic() - began, time.monotonic()
            weights = await kept(into / WEIGHTS, self.checkpoints.blobs)
            uploaded = time.monotonic() - began
            state = await kept(into / STATE, self.checkpoints.blobs) if (into / STATE).exists() else None
            metrics = {name: float(value) for name, value in step.metrics.items()}
            metrics |= {"step_seconds": round(stepped, 3), "upload_adapter_seconds": round(uploaded, 3)}
            keeping = step.keeping and isinstance(trainer, Keeps)
            made = StepMade(asked.into, metrics, weights, state, complete=not keeping)
            await self._answered(made)
            if keeping:
                assert isinstance(trainer, Keeps)
                self.keeping[asked.into] = asyncio.create_task(self._kept(trainer, made, work))
        except Exception as error:  # (a step that failed: taken again when it is asked for again)
            keeping = False
            self.states[asked.into] = StepState(FAILED, error=f"{type(error).__name__}: {error}"[-2000:])
        finally:
            await asyncio.to_thread(shutil.rmtree, work / "parent" if keeping else work, ignore_errors=True)

    async def _kept(self, trainer: Keeps, made: StepMade, work: Path) -> None:
        """Wait until `trainer` has kept the rest of the step's state, and answer with the whole state; or, if it never
        will, say why (loudly), leaving the state incomplete. The step's files on the pod are deleted after."""
        try:
            rest = await trainer.kept(made.into)
            whole = {**(made.state.files if made.state is not None else {}), **rest.files}
            done = dataclasses.replace(made, state=Manifest(whole), complete=True)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            said = f"{type(error).__name__}: {error}"[-2000:]
            logger.error("the state of step %s was not kept, and stays incomplete: %s", made.into, said)
            done = dataclasses.replace(made, state_failed=said)
        try:
            await self._answered(done)
        finally:
            self.keeping.pop(made.into, None)
            await asyncio.to_thread(shutil.rmtree, work, ignore_errors=True)

    async def _answered(self, made: StepMade) -> None:
        """What a step made, as its answer: kept on disk (an answer for the step asked for again, after a restart too),
        and here."""
        answer = self._answer(made.into)
        await asyncio.to_thread(answer.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(_written, answer, _MADE.dump_json(made))
        self.states[made.into] = StepState(MADE, made)

    async def _parent(self, asked: StepAsked, directory: Path) -> Files | None:
        """The step's parent's files under `directory`: only the state's `HELD` where the trainer holds the parent,
        else all of them, the state the whole one where this service made the parent (waiting while it keeps it)."""
        if asked.parent is None:
            return None
        state = asked.parent.state
        holding = self.trainer.holding if isinstance(self.trainer, Resident) else None
        if holding is not None and state is not None and HELD in state.files:
            named = await self.checkpoints.blobs.read(state.files[HELD])
            if named.decode(errors="replace").strip() == holding:
                kept_state = await self.checkpoints.files(Manifest({HELD: state.files[HELD]}), directory / STATE)
                await asyncio.to_thread((directory / WEIGHTS).mkdir, parents=True, exist_ok=True)
                return Files(directory / WEIGHTS, kept_state)
        if asked.parent.id is not None and (whole := await self._whole(asked.parent.id)) is not None:
            state = whole
        weights = await self.checkpoints.files(asked.parent.weights, directory / WEIGHTS)
        return Files(weights, await self.checkpoints.files(state, directory / STATE) if state else None)

    async def _whole(self, into: str) -> Manifest | None:
        """The whole state of a step this service made, once kept (waiting while it is being kept); None where it has
        none (another made the step, or its state was not kept)."""
        if (task := self.keeping.get(into)) is not None:
            await asyncio.shield(task)
        said = await self.state(into) if CHECKPOINT_ID.fullmatch(into) else None
        made = said.made if said is not None else None
        return made.state if made is not None and made.complete else None

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


def made(implementation: str, model: str, settings: Mapping[str, Any]) -> Trainer:
    """A trainer: `implementation` (`module:name`) called with the model and those of `settings` it takes (each one,
    where it takes any keyword)."""
    import inspect

    making = named(implementation)
    try:
        parameters = inspect.signature(making).parameters.values()
    except (TypeError, ValueError):
        return making(model, **settings)
    if not any(each.kind is inspect.Parameter.VAR_KEYWORD for each in parameters):
        names = {each.name for each in parameters}
        settings = {key: value for key, value in settings.items() if key in names}
    return making(model, **settings)


class _LocalServer:
    """The pod's own vLLM server as what a colocated trainer pauses (`rollout_train.colocated.Pausable`): it sleeps
    while a step is taken. Requests to it are not held back here: the gateway's to a sleeping server fail, and are
    sampled again on its next look."""

    def __init__(self, address: str) -> None:
        from rollout_train.inference import RemoteEngine

        self.engine = RemoteEngine("", address=address)

    async def pause(self) -> None:
        return None

    def resume(self) -> None:
        return None

    async def sleep(self) -> None:
        await self.engine.sleep()

    async def wake(self) -> None:
        await self.engine.wake()


async def following(
    service: TrainerService,
    name: str,
    make: Callable[[Mapping[str, JsonValue]], Trainer],
    *,
    every: float = 2.0,
) -> None:
    """Keep `service`'s trainer the one the run that holds pod `name` asks for (its lease's settings, made by `make`),
    and its ledger token that run's, from when no step runs; until cancelled."""
    from rollout_train.pods.leases import IDLE, pod_leases_of

    leases = pod_leases_of(service.checkpoints.ledger)
    if leases is None:
        raise ValueError("this ledger keeps no pods' leases")
    while True:
        with contextlib.suppress(Exception):  # (looked at again at the next look)
            lease = await leases.get(name)
            run = lease.run if lease is not None and lease.state != IDLE else None
            if lease is not None and run is not None and run != service.run and service.running is None:
                if lease.token and callable(use := getattr(service.checkpoints.ledger, "use", None)):
                    use(lease.token)
                await asyncio.to_thread(closed, service.trainer)  # (its processes end: the GPUs are the next one's)
                service.trainer = await asyncio.to_thread(make, lease.settings)
                service.run = run
            elif run is None and service.running is None:
                if service.run is not None:  # (released: what the trainer holds is freed)
                    await asyncio.to_thread(closed, service.trainer)
                service.run = None
        await asyncio.sleep(every)


def closed(trainer: Trainer) -> None:
    """End what a trainer keeps running between steps (`Resident`), if it keeps anything."""
    if isinstance(trainer, Resident):
        trainer.close()


async def main(environ: Mapping[str, str]) -> None:
    """Serve the trainer the environment names (and then the one the run that holds the pod asks for), and beat,
    until cancelled."""
    ledger, blobs = stores(environ)
    settings: Any = json.loads(environ.get("ROLLOUT_TRAINER_SETTINGS") or "{}")
    model = required(environ, "ROLLOUT_TRAINER_MODEL")
    implementation = required(environ, "ROLLOUT_TRAINER")
    sleeps = environ.get("ROLLOUT_SLEEP_VLLM", "") in ("1", "true")
    gpus = int(environ["ROLLOUT_TRAINER_GPUS"]) if environ.get("ROLLOUT_TRAINER_GPUS") else None

    kept_in = location(environ, "ROLLOUT_BLOBS")

    def make(said: Mapping[str, JsonValue]) -> Trainer:
        given: Any = said.get("trainer")
        chosen = cast(dict[str, Any], given) if isinstance(given, dict) else cast(dict[str, Any], settings)
        if gpus is not None:
            chosen = {**chosen, "gpus": gpus}
        if sleeps:  # (beside the pod's vLLM: its processes end after each step, and give the memory back)
            chosen = {**chosen, "colocated": True}
        trainer = made(str(said.get("implementation") or implementation), str(said.get("model") or model), chosen)
        if sleeps:
            from rollout_train.colocated import Colocated

            local = _LocalServer(environ.get("ROLLOUT_VLLM", "http://127.0.0.1:8000"))
            return cast(Trainer, Colocated(trainer, [local]))
        if isinstance(trainer, Keeps):  # (its steps' state kept after they return: the weights served first)
            trainer.keep_in(kept_in)
        return trainer

    work = Path(environ.get("ROLLOUT_WORK", "/workspace/rollout"))
    service = TrainerService(make({}), Checkpoints(ledger, blobs), work, model=model)
    name = required(environ, "ROLLOUT_POD_NAME")
    serials = serial_file(environ)
    identity = pod_identity(name)
    role = environ.get("ROLLOUT_ROLE") or TRAINER

    def about() -> Mapping[str, JsonValue]:
        pod: dict[str, JsonValue] = {"name": name, "identity": identity, "role": TRAINER,
                                     "ready": service.run is not None, "run": service.run, "running": service.running,
                                     "serial": serial(serials), "steps": True}  # fmt: skip
        return {"host": socket.gethostname(), "kind": TRAINER, POD: pod}

    host, port = listening(environ, "ROLLOUT_LISTEN", "127.0.0.1:8001")
    waits = [served(app(service), host, port), following(service, name, make)]
    if (presence := presence_of(ledger)) is not None and role == TRAINER:  # (a host's follower beats for the pod)
        waits.append(beating(presence, name, about))
    await asyncio.gather(*waits)


if __name__ == "__main__":
    asyncio.run(main(os.environ))
