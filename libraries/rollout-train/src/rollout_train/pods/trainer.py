"""`RemoteTrainer`: a `Trainer` whose steps run on a training pod, through its training service
(`rollout_train.pods.training`).

It is a `rollout_train.trainer.Remote`: the training loop asks it for a step (`made`) with the parent as the checkpoint,
whose manifests it hands the pod, and gets back manifests of the new weights and state, which the pod kept in the blob
store; nothing is read to the loop's machine. A step puts its batch in the blob store, asks the pod for the step by the
checkpoint it makes (its id), and asks after it every few seconds until it is made or has failed. The pod answers once
the weights are kept; the rest of the state may still be being kept then (`Made.complete` false), and `state` asks
after the step until it is (or raises `StateLost` where it never will be). Asking again is safe: the pod knows a step by
its checkpoint, so a step asked for again after a connection dropped, or by a loop started again, is the same step.

As a plain `Trainer` (`step`, for a caller that works with files), it keeps the parent's files from disk in the blob
store (content-addressed: files the store has are not written again), and fetches the whole step's files into `into`
once the state is kept.

The settings the trainer takes between steps (`Changeable`) are kept here, and those that differ from what it was made
with are sent with every step, so the pod's trainer steps with the settings the run has now, whatever it took before (a
pod's trainer is made with the settings it was given at the start, as its lease says).

Every way a step can fail is a `StepFailed` (the policy stays as it was, and a later step may succeed), with a kind of
its own where the pod is the cause: `TrainerUnreachable` when it did not answer for `patience` seconds,
`TrainerRefused` when it refused the request (the proxy's 404 for a path it does not pass, an error of the service),
`TrainerBusy` when it is taking another run's step.
"""

import asyncio
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
from pydantic import JsonValue

from rollout_train.checkpoints import Checkpoint, Checkpoints, Manifest, kept
from rollout_train.http import answer_of
from rollout_train.inference.remote import Connection, https_only
from rollout_train.objectives import DEFAULT, Objective
from rollout_train.pods.training import (
    FAILED,
    MADE,
    Parent,
    StepAsked,
    StepMade,
    StepState,
    asked_json,
    batch_bytes,
    state_of,
)
from rollout_train.trainer import STATE, WEIGHTS, Budget, Files, Item, Made, StateLost, Step, StepFailed


class TrainerUnreachable(StepFailed):
    """The training pod did not answer for as long as a step waits for it."""


class TrainerRefused(StepFailed):
    """The training pod (or the proxy in front of it) refused the request."""


class TrainerBusy(TrainerRefused):
    """The training pod is taking another step, which it was asked for by someone else."""


class RemoteTrainer:
    """Takes steps on the training service at `address`, its files through `checkpoints`' blob store. `weights` and
    `budget` are what the pod's trainer makes and can take (as the cluster says of it; `describe` asks the pod);
    `objective` what it trains with (the `default` preset unless given); `changeable` the settings it takes between
    steps, with their values now. The pod is reached as `connection` says
    (a client certificate, the CA, and the identity the pod's certificate must carry). It is asked after a step every
    `every` seconds; a step fails as `TrainerUnreachable` after `patience` seconds without an answer."""

    def __init__(
        self,
        address: str,
        checkpoints: Checkpoints,
        *,
        weights: str = "lora",
        budget: Budget | None = None,
        objective: Objective = DEFAULT,
        changeable: Mapping[str, JsonValue] | None = None,
        connection: Connection | None = None,
        client: httpx.AsyncClient | None = None,
        every: float = 2.0,
        patience: float = 300.0,
    ) -> None:
        self.address = address.rstrip("/")
        https_only(self.address, connection)
        self.checkpoints = checkpoints
        self.weights = weights
        self.budget = budget or Budget()
        self.objective = objective
        self._changeable: dict[str, JsonValue] = dict(changeable or {})
        self._made_with: dict[str, JsonValue] = dict(changeable or {})
        self._http = client or (connection or Connection()).client(timeout=60.0)
        self._owned = client is None
        self.every = every
        self.patience = patience
        self._lock = asyncio.Lock()

    @property
    def changeable(self) -> Mapping[str, JsonValue]:
        return dict(self._changeable)

    def change(self, settings: Mapping[str, JsonValue]) -> None:
        if unknown := sorted(set(settings) - set(self._changeable)):
            known = ", ".join(self._changeable) or "none"
            raise ValueError(f"{', '.join(unknown)} cannot change between steps (these can: {known})")
        self._changeable.update(settings)

    async def describe(self) -> dict[str, Any]:
        """What the pod says its trainer is: its kind, model, `weights`, `budget`, and the settings it takes between
        steps."""
        return await self._call("GET", "/v1/trainer")

    async def made(self, batch: Sequence[Item], *, seed: int, parent: Checkpoint | None, into: str) -> Made:
        """A step from `parent`'s files where they are, in the blob store, making the checkpoint `into` (its id): what
        the pod kept, as manifests, its state whole or (`complete` false) only what was kept with the weights so far."""
        given = None
        if parent is not None:
            if parent.weights is None:
                raise StepFailed(f"{parent.id} was released: its weights are gone")
            given = Parent(parent.weights, parent.state, parent.id)
        made = await self._made(batch, seed, given, into)
        return Made(dict(made.metrics), made.weights, made.state, made.complete)

    async def state(self, into: str) -> Manifest:
        """The whole state of the step that made `into`, once the pod has kept it; `StateLost` where it never will (the
        pod says keeping it failed, or no longer knows the step)."""
        while True:
            try:
                said = state_of(await self._call("GET", f"/v1/steps/{into}"))
            except _Forgotten:
                raise StateLost(f"{self.address} no longer knows step {into}: its state was not kept") from None
            made = said.made
            if said.state != MADE or made is None:
                raise StateLost(f"{self.address} says step {into} is {said.state}: {said.error}")
            if made.state_failed is not None:
                raise StateLost(f"{self.address} did not keep the state of step {into}: {made.state_failed}")
            if made.complete:
                return made.state if made.state is not None else Manifest({})
            await asyncio.sleep(self.every)

    async def step(self, batch: Sequence[Item], *, seed: int, parent: Files | None, into: Path) -> Step:
        """The step, its parent's files kept in the blob store from disk, and the files it made fetched into `into`
        once its state is all kept."""
        given = None
        if parent is not None:
            blobs = self.checkpoints.blobs
            left = await kept(parent.state, blobs) if parent.state is not None else None
            given = Parent(await kept(parent.weights, blobs), left)
        made = await self._made(batch, seed, given, into.name)
        state = made.state if made.complete else await self.state(into.name)
        await self.checkpoints.files(made.weights, into / WEIGHTS)
        if state is not None:
            await self.checkpoints.files(state, into / STATE)
        return Step(dict(made.metrics))

    async def _made(self, batch: Sequence[Item], seed: int, given: Parent | None, into: str) -> StepMade:
        """Ask the pod for the step, with the batch in the blob store, and wait until it is made."""
        async with self._lock:
            reference = await self.checkpoints.blobs.put(batch_bytes(batch), "application/json")
            changed = {key: value for key, value in self._changeable.items() if value != self._made_with.get(key)}
            made = (await self._asked(StepAsked(into, seed, reference, given, changed))).made
            if made is None:
                raise StepFailed(f"{self.address} took step {into} and said it made nothing")
            return made

    async def _asked(self, asked: StepAsked) -> StepState:
        """Ask for the step, and after it until it is made or failed."""
        said = state_of(await self._call("POST", "/v1/steps", asked_json(asked), busy=True))
        while said.state not in (MADE, FAILED):
            await asyncio.sleep(self.every)
            try:
                said = state_of(await self._call("GET", f"/v1/steps/{asked.into}"))
            except _Forgotten:  # (the pod started again and the step went with it: asked for again)
                said = state_of(await self._call("POST", "/v1/steps", asked_json(asked), busy=True))
        if said.state == FAILED:
            raise StepFailed(f"{self.address}: step {asked.into} failed: {said.error}")
        return said

    async def _call(self, method: str, path: str, body: JsonValue = None, *, busy: bool = False) -> dict[str, Any]:
        """One request, tried again while the pod does not answer, for `patience` seconds."""
        given_up = time.monotonic() + self.patience
        while True:
            try:
                response = await self._http.request(method, self.address + path, json=body)
                break
            except httpx.TransportError as error:
                if time.monotonic() > given_up:
                    raise TrainerUnreachable(
                        f"{self.address} did not answer for {self.patience:.0f} seconds: {type(error).__name__}: "
                        f"{error}"
                    ) from error
                await asyncio.sleep(self.every)
        said = answer_of(response)
        if response.status_code == 409 and busy:
            raise TrainerBusy(f"{self.address} is taking another step ({said.get('running')})")
        if response.status_code == 404 and path.startswith("/v1/steps/") and said.get("error") == "no such step":
            raise _Forgotten(path)
        if response.status_code >= 400:
            message = said.get("error") or response.text[:300] or response.reason_phrase
            raise TrainerRefused(f"{self.address} refused {method} {path}: {response.status_code} {message}")
        return said

    async def aclose(self) -> None:
        """Close the client it made (one it was given is its giver's)."""
        if self._owned:
            await self._http.aclose()


class _Forgotten(Exception):
    """The pod has never heard of the step."""
