"""`RemoteTrainer`: a `Trainer` whose steps run on a training pod, through its training service
(`rollout_train.pods.training`).

A step puts its batch and its parent's files in the blob store (content-addressed: files the store has are not written
again), asks the pod for the step by the checkpoint it makes (`into`'s name, the checkpoint's id), and asks after it
every few seconds until it is made or has failed. Then it fetches the new weights and state from the blob store into
`into`, where the training loop looks for them. Asking again is safe: the pod knows a step by its checkpoint, so a step
asked for again after a connection dropped, or by a loop started again, is the same step.

The settings the trainer takes between steps (`Changeable`) are kept here and sent with every step, so the pod's
trainer steps with the settings the run has now, whatever it took before.

Every way a step can fail is a `StepFailed` (the policy stays as it was, and a later step may succeed), with a kind of
its own where the pod is the cause: `TrainerUnreachable` when it did not answer for `patience` seconds,
`TrainerRefused` when it refused the request (the proxy's 404 for a path it does not pass, an error of the service),
`TrainerBusy` when it is taking another run's step.
"""

import asyncio
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import httpx
from pydantic import JsonValue

from rollout_train.checkpoints import Checkpoints, kept
from rollout_train.inference.remote import Connection
from rollout_train.pods.training import FAILED, MADE, Parent, StepAsked, StepState, asked_json, batch_bytes, state_of
from rollout_train.trainer import STATE, WEIGHTS, Budget, Files, Step, StepFailed, Weighted


class TrainerUnreachable(StepFailed):
    """The training pod did not answer for as long as a step waits for it."""


class TrainerRefused(StepFailed):
    """The training pod (or the proxy in front of it) refused the request."""


class TrainerBusy(TrainerRefused):
    """The training pod is taking another step, which it was asked for by someone else."""


class RemoteTrainer:
    """Takes steps on the training service at `address`, its files through `checkpoints`' blob store. `weights` and
    `budget` are what the pod's trainer makes and can take (as the cluster says of it; `describe` asks the pod);
    `changeable` the settings it takes between steps, with their values now. The pod is reached as `connection` says
    (a client certificate, the CA, and the identity the pod's certificate must carry). It is asked after a step every
    `every` seconds; a step fails as `TrainerUnreachable` after `patience` seconds without an answer."""

    def __init__(
        self,
        address: str,
        checkpoints: Checkpoints,
        *,
        weights: str = "lora",
        budget: Budget | None = None,
        changeable: Mapping[str, JsonValue] | None = None,
        connection: Connection | None = None,
        client: httpx.AsyncClient | None = None,
        every: float = 2.0,
        patience: float = 300.0,
    ) -> None:
        self.address = address.rstrip("/")
        self.checkpoints = checkpoints
        self.weights = weights
        self.budget = budget or Budget()
        self._changeable: dict[str, JsonValue] = dict(changeable or {})
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

    async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Files | None, into: Path) -> Step:
        async with self._lock:
            blobs = self.checkpoints.blobs
            reference = await blobs.put(batch_bytes(batch), "application/json")
            given = None
            if parent is not None:
                left = await kept(parent.state, blobs) if parent.state is not None else None
                given = Parent(await kept(parent.weights, blobs), left)
            asked = StepAsked(into.name, seed, reference, given, dict(self._changeable))
            made = (await self._asked(asked)).made
            if made is None:
                raise StepFailed(f"{self.address} took step {into.name} and said it made nothing")
            await self.checkpoints.files(made.weights, into / WEIGHTS)
            if made.state is not None:
                await self.checkpoints.files(made.state, into / STATE)
            return Step(dict(made.metrics))

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
        said = _answer(response)
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


def _answer(response: httpx.Response) -> dict[str, Any]:
    try:
        said: Any = response.json()
    except ValueError:
        return {}
    return cast(dict[str, Any], said) if isinstance(said, dict) else {}
