"""What a runner needs to have its recorded slots served by the gateway: endpoints that sample there, a key for each
harness it starts, and the segments its runs recorded, read back from the turn store.

`GatewayEndpoints` stands where a runner's `Recorder` stands. A runner tells it, before a run starts, which attempt
the run plays (`admit`: the run, the episode and attempt, the fence its turns are recorded under); from then on each
of the run's recorded slots samples through the gateway under a key minted for it, and a harness the run starts is
handed the gateway's address and such a key. When the run ends, `sessions` reads what each slot recorded.

A sample is retried under its effect id when the gateway cannot be reached or a replica fails, so a replica that dies
mid-turn costs a retry on another, never a second recorded turn.
"""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import httpx

from rollout.contracts import (
    CapabilityContract,
    ContextOverflow,
    ModelAddress,
    ModelEndpoint,
    ModelEndpointError,
    SampleRequest,
    SampleResult,
    SessionIdentity,
)
from rollout.harness.runner import RecordedModel
from rollout_train.gateway.keys import Grant, Keyring, granted
from rollout_train.gateway.turns import TurnStore
from rollout_train.ledger import Fence
from rollout_train.recorder.recorder import SERVED_UNDER, Segment

LIFETIME = 6 * 3600.0
"""Seconds a key minted for a run's slot is good for."""


@dataclass(frozen=True)
class Attempt:
    """What a program's run plays, as the keys of its slots say."""

    run: str
    """The run whose tables its turns go under."""
    fence: Fence
    """What its turns are appended under."""
    episode: str = ""
    """`GROUP/EPISODE`."""
    attempt: int = 0


class GatewayEndpoints:
    """Implements `RecordedEndpoints` over the gateway at `url` (its base URL, without `/v1`), with keys signed by
    `keyring`; `contracts` is each channel's capability contract, and `store` the turn store the gateway records in."""

    def __init__(
        self,
        url: str,
        keyring: Keyring,
        store: TurnStore,
        contracts: Mapping[str, CapabilityContract],
        *,
        lifetime: float = LIFETIME,
        http: httpx.AsyncClient | None = None,
        retries: int = 5,
        backoff: float = 0.5,
    ) -> None:
        self.url = url.rstrip("/")
        self.keyring = keyring
        self.store = store
        self.contracts = contracts
        self.lifetime = lifetime
        self.http = http or httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
        self.retries = retries
        self.backoff = backoff
        self._attempts: dict[str, Attempt] = {}

    @property
    def channels(self) -> Mapping[str, CapabilityContract]:
        return self.contracts

    def admit(self, run_id: str, attempt: Attempt) -> None:
        """Say which attempt a program's run plays, before it starts."""
        self._attempts[run_id] = attempt

    def forget(self, run_id: str) -> None:
        self._attempts.pop(run_id, None)

    def endpoint(self, binding: RecordedModel) -> "GatewayEndpoint":
        if binding.channel not in self.contracts:
            raise ValueError(f"no recorded channel {binding.channel!r}")
        return GatewayEndpoint(self, binding)

    def key(self, session_id: str, binding: RecordedModel) -> str:
        """A key for a session of an admitted run."""
        identity = SessionIdentity.parse(session_id)
        attempt = self._attempts.get(identity.owner)
        if attempt is None:
            raise RuntimeError(f"run {identity.owner} was not admitted: the gateway cannot record its turns")
        grant = Grant(
            run=attempt.run,
            run_id=identity.owner,
            slot=identity.model_slot,
            channel=binding.channel,
            fence=attempt.fence,
            expires=0.0,
            episode=attempt.episode,
            attempt=attempt.attempt,
            temperature=binding.sampling.temperature,
            top_p=binding.sampling.top_p,
        )
        return self.keyring.mint(granted(grant, self.lifetime))

    async def sessions(self, run: str, run_id: str) -> dict[str, list[Segment]]:
        """What each slot of a program's run recorded, by slot."""
        return await self.store.sessions(run, run_id)


class GatewayEndpoint:
    """Implements `AddressableEndpoint` for one recorded binding, through the gateway."""

    def __init__(self, endpoints: GatewayEndpoints, binding: RecordedModel) -> None:
        self._endpoints = endpoints
        self._binding = binding

    def describe(self, session_id: str) -> CapabilityContract:
        return self._endpoints.contracts[self._binding.channel]

    def address(self, session_id: str, *, through: ModelEndpoint | None = None) -> ModelAddress:
        """The gateway, and a key for the session. What a harness samples there is recorded by the gateway, so it does
        not go `through` the runner's endpoint (its hooks do not see it)."""
        key = self._endpoints.key(session_id, self._binding)
        return ModelAddress(base_url=f"{self._endpoints.url}{SERVED_UNDER}", api_key=key, model=self._binding.channel)

    async def cancel(self, effect_id: str) -> None:
        """Nothing to do: the gateway records the turn whether or not it is awaited."""

    async def sample(self, request: SampleRequest) -> SampleResult:
        endpoints = self._endpoints
        key = endpoints.key(request.session_id, self._binding)
        body = request.model_dump(mode="json")
        failure = "no attempt was made"
        for attempt in range(endpoints.retries + 1):
            if attempt:
                await asyncio.sleep(endpoints.backoff * 2 ** (attempt - 1))
            try:
                response = await endpoints.http.post(
                    f"{endpoints.url}{SERVED_UNDER}/samples", json=body, headers={"Authorization": f"Bearer {key}"}
                )
            except httpx.TransportError as error:  # (a replica that died, or none reachable: the same effect again)
                failure = f"{type(error).__name__}: {error}"
                continue
            if response.status_code == 200:
                return SampleResult.model_validate(response.json())
            error = _error(response)
            failure = f"{response.status_code} {error.get('type', '')}: {error.get('message', response.text)}"
            if error.get("type") == "ContextOverflow":
                raise ContextOverflow(int(error["context_limit"]))
            if response.status_code < 500:
                raise ModelEndpointError(f"the gateway refused the sample: {failure}")
        raise ModelEndpointError(f"the gateway did not answer after {endpoints.retries + 1} tries: {failure}")


def _error(response: httpx.Response) -> dict[str, Any]:
    """What an error response of the gateway's says: its `type`, `message` and (for a context too long) its
    `context_limit`."""
    if not response.headers.get("content-type", "").startswith("application/json"):
        return {}
    said: Any = response.json()
    error: Any = cast(dict[str, Any], said).get("error") if isinstance(said, dict) else None
    return cast(dict[str, Any], error) if isinstance(error, dict) else {}
