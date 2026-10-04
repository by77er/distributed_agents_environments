"""What a runner needs to have its recorded slots served by the gateway: endpoints that sample there, a key for each
harness it starts, and the segments its runs recorded, read back from the turn store.

`GatewayEndpoints` is a runner's `RecordedEndpoints`. A runner tells it, before a run starts, which attempt the run
plays (`admit`: the run, the episode and attempt, the fence its turns are recorded under); from then on each of the
run's recorded slots samples through the gateway under a key minted for it, and a harness the run starts is handed the
gateway's address and such a key. When the run ends, `sessions` reads what each slot recorded.

The gateway is in this process (`GatewayEndpoints.of`: a sample is a call, as the gateway's own HTTP handlers make it)
or elsewhere, at a URL. There, a sample is retried under its effect id when the gateway cannot be reached or a replica
fails, so a replica that dies mid-turn costs a retry on another, never a second recorded turn. What a channel the
gateway there hosts (its engines in the gateway's own processes) guarantees is what the gateway says of it (`hosted`).
"""

import asyncio
import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, cast

import httpx

from rollout.contracts import (
    CapabilityContract,
    ContextOverflow,
    ModelAddress,
    ModelEndpointError,
    SampleRequest,
    SampleResult,
    SessionIdentity,
)
from rollout.harness.runner import RecordedModel, RunBinding
from rollout_train.gateway.keys import Grant, Keyring, granted
from rollout_train.gateway.service import Gateway, Refused, contract_of, limits_of
from rollout_train.gateway.turns import TurnStore
from rollout_train.inference import Routes
from rollout_train.inference.channel import Sampler
from rollout_train.ledger import Fence
from rollout_train.recorder.compat import SERVED_UNDER
from rollout_train.recorder.segments import Segment
from rollout_train.serving import parts

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
    """Implements `RecordedEndpoints` over a gateway: the one in this process (`gateway`), else the one at `url` (its
    base URL, without `/v1`), with keys signed by `keyring`; `store` is the turn store the gateway records in. What a
    channel guarantees is the gateway's to say, in this process; else, for a routed channel, what `routes` say of it
    (the runner's view of the same servers), and for any other, `contracts` (for a channel the gateway at `url` hosts,
    what it says of it: `hosted`). `url` is also what a harness is handed:
    without one, a run cannot give a harness an address."""

    def __init__(
        self,
        url: str | None,
        keyring: Keyring,
        store: TurnStore,
        contracts: Mapping[str, CapabilityContract] | None = None,
        *,
        gateway: Gateway | None = None,
        routes: Routes | None = None,
        lifetime: float = LIFETIME,
        http: httpx.AsyncClient | None = None,
        retries: int = 5,
        backoff: float = 0.5,
    ) -> None:
        if url is None and gateway is None:
            raise ValueError("a gateway in this process, or the URL of one")
        self.url = url.rstrip("/") if url is not None else None
        self.keyring = keyring
        self.store = store
        self.contracts = dict(contracts or {})
        self.gateway = gateway
        self.routes = gateway.routes if gateway is not None else routes
        self.lifetime = lifetime
        self._http = http
        self.retries = retries
        self.backoff = backoff
        self._attempts: dict[str, Attempt] = {}

    @classmethod
    def of(cls, gateway: Gateway, url: str | None = None, *, lifetime: float = LIFETIME) -> "GatewayEndpoints":
        """Endpoints over a gateway in this process; `url`, where it is also served over HTTP, for harnesses."""
        return cls(url, gateway.keyring, gateway.store, gateway=gateway, lifetime=lifetime)

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
        return self._http

    @property
    def channels(self) -> list[str]:
        """The channels it samples, by name."""
        routed: Mapping[str, object] = self.routes.routes if self.routes is not None else {}
        local = self.gateway.names if self.gateway is not None else list(self.contracts)
        return [*local, *(name for name in [*self.contracts, *routed] if name not in local)]

    async def hosted(self, channels: Collection[str], *, patience: float = 60.0) -> None:
        """Learn what the gateway at the URL guarantees of each of `channels`, which its own engines serve (its
        `/v1/models` says each one's contract), asking again while it cannot be reached, for up to `patience` seconds.
        Raises `ModelEndpointError` when it cannot be reached, or does not host one of them."""
        if self.url is None:
            raise ValueError("a gateway in this process says what its channels guarantee")
        deadline, failure, wait = time.monotonic() + patience, "no attempt was made", self.backoff
        while True:
            try:
                response = await self.http.get(f"{self.url}{SERVED_UNDER}/models")
                if response.status_code == 200:
                    break
                failure = f"{response.status_code}: {response.text[:200]}"
            except httpx.TransportError as error:
                failure = f"{type(error).__name__}: {error}"
            if time.monotonic() + wait > deadline:
                raise ModelEndpointError(f"the gateway at {self.url} did not say its channels: {failure}")
            await asyncio.sleep(wait)
            wait = min(wait * 2, 5.0)
        said: Any = response.json()
        listed = cast(
            list[dict[str, Any]], cast(dict[str, Any], said).get("data", []) if isinstance(said, dict) else []
        )
        contracts = {str(each.get("id")): each["contract"] for each in listed if isinstance(each.get("contract"), dict)}
        if missing := sorted(set(channels) - set(contracts)):
            raise ModelEndpointError(f"the gateway at {self.url} hosts no channel {', '.join(missing)}")
        self.contracts |= {name: CapabilityContract.model_validate(contracts[name]) for name in channels}

    def admit(self, run_id: str, attempt: Attempt) -> None:
        """Say which attempt a program's run plays, before it starts (and again when the attempt is taken up anew)."""
        self._attempts[run_id] = attempt

    def forget(self, run_id: str) -> None:
        self._attempts.pop(run_id, None)

    def endpoint(self, binding: RecordedModel) -> "GatewayEndpoint":
        if _name(binding.channel) not in self.channels:
            raise ValueError(f"no recorded channel {binding.channel!r}")
        return GatewayEndpoint(self, binding)

    def attempt(self, run_id: str) -> Attempt:
        """The attempt an admitted run plays."""
        attempt = self._attempts.get(run_id)
        if attempt is None:
            raise RuntimeError(f"run {run_id} was not admitted: the gateway cannot record its turns")
        return attempt

    def key(self, session_id: str, binding: RecordedModel) -> str:
        """A key for a session of an admitted run."""
        identity = SessionIdentity.parse(session_id)
        attempt = self.attempt(identity.owner)
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
            thinking=binding.sampling.thinking_tokens,
            answer=binding.sampling.answer_tokens,
        )
        return self.keyring.mint(granted(grant, self.lifetime))

    def contract(self, session_id: str, binding: RecordedModel) -> CapabilityContract:
        """What a binding's channel guarantees a session, with the thinking and answer room the binding gives in place
        of the channel's: a routed channel, as its run's servers say (the run is admitted)."""
        channel, said = binding.channel, binding.sampling
        name = _name(channel)
        sampler: Sampler | None = None
        if self.gateway is not None and name in self.gateway.channels:
            sampler = self.gateway.channels[name]
        elif self.routes is not None and self.routes.routed(name):
            run = parts(channel)[0] if "/" in channel else self.attempt(SessionIdentity.parse(session_id).owner).run
            sampler = self.routes.channel(run, name)
        if sampler is not None:
            return contract_of(sampler, limits_of(sampler.limits, said.thinking_tokens, said.answer_tokens))
        if name not in self.contracts:
            raise ModelEndpointError(f"no recorded channel {channel!r}")
        given = self.contracts[name]
        if said.thinking_tokens is None or said.answer_tokens is None:
            return given
        return given.model_copy(update={"max_output_tokens": said.thinking_tokens + said.answer_tokens})

    async def reaches(self, run: str, binding: RunBinding) -> bool:
        """Whether every recorded model of a run's binding can be sampled now: a channel the gateway in this process
        samples (a routed one only once its servers have a checkpoint close enough to what the run says it should
        serve), or one the gateway elsewhere serves (a routed one likewise, as this process sees its servers)."""
        for model in binding.models.values():
            recorded = model.recorded
            if recorded is None:
                continue
            if self.gateway is not None:
                if not await self.gateway.reaches(run, recorded.channel):
                    return False
                continue
            name = _name(recorded.channel)
            if self.routes is not None and self.routes.routed(name):
                if not await self.routes.reaches(run, name):
                    return False
            elif name not in self.contracts:
                return False
        return True

    async def sessions(self, run: str, run_id: str) -> dict[str, list[Segment]]:
        """What each slot of a program's run recorded, by slot."""
        return await self.store.sessions(run, run_id)


class GatewayEndpoint:
    """Implements `AddressableEndpoint` for one recorded binding, through the gateway."""

    def __init__(self, endpoints: GatewayEndpoints, binding: RecordedModel) -> None:
        self._endpoints = endpoints
        self._binding = binding

    def describe(self, session_id: str) -> CapabilityContract:
        return self._endpoints.contract(session_id, self._binding)

    def address(self, session_id: str) -> ModelAddress:
        """The gateway, and a key for the session. What a harness samples there is recorded by the gateway (a gateway
        in this process tells the runner's hooks of it)."""
        endpoints = self._endpoints
        if endpoints.url is None:
            raise RuntimeError("the gateway is not served over HTTP: a harness cannot be given an address")
        key = endpoints.key(session_id, self._binding)
        return ModelAddress(base_url=f"{endpoints.url}{SERVED_UNDER}", api_key=key, model=self._binding.channel)

    async def cancel(self, effect_id: str) -> None:
        """Nothing to do: a turn whose sampling is cancelled in this process is not recorded, and a gateway elsewhere
        records the turn whether or not it is awaited."""

    async def sample(self, request: SampleRequest) -> SampleResult:
        endpoints = self._endpoints
        key = endpoints.key(request.session_id, self._binding)
        if endpoints.gateway is not None:
            try:
                grant = endpoints.gateway.granted(key)
                return (await endpoints.gateway.sample(grant, request)).result
            except Refused as error:
                raise ModelEndpointError(f"the gateway refused the sample: {error}") from None
        return await self._posted(key, request)

    async def _posted(self, key: str, request: SampleRequest) -> SampleResult:
        """A sample of the gateway at its URL, posted again under the same effect id while it cannot be reached or a
        replica fails."""
        endpoints = self._endpoints
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


def _name(channel: str) -> str:
    """A channel's name, where it is named within its run (`RUN/NAME`) or not."""
    return parts(channel)[1] if "/" in channel else channel


def _error(response: httpx.Response) -> dict[str, Any]:
    """What an error response of the gateway's says: its `type`, `message` and (for a context too long) its
    `context_limit`."""
    if not response.headers.get("content-type", "").startswith("application/json"):
        return {}
    said: Any = response.json()
    error: Any = cast(dict[str, Any], said).get("error") if isinstance(said, dict) else None
    return cast(dict[str, Any], error) if isinstance(error, dict) else {}
