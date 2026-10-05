"""Channels on hosted APIs: a channel whose provider is of the kind `api` (OpenAI's Responses, Anthropic's Messages),
sampled by message rather than by token.

A hosted API takes messages and returns a message: there are no exact tokens and no behaviour logprobs, so what it
samples is never trained on. The gateway samples such a channel through the endpoint its provider names
(`endpoint = "module:name"`, called as `hosted` is in `rollout_openai` and `rollout_anthropic`: the model, the API key,
the model's context and most output and its options from the catalog, the provider's base URL), made the first time
the channel samples, with the key read then (`Hosted`: the provider's `api_key_env`, or its auth's key or token).

`ApiChannel` samples one model of one such provider:

- **at most `concurrency` requests at once** across every channel of the provider in the process (`Hosted.admission`,
  shared), where the provider says a concurrency;
- **with backoff**: a reply the API cannot give now (`Overloaded`: a rate limit, an overloaded or unavailable API; an
  `InternalError`: a failed stream or connection) is asked for again after a wait that doubles each time, at least what
  the API says to wait (`retry-after`), up to `attempts` times, then raised. Credentials the API refuses, a request it
  rejects or a context too long are raised at once (a `ModelEndpointError`, a `ContextOverflow`): asking again would
  not change the answer, and the episode fails with the reason;
- **with each binding's sampling**: temperature, top-p and the thinking and answer budgets (the binding's, else the
  channel's) are sent to the endpoint with each request.

What a reply cost is counted from its usage at the model's catalog prices (`priced`): dollars per million tokens of
input, of cached input (else the input price), of output and of thinking (else the output price).
"""

import asyncio
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import JsonValue

from rollout.contracts import (
    InternalError,
    ModelEndpointError,
    Overloaded,
    SampleRequest,
    SampleResult,
    Usage,
)
from rollout.harness.runner import SamplingParameters
from rollout.names import named
from rollout_train.inference.channel import Limits, Throughput

if TYPE_CHECKING:
    from rollout_train.providers import InferenceProvider, ModelOffer, Secret

__all__ = ["ATTEMPTS", "ApiChannel", "Hosted", "HostedEndpoint", "priced"]

ATTEMPTS = 6
"""Times a reply the API cannot give now is asked for, before the channel gives up."""
BACKOFF = 1.0
"""Seconds before the first retry; each later one waits twice as long."""
LONGEST_WAIT = 60.0
"""The longest wait between two tries."""


class HostedEndpoint(Protocol):
    """What samples a hosted model: an endpoint that takes the sampling parameters of each request."""

    async def sample(self, request: SampleRequest, *, sampling: SamplingParameters | None = None) -> SampleResult: ...


def priced(usage: Usage, cost: Mapping[str, float]) -> float | None:
    """What a reply cost, in dollars, from its usage at a model's catalog prices (dollars per million tokens of
    `input`, `cached_input`, `output` and `thinking`); none where the catalog prices nothing or the usage counts no
    tokens."""
    if not cost or usage.input_tokens is None or usage.output_tokens is None:
        return None
    cached = min(usage.cached_input_tokens or 0, usage.input_tokens)
    thinking = min(usage.thinking_tokens or 0, usage.output_tokens)
    input_price, output_price = cost.get("input", 0.0), cost.get("output", 0.0)
    dollars = (
        (usage.input_tokens - cached) * input_price
        + cached * cost.get("cached_input", input_price)
        + (usage.output_tokens - thinking) * output_price
        + thinking * cost.get("thinking", output_price)
    )
    return dollars / 1e6


@dataclass
class Hosted:
    """An `api` provider as a gateway reaches it: what makes its models' endpoints (`module:name`), its key, its
    catalog, its base URL, and its concurrency cap with what holds it (`admission`, shared by its channels)."""

    name: str
    endpoint: str
    key: "Secret | None"
    models: Mapping[str, "ModelOffer"]
    base_url: str | None = None
    concurrency: int | None = None
    admission: asyncio.Semaphore | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.admission is None and self.concurrency is not None:
            self.admission = asyncio.Semaphore(self.concurrency)

    @classmethod
    def of(cls, provider: "InferenceProvider") -> "Hosted":
        """A cluster's `api` provider: its key is its `api_key_env` (or `api_key_file`), else its auth's key or
        token."""
        key = provider.secrets.get("api_key") or provider.auth.key or provider.auth.token
        base_url = provider.settings.get("base_url")
        return cls(
            provider.name,
            str(provider.settings["endpoint"]),
            key,
            provider.models,
            str(base_url) if base_url else None,
            provider.concurrency,
        )


class ApiChannel:
    """One model of a hosted API, as a channel (the module's docstring). It serves no checkpoint: every turn is the
    model's own, and none is trained on (`sampled_with` is empty)."""

    sampled_with: tuple[str, ...] = ()

    def __init__(
        self,
        name: str,
        provider: Hosted,
        model: str,
        limits: Limits | None = None,
        *,
        endpoint: HostedEndpoint | None = None,
        attempts: int = ATTEMPTS,
        backoff: float = BACKOFF,
        longest_wait: float = LONGEST_WAIT,
    ) -> None:
        if model not in provider.models:
            raise ValueError(f"provider {provider.name} offers no model {model}")
        self.name = name
        self.provider = provider
        self.model = model
        self.limits = limits or Limits()
        self.offer = provider.models[model]
        self.attempts = attempts
        self.backoff = backoff
        self.longest_wait = longest_wait
        self._endpoint = endpoint
        self._throughput = Throughput()
        self._in_flight = 0
        self._retries = 0
        self._dollars = 0.0

    @property
    def context_limit(self) -> int:
        return self.offer.context

    @property
    def max_output_tokens(self) -> int:
        """The most a reply may take: the model's `max_output_tokens` option, else its context."""
        said = self.offer.options.get("max_output_tokens")
        return min(int(said), self.offer.context) if isinstance(said, int) else self.offer.context

    @property
    def held(self) -> str:
        """What every turn is recorded as served by: the model."""
        return self.model

    def dollars(self, usage: Usage) -> float | None:
        """What a reply with this usage cost (`priced`, at the model's catalog prices)."""
        return priced(usage, self.offer.cost)

    async def weights(self, session: str) -> tuple[str | None, int]:
        """The model's own weights, at depth 0: a hosted model serves no checkpoint."""
        return None, 0

    async def reaches(self) -> bool:
        """Always: whether the API answers is known once it is asked."""
        return True

    async def refresh(self) -> None:
        """Nothing to learn: the catalog says what the model takes."""

    def endpoint(self) -> HostedEndpoint:
        """The endpoint the provider names for the model, made the first time, with the key read then. Raises
        `ModelEndpointError` where it cannot be made (no key, an endpoint that does not import)."""
        if self._endpoint is None:
            provider, offer = self.provider, self.offer
            options: Mapping[str, JsonValue] = offer.options
            try:
                key = provider.key.resolve() if provider.key is not None else None
                made: Any = named(provider.endpoint)(
                    self.model, api_key=key, context_limit=offer.context, max_output_tokens=self.max_output_tokens,
                    options=options, base_url=provider.base_url,
                )  # fmt: skip
            except PermissionError as error:
                where = f" ({provider.key})" if provider.key is not None else ""
                raise ModelEndpointError(f"provider {provider.name} cannot be reached{where}: {error}") from None
            except (ImportError, AttributeError, ValueError, TypeError) as error:
                raise ModelEndpointError(f"provider {provider.name}'s endpoint {provider.endpoint}: {error}") from None
            self._endpoint = made
        assert self._endpoint is not None
        return self._endpoint

    async def sample(
        self,
        request: SampleRequest,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        thinking: int | None = None,
        answer: int | None = None,
    ) -> SampleResult:
        """One reply, with the binding's sampling and budgets (none: the channel's), asked for again while the API
        cannot give it now, up to `attempts` times (the module's docstring)."""
        sampling = SamplingParameters(
            temperature=temperature,
            top_p=top_p,
            thinking_tokens=thinking if thinking is not None else self.limits.thinking,
            answer_tokens=answer if answer is not None else self.limits.answer,
        )
        endpoint = self.endpoint()
        wait = self.backoff
        for attempt in range(1, self.attempts + 1):
            try:
                result = await self._admitted(endpoint, request, sampling)
            except (Overloaded, InternalError) as error:
                if attempt == self.attempts:
                    raise
                said = error.retry_after if isinstance(error, Overloaded) else None
                self._retries += 1
                await asyncio.sleep(min(max(wait * (1 + random.random() / 4), said or 0.0), self.longest_wait))
                wait *= 2
                continue
            except PermissionError as error:
                raise ModelEndpointError(f"provider {self.provider.name} refused the credentials: {error}") from None
            spent = self.dollars(result.usage)
            self._dollars += spent or 0.0
            return result
        raise AssertionError("unreachable")

    async def _admitted(
        self, endpoint: HostedEndpoint, request: SampleRequest, sampling: SamplingParameters
    ) -> SampleResult:
        """One request, once the provider's concurrency admits it."""
        admission = self.provider.admission
        if admission is not None:
            await admission.acquire()
        started = time.monotonic()
        if self._in_flight == 0:
            self._throughput.busy(started)
        self._in_flight += 1
        try:
            result = await endpoint.sample(request, sampling=sampling)
            usage = result.usage
            self._throughput.counted(usage.input_tokens or 0, usage.output_tokens or 0)
            return result
        finally:
            self._in_flight -= 1
            self._throughput.ended(started, idle=self._in_flight == 0)
            if admission is not None:
                admission.release()

    def take(self) -> dict[str, float]:
        """What passed through since the last call: requests, tokens in and out, throughput, the retries and the
        dollars spent."""
        taken = self._throughput.take(busy=self._in_flight > 0)
        taken |= {"retries": float(self._retries), "dollars": round(self._dollars, 6)}
        self._retries, self._dollars = 0, 0.0
        return taken

    def close(self) -> None:
        """Nothing to end: the endpoint's connections close with the process."""
