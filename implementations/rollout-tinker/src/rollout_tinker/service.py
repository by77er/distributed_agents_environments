"""What the trainer and the engine ask of Tinker: the few calls of its SDK they make, as protocols, so that a fake
service (`rollout_tinker.testing`) stands in for it in tests.

The key is the SDK's business: it reads `TINKER_API_KEY`, else the credential `tinker auth login` stored in
`~/.tinker/credentials.json`. Nothing here prints or records it: `said` looks at it only to take it out of an error's
text before that goes anywhere, and `has_key` only asks whether there is one.

A call Tinker refuses for billing (402) is fatal (`Unpaid`): the trainer stops the run rather than fail one step, the
engine refuses every later turn without calling Tinker, and the SDK's own pause-and-retry on 402 is turned off.
"""

import importlib
import os
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from tinker import (
    AdamParams,
    Datum,
    ForwardBackwardOutput,
    ModelInput,
    OptimStepResponse,
    SampleResponse,
    SamplingParams,
)

from rollout.contracts import ModelEndpointError
from rollout.names import named

__all__ = [
    "CREDENTIALS",
    "Rest",
    "Sampler",
    "Service",
    "Trainable",
    "Unpaid",
    "connected",
    "has_key",
    "no_billing_pause",
    "said",
    "service_of",
    "unpaid",
]

CREDENTIALS = Path("~/.tinker/credentials.json")
"""Where `tinker auth login` keeps the key."""
PAYMENT_REQUIRED = 402
"""What Tinker answers a call it will not bill: the balance ran out (no automatic top-up), or billing is not set up."""


class Saved(Protocol):
    path: str


class Sampler(Protocol):
    async def sample_async(
        self, prompt: ModelInput, num_samples: int, sampling_params: SamplingParams
    ) -> SampleResponse: ...


class Trainable(Protocol):
    """A training run: what `tinker.TrainingClient` is."""

    async def forward_async(
        self, data: list[Datum], loss_fn: Any, loss_fn_config: Mapping[str, float | str] | None = None
    ) -> Awaitable[ForwardBackwardOutput]: ...

    async def forward_backward_async(
        self, data: list[Datum], loss_fn: Any, loss_fn_config: Mapping[str, float | str] | None = None
    ) -> Awaitable[ForwardBackwardOutput]: ...

    async def forward_backward_custom_async(
        self, data: list[Datum], loss_fn: Callable[[list[Datum], list[Any]], tuple[Any, dict[str, float]]]
    ) -> Awaitable[ForwardBackwardOutput]: ...

    async def optim_step_async(self, optim_params: AdamParams | None = None) -> Awaitable[OptimStepResponse]: ...

    async def save_state_async(
        self, name: str, ttl_seconds: int | None = None, overwrite: bool = False
    ) -> Awaitable[Saved]: ...

    async def save_weights_for_sampler_async(self, name: str, ttl_seconds: int | None = None) -> Awaitable[Saved]: ...


class Archive(Protocol):
    url: str


class Rest(Protocol):
    async def get_checkpoint_archive_url_from_tinker_path_async(self, tinker_path: str) -> Archive: ...

    async def delete_checkpoint_from_tinker_path_async(self, tinker_path: str) -> None: ...


class Service(Protocol):
    """What `tinker.ServiceClient` is, as far as a trainer and an engine use it."""

    async def create_lora_training_client_async(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
    ) -> Trainable: ...

    async def create_training_client_from_state_async(self, path: str) -> Trainable: ...

    async def create_training_client_from_state_with_optimizer_async(self, path: str) -> Trainable: ...

    async def create_sampling_client_async(
        self, model_path: str | None = None, base_model: str | None = None
    ) -> Sampler: ...

    def create_rest_client(self) -> Rest: ...


class Unpaid(ModelEndpointError):
    """Tinker refused a call for billing (402): the balance ran out, or billing is not set up. It is fatal: what
    raises it never calls Tinker again, and nothing retries it."""


def unpaid(error: BaseException) -> bool:
    """Whether an error is (or was raised from) Tinker's refusal for billing: a 402, which the SDK may wrap."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, Unpaid) or getattr(current, "status_code", None) == PAYMENT_REQUIRED:
            return True
        current = current.__cause__ or current.__context__
    return False


def no_billing_pause() -> None:
    """Have the SDK raise a 402 at once. It otherwise pauses a call refused for billing and asks again every five
    seconds, for up to an hour, as if the balance might come back by itself (with no automatic top-up it does not)."""
    holder: Any = importlib.import_module("tinker.lib.internal_client_holder").InternalClientHolder  # (no stubs)

    def never(self: object, status_code: int, detail: str) -> bool:
        return False

    holder._should_pause_on_billing = never


def connected(project: str | None = None) -> Service:
    """A session with Tinker (in `project`, else `TINKER_PROJECT_ID`'s), its key read by the SDK; a call refused for
    billing raises at once (`no_billing_pause`)."""
    import tinker

    no_billing_pause()
    service: Any = tinker.ServiceClient(project_id=project)
    return service


def service_of(service: "Service | str | None", project: str | None) -> Service:
    """The service a trainer or an engine is given: a service, `module:name` of what makes one (as a profile names
    it), or None for a session with Tinker."""
    if service is None:
        return connected(project)
    if isinstance(service, str):
        return named(service)()
    return service


def has_key() -> bool:
    """Whether the SDK will find a key (without reading it)."""
    return bool(os.environ.get("TINKER_API_KEY")) or CREDENTIALS.expanduser().is_file()


def said(error: BaseException) -> str:
    """What an error says, as its type and message, with the key (if the environment holds it) taken out."""
    text = f"{type(error).__name__}: {error}"
    key = os.environ.get("TINKER_API_KEY")
    return text.replace(key, "[the key]") if key else text
