"""`TinkerEngine`: an `Engine` that samples at Thinking Machines.

Token ids in; ids, their logprobs and why sampling stopped out. No chat template is applied there: the channel's
renderer is the only token format. An adapter is a version's weights directory, whose pointer
(`rollout_tinker.weights`) names the sampler checkpoint to sample from: publishing makes a sampling client for it, at
once, and requests that name the adapter use it. Nothing runs on this machine, so sleeping and waking do nothing.

A call Tinker refuses for billing (402) is fatal: the engine raises `Unpaid` (a `ModelEndpointError`, which the gateway
answers as the endpoint failing) for that turn and every later one, without calling Tinker again.
"""

from collections.abc import Sequence
from pathlib import Path

from tinker import ModelInput, SamplingParams

from rollout_tinker.service import Sampler, Service, Unpaid, said, service_of, unpaid
from rollout_tinker.weights import pointer
from rollout_train.inference import Generation

__all__ = ["TinkerEngine"]


class TinkerEngine:
    """Samples `model` (Tinker's id for it), or the sampler checkpoint an adapter's pointer names. `max_model_len` is
    the longest turn it takes (prompt and completion; Tinker's context for `Qwen/Qwen3.5-9B` is 64K). `service` is
    as `TinkerTrainer` takes it."""

    processes: Sequence[int] = ()

    def __init__(
        self,
        model: str,
        *,
        max_model_len: int = 32_768,
        project: str | None = None,
        service: "Service | str | None" = None,
    ) -> None:
        self.model = model
        self.max_model_len = max_model_len
        self._service = service_of(service, project)
        self._base: Sampler | None = None
        self._adapters: dict[str, Sampler] = {}
        self._unpaid: str | None = None
        """What Tinker said when it refused a call for billing: every call since is refused here."""

    async def generate(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_token_ids: Sequence[int],
        adapter: str | None,
    ) -> Generation:
        self._paid()
        parameters = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=list(stop_token_ids) or None,  # (an empty list would stop it stopping even at the end of text)
        )
        try:
            client = self._adapters[adapter] if adapter is not None else await self._base_client()
            response = await client.sample_async(ModelInput.from_ints(list(prompt)), 1, parameters)
        except Exception as error:
            self._refused(error)
            raise
        sequence = response.sequences[0]
        tokens, logprobs = list(sequence.tokens), sequence.logprobs
        if logprobs is None or len(logprobs) != len(tokens):  # (it could not be trained on)
            raise RuntimeError("the sampler returned a token without its logprob")
        return Generation(
            tokens, [float(each) for each in logprobs], "length" if sequence.stop_reason == "length" else "stop"
        )

    async def load_adapter(self, name: str, path: str) -> None:
        """Sample from the sampler checkpoint that the pointer in `path` (a version's weights) names."""
        sampler = pointer(Path(path), "sampler")
        if sampler is None:
            raise ValueError(f"{path} names no Tinker checkpoint: its version was not trained on Tinker")
        self._paid()
        try:
            self._adapters[name] = await self._service.create_sampling_client_async(model_path=sampler)
        except Exception as error:
            self._refused(error)
            raise

    async def remove_adapter(self, name: str) -> None:
        self._adapters.pop(name, None)

    async def load_weights(self, path: str) -> None:
        raise ValueError("Tinker serves LoRA adapters over its own models, not full weights")

    async def sleep(self) -> None:
        """Nothing to free on this machine."""

    async def wake(self) -> None:
        """Nothing was freed."""

    def close(self) -> None:
        self._adapters.clear()
        self._base = None

    def _paid(self) -> None:
        """Refuse a call once Tinker has refused one for billing."""
        if self._unpaid is not None:
            raise Unpaid(f"tinker refused sampling for billing, so this engine samples no more: {self._unpaid}")

    def _refused(self, error: Exception) -> None:
        """Raise `Unpaid` for an error that is Tinker's refusal for billing, and remember it."""
        if unpaid(error):
            self._unpaid = said(error)
            raise Unpaid(f"tinker refused sampling for billing: {self._unpaid}") from error

    async def _base_client(self) -> Sampler:
        if self._base is None:
            self._base = await self._service.create_sampling_client_async(base_model=self.model)
        return self._base
