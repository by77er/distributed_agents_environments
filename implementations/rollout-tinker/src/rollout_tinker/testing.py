# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave parts of autograd and its optimizers untyped.)
"""A fake Tinker, for tests: the calls `TinkerTrainer` and `TinkerEngine` make, on a model small enough to compute.

Its model is a bigram: a token's logits are a fixed table's row for the token before it, plus a learnable table of the
same size (zero at first, as a new LoRA adapter changes nothing). Training runs, their checkpoints and sampling all act
on that learnable table:

- `forward` and `forward_backward` compute each datum's logprobs, and the losses as Tinker's documentation writes them
  (`cross_entropy`, `importance_sampling`, `ppo`, `cispo`, each a sum over rows); the RL losses refuse any input but
  their three.
- `forward_backward_custom` does what the SDK does: logprobs, the caller's loss and its gradient, then a
  `cross_entropy` pass whose weights are minus that gradient.
- `optim_step` is torch's AdamW with the given `AdamParams`, after clipping the gradient's norm.
- `save_state` keeps the table and the optimizer's state, `save_weights_for_sampler` the table, by `tinker://` paths
  named as Tinker names them; a sampling client samples from either, or from the base model.
- The rest client hands out an archive of a checkpoint's adapter, with Tinker's names, made by the `adapter` it is
  given (a test's model of a real one), as a `file://` URL.

`fake_service()` is one service shared by whatever names it (a profile's trainer and engine, say); `reset()` starts it
afresh.
"""

import asyncio
import copy
import io
import itertools
import json
import tarfile
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy
import torch
from tinker import (
    AdamParams,
    Datum,
    ForwardBackwardOutput,
    ModelInput,
    OptimStepResponse,
    SampledSequence,
    SampleResponse,
    SamplingParams,
    TensorData,
    TinkerError,
)

__all__ = ["FakeService", "fake_service", "reset"]

RL_INPUTS = {"target_tokens", "logprobs", "advantages"}
DEFAULT_CLIPS = {"ppo": (0.8, 1.2), "cispo": (0.0, 4.0)}


def _done(value: Any) -> "asyncio.Future[Any]":
    future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
    future.set_result(value)
    return future


@dataclass
class _Saved:
    table: torch.Tensor
    optimizer: dict[str, Any] | None = None
    """For a training state: AdamW's."""
    base_model: str = ""
    rank: int = 32


@dataclass
class _Archive:
    url: str


@dataclass
class _Response:
    path: str


class FakeService:
    """A fake of `tinker.ServiceClient` over a bigram of `vocabulary` tokens. `favored` tokens get `lift` more in
    the base model's logits (a test's environment wants some answers likelier than others). `adapter` makes the
    tensors of a checkpoint's archive (Tinker's names, from its path), for tests of `weights = "peft"`."""

    def __init__(
        self,
        vocabulary: int = 128,
        *,
        seed: int = 0,
        favored: Sequence[int] = (),
        lift: float = 4.0,
        adapter: Callable[[str], Mapping[str, torch.Tensor]] | None = None,
        rank: int = 32,
        alpha: float = 32.0,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.base = torch.randn(vocabulary, vocabulary, generator=generator, dtype=torch.float64)
        for token in favored:
            self.base[:, token] += lift
        self.saved: dict[str, _Saved] = {}
        self.calls: list[str] = []
        """What was asked, in order: `lora RANK`, `state PATH`, `state with optimizer PATH`, `sampler PATH`,
        `sampler BASE`, `forward`, `forward_backward LOSS`, `custom`, `optim`, `save PATH`, `archive PATH`, `delete
        PATH`."""
        self.sampled: list[str | None] = []
        """The checkpoint each sample came from (None: the base model)."""
        self.adapter = adapter
        self.rank, self.alpha = rank, alpha
        self._runs = itertools.count(1)
        self._samples = itertools.count(1)
        self._archives = Path(tempfile.mkdtemp(prefix="fake-tinker-"))

    def table(self, path: str) -> torch.Tensor:
        """The learnable table a checkpoint holds."""
        return self.saved[path].table

    async def create_lora_training_client_async(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
    ) -> "FakeTraining":
        self.calls.append(f"lora {rank}")
        return FakeTraining(self, base_model, rank)

    async def create_training_client_from_state_async(self, path: str) -> "FakeTraining":
        self.calls.append(f"state {path}")
        saved = self._found(path, "weights")
        return FakeTraining(self, saved.base_model, saved.rank, table=saved.table)

    async def create_training_client_from_state_with_optimizer_async(self, path: str) -> "FakeTraining":
        self.calls.append(f"state with optimizer {path}")
        saved = self._found(path, "weights")
        return FakeTraining(self, saved.base_model, saved.rank, table=saved.table, optimizer=saved.optimizer)

    async def create_sampling_client_async(
        self, model_path: str | None = None, base_model: str | None = None
    ) -> "FakeSampler":
        if model_path is not None:
            self.calls.append(f"sampler {model_path}")
            return FakeSampler(self, model_path, self._found(model_path, "sampler_weights").table)
        if base_model is None:
            raise ValueError("Either model_path or base_model must be provided")
        self.calls.append(f"sampler {base_model}")
        return FakeSampler(self, None, torch.zeros_like(self.base))

    def create_rest_client(self) -> "FakeRest":
        return FakeRest(self)

    def logprobs(self, table: torch.Tensor, inputs: Sequence[int], targets: Sequence[int]) -> torch.Tensor:
        """Each target's logprob after its input token, under the base model plus `table`."""
        index = torch.tensor(list(inputs))
        logits = self.base[index] + table[index]
        return torch.log_softmax(logits, dim=-1).gather(1, torch.tensor(list(targets))[:, None])[:, 0]

    def _found(self, path: str, kind: str) -> _Saved:
        if f"/{kind}/" not in path or path not in self.saved:
            raise TinkerError(f"no {kind} checkpoint at {path}")
        return self.saved[path]

    def _run(self) -> str:
        return f"run-{next(self._runs)}"


class FakeTraining:
    """A fake of `tinker.TrainingClient`: a training run of the bigram's learnable table."""

    def __init__(
        self,
        service: FakeService,
        base_model: str,
        rank: int,
        *,
        table: torch.Tensor | None = None,
        optimizer: dict[str, Any] | None = None,
    ) -> None:
        self.service = service
        self.base_model, self.rank = base_model, rank
        self.run = service._run()  # pyright: ignore[reportPrivateUsage]
        start = table.clone() if table is not None else torch.zeros_like(service.base)
        self.table = torch.nn.Parameter(start)
        self.optimizer = torch.optim.AdamW([self.table], lr=0.0, weight_decay=0.0)
        if optimizer is not None:
            self.optimizer.load_state_dict(copy.deepcopy(optimizer))

    async def forward_async(
        self, data: list[Datum], loss_fn: Any, loss_fn_config: Mapping[str, float | str] | None = None
    ) -> "asyncio.Future[ForwardBackwardOutput]":
        self.service.calls.append("forward")
        with torch.no_grad():
            return _done(self._passed(data, str(loss_fn), loss_fn_config, backward=False))

    async def forward_backward_async(
        self, data: list[Datum], loss_fn: Any, loss_fn_config: Mapping[str, float | str] | None = None
    ) -> "asyncio.Future[ForwardBackwardOutput]":
        self.service.calls.append(f"forward_backward {loss_fn}")
        return _done(self._passed(data, str(loss_fn), loss_fn_config, backward=True))

    async def forward_backward_custom_async(
        self, data: list[Datum], loss_fn: Callable[[list[Datum], list[Any]], tuple[Any, dict[str, float]]]
    ) -> "asyncio.Future[ForwardBackwardOutput]":
        self.service.calls.append("custom")
        logprobs: list[torch.Tensor] = []
        with torch.no_grad():
            for each in data:
                found = self.service.logprobs(self.table, each.model_input.to_ints(), _ints(each, "target_tokens"))
                logprobs.append(found.clone().requires_grad_(True))
        loss, metrics = loss_fn(data, logprobs)
        custom = float(loss.detach().sum())
        loss.backward()
        stand_in = [
            Datum(
                model_input=each.model_input,
                loss_fn_inputs={
                    "target_tokens": each.loss_fn_inputs["target_tokens"],
                    "weights": TensorData(data=(-found.grad).tolist(), dtype="float32", shape=[len(found)]),
                },
            )
            for each, found in zip(data, logprobs, strict=True)
            if found.grad is not None
        ]
        output = self._passed(stand_in, "cross_entropy", None, backward=True)
        output.metrics.update({**metrics, "surrogate_loss:sum": output.metrics["loss:sum"], "loss:sum": custom})
        return _done(output)

    async def optim_step_async(self, optim_params: AdamParams | None = None) -> "asyncio.Future[OptimStepResponse]":
        self.service.calls.append("optim")
        params = optim_params or AdamParams()
        if self.table.grad is None:
            self.table.grad = torch.zeros_like(self.table)
        limit = params.grad_clip_norm if params.grad_clip_norm > 0 else float("inf")
        norm = torch.nn.utils.clip_grad_norm_([self.table], limit)
        for group in self.optimizer.param_groups:
            group.update(lr=params.learning_rate, betas=(params.beta1, params.beta2), eps=params.eps,
                         weight_decay=params.weight_decay)  # fmt: skip
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return _done(OptimStepResponse(metrics={"grad_norm": float(norm)}))

    async def save_state_async(
        self, name: str, ttl_seconds: int | None = None, overwrite: bool = False
    ) -> "asyncio.Future[_Response]":
        path = f"tinker://{self.run}/weights/{name}"
        self.service.calls.append(f"save {path}")
        if path in self.service.saved and not overwrite:
            raise TinkerError(f"{path} exists")
        state = copy.deepcopy(self.optimizer.state_dict())
        self.service.saved[path] = _Saved(self.table.detach().clone(), state, self.base_model, self.rank)
        return _done(_Response(path))

    async def save_weights_for_sampler_async(
        self, name: str, ttl_seconds: int | None = None
    ) -> "asyncio.Future[_Response]":
        path = f"tinker://{self.run}/sampler_weights/{name}"
        self.service.calls.append(f"save {path}")
        if path in self.service.saved:
            raise TinkerError(f"{path} exists")
        self.service.saved[path] = _Saved(self.table.detach().clone(), None, self.base_model, self.rank)
        return _done(_Response(path))

    def _passed(
        self, data: list[Datum], loss_fn: str, config: Mapping[str, float | str] | None, *, backward: bool
    ) -> ForwardBackwardOutput:
        outputs: list[dict[str, TensorData]] = []
        total = torch.zeros((), dtype=torch.float64)
        for each in data:
            inputs = each.model_input.to_ints()
            targets = _ints(each, "target_tokens")
            if len(inputs) != len(targets):
                raise TinkerError(f"{len(inputs)} input tokens for {len(targets)} targets")
            logprobs = self.service.logprobs(self.table, inputs, targets)
            total = total + _loss(loss_fn, each, logprobs, config)
            outputs.append(
                {"logprobs": TensorData(data=logprobs.detach().tolist(), dtype="float32", shape=[len(targets)])}
            )
        if backward:
            total.backward()
        return ForwardBackwardOutput(loss_fn_output_type="ArrayRecord", loss_fn_outputs=outputs,
                                     metrics={"loss:sum": float(total.detach())})  # fmt: skip


def _loss(loss_fn: str, each: Datum, logprobs: torch.Tensor, config: Mapping[str, float | str] | None) -> torch.Tensor:
    """One datum's loss, as Tinker's documentation writes each (a sum over its rows)."""
    given = set(each.loss_fn_inputs)
    if loss_fn == "cross_entropy":
        if given != {"target_tokens", "weights"}:
            raise TinkerError(f"cross_entropy takes target_tokens and weights, not {sorted(given)}")
        return -(logprobs * _floats(each, "weights")).sum()
    if given != RL_INPUTS:
        raise TinkerError(f"{loss_fn} takes target_tokens, logprobs and advantages, not {sorted(given)}")
    sampling, advantages = _floats(each, "logprobs"), _floats(each, "advantages")
    ratio = torch.exp(logprobs - sampling)
    if loss_fn == "importance_sampling":
        return -(ratio * advantages).sum()
    low, high = DEFAULT_CLIPS.get(loss_fn, (0.0, 0.0))
    if config:
        low, high = float(config.get("clip_low_threshold", low)), float(config.get("clip_high_threshold", high))
    clipped = torch.clamp(ratio, low, high)
    if loss_fn == "ppo":
        return -torch.min(ratio * advantages, clipped * advantages).sum()
    if loss_fn == "cispo":
        return -(clipped.detach() * logprobs * advantages).sum()
    raise TinkerError(f"no loss {loss_fn}")


def _ints(each: Datum, key: str) -> list[int]:
    return [int(value) for value in each.loss_fn_inputs[key].data]


def _floats(each: Datum, key: str) -> torch.Tensor:
    return torch.tensor([float(value) for value in each.loss_fn_inputs[key].data], dtype=torch.float64)


class FakeSampler:
    """A fake of `tinker.SamplingClient`: samples from the bigram plus a checkpoint's table."""

    def __init__(self, service: FakeService, path: str | None, table: torch.Tensor) -> None:
        self.service, self.path, self.table = service, path, table
        self._generator = torch.Generator().manual_seed(len(service.sampled))

    async def sample_async(
        self, prompt: ModelInput, num_samples: int, sampling_params: SamplingParams
    ) -> SampleResponse:
        if sampling_params.top_p != 1:
            raise TinkerError("the fake samples the whole distribution only (top_p 1)")
        stops = set(sampling_params.stop) if isinstance(sampling_params.stop, list) else set[Any]()
        sequences: list[SampledSequence] = []
        for _ in range(num_samples):
            self.service.sampled.append(self.path)
            last, tokens, logprobs, reason = prompt.to_ints()[-1], list[int](), list[float](), "length"
            for _ in range(sampling_params.max_tokens or 16):
                logits = (self.service.base[last] + self.table[last]) / max(sampling_params.temperature, 1e-6)
                distribution = torch.log_softmax(logits, dim=-1)
                last = int(torch.multinomial(distribution.exp(), 1, generator=self._generator))
                tokens.append(last)
                logprobs.append(float(distribution[last]))
                if last in stops:
                    reason = "stop"
                    break
            sequences.append(
                SampledSequence(
                    stop_reason="stop" if reason == "stop" else "length",
                    sequence_id=f"s{next(self.service._samples)}",  # pyright: ignore[reportPrivateUsage]
                    tokens_np=numpy.array(tokens, dtype=numpy.int32),
                    logprobs_np=numpy.array(logprobs, dtype=numpy.float32),
                )
            )
        return SampleResponse(sequences=sequences)


@dataclass
class FakeRest:
    """A fake of `tinker.RestClient`: archives of checkpoints' adapters, and deleting checkpoints."""

    service: FakeService
    deleted: list[str] = field(default_factory=list[str])

    async def get_checkpoint_archive_url_from_tinker_path_async(self, tinker_path: str) -> _Archive:
        self.service.calls.append(f"archive {tinker_path}")
        saved = self.service._found(tinker_path, "sampler_weights")  # pyright: ignore[reportPrivateUsage]
        if self.service.adapter is None:
            raise TinkerError("this fake was given no adapter to archive")
        from safetensors.torch import save as saved_tensors

        tensors = {key: value.contiguous() for key, value in self.service.adapter(tinker_path).items()}
        config = {"r": saved.rank, "lora_alpha": self.service.alpha, "base_model": saved.base_model}
        archive = self.service._archives / f"{len(self.service.calls)}.tar"  # pyright: ignore[reportPrivateUsage]
        with tarfile.open(archive, "w") as tar:
            for name, content in (
                ("adapter_model.safetensors", saved_tensors(tensors)),
                ("adapter_config.json", json.dumps(config).encode()),
            ):
                entry = tarfile.TarInfo(name)
                entry.size = len(content)
                tar.addfile(entry, io.BytesIO(content))
        return _Archive(archive.as_uri())

    async def delete_checkpoint_from_tinker_path_async(self, tinker_path: str) -> None:
        self.service.calls.append(f"delete {tinker_path}")
        self.service.saved.pop(tinker_path, None)
        self.deleted.append(tinker_path)


SHARED: list[FakeService] = []


def fake_service() -> FakeService:
    """The one fake service of this process (what a profile names: `rollout_tinker.testing:fake_service`)."""
    if not SHARED:
        SHARED.append(FakeService(favored=[ord("a"), ord("b"), ord("\n")]))
    return SHARED[0]


def reset(service: FakeService | None = None) -> FakeService:
    """Start the shared fake afresh (as `service`, if given)."""
    SHARED.clear()
    SHARED.append(service or FakeService(favored=[ord("a"), ord("b"), ord("\n")]))
    return SHARED[0]
