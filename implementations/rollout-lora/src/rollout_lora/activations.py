# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave autograd functions partly untyped.)
"""Training activations that take less of the GPU: two changes to gradient checkpointing.

With gradient checkpointing a decoder layer keeps only its input for backward and computes itself again there. Two
things still grow with the segment's length:

- **The layer inputs.** One per layer, `hidden` bf16 values per token (256 KiB a token for Qwen3.5-9B, 2 GiB at
  8,000 tokens). `HostStore` keeps them in (pinned) system memory between forward and backward.
- **The MLP inside a recomputed layer.** Its intermediate activations (12,288 wide, and LoRA's float32 copies) are
  the largest part of the layer's peak. `piecewise_mlp` runs it over the sequence in pieces of `rows` tokens. In
  backward it does so without computing anything twice more than today: a layer's output is
  `residual + mlp(norm(residual))`, so when the layer is computed again for its backward pass, the MLP's output value
  is not needed (only gradients flow back through that sum). The MLP then returns zeros at once and, when the
  gradient arrives, computes each piece with gradients and takes that piece's backward step before the next piece.

`checkpoint_layers` installs both on a transformers model whose layers checkpoint themselves
(`GradientCheckpointingLayer`). Its checkpoint is the reentrant kind: the layer's forward runs without gradients, and
its backward runs the layer again with gradients and calls backward on it. Without gradients (a pass that only
reads logprobs) nothing is stored and the MLP runs in pieces.
"""

import threading
from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn


class HostStore:
    """Layer inputs in system memory between forward and backward. Buffers are kept for reuse (pinning memory is
    slow); a slot is free again once its input has been read back, or once whatever held it is gone (a pass that
    failed half way)."""

    def __init__(self, *, pin: bool) -> None:
        self.pin = pin
        self._buffers: list[torch.Tensor] = []
        self._free: list[int] = []
        self._lock = threading.Lock()

    def put(self, tensor: torch.Tensor) -> "Held":
        size = tensor.numel() * tensor.element_size()
        with self._lock:
            fitting = [index for index in self._free if self._buffers[index].numel() >= size]
            if fitting:
                index = min(fitting, key=lambda each: self._buffers[each].numel())
                self._free.remove(index)
            else:
                index = len(self._buffers)
                self._buffers.append(torch.empty(0, dtype=torch.uint8))
            if self._buffers[index].numel() < size:
                self._buffers[index] = torch.empty(size, dtype=torch.uint8, pin_memory=self.pin)
        host = self._buffers[index][:size].view(tensor.dtype).view(tensor.shape)
        host.copy_(tensor.detach(), non_blocking=self.pin)  # (ordered on the stream before the memory is reused)
        return Held(self, index, host, tensor.device)

    def _release(self, index: int) -> None:
        with self._lock:
            if index not in self._free:
                self._free.append(index)

    @property
    def held(self) -> int:
        """Slots holding an input now."""
        with self._lock:
            return len(self._buffers) - len(self._free)

    @property
    def nbytes(self) -> int:
        """System memory the buffers take."""
        with self._lock:
            return sum(buffer.numel() for buffer in self._buffers)

    def clear(self) -> None:
        """Give the buffers' memory back (between steps, say). Only when nothing is held."""
        with self._lock:
            if len(self._free) != len(self._buffers):
                raise RuntimeError("inputs are still held")
            self._buffers, self._free = [], []


class Held:
    """One stored input; reading it back frees its slot."""

    def __init__(self, store: HostStore, index: int, host: torch.Tensor, device: torch.device) -> None:
        self._store, self._index, self._host, self._device = store, index, host, device
        self._done = False

    def take(self) -> torch.Tensor:
        if self._done:
            raise RuntimeError("this input was read back already")
        tensor = self._host.to(self._device, non_blocking=self._store.pin)
        if self._device.type == "cpu":
            tensor = tensor.clone()  # (on the CPU `to` is the buffer itself)
        self._done = True
        self._store._release(self._index)  # pyright: ignore[reportPrivateUsage]
        return tensor

    def __del__(self) -> None:
        if not self._done:
            self._store._release(self._index)  # pyright: ignore[reportPrivateUsage]


_recomputing = threading.local()
"""Set on the thread that computes a layer again for its backward pass (autograd's own thread)."""


def _is_recomputing() -> bool:
    return bool(getattr(_recomputing, "active", False))


class _LayerCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, function: Callable[[torch.Tensor], torch.Tensor], store: HostStore | None, hidden: torch.Tensor
    ):
        ctx.function = function
        ctx.held = store.put(hidden) if store is not None else hidden.detach()
        with torch.no_grad():
            return function(hidden)

    @staticmethod
    def backward(ctx: Any, *gradients: torch.Tensor) -> tuple[None, None, torch.Tensor | None]:
        (gradient,) = gradients
        held = ctx.held
        hidden = (held.take() if isinstance(held, Held) else held).detach().requires_grad_(True)
        ctx.held = None
        previous = _is_recomputing()
        _recomputing.active = True
        try:
            with torch.enable_grad():
                output = ctx.function(hidden)
        finally:
            _recomputing.active = previous
        torch.autograd.backward(output, gradient)
        return None, None, hidden.grad


def layer_checkpoint(store: HostStore | None) -> Callable[..., torch.Tensor]:
    """A `_gradient_checkpointing_func` for transformers' layers: the layer input in `store` (None: on the GPU)."""

    def checkpointed(function: Callable[..., torch.Tensor], *arguments: Any) -> torch.Tensor:
        if len(arguments) != 1 or not isinstance(arguments[0], torch.Tensor):
            raise TypeError("a checkpointed layer takes its hidden states as its one positional argument")
        (hidden,) = arguments
        if not torch.is_grad_enabled():
            return function(hidden)
        return cast(torch.Tensor, _LayerCheckpoint.apply(function, store, hidden))

    return checkpointed


class _DeferredMlp(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, whole: Callable[[torch.Tensor], torch.Tensor], rows: int):
        ctx.save_for_backward(inputs)
        ctx.whole, ctx.rows = whole, rows
        return torch.zeros_like(inputs)  # (the value is never read: see the module's docstring)

    @staticmethod
    def backward(ctx: Any, *gradients: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (gradient,) = gradients
        (inputs,) = ctx.saved_tensors
        result = torch.empty_like(inputs)
        length = inputs.shape[-2]
        for start in range(0, length, ctx.rows):
            count = min(ctx.rows, length - start)
            piece = inputs.narrow(-2, start, count).detach().requires_grad_(True)
            with torch.enable_grad():
                output = ctx.whole(piece)
            torch.autograd.backward(output, gradient.narrow(-2, start, count))
            result.narrow(-2, start, count).copy_(cast(torch.Tensor, piece.grad))
        return result, None, None


def piecewise_mlp(mlp: nn.Module, rows: int) -> None:
    """Run `mlp` over the sequence (the next-to-last dimension) in pieces of `rows`: without gradients, and when its
    layer is computed again for backward. Its parameters and their names stay as they are."""
    whole = cast(Callable[[torch.Tensor], torch.Tensor], mlp.forward)

    def forward(inputs: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            length = inputs.shape[-2]
            pieces = [whole(inputs.narrow(-2, start, min(rows, length - start))) for start in range(0, length, rows)]
            return torch.cat(pieces, dim=-2)
        if _is_recomputing():
            return cast(torch.Tensor, _DeferredMlp.apply(inputs, whole, rows))
        return whole(inputs)

    object.__setattr__(mlp, "forward", forward)


def checkpoint_layers(model: nn.Module, *, store: HostStore | None, rows: int | None) -> int:
    """Checkpoint each of the model's checkpointing layers with `layer_checkpoint(store)` and, with `rows`, run each
    one's `mlp` in pieces; returns how many layers. Gradient checkpointing must be enabled on the model."""
    from transformers.modeling_layers import GradientCheckpointingLayer

    layers = [module for module in model.modules() if isinstance(module, GradientCheckpointingLayer)]
    if not layers or not all(layer.gradient_checkpointing for layer in layers):
        raise ValueError("enable gradient checkpointing on the model first")
    checkpointed = layer_checkpoint(store)
    for layer in layers:
        object.__setattr__(layer, "_gradient_checkpointing_func", checkpointed)
        mlp = getattr(layer, "mlp", None)
        if rows is not None:
            if not isinstance(mlp, nn.Module):
                raise ValueError(f"{type(layer).__name__} has no MLP to run in pieces")
            piecewise_mlp(mlp, rows)
    return len(layers)
