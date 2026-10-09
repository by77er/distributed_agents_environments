# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave module iteration and autograd functions partly untyped.)
"""4-bit linear layers that train LoRA without ever holding a dequantized model.

The trainer and the engine read the same checkpoint (compressed-tensors `pack-quantized`: symmetric int4 packed
eight to an int32, one bf16 scale per group of input columns), so behavior and trainer logprobs agree. `Int4Linear`
dequantizes its weight inside the matrix multiply, in forward and again in backward: only the packed weight is kept
for backward, so at most one layer's bf16 weight exists at a time, whether or not activations are checkpointed.

On a GPU, a Triton kernel (`rollout_lora.kernels`) dequantizes a weight in one pass, reading the packed words and
writing the weight; torch's own operations, which the CPU runs, write and read whole int32, int8 and bf16 copies of
it between. Both round each weight once, from its value times its scale in float32, and agree bit for bit.
"""

import functools
import importlib.util
from typing import Any, cast

import torch
from torch import nn


@functools.cache
def _kernels() -> bool:
    """Whether Triton is here to compile `rollout_lora.kernels`."""
    return importlib.util.find_spec("triton") is not None


def unpack_int4(packed: torch.Tensor, columns: int) -> torch.Tensor:
    """Signed int4 values (-8..7) from int32 words holding eight each, lowest bits first."""
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    nibbles = (packed.unsqueeze(-1) >> shifts) & 0xF
    return (nibbles.reshape(packed.shape[0], -1)[:, :columns] - 8).to(torch.int8)


def dequantize(packed: torch.Tensor, scale: torch.Tensor, columns: int, dtype: torch.dtype) -> torch.Tensor:
    """The weight in `dtype`: each int4 value times its group's scale (`scale`, one for each `columns // groups`
    columns of a row), by the Triton kernel on a GPU, else by torch's operations."""
    if packed.is_cuda and _kernels():
        from rollout_lora.kernels import dequantized

        return dequantized(packed, scale, columns, dtype)
    values = unpack_int4(packed, columns)
    group = columns // scale.shape[1]
    return (values.reshape(values.shape[0], -1, group).to(dtype) * scale.unsqueeze(-1).to(dtype)).reshape(
        values.shape[0], columns
    )


class _Int4MatMul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, inputs: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, columns: int
    ) -> torch.Tensor:
        ctx.save_for_backward(packed, scale)
        ctx.columns = columns
        weight = dequantize(packed, scale, columns, inputs.dtype)
        return inputs @ weight.t()

    @staticmethod
    def backward(ctx: Any, *gradients: torch.Tensor) -> tuple[torch.Tensor | None, None, None, None]:
        (gradient,) = gradients
        packed, scale = cast(tuple[torch.Tensor, torch.Tensor], ctx.saved_tensors)
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        weight = dequantize(packed, scale, ctx.columns, gradient.dtype)
        return gradient @ weight, None, None, None


class Int4Linear(nn.Module):
    """A frozen int4 linear layer (no bias, as in the checkpoints this reads)."""

    def __init__(self, packed: torch.Tensor, scale: torch.Tensor, in_features: int, out_features: int) -> None:
        super().__init__()
        self.register_buffer("weight_packed", packed)
        self.register_buffer("weight_scale", scale)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        packed = cast(torch.Tensor, self.weight_packed)
        scale = cast(torch.Tensor, self.weight_scale)
        return cast(torch.Tensor, _Int4MatMul.apply(inputs, packed, scale, self.in_features))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int4"


def replace_compressed_linears(model: nn.Module) -> int:
    """Swap every compressed-tensors packed linear layer for an `Int4Linear` sharing its tensors; returns how many."""
    replaced = 0
    for name, module in list(model.named_modules()):
        packed = getattr(module, "weight_packed", None)
        scale = getattr(module, "weight_scale", None)
        shape = getattr(module, "weight_shape", None)
        if not isinstance(module, nn.Linear) or packed is None or scale is None or shape is None:
            continue
        out_features, in_features = (int(value) for value in cast(torch.Tensor, shape).tolist())
        layer = Int4Linear(cast(torch.Tensor, packed).data, cast(torch.Tensor, scale).data, in_features, out_features)
        parent_name, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent_name) if parent_name else model, child, layer)
        replaced += 1
    return replaced
