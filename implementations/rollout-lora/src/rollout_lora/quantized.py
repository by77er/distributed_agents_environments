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

Where it is faster on the GPU at hand (`accumulates_in_fp16`: a consumer GPU, whose fp32 accumulation runs at half
the rate of fp16's), bf16 activations are multiplied on fp16 tensor cores accumulating in fp16 a tile at a time
(`rollout_lora.kernels.half_matmul`), by the weight dequantized to fp16 (times `WEIGHT_FACTOR`, divided back out
after). fp16 keeps three more bits than bf16 of the activations and of the weights, so the products are closer to
exact than bf16's with fp32 accumulation; on an RTX 5080 the matmuls take about 70% of the time.
"""

import functools
import importlib.util
from typing import Any, cast

import torch
from torch import nn

WEIGHT_FACTOR = 64.0
"""What a weight dequantized to fp16 is multiplied by (exactly: a power of two), so that its smallest weights (an int4
value of 1 times a scale of about 4e-5) are normal fp16 numbers; its largest (8 times a scale of at most about
0.26, times this) are far below fp16's largest, as are the tile sums of products with rows of magnitude 1."""


@functools.cache
def _kernels() -> bool:
    """Whether Triton is here to compile `rollout_lora.kernels`."""
    return importlib.util.find_spec("triton") is not None


@functools.cache
def accumulates_in_fp16(device: torch.device) -> bool:
    """Whether `rollout_lora.kernels.half_matmul` multiplies a 9B model's projection faster than torch's bf16 matmul
    on this GPU (by a tenth or more), timed once: a consumer GPU's fp32 accumulation runs at half the rate of fp16's,
    a datacenter GPU's at the same."""
    from rollout_lora.kernels import half_matmul

    generator = torch.Generator(device=device).manual_seed(0)
    inputs = torch.randn(2048, 4096, device=device, generator=generator).to(torch.bfloat16)
    weight = torch.randn(4096, 4096, device=device, generator=generator)

    def timed(multiply: Any) -> float:
        for _ in range(3):
            multiply()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            multiply()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    with torch.cuda.device(device):
        half, bf16 = weight.half(), weight.to(torch.bfloat16)
        return timed(lambda: half_matmul(inputs, half.t())) < 0.9 * timed(lambda: inputs @ bf16.t())


def _half(tensor: torch.Tensor) -> bool:
    """Whether a matmul of these activations runs on fp16 tensor cores (`accumulates_in_fp16`)."""
    return (
        tensor.is_cuda
        and tensor.dtype in (torch.bfloat16, torch.float16)
        and _kernels()
        and accumulates_in_fp16(tensor.device)
    )


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
        if _half(inputs):
            from rollout_lora.kernels import dequantized, half_matmul

            weight = dequantized(packed, scale, columns, torch.float16, WEIGHT_FACTOR)
            found = half_matmul(inputs.reshape(-1, columns), weight.t(), 1 / WEIGHT_FACTOR)
            return found.reshape(*inputs.shape[:-1], packed.shape[0])
        weight = dequantize(packed, scale, columns, inputs.dtype)
        return inputs @ weight.t()

    @staticmethod
    def backward(ctx: Any, *gradients: torch.Tensor) -> tuple[torch.Tensor | None, None, None, None]:
        (gradient,) = gradients
        packed, scale = cast(tuple[torch.Tensor, torch.Tensor], ctx.saved_tensors)
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        if _half(gradient):
            from rollout_lora.kernels import dequantized, half_matmul

            weight = dequantized(packed, scale, ctx.columns, torch.float16, WEIGHT_FACTOR)
            found = half_matmul(gradient.reshape(-1, gradient.shape[-1]), weight, 1 / WEIGHT_FACTOR)
            return found.reshape(*gradient.shape[:-1], ctx.columns), None, None, None
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
