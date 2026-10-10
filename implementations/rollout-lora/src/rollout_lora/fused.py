# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave module iteration and autograd functions partly untyped.)
"""Qwen3.5's elementwise operations fused into one Triton kernel each, forward and backward, on a GPU.

Eager, each norm is half a dozen kernels (a cast to float32, a square, a mean, an rsqrt, two multiplies, a cast back),
each reading and writing the whole activation, and the MLP's `silu(gate) * up` three; a training pass runs each twice
(activations are checkpointed) and their backward passes once. `fuse` gives a model's

- zero-centred norms (`Qwen3_5RMSNorm`: each layer's two, attention's query and key norms, the last norm),
- gated norms (`Qwen3_5RMSNormGated`, after the gated delta rule), and
- MLPs (`Qwen3_5MLP`: `silu(gate) * up` before the down projection)

forwards that run `rollout_lora.kernels`' fused kernels where their inputs are on a GPU, rounding where the eager
operations round: on a 9B model's activations a norm's output differs from eager in about one element in a hundred
thousand, by one bf16 unit in the last place (its mean of squares added up in another order), and `silu(gate) * up`'s
not at all. A norm whose weight trains (full weights) keeps its eager forward: these kernels give no weight a
gradient. On an RTX 5080 a 9B gradient pass takes about a sixth less time, and its peak memory about 0.7 GiB less.
"""

import importlib.util
import types
from typing import Any, cast

import torch
from torch import nn

__all__ = ["FUSED", "fuse"]

FUSED = ("Qwen3_5RMSNorm", "Qwen3_5RMSNormGated", "Qwen3_5MLP")
"""The modules `fuse` gives fused forwards, by class name."""


class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        from rollout_lora.kernels import norm_forward

        out, scales, stride = norm_forward(x, weight, eps)
        ctx.save_for_backward(x, weight, scales)
        ctx.stride = stride
        return out

    @staticmethod
    def backward(ctx: Any, *gradients: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        from rollout_lora.kernels import norm_backward

        x, weight, scales = cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor], ctx.saved_tensors)
        rows = x if ctx.stride != x.shape[-1] or x.is_contiguous() else x.contiguous()
        return norm_backward(gradients[0], rows, ctx.stride, weight, scales).view(x.shape), None, None


class _GatedNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        from rollout_lora.kernels import gated_norm_forward

        out, scales, rows, gates = gated_norm_forward(x, gate, weight, eps)
        ctx.save_for_backward(rows, gates, weight, scales)
        return out

    @staticmethod
    def backward(ctx: Any, *gradients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        from rollout_lora.kernels import gated_norm_backward

        rows, gates, weight, scales = cast(tuple[torch.Tensor, ...], ctx.saved_tensors)
        dx, dgate = gated_norm_backward(gradients[0], rows, gates, weight, scales)
        return dx, dgate, None, None


class _SiluProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        from rollout_lora.kernels import silu_product_forward

        out, gates, ups = silu_product_forward(gate, up)
        ctx.save_for_backward(gates, ups)
        return out

    @staticmethod
    def backward(ctx: Any, *gradients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from rollout_lora.kernels import silu_product_backward

        gates, ups = cast(tuple[torch.Tensor, torch.Tensor], ctx.saved_tensors)
        return silu_product_backward(gradients[0], gates, ups)


def _on_gpu(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and tensor.dtype in (torch.bfloat16, torch.float16)


def _norm(original: Any) -> Any:
    def forward(self: Any, x: torch.Tensor) -> torch.Tensor:
        if not _on_gpu(x) or self.weight.requires_grad:
            return original(x)
        return cast(torch.Tensor, _Norm.apply(x, self.weight, self.eps))

    return forward


def _gated_norm(original: Any) -> Any:
    def forward(self: Any, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if not _on_gpu(hidden_states) or self.weight.requires_grad or self.activation != "silu":
            return original(hidden_states, gate)
        return cast(torch.Tensor, _GatedNorm.apply(hidden_states, gate, self.weight, self.variance_epsilon))

    return forward


def _mlp(original: Any) -> Any:
    def forward(self: Any, x: torch.Tensor) -> torch.Tensor:
        if not _on_gpu(x) or self.config.hidden_act != "silu":
            return original(x)
        return cast(torch.Tensor, self.down_proj(_SiluProduct.apply(self.gate_proj(x), self.up_proj(x))))

    return forward


def fuse(model: nn.Module) -> int:
    """Give each of a model's modules of `FUSED` its fused forward (where Triton is here to compile the kernels);
    returns how many."""
    if importlib.util.find_spec("triton") is None:
        return 0
    wrappers = {"Qwen3_5RMSNorm": _norm, "Qwen3_5RMSNormGated": _gated_norm, "Qwen3_5MLP": _mlp}
    fused = 0
    for module in model.modules():
        wrap = wrappers.get(type(module).__name__)
        if wrap is not None:
            original = type(module).forward.__get__(module)
            module.forward = types.MethodType(wrap(original), module)
            fused += 1
    return fused
