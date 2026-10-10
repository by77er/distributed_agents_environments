# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Qwen3.5's norms and MLP fused into Triton kernels (`rollout_lora.fused`): what eager computes, forward and
backward, rounded as it rounds."""

import copy
from typing import Any, cast

import pytest
import torch
from torch import nn

from rollout_lora.fused import fuse

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def differing(found: torch.Tensor, wanted: torch.Tensor) -> float:
    """The share of elements that are not bit for bit the same."""
    return float((found != wanted).float().mean())


def close(found: torch.Tensor, wanted: torch.Tensor) -> None:
    assert found.dtype == wanted.dtype and found.shape == wanted.shape
    assert differing(found, wanted) < 1e-3  # (a mean of squares added up in another order: one unit, rarely)
    assert float((found.float() - wanted.float()).norm() / wanted.float().norm()) < 1e-3


def modules() -> dict[str, Any]:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    torch.manual_seed(0)
    config = Qwen3_5TextConfig(hidden_size=256, intermediate_size=768, hidden_act="silu")
    norm = qwen.Qwen3_5RMSNorm(256)
    nn.init.normal_(norm.weight, std=0.1)
    gated = qwen.Qwen3_5RMSNormGated(128)
    nn.init.normal_(gated.weight, mean=1.0, std=0.1)
    mlp = qwen.Qwen3_5MLP(cast(Any, config), 768)
    found = {"norm": norm, "gated": gated, "mlp": mlp}
    for module in found.values():
        module.to("cuda", torch.bfloat16).requires_grad_(False)
    return found


def test_a_zero_centred_norm_fused_is_eagers_forward_and_backward() -> None:
    eager = modules()["norm"]
    fused = copy.deepcopy(eager)
    assert fuse(fused) == 1
    x = (torch.randn(3, 700, 256, device="cuda") * 5).to(torch.bfloat16).requires_grad_(True)
    upstream = torch.randn(3, 700, 256, device="cuda").to(torch.bfloat16)
    wanted = eager(x)
    (wanted_dx,) = torch.autograd.grad(wanted, x, upstream)
    found = fused(x)
    (found_dx,) = torch.autograd.grad(found, x, upstream)
    close(found, wanted)
    close(found_dx, wanted_dx)


def test_a_norm_of_a_heads_slice_of_a_projection_reads_it_where_it_is() -> None:
    eager = modules()["norm"]
    fused = copy.deepcopy(eager)
    fuse(fused)
    projected = torch.randn(1, 300, 4, 512, device="cuda").to(torch.bfloat16).requires_grad_(True)
    upstream = torch.randn(1, 300, 4, 256, device="cuda").to(torch.bfloat16)
    wanted = eager(projected[..., :256])
    (wanted_dx,) = torch.autograd.grad(wanted, projected, upstream)
    found = fused(projected[..., :256])  # (rows 512 apart: a query and its gate, as attention's projection holds them)
    (found_dx,) = torch.autograd.grad(found, projected, upstream)
    close(found, wanted)
    close(found_dx, wanted_dx)
    assert torch.all(found_dx[..., 256:] == 0)


def test_a_gated_norm_fused_is_eagers_forward_and_backward() -> None:
    eager = modules()["gated"]
    fused = copy.deepcopy(eager)
    fuse(fused)
    x = torch.randn(2000, 128, device="cuda").to(torch.bfloat16).requires_grad_(True)
    gate = (torch.randn(2000, 128, device="cuda") * 3).to(torch.bfloat16).requires_grad_(True)
    upstream = torch.randn(2000, 128, device="cuda").to(torch.bfloat16)
    wanted = eager(x, gate)
    wanted_grads = torch.autograd.grad(wanted, (x, gate), upstream)
    found = fused(x, gate)
    found_grads = torch.autograd.grad(found, (x, gate), upstream)
    close(found, wanted)
    for found_grad, wanted_grad in zip(found_grads, wanted_grads, strict=True):
        close(found_grad, wanted_grad)


def test_an_mlps_silu_product_fused_is_eagers_bit_for_bit() -> None:
    eager = modules()["mlp"]
    fused = copy.deepcopy(eager)
    fuse(fused)
    x = torch.randn(1, 900, 256, device="cuda").to(torch.bfloat16).requires_grad_(True)
    upstream = torch.randn(1, 900, 256, device="cuda").to(torch.bfloat16)
    wanted = eager(x)
    (wanted_dx,) = torch.autograd.grad(wanted, x, upstream)
    found = fused(x)
    (found_dx,) = torch.autograd.grad(found, x, upstream)
    assert torch.equal(found, wanted)  # (silu(gate) * up rounds where eager rounds, and nothing is added up)
    close(found_dx, wanted_dx)


def test_a_norm_whose_weight_trains_keeps_its_eager_forward() -> None:
    eager = modules()["norm"]
    fused = copy.deepcopy(eager)
    fuse(fused)
    fused.weight.requires_grad_(True)
    x = torch.randn(50, 256, device="cuda").to(torch.bfloat16)
    found = fused(x)
    assert torch.equal(found, eager(x))
    found.sum().backward()
    assert fused.weight.grad is not None and fused.weight.grad.abs().sum() > 0
