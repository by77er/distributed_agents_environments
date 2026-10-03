# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""The store of layer inputs and the MLP in pieces: on a toy layer of the same shape as a transformer's, and on a
tiny Qwen3.5."""

from typing import Any

import pytest
import torch
from torch import nn

from rollout_lora.activations import HostStore, checkpoint_layers, layer_checkpoint, piecewise_mlp


def test_a_slot_is_reused_once_its_input_is_read_back() -> None:
    store = HostStore(pin=False)
    first = store.put(torch.arange(6.0).reshape(2, 3))
    assert store.held == 1
    assert torch.equal(first.take(), torch.arange(6.0).reshape(2, 3))
    assert store.held == 0
    second = store.put(torch.ones(4))  # fits in the first buffer
    assert store.nbytes == 24 and store.held == 1
    third = store.put(torch.zeros(10))  # does not: a buffer of its own
    assert store.nbytes == 24 + 40 and store.held == 2
    assert torch.equal(second.take(), torch.ones(4)) and torch.equal(third.take(), torch.zeros(10))
    with pytest.raises(RuntimeError):
        second.take()
    store.clear()
    assert store.nbytes == 0


def test_an_input_dropped_unread_frees_its_slot_and_clear_waits_for_held_ones() -> None:
    store = HostStore(pin=False)
    held = store.put(torch.ones(3))
    with pytest.raises(RuntimeError, match="still held"):
        store.clear()
    del held
    assert store.held == 0
    store.clear()


def test_what_is_read_back_is_a_copy_not_the_buffer() -> None:
    store = HostStore(pin=False)
    original = torch.ones(3)
    held = store.put(original)
    original.fill_(5.0)
    back = held.take()
    store.put(torch.zeros(3))  # reuses the buffer
    assert torch.equal(back, torch.ones(3))


class Block(nn.Module):
    """`residual + mlp(norm(residual))` after a mixing step, as a decoder layer is."""

    def __init__(self) -> None:
        super().__init__()
        self.mix = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.mlp = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 4))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.mix(hidden)
        return hidden + self.mlp(self.norm(hidden))


@pytest.mark.parametrize("stored", [False, True])
def test_a_checkpointed_block_with_its_mlp_in_pieces_has_the_plain_gradients(stored: bool) -> None:
    torch.manual_seed(0)
    plain, paced = Block(), Block()
    paced.load_state_dict(plain.state_dict())
    piecewise_mlp(paced.mlp, rows=3)
    checkpointed = layer_checkpoint(HostStore(pin=False) if stored else None)
    inputs = torch.randn(2, 10, 4)
    weights = torch.randn(2, 10, 4)
    first, second = inputs.clone().requires_grad_(True), inputs.clone().requires_grad_(True)
    (plain(plain(first)) * weights).sum().backward()
    (checkpointed(paced, checkpointed(paced, second)) * weights).sum().backward()
    assert first.grad is not None and second.grad is not None
    assert torch.allclose(first.grad, second.grad, atol=1e-6)
    for (name, expected), (_, ours) in zip(plain.named_parameters(), paced.named_parameters(), strict=True):
        assert expected.grad is not None and ours.grad is not None
        assert torch.allclose(expected.grad, ours.grad, atol=1e-6), name


LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]


def tiny_qwen() -> Any:
    """A Qwen3.5 text model of four layers, small enough for the CPU, with gradient checkpointing on."""
    qwen = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=96,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=len(LAYER_TYPES),
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_num_value_heads=4,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        layer_types=LAYER_TYPES,
    )
    model = qwen.Qwen3_5TextModel(config).float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model


@pytest.fixture
def plain_torch_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """transformers' plain-torch linear attention: the kernels it prefers need a GPU."""
    qwen = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    for name in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn"):
        function = getattr(qwen, name, None)
        if function is not None:
            monkeypatch.setattr(qwen, name, getattr(function, "__wrapped__", function))


@pytest.mark.usefixtures("plain_torch_kernels")
def test_on_a_qwen_the_inputs_on_the_host_and_the_mlp_in_pieces_change_no_gradient() -> None:
    torch.manual_seed(0)
    plain, paced = tiny_qwen(), tiny_qwen()
    paced.load_state_dict(plain.state_dict())
    store = HostStore(pin=False)
    assert checkpoint_layers(paced, store=store, rows=5) == len(LAYER_TYPES)
    seen: list[int] = []  # the tokens each call of the first layer's up projection saw

    def record(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        seen.append(inputs[0].shape[-2])

    paced.layers[0].mlp.up_proj.register_forward_hook(record)
    embedded, weights = torch.randn(1, 24, 128), torch.randn(1, 24, 128)
    first, second = embedded.clone().requires_grad_(True), embedded.clone().requires_grad_(True)
    (plain(inputs_embeds=first).last_hidden_state * weights).sum().backward()
    (paced(inputs_embeds=second).last_hidden_state * weights).sum().backward()
    assert first.grad is not None and second.grad is not None and torch.allclose(first.grad, second.grad, atol=1e-5)
    for (name, expected), (_, ours) in zip(plain.named_parameters(), paced.named_parameters(), strict=True):
        if expected.grad is not None:
            assert ours.grad is not None and torch.allclose(expected.grad, ours.grad, atol=1e-5, rtol=1e-4), name
    assert seen and max(seen) <= 5  # never the whole segment at once
    assert store.held == 0 and store.nbytes > 0  # every input was read back
    with torch.no_grad():  # a pass without gradients stores nothing
        held_before = store.nbytes
        assert torch.allclose(
            paced(inputs_embeds=embedded).last_hidden_state, plain(inputs_embeds=embedded).last_hidden_state, atol=1e-5
        )
    assert store.held == 0 and store.nbytes == held_before
