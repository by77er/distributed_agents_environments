# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# (torch's and transformers' annotations leave modules and attention functions partly untyped.)
"""A transformers model run over a pack (`rollout_objectives.packing.Pack`): many segments in one row, each token
seeing only the tokens of its own segment before it.

`prepare` readies a model for packs, if it is one of `PACKABLE`, and says whether it did:

- its softmax-attention layers get the attention function `PACKED`, which runs each of a pack's runs on its own
  (`scaled_dot_product_attention`, causal; a branch's queries over its prefix's keys and values and its own,
  causal from the bottom right) and, without a pack, is transformers' `sdpa`;
- its linear-attention layers (Qwen3.5's gated delta rule) get a forward that, given a pack, runs the short
  convolution over each run with the tokens before it in its segment (none for a root, the prefix's last for a
  branch), and the recurrence once over the roots (each from a zero state, as a segment alone starts) and once
  over the branches, each branch from the state its prefix ended in. The flash-linear-attention kernel takes the
  runs' boundaries (`cu_seqlens`) and the branches' starting states, and passes gradients back through them; on a
  CPU, or without it, transformers' own recurrence runs a run at a time.

Without a pack (`packing` not given) every layer is what it was. A shared prefix is exact: its keys, values and
recurrent state are those every segment under it would compute alone, and a backward pass through it adds up every
branch's gradient.

`hidden` runs a pack through a model's decoder and returns the last hidden state of each row token.
"""

import importlib
import inspect
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional

from rollout_objectives.packing import Pack, Run

__all__ = ["PACKABLE", "PACKED", "Layout", "hidden", "packable", "prepare"]

PACKED = "rollout_packed"
"""The attention implementation a prepared model's softmax-attention layers use."""
PACKABLE = ("llama", "qwen2", "qwen3", "qwen3_5_text")
"""The decoders `prepare` readies for packs (by `model_type`): softmax attention over the whole sequence, and
Qwen3.5's gated delta rule."""


@dataclass
class Layout:
    """A pack's runs as a model's layers read them, on its device."""

    runs: list[Run]
    roots: int
    """Tokens of the root runs, which come first in the row."""
    root_bounds: torch.Tensor
    """Where each root starts in the row, and where the last ends (`cu_seqlens`)."""
    branch_bounds: torch.Tensor
    """Where each branch starts, counted from the first branch, and where the last ends."""
    parents: torch.Tensor
    """For each branch, which root (counted among the roots) it continues."""
    conv_index: dict[int, tuple[torch.Tensor, torch.Tensor]]
    """By a convolution's context (its width less one): the rows each run reads, its context first (`-1` for none),
    and where in those rows each row token's output is."""

    @classmethod
    def of(cls, pack: Pack, device: torch.device) -> "Layout":
        roots = [run for run in pack.runs if run.parent is None]
        branches = [run for run in pack.runs if run.parent is not None]
        root_number = {
            index: number for number, index in enumerate(i for i, run in enumerate(pack.runs) if run.parent is None)
        }
        return cls(
            runs=pack.runs,
            roots=sum(run.length for run in roots),
            root_bounds=_bounds([run.length for run in roots], device),
            branch_bounds=_bounds([run.length for run in branches], device),
            parents=torch.tensor(
                [root_number[cast(int, run.parent)] for run in branches], device=device, dtype=torch.long
            ),
            conv_index={},
        )

    def convolved(self, context: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """The rows (and outputs) a short convolution of `context` tokens before each reads, as `conv_index` holds."""
        if context not in self.conv_index:
            reads: list[int] = []
            outputs: list[int] = []
            for run in self.runs:
                before = [-1] * context
                if run.parent is not None:
                    prefix = self.runs[run.parent]
                    tail = list(range(max(prefix.start, prefix.end - context), prefix.end))
                    before = [-1] * (context - len(tail)) + tail
                reads.extend(before)
                outputs.extend(range(len(reads) - context, len(reads) - context + run.length))
                reads.extend(range(run.start, run.end))
            self.conv_index[context] = (
                torch.tensor(reads, device=device, dtype=torch.long),
                torch.tensor(outputs, device=device, dtype=torch.long),
            )
        return self.conv_index[context]


def _bounds(lengths: list[int], device: torch.device) -> torch.Tensor:
    found = [0]
    for length in lengths:
        found.append(found[-1] + length)
    return torch.tensor(found, device=device, dtype=torch.long)


def packed_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Attention over a pack's runs (`packing`, a `Layout`), each on its own; without one, transformers' `sdpa`."""
    layout = cast(Layout | None, kwargs.pop("packing", None))
    if layout is None:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout, scaling, **kwargs)
    if kwargs.get("sliding_window") is not None:
        raise ValueError("a pack's attention is over whole segments, not a sliding window")
    if query.shape[1] != key.shape[1]:  # (grouped queries: each key and value head serves a group of query heads)
        groups = query.shape[1] // key.shape[1]
        key, value = key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)
    from torch.nn.attention.bias import causal_lower_right

    outputs: list[torch.Tensor] = []
    for run in layout.runs:
        asked = query[:, :, run.start : run.end]
        if run.parent is None:
            keys, values = key[:, :, run.start : run.end], value[:, :, run.start : run.end]
            outputs.append(
                functional.scaled_dot_product_attention(
                    asked, keys, values, dropout_p=dropout, is_causal=True, scale=scaling
                )
            )
            continue
        prefix = layout.runs[run.parent]
        keys = torch.cat([key[:, :, prefix.start : prefix.end], key[:, :, run.start : run.end]], dim=2)
        values = torch.cat([value[:, :, prefix.start : prefix.end], value[:, :, run.start : run.end]], dim=2)
        mask = causal_lower_right(run.length, prefix.length + run.length)
        outputs.append(
            functional.scaled_dot_product_attention(
                asked, keys, values, attn_mask=mask, dropout_p=dropout, scale=scaling
            )
        )
    return torch.cat(outputs, dim=2).transpose(1, 2).contiguous(), None


def _fla_rule() -> Callable[..., Any] | None:
    try:
        found = importlib.import_module("fla.ops.gated_delta_rule")
    except Exception:  # (not installed, or no GPU to build it for)
        return None
    return cast(Callable[..., Any], found.chunk_gated_delta_rule)


def _torch_rule() -> Callable[..., Any]:
    from transformers.models.qwen3_5 import modeling_qwen3_5

    return cast(Callable[..., Any], inspect.unwrap(modeling_qwen3_5.torch_chunk_gated_delta_rule))


def delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    layout: Layout,
) -> torch.Tensor:
    """The gated delta rule over a pack's row (batch of one): each root from a zero state, each branch from the state
    its root ended in."""
    roots = layout.roots
    has_branches = len(layout.parents) > 0
    found = _fla_rule() if query.is_cuda else None
    if found is not None:
        rule = found
        root_out, final = rule(
            query[:, :roots], key[:, :roots], value[:, :roots], g=g[:, :roots], beta=beta[:, :roots],
            output_final_state=has_branches, use_qk_l2norm_in_kernel=True, cu_seqlens=layout.root_bounds,
        )  # fmt: skip
        if not has_branches:
            return root_out
        branch_out, _ = rule(
            query[:, roots:], key[:, roots:], value[:, roots:], g=g[:, roots:], beta=beta[:, roots:],
            initial_state=final.index_select(0, layout.parents), output_final_state=False,
            use_qk_l2norm_in_kernel=True, cu_seqlens=layout.branch_bounds,
        )  # fmt: skip
        return torch.cat([root_out, branch_out], dim=1)
    rule = _torch_rule()
    outputs: list[torch.Tensor] = []
    states: dict[int, torch.Tensor] = {}
    for index, run in enumerate(layout.runs):
        part = slice(run.start, run.end)
        out, state = rule(
            query[:, part], key[:, part], value[:, part], g=g[:, part], beta=beta[:, part],
            initial_state=states.get(run.parent) if run.parent is not None else None,
            output_final_state=run.parent is None, use_qk_l2norm_in_kernel=True,
        )  # fmt: skip
        if run.parent is None:
            states[index] = state
        outputs.append(out)
    return torch.cat(outputs, dim=1)


def _gated_delta_forward(original: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """A Qwen3.5 gated delta rule layer's forward that, given a pack (`packing`), keeps its segments apart; without
    one, `original`."""

    def forward(
        self: Any,
        hidden_states: torch.Tensor,
        cache_params: Any = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        layout = cast(Layout | None, kwargs.pop("packing", None))
        if layout is None:
            return original(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
        if cache_params is not None or hidden_states.shape[0] != 1:
            raise ValueError("a pack is one row, run without a cache")
        length = hidden_states.shape[1]
        mixed = self.in_proj_qkv(hidden_states)[0]  # (row, channels)
        z = self.in_proj_z(hidden_states).reshape(1, length, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # The short convolution, each run reading the tokens before it in its segment (zeros where there are none).
        weight = self.conv1d.weight  # (channels, 1, width)
        context = weight.shape[-1] - 1
        reads, outputs = layout.convolved(context, mixed.device)
        padded = torch.cat([mixed.new_zeros(1, mixed.shape[1]), mixed]).to(weight.dtype)
        read = padded.index_select(0, reads + 1).transpose(0, 1).unsqueeze(0)
        convolved = functional.conv1d(read, weight, self.conv1d.bias, groups=mixed.shape[1])[0]
        mixed = convolved.index_select(1, outputs)
        from transformers.activations import ACT2FN

        mixed = ACT2FN[self.activation](mixed).to(hidden_states.dtype).transpose(0, 1).unsqueeze(0)

        query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(1, length, -1, self.head_k_dim)
        key = key.reshape(1, length, -1, self.head_k_dim)
        value = value.reshape(1, length, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * functional.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        core = delta_rule(query, key, value, g, beta, layout)
        core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(core.reshape(1, length, -1))

    return forward


def packable(model: nn.Module) -> bool:
    """Whether a model's decoder (an image-text model's language part, or a text model's own) is one of
    `PACKABLE`."""
    from rollout_lora.policy import body

    decoder = body(model)
    said = decoder.config
    if getattr(said, "model_type", None) not in PACKABLE:
        return False
    kinds = set(getattr(said, "layer_types", None) or ["full_attention"])
    return kinds <= {"full_attention", "linear_attention"} and not getattr(said, "use_sliding_window", False)


def prepare(model: nn.Module) -> bool:
    """Ready a model for packs (as the module says), if it is `packable`; whether it is."""
    if not packable(model):
        return False
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    del sdpa_attention_forward  # (imported to fail here, not mid-step, if transformers moved it)
    AttentionInterface.register(PACKED, packed_attention)  # (and no mask: each sequence alone is causal, unpadded)
    from rollout_lora.policy import body

    for config in {id(each): each for each in (cast(Any, model).config, body(model).config)}.values():
        config._attn_implementation = PACKED
        for name in ("text_config",):
            inner = getattr(config, name, None)
            if inner is not None:
                inner._attn_implementation = PACKED
    for module in model.modules():
        if type(module).__name__.endswith("GatedDeltaNet"):
            original = type(module).forward.__get__(module)
            module.forward = types.MethodType(_gated_delta_forward(original), module)
    return True


def hidden(
    model: nn.Module, pack: Pack, *, ids: torch.Tensor | None = None, embeddings: torch.Tensor | None = None
) -> torch.Tensor:
    """The last hidden state of each of a pack's row tokens, from a `prepare`d model: of the row's token `ids`, or of
    its `embeddings` where the model reads them from elsewhere (one of the two, each a batch of one row)."""
    from rollout_lora.policy import body

    decoder = body(model)
    given = ids if embeddings is None else embeddings
    if given is None:
        raise ValueError("a pack's row is run from its tokens or its embeddings")
    layout = Layout.of(pack, given.device)
    options: dict[str, Any] = {
        "position_ids": torch.tensor([pack.positions], device=given.device),
        "use_cache": False,
        "packing": layout,
    }
    if embeddings is None:
        options["input_ids"] = ids
    else:
        options["inputs_embeds"] = embeddings
    return decoder(**options).last_hidden_state[0]
