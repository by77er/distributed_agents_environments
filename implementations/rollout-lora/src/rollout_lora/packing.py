# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# (torch's and transformers' annotations leave modules and attention functions partly untyped.)
"""A transformers model run over a pack (`rollout_objectives.packing.Pack`): many segments in one row, each token
seeing only the tokens of its own segment before it.

`prepare` readies a model for packs, if it is one of `PACKABLE`, and says whether it did:

- its softmax-attention layers get the attention function `PACKED`, which runs each root on its own
  (`scaled_dot_product_attention`, causal) and every branch's queries in one call over the row's keys and values,
  each seeing its prefix and its own branch's tokens before it, and nothing is copied for a branch. On the GPU that
  call is FlexAttention, compiled, with a block mask that skips the blocks no branch token sees; on the CPU it is
  `scaled_dot_product_attention` with the mask whole. Without a pack it is transformers' `sdpa`;
- its linear-attention layers (Qwen3.5's gated delta rule) get a forward that, given a pack, runs the short
  convolution over each run with the tokens before it in its segment (none for a root, the prefix's last for a
  branch), and the recurrence once over the roots (each from a zero state, as a segment alone starts) and once
  over the branches, each branch from the state its prefix ended in. The flash-linear-attention kernel takes the
  runs' boundaries (`cu_seqlens`) and the branches' starting states, and passes gradients back through them; on a
  CPU, or without it, transformers' own recurrence runs a run at a time.

Without a pack (`packing` not given) every layer is what it was. A shared prefix is exact: its keys, values and
recurrent state are those every segment under it would compute alone, and a backward pass through it adds up every
branch's gradient. A pack's activations are those of a row of as many tokens, however many branches share a prefix.

`hidden` runs a pack through a model's decoder and returns the last hidden state of each row token.
"""

import functools
import importlib
import inspect
import math
import types
from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional

from rollout_lora.models import body
from rollout_objectives.packing import Pack, Run

__all__ = ["BLOCK", "PACKABLE", "PACKED", "Layout", "hidden", "packable", "prepare"]

PACKED = "rollout_packed"
"""The attention implementation a prepared model's softmax-attention layers use."""
PACKABLE = ("llama", "qwen2", "qwen3", "qwen3_5_text")
"""The decoders `prepare` readies for packs (by `model_type`): softmax attention over the whole sequence, and
Qwen3.5's gated delta rule."""
UNPACKABLE_ROPE = ("dynamic", "longrope")
"""Rotary scalings that rescale by the longest position of the sequence: in a pack, the row's, not the segment's."""
BLOCK = 128
"""Tokens of a block of the branches' block mask, each side (FlexAttention's)."""
NARROW_TILES = {"BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32}
"""FlexAttention's backward tiles where a head is wider than 128 on a GPU with less than `WIDE_SHARED_MEMORY`: a
consumer card has 99 KiB a block, and the default tiles need 112 for heads of 256 (Qwen3.5's)."""
WIDE_SHARED_MEMORY = 160 * 2**10


class Layout:
    """A pack's runs as a model's layers read them, on its device, with what its layers build from them once (the
    branches' mask, the convolution's indices). Not a dataclass: FSDP casts a layer's inputs, and rebuilds any
    dataclass among them, which would leave each layer to build these again."""

    def __init__(self, pack: Pack, device: torch.device) -> None:
        self.runs: list[Run] = pack.runs
        self.length = pack.length
        roots = [run for run in pack.runs if run.parent is None]
        branches = [run for run in pack.runs if run.parent is not None]
        self.roots = sum(run.length for run in roots)
        """Tokens of the root runs, which come first in the row."""
        self.root_bounds_cpu = _bounds([run.length for run in roots])
        """Where each root starts in the row, and where the last ends (`cu_seqlens`)."""
        self.branch_bounds_cpu = _bounds([run.length for run in branches])
        """Where each branch starts, counted from the first branch, and where the last ends."""
        self.root_bounds = self.root_bounds_cpu.to(device)
        self.branch_bounds = self.branch_bounds_cpu.to(device)
        indices = [index for index, run in enumerate(pack.runs) if run.parent is None]
        number = {index: count for count, index in enumerate(indices)}
        self.parents = torch.tensor([number[cast(int, run.parent)] for run in branches], dtype=torch.long).to(device)
        """For each branch, which root (counted among the roots) it continues."""
        self.device = device
        self._convolved: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._seen: Any = None

    def convolved(self, context: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The rows a short convolution of `context` tokens before each reads, run by run, its context first (`-1` for
        none: a root's; a branch's is its prefix's last), and where in those rows each row token's output is."""
        if context not in self._convolved:
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
            self._convolved[context] = (
                torch.tensor(reads, dtype=torch.long).to(self.device),
                torch.tensor(outputs, dtype=torch.long).to(self.device),
            )
        return self._convolved[context]

    def _seeing(self, padded: int) -> tuple[torch.Tensor, ...]:
        """For each branch token (the row's tokens after the roots, then `padded` of them in all, the rest seeing
        nothing), the keys it sees as two spans of the row: its own branch's tokens up to itself, `[own, last]`, and
        its prefix, `[start, end)`."""
        own: list[int] = []
        start: list[int] = []
        end: list[int] = []
        for run in self.runs:
            if run.parent is not None:
                prefix = self.runs[run.parent]
                own += [run.start] * run.length
                start += [prefix.start] * run.length
                end += [prefix.end] * run.length
        last = torch.arange(self.roots, self.roots + padded)
        rest = padded - len(own)
        own_tensor = torch.tensor(own + [0] * rest, dtype=torch.long)
        own_tensor[len(own) :] = last[len(own) :] + 1  # (an empty span)
        spans = (own_tensor, last, torch.tensor(start + [0] * rest), torch.tensor(end + [0] * rest))
        return tuple(each.to(self.device) for each in spans)

    def seen(self) -> Any:
        """Which keys of the row each branch token sees: on the GPU as a block mask (`block_mask`), on the CPU the
        whole mask (branch tokens by row tokens)."""
        if self._seen is None:
            self._seen = self.block_mask() if self.device.type == "cuda" else self.mask()
        return self._seen

    def mask(self) -> torch.Tensor:
        """Which keys of the row each branch token sees, whole (branch tokens by row tokens)."""
        own, last, start, end = self._seeing(self.length - self.roots)
        keys = torch.arange(self.length, device=self.device)
        return ((keys >= own[:, None]) & (keys <= last[:, None])) | ((keys >= start[:, None]) & (keys < end[:, None]))

    def block_mask(self) -> Any:
        """Which keys of the row each branch token sees, as FlexAttention's block mask: blocks of `BLOCK` tokens a
        side, a block held where any of its branch tokens sees one of its keys, and the mask read within it only where
        some do not see all."""
        from torch.nn.attention.flex_attention import BlockMask

        branch_tokens = self.length - self.roots
        query_blocks, key_blocks = math.ceil(branch_tokens / BLOCK), math.ceil(self.length / BLOCK)
        own, last, start, end = self._seeing(query_blocks * BLOCK)

        def visible(batch: Any, head: Any, query: Any, key: Any) -> Any:
            return ((key >= own[query]) & (key <= last[query])) | ((key >= start[query]) & (key < end[query]))

        # How many keys of each block each branch token sees (its spans' overlaps with the block): a block is full
        # where every token of the query block sees all of it (a padded token sees nothing), held where any sees one.
        first = torch.arange(key_blocks, device=self.device) * BLOCK
        after = first + BLOCK
        seen = _overlap(own, last + 1, first, after) + _overlap(start, end, first, after)
        seen = seen.view(query_blocks, BLOCK, key_blocks)
        full = (seen == BLOCK).all(dim=1)
        partial = (seen > 0).any(dim=1) & ~full
        return BlockMask.from_kv_blocks(
            *_listed(partial), *_listed(full), BLOCK_SIZE=BLOCK, mask_mod=visible,
            seq_lengths=(branch_tokens, self.length),
        )  # fmt: skip


def _bounds(lengths: list[int]) -> torch.Tensor:
    found = [0]
    for length in lengths:
        found.append(found[-1] + length)
    return torch.tensor(found, dtype=torch.long)


def _overlap(starts: torch.Tensor, ends: torch.Tensor, first: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    """For each span `[starts[i], ends[i])` and each block `[first[j], after[j])`, how many tokens both hold."""
    return (torch.minimum(ends[:, None], after[None]) - torch.maximum(starts[:, None], first[None])).clamp(min=0)


def _listed(held: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A block mask's blocks (query blocks by key blocks) as FlexAttention lists them: how many each query block
    holds, and their indices first, in order."""
    count = held.sum(dim=-1, dtype=torch.int32)
    order = torch.argsort((~held).to(torch.int8), dim=-1, stable=True).to(torch.int32)
    return count[None, None], order[None, None]


@functools.cache
def _flex() -> Callable[..., torch.Tensor]:
    """FlexAttention, compiled once a process, for rows of any length."""
    from torch.nn.attention.flex_attention import flex_attention

    return cast(Callable[..., torch.Tensor], torch.compile(flex_attention, dynamic=True))


def _kernel_options(query: torch.Tensor) -> dict[str, Any]:
    """FlexAttention's kernel options for `query`: its attention kernel for any number of branch tokens (not its
    decoding kernel, which it would choose for fewer than 128, and which has no kernel for some shapes), with
    `NARROW_TILES` where its default tiles do not fit the GPU."""
    options: dict[str, Any] = {"FORCE_USE_FLEX_ATTENTION": True}
    shared = torch.cuda.get_device_properties(query.device).shared_memory_per_block_optin
    if query.shape[-1] > 128 and shared < WIDE_SHARED_MEMORY:
        options |= NARROW_TILES
    return options


def _flexed(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, seen: Any, scaling: float | None, grouped: bool
) -> torch.Tensor:
    """FlexAttention of `query` over `key` and `value` where `seen` (a block mask) says, compiled. A pass without a
    gradient runs the kernels a pass with one does (inputs that ask for a gradient, the output detached): inductor
    compiles a pass without one apart, and rounds it differently, so that a step's start would not be what its
    minibatches compute on the same weights."""
    options = _kernel_options(query)
    if torch.is_grad_enabled() and any(each.requires_grad for each in (query, key, value)):
        return _flex()(query, key, value, block_mask=seen, scale=scaling, enable_gqa=grouped, kernel_options=options)
    with torch.enable_grad():
        asked, keys, values = (each.detach().requires_grad_() for each in (query, key, value))
        found = _flex()(asked, keys, values, block_mask=seen, scale=scaling, enable_gqa=grouped, kernel_options=options)
    return found.detach()


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
    """Attention over a pack (`packing`, a `Layout`): each root on its own, then every branch token over the keys it
    sees (`Layout.seen`); without one, transformers' `sdpa`."""
    layout = cast(Layout | None, kwargs.pop("packing", None))
    if layout is None:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout, scaling, **kwargs)
    if kwargs.get("sliding_window") is not None:
        raise ValueError("a pack's attention is over whole segments, not a sliding window")
    if dropout:
        raise ValueError("a pack's attention takes no dropout")
    grouped = query.shape[1] != key.shape[1]  # (each key and value head serves a group of query heads)
    outputs: list[torch.Tensor] = []
    for run in layout.runs:
        if run.parent is None:
            part = slice(run.start, run.end)
            outputs.append(
                functional.scaled_dot_product_attention(
                    query[:, :, part], key[:, :, part], value[:, :, part], is_causal=True, scale=scaling,
                    enable_gqa=grouped,
                )
            )  # fmt: skip
    if layout.roots < layout.length:
        asked = query[:, :, layout.roots :]
        if query.is_cuda:
            found = _flexed(asked, key, value, layout.seen(), scaling, grouped)
        else:
            found = functional.scaled_dot_product_attention(
                asked, key, value, attn_mask=layout.seen(), scale=scaling, enable_gqa=grouped
            )
        outputs.append(found)
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
            cu_seqlens_cpu=layout.root_bounds_cpu,
        )  # fmt: skip
        if not has_branches:
            return root_out
        branch_out, _ = rule(
            query[:, roots:], key[:, roots:], value[:, roots:], g=g[:, roots:], beta=beta[:, roots:],
            initial_state=final.index_select(0, layout.parents), output_final_state=False,
            use_qk_l2norm_in_kernel=True, cu_seqlens=layout.branch_bounds, cu_seqlens_cpu=layout.branch_bounds_cpu,
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
        reads, outputs = layout.convolved(context)
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
    """Whether a model's decoder (an image-text model's language part, or a text model's own) is one of `PACKABLE`,
    with no sliding window and no rotary scaling of `UNPACKABLE_ROPE`."""
    said = body(model).config
    if getattr(said, "model_type", None) not in PACKABLE:
        return False
    kinds = set(getattr(said, "layer_types", None) or ["full_attention"])
    if not kinds <= {"full_attention", "linear_attention"} or getattr(said, "use_sliding_window", False):
        return False
    return not _rope_types(said) & set(UNPACKABLE_ROPE)


def _rope_types(said: Any) -> set[str]:
    """The kinds of rotary scaling a decoder's configuration names (`rope_parameters`, by layer type or for every
    layer, or the older `rope_scaling`)."""
    found: set[str] = set()
    for name in ("rope_parameters", "rope_scaling"):
        given = getattr(said, name, None)
        if not isinstance(given, dict):
            continue
        given = cast(dict[str, Any], given)
        for each in [given, *(value for value in given.values() if isinstance(value, dict))]:
            kind = cast(dict[str, Any], each).get("rope_type", cast(dict[str, Any], each).get("type"))
            if isinstance(kind, str):
                found.add(kind)
    return found


def prepare(model: nn.Module) -> bool:
    """Ready a model for packs (as the module says), if it is `packable`; whether it is."""
    if not packable(model):
        return False
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    del sdpa_attention_forward  # (imported to fail here, not mid-step, if transformers moved it)
    AttentionInterface.register(PACKED, packed_attention)  # (and no mask: each sequence alone is causal, unpadded)
    for config in {id(each): each for each in (cast(Any, model).config, body(model).config)}.values():
        config._attn_implementation = PACKED
        inner = getattr(config, "text_config", None)
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
    decoder = body(model)
    given = ids if embeddings is None else embeddings
    if given is None:
        raise ValueError("a pack's row is run from its tokens or its embeddings")
    layout = Layout(pack, given.device)
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
