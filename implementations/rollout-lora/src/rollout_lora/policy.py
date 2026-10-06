# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave module iteration and autograd functions partly untyped.)
"""A trainable policy: the model the engine serves (a 4-bit image-text checkpoint, or a text model in bfloat16) with
LoRA on its linear layers.

`Policy.logprobs` computes the logprobs of sampled tokens. It runs the transformer over the whole sequence but the
output layer (the vocabulary projection, the largest activation by far) only at the positions being scored. Both run
in one call of a `Scorer`, the module a sharded policy shards as its root (`rollout_lora.sharded`): what lies outside
its decoder layers (the output layer, the last norm) is gathered once a call, however many chunks of rows it scores.
`Policy.logprobs_and_entropy` adds each position's entropy, `Policy.logprobs_among` the logprobs of given tokens at
each position (a teacher's top-k, for distillation); `Policy.reference` gives the logprobs of the model trained over,
the adapter switched off. `Policy.packed` gives the same of every segment of a pack in one pass
(`rollout_lora.packing`).

A step shared among processes (`rollout_objectives.ranks`) asks two more things of a sharded policy, whose gradients
are added up across the processes (`rollout_lora.sharded.summed`), not averaged: an idle pass (`idle`, a two-token
sequence of nothing learnt), for a process with fewer passes than the others, and the gradient's norm over every shard,
clipped (`clip_gradients`). An adapter sharded with its frozen model whole on each GPU also reduces a minibatch's
gradient once, in its last pass (`gradient_sync`).

Only what training text needs is kept on the GPU: a vision tower is dropped, and the token embedding table (as
large as the output layer, and used only to look up a sequence's rows) is read from the checkpoint file as needed.
On a 16 GB card that is the difference between turns of 5,000 tokens and turns of about twice that.
"""

import json
import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from rollout_lora.layers import adapter_off, add_lora, lora_parameters, save_adapter
from rollout_lora.models import body, config, local, multimodal, quantized
from rollout_lora.packing import hidden, prepare
from rollout_lora.quantized import replace_compressed_linears
from rollout_objectives.packing import Pack, Scores
from rollout_objectives.ranks import Ranks
from rollout_train.recorder import Segment, Span

TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "out_proj",
)  # fmt: skip
"""Every attention, linear-attention and MLP projection of Qwen3.5 except the tiny gate projections."""


IDLE = Segment([0, 0], [Span(1, 2, 0)], [0.0])
"""What an idle pass computes: the passes the other processes take, with nothing learnt."""
LOGIT_ROWS = 128
"""Positions sent through the output layer at a time (each row of logits is a vocabulary wide)."""


class FileEmbedding:
    """A token embedding table left in its checkpoint file (memory-mapped): only the rows a sequence uses are read."""

    def __init__(self, file: Path, name: str) -> None:
        with file.open("rb") as opened:
            (length,) = struct.unpack("<Q", opened.read(8))
            entry = json.loads(opened.read(length))[name]
        if entry["dtype"] != "BF16":
            raise ValueError(f"{name} is stored as {entry['dtype']}, not bfloat16")
        rows, width = entry["shape"]
        self.table = numpy.memmap(file, dtype=numpy.int16, mode="r", offset=8 + length + entry["data_offsets"][0])
        self.table = self.table[: rows * width].reshape(rows, width)

    def __call__(self, ids: Sequence[int], device: torch.device) -> torch.Tensor:
        """The rows for `ids`, on `device`, as bfloat16."""
        rows = numpy.ascontiguousarray(self.table[numpy.asarray(ids)])
        return torch.from_numpy(rows).view(torch.bfloat16).to(device)

    @classmethod
    def find(cls, checkpoint: str, name: str) -> "FileEmbedding | None":
        """The table named `name` in a checkpoint (a directory, or a model already in the Hugging Face cache)."""
        for file in sorted(local(checkpoint).glob("*.safetensors")):
            with file.open("rb") as opened:
                (length,) = struct.unpack("<Q", opened.read(8))
                if name in json.loads(opened.read(length)):
                    return cls(file, name)
        return None


EMBEDDING = "model.language_model.embed_tokens.weight"
"""The token embeddings of an image-text model's language part (a text model's are `model.embed_tokens.weight`)."""


def scored(model: nn.Module, hidden: torch.Tensor, ids: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
    """Logprobs of `ids[p]` from the hidden state before it, for each p in `positions`, the output layer run a chunk
    of rows at a time (each recomputed for the backward pass, so the peak is one chunk's logits)."""
    return scored_with_entropy(model, hidden, ids, positions, entropy=False)[0]


def _at(ids: torch.Tensor, positions: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """The rows of the hidden states before `positions`, and the tokens at them."""
    device = ids.device
    rows = torch.tensor([position - 1 for position in positions], device=device, dtype=torch.long)
    return rows, ids[0].index_select(0, torch.tensor(list(positions), device=device, dtype=torch.long))


def scored_with_entropy(
    model: nn.Module, hidden: torch.Tensor, ids: torch.Tensor, positions: Sequence[int], *, entropy: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """`scored`, and the entropy of the distribution at each of `positions` (with `entropy`; else zeros)."""
    rows, targets = _at(ids, positions)
    return scored_rows(model, hidden, rows, targets, entropy=entropy)


def scored_rows(
    model: nn.Module, hidden: torch.Tensor, rows: torch.Tensor, targets: torch.Tensor, *, entropy: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logprobs of `targets[i]` from the hidden state in row `rows[i]`, and the entropy of the distribution there
    (with `entropy`; else zeros), the output layer run a chunk of rows at a time (each recomputed for the backward
    pass, so the peak is one chunk's logits)."""
    picked = hidden.index_select(0, rows.to(hidden.device))
    targets = targets.to(hidden.device)
    head = cast(Any, model).lm_head

    def chunk(rows: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = head(rows).float()
        found = -torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        if not entropy:
            return found, torch.zeros_like(found)
        logged = torch.log_softmax(logits, dim=-1)
        return found, -(logged.exp() * logged).sum(-1)

    parts = [
        cast(
            tuple[torch.Tensor, torch.Tensor],
            checkpoint(
                chunk, picked[start : start + LOGIT_ROWS], targets[start : start + LOGIT_ROWS], use_reentrant=False
            ),
        )
        for start in range(0, len(targets), LOGIT_ROWS)
    ]
    if not parts:
        empty = hidden.new_zeros(0, dtype=torch.float32)
        return empty, empty
    return torch.cat([part[0] for part in parts]), torch.cat([part[1] for part in parts])


def scored_among(
    model: nn.Module, hidden: torch.Tensor, ids: torch.Tensor, positions: Sequence[int], candidates: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`scored`, and the logprobs of `candidates[i]` (token ids) at the i-th of `positions`, a chunk of rows at a time
    as `scored_with_entropy` runs them."""
    rows, targets = _at(ids, positions)
    return scored_rows_among(model, hidden, rows, targets, candidates)


def scored_rows_among(
    model: nn.Module, hidden: torch.Tensor, rows: torch.Tensor, targets: torch.Tensor, candidates: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`scored_rows`' logprobs, and those of `candidates[i]` (token ids) from the hidden state in row `rows[i]`."""
    device = hidden.device
    picked = hidden.index_select(0, rows.to(device))
    targets = targets.to(device)
    wanted = candidates.to(device)
    head = cast(Any, model).lm_head

    def chunk(rows: torch.Tensor, targets: torch.Tensor, wanted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logged = torch.log_softmax(head(rows).float(), dim=-1)
        return logged.gather(-1, targets.unsqueeze(-1)).squeeze(-1), logged.gather(-1, wanted)

    parts = [
        cast(
            tuple[torch.Tensor, torch.Tensor],
            checkpoint(
                chunk,
                picked[start : start + LOGIT_ROWS],
                targets[start : start + LOGIT_ROWS],
                wanted[start : start + LOGIT_ROWS],
                use_reentrant=False,
            ),
        )
        for start in range(0, len(targets), LOGIT_ROWS)
    ]
    return torch.cat([part[0] for part in parts]), torch.cat([part[1] for part in parts])


def pack_rows(pack: Pack, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """For each of a pack's segments in turn, the rows of the hidden states before its sampled tokens, and the
    tokens."""
    rows: list[int] = []
    targets: list[int] = []
    for index in range(len(pack.segments)):
        found_rows, found_targets = pack.scored(index)
        rows.extend(found_rows)
        targets.extend(found_targets)
    return torch.tensor(rows, device=device, dtype=torch.long), torch.tensor(targets, device=device, dtype=torch.long)


def split_scores(found: tuple[torch.Tensor, torch.Tensor], pack: Pack, *, entropy: bool, among: bool) -> list[Scores]:
    """A `Scorer`'s scores of a pack, one segment's after another, as each segment's `Scores`."""
    counts = [len(pack.scored(index)[0]) for index in range(len(pack.segments))]
    logprobs, other = found[0].split(counts), found[1].split(counts)
    if among:
        return [Scores(each, among=wanted) for each, wanted in zip(logprobs, other, strict=True)]
    return [Scores(each, entropy=spread if entropy else None) for each, spread in zip(logprobs, other, strict=True)]


class Scorer(nn.Module):
    """A model's last hidden states and the output layer's scores at given positions, in one call (`forward`): what
    a policy computes of a sequence."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        ids: torch.Tensor,
        embedded: torch.Tensor | None,
        positions: Sequence[int],
        entropy: bool = False,
        candidates: torch.Tensor | None = None,
        pack: Pack | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`scored_with_entropy` (or `scored_among`, given `candidates`) over the decoder's last hidden states of `ids`
        (or of their rows `embedded`, where the embeddings are read from a file). Given a `pack`, `ids` are its row,
        and the scores are of each of its segments' sampled tokens, one segment's after another (`split_scores`;
        `candidates` theirs, likewise); the model must be one `rollout_lora.packing.prepare` readied."""
        if pack is not None:
            found = hidden(self.model, pack, ids=ids, embeddings=embedded)
            rows, targets = pack_rows(pack, found.device)
            if candidates is not None:
                return scored_rows_among(self.model, found, rows, targets, candidates)
            return scored_rows(self.model, found, rows, targets, entropy=entropy)
        inner = body(self.model)
        states = (inner(input_ids=ids) if embedded is None else inner(inputs_embeds=embedded)).last_hidden_state[0]
        if candidates is not None:
            return scored_among(self.model, states, ids, positions, candidates)
        return scored_with_entropy(self.model, states, ids, positions, entropy=entropy)


def idle_pass(policy: Any, *, gradient: bool = False, reference: bool = False) -> None:
    """A pass of `IDLE` through a policy (`Policy` or `rollout_lora.full.FullPolicy`), as one of a pack: under the
    reference (`reference`), or with a gradient and its loss times zero (`gradient`), or without one."""
    pack = Pack.single(0, IDLE)
    if reference:
        if policy.packing:
            policy.packed_reference(pack)
        else:
            policy.reference(IDLE.tokens, [1])
        return
    with torch.set_grad_enabled(gradient):
        logprobs = policy.packed(pack)[0].logprobs if policy.packing else policy.logprobs(IDLE.tokens, [1])
        if gradient:
            (logprobs.sum() * 0.0).backward()


def clip_gradients(parameters: Sequence[nn.Parameter], maximum: float, ranks: Ranks) -> float:
    """The gradient's norm before clipping, after scaling it to at most `maximum` (as `clip_grad_norm_` does).
    Shared among processes, each holds a shard of each gradient (a sharded model's), and their squares are added up;
    a gradient that is not a shard (a trainable weight outside the sharded model, which no process adds up) is an
    error."""
    if not ranks.shared:
        return float(torch.nn.utils.clip_grad_norm_(parameters, maximum))
    from torch.distributed.tensor import DTensor

    gradients = [each.grad for each in parameters if each.grad is not None]
    norms: list[torch.Tensor] = []  # (each in its gradient's dtype, as clip_grad_norm_ takes them)
    for gradient in gradients:
        if not isinstance(gradient, DTensor):
            raise ValueError("a trainable weight is outside the sharded model: its gradient is not added up")
        norms.append(torch.linalg.vector_norm(gradient.to_local()))
    local = float(torch.stack(norms).double().square().sum()) if norms else 0.0
    total = math.sqrt(ranks.summed([local])[0])
    coefficient = min(1.0, maximum / (total + 1e-6))
    if coefficient < 1.0:
        for gradient in gradients:
            gradient.mul_(coefficient)
    return total


def recover(scorers: Sequence[nn.Module | None]) -> None:
    """After a pass that ran out of memory part way, for each of `scorers` sharded with FSDP2 (`rollout_lora.sharded`,
    on one process too): every gradient the pass left on a gathered weight, not yet reduced to its shard, dropped (the
    output layer's, the last norm's, a layer's half done: the next pass would add them to its own), and FSDP's state of
    the pass reset (`reset_iter_state`). A scorer that is not sharded has none: the optimizer's `zero_grad` clears its
    gradients."""
    from torch.distributed.fsdp import FSDPModule

    for scorer in scorers:
        if not isinstance(scorer, FSDPModule):
            continue
        for module in scorer.modules():
            if not isinstance(module, FSDPModule):
                continue
            state = module._get_fsdp_state()  # pyright: ignore[reportPrivateUsage]  (FSDP2 keeps these to itself)
            for group in state._fsdp_param_groups:  # pyright: ignore[reportPrivateUsage]
                for each in group.fsdp_params:
                    gathered = getattr(each, "_unsharded_param", None)
                    if gathered is not None:
                        gathered.grad = None
                    each.unsharded_accumulated_grad = None
        scorer.reset_iter_state()


def layers_of(model: nn.Module) -> list[nn.Module]:
    """The decoder's layers, in order."""
    return list(cast(nn.ModuleList, body(model).layers))


@dataclass
class Policy:
    model: nn.Module
    checkpoint: str
    rank: int
    alpha: float
    embedding: FileEmbedding | None = None
    """The token embeddings, if they are read from the checkpoint file rather than held on the GPU."""
    packing: bool = False
    """Whether it runs packs (`packed`): its model is one `rollout_lora.packing.prepare` readies for them."""
    scorer: Scorer = field(init=False)
    gradient_sync: Callable[[bool], None] | None = field(default=None, init=False)
    """Told before each of a minibatch's gradient passes whether it is the last, where the adapter's gradients are
    reduced across processes once a minibatch (`rollout_lora.sharded.gradient_sync`); none: each pass reduces its
    own."""

    def __post_init__(self) -> None:
        self.scorer = Scorer(self.model)

    @classmethod
    def load(cls, checkpoint: str, *, rank: int, alpha: float, device: str = "cuda") -> "Policy":
        """The checkpoint on `device` (the GPU; the CPU, for a policy to shard) with a new adapter, its layers
        checkpointed for the backward pass."""
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, CompressedTensorsConfig

        where = str(local(checkpoint))
        if multimodal(checkpoint):
            options: dict[str, Any] = {"dtype": torch.bfloat16, "device_map": {"": device}}
            if quantized(checkpoint):
                options["quantization_config"] = CompressedTensorsConfig(run_compressed=True)
            model = cast(nn.Module, AutoModelForImageTextToText.from_pretrained(where, **options))  # pyright: ignore[reportUnknownMemberType]
            replace_compressed_linears(model)
        else:
            model = cast(
                nn.Module, AutoModelForCausalLM.from_pretrained(where, dtype=torch.bfloat16, device_map={"": device})
            )  # pyright: ignore[reportUnknownMemberType]
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        inner = cast(Any, model).model
        if getattr(inner, "visual", None) is not None:  # agents read text: the vision tower is never run
            inner.visual = None
        tied = bool(config(checkpoint).get("tie_word_embeddings")) or bool(
            config(checkpoint).get("text_config", {}).get("tie_word_embeddings")
        )
        embedding = None if tied else FileEmbedding.find(checkpoint, EMBEDDING)  # (tied: the output layer holds it)
        if embedding is not None:
            body(model).embed_tokens = None
        torch.cuda.empty_cache()
        within = "language_model" if multimodal(checkpoint) else "layers"
        add_lora(model, TARGETS, rank=rank, alpha=alpha, within=within, dtype=torch.float32)
        cast(Any, model).gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if embedding is None:  # (embeddings read from the file are marked as needing gradients where used)
            cast(Any, model).enable_input_require_grads()
        return cls(model, checkpoint, rank, alpha, embedding, prepare(model))

    def parameters(self) -> list[nn.Parameter]:
        return lora_parameters(self.model)

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        """Logprobs of `tokens[p]` given `tokens[:p]`, for each p in `positions` (all at least 1)."""
        return self.logprobs_and_entropy(tokens, positions, entropy=False)[0]

    def logprobs_and_entropy(
        self, tokens: Sequence[int], positions: Sequence[int], *, entropy: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`logprobs`, and the entropy of the policy's distribution at each of `positions`."""
        ids, embedded = self._inputs(tokens)
        return self.scorer(ids, embedded, positions, entropy)

    def logprobs_among(
        self, tokens: Sequence[int], positions: Sequence[int], candidates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`logprobs`, and the logprobs of `candidates[i]` (a row of token ids) at the i-th of `positions`: what the
        top-k form of distillation reads at the teacher's top tokens."""
        ids, embedded = self._inputs(tokens)
        return self.scorer(ids, embedded, positions, False, candidates)

    def _inputs(self, tokens: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The sequence's ids, and their embeddings where they are read from the checkpoint file."""
        device = next(iter(self.model.buffers())).device
        ids = torch.tensor([list(tokens)], device=device)
        if self.embedding is None:
            return ids, None
        return ids, self.embedding(tokens, device).unsqueeze(0).requires_grad_(True)  # (for checkpointing)

    def reference(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        """`logprobs` under the reference: the model trained over, the adapter switched off (no gradient)."""
        with torch.no_grad(), adapter_off(self.model):
            return self.logprobs(tokens, positions)

    def packed(
        self, pack: Pack, *, entropy: bool = False, candidates: Sequence[torch.Tensor] | None = None
    ) -> list[Scores]:
        """Each of a pack's segments' sampled tokens' logprobs, as `logprobs` gives one segment's (and their
        positions' entropies with `entropy`, or the logprobs of `candidates`, a tensor for each segment, as
        `logprobs_among`), from one pass over the pack's row."""
        ids, embedded = self._inputs(pack.tokens)
        joined = None if candidates is None else torch.cat(list(candidates))
        found = self.scorer(ids, embedded, (), entropy, joined, pack)
        return split_scores(found, pack, entropy=entropy, among=candidates is not None)

    def packed_reference(self, pack: Pack) -> list[torch.Tensor]:
        """`packed` logprobs under the reference (no gradient)."""
        with torch.no_grad(), adapter_off(self.model):
            return [each.logprobs for each in self.packed(pack)]

    def idle(self, *, gradient: bool = False, reference: bool = False) -> None:
        """An idle pass (`idle_pass`)."""
        idle_pass(self, gradient=gradient, reference=reference)

    def clip_gradients(self, maximum: float, ranks: Ranks) -> float:
        """The adapter's gradient clipped (`clip_gradients`); its norm before."""
        return clip_gradients(self.parameters(), maximum, ranks)

    def recover(self) -> None:
        """After a pass that ran out of memory part way: what it left of a sharded adapter's dropped (`recover`)."""
        recover([self.scorer])

    def save(self, directory: Path) -> Path:
        return save_adapter(self.model, directory, base_model=self.checkpoint, rank=self.rank, alpha=self.alpha)
