# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch's annotations leave module iteration partly untyped.)
"""A policy whose every weight is trained: a text model, held in float32 and computed in bfloat16.

The weights stay in float32 between steps (each step starts from the files the one before saved): an update is far
smaller than a bfloat16 weight resolves, and saved in bfloat16 most of it would be lost. Engines cast the files to
their own dtype when they load them. A checkpoint keeps the model's configuration and tokenizer beside its weights,
so that it can also be served, or trained from, as a model of its own.

Its memory: the weights, their gradients and Adam's two moments, four copies in float32 (16 bytes a weight), and the
activations of one segment with gradient checkpointing. Qwen3-0.6B takes about 10 GiB. An objective that reads the
reference needs a frozen copy of the model trained over beside it, in bfloat16 (2 bytes a weight more), which it holds
only when asked (`LoraSettings.frozen_reference`).
"""

import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from rollout_lora.models import COPIED, local, multimodal
from rollout_lora.packing import prepare
from rollout_lora.policy import Scorer, clip_gradients, idle_pass, recover, split_scores
from rollout_objectives.packing import Pack, Scores
from rollout_objectives.ranks import Ranks


@dataclass
class FullPolicy:
    model: nn.Module
    checkpoint: str
    """Where it was loaded from: a model's name or directory, or a full checkpoint's files."""
    frozen: nn.Module | None = None
    """The reference, if it holds one: a frozen copy of the model trained over, in bfloat16."""
    packing: bool = False
    """Whether it runs packs (`packed`), as `rollout_lora.policy.Policy` does."""
    scorer: Scorer = field(init=False)
    frozen_scorer: Scorer | None = field(init=False)
    gradient_sync: Callable[[bool], None] | None = field(default=None, init=False)
    """None: each pass reduces its own gradients (kept to a minibatch's last pass, full weights' unsharded gradients
    would be on every GPU)."""

    def __post_init__(self) -> None:
        self.scorer = Scorer(self.model)
        self.frozen_scorer = Scorer(self.frozen) if self.frozen is not None else None

    @classmethod
    def load(
        cls,
        checkpoint: str,
        *,
        gradient_checkpointing: bool = True,
        reference: str | None = None,
        device: str = "cuda",
    ) -> "FullPolicy":
        """The policy from `checkpoint` on `device` (the GPU; the CPU, for a policy to shard), and with `reference`
        (a model's name or directory) a frozen copy of it."""
        from transformers import AutoModelForCausalLM

        if multimodal(checkpoint):
            raise ValueError(f"{checkpoint} is an image-text model: full weights are trained on text models only")
        model = cast(
            nn.Module,
            AutoModelForCausalLM.from_pretrained(str(local(checkpoint)), dtype=torch.float32, device_map={"": device}),
        )
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        if gradient_checkpointing:
            cast(Any, model).gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        frozen: nn.Module | None = None
        if reference is not None:
            frozen = cast(
                nn.Module,
                AutoModelForCausalLM.from_pretrained(
                    str(local(reference)), dtype=torch.bfloat16, device_map={"": device}
                ),
            )
            frozen.eval()
            for parameter in frozen.parameters():
                parameter.requires_grad_(False)
        packing = prepare(model) and (frozen is None or prepare(frozen))
        return cls(model, checkpoint, frozen, packing)

    def parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        """Logprobs of `tokens[p]` given `tokens[:p]`, for each p in `positions` (all at least 1)."""
        return self.logprobs_and_entropy(tokens, positions, entropy=False)[0]

    def logprobs_and_entropy(
        self, tokens: Sequence[int], positions: Sequence[int], *, entropy: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`logprobs`, and the entropy of the policy's distribution at each of `positions`."""
        ids = self._ids(tokens)
        with torch.autocast(ids.device.type, dtype=torch.bfloat16):
            return self.scorer(ids, None, positions, entropy)

    def logprobs_among(
        self, tokens: Sequence[int], positions: Sequence[int], candidates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`logprobs`, and the logprobs of `candidates[i]` (a row of token ids) at the i-th of `positions`."""
        ids = self._ids(tokens)
        with torch.autocast(ids.device.type, dtype=torch.bfloat16):
            return self.scorer(ids, None, positions, False, candidates)

    def reference(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        """`logprobs` under the frozen copy of the model trained over (no gradient)."""
        if self.frozen_scorer is None:
            raise ValueError("this full-weight policy holds no reference (LoraSettings.frozen_reference)")
        with torch.no_grad():
            return self.frozen_scorer(self._ids(tokens), None, positions)[0]

    def packed(
        self, pack: Pack, *, entropy: bool = False, candidates: Sequence[torch.Tensor] | None = None
    ) -> list[Scores]:
        """Each of a pack's segments' sampled tokens' logprobs (`rollout_lora.policy.Policy.packed`)."""
        ids = self._ids(pack.tokens)
        joined = None if candidates is None else torch.cat(list(candidates))
        with torch.autocast(ids.device.type, dtype=torch.bfloat16):
            found = self.scorer(ids, None, (), entropy, joined, pack)
        return split_scores(found, pack, entropy=entropy, among=candidates is not None)

    def packed_reference(self, pack: Pack) -> list[torch.Tensor]:
        """`packed` logprobs under the frozen copy of the model trained over (no gradient)."""
        if self.frozen_scorer is None:
            raise ValueError("this full-weight policy holds no reference (LoraSettings.frozen_reference)")
        with torch.no_grad():
            found = self.frozen_scorer(self._ids(pack.tokens), None, (), False, None, pack)
        return [each.logprobs for each in split_scores(found, pack, entropy=False, among=False)]

    def idle(self, *, gradient: bool = False, reference: bool = False) -> None:
        """An idle pass (`rollout_lora.policy.idle_pass`)."""
        idle_pass(self, gradient=gradient, reference=reference)

    def clip_gradients(self, maximum: float, ranks: Ranks) -> float:
        """Every weight's gradient clipped (`rollout_lora.policy.clip_gradients`); its norm before."""
        return clip_gradients(self.parameters(), maximum, ranks)

    def recover(self) -> None:
        """After a pass that ran out of memory part way: what it left of the sharded weights and the reference dropped
        (`rollout_lora.policy.recover`)."""
        recover([self.scorer, self.frozen_scorer])

    def _ids(self, tokens: Sequence[int]) -> torch.Tensor:
        device = next(iter(self.model.parameters())).device
        return torch.tensor([list(tokens)], device=device)

    def save(self, directory: Path) -> Path:
        """The weights (float32, in safetensors) and the model's configuration and tokenizer."""
        directory.mkdir(parents=True, exist_ok=True)
        cast(Any, self.model).save_pretrained(directory, safe_serialization=True, max_shard_size="4GB")
        source = local(self.checkpoint)
        for name in COPIED:
            if (source / name).exists() and not (directory / name).exists():
                shutil.copy2(source / name, directory / name)
        said = json.loads((directory / "config.json").read_text())  # served in bfloat16: the files stay float32
        said |= {key: "bfloat16" for key in ("torch_dtype", "dtype") if key in said}
        (directory / "config.json").write_text(json.dumps(said, indent=2))
        return directory
