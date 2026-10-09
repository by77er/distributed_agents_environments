# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportArgumentType=false
# (Triton has no type stubs, and types a kernel's parameters itself as it compiles it.)
"""Triton kernels of the 4-bit layers (`rollout_lora.quantized`), imported only where a weight is on a GPU."""

import torch
import triton
import triton.language as tl

__all__ = ["WORDS", "dequantized"]

WORDS = 128
"""Packed words (eight weights each) one program of the dequantizing kernel takes."""


@triton.jit
def _dequantize(packed, scale, out, words, columns, groups, group, WORDS: tl.constexpr):
    """One row's `WORDS` packed words, from the `program_id(1)`-th block on, as their weights in `out`: each int4
    value (lowest bits first, less 8) times its group's scale, in float32, rounded once to `out`'s dtype."""
    row = tl.program_id(0).to(tl.int64)
    word = tl.program_id(1) * WORDS + tl.arange(0, WORDS)
    loaded = tl.load(packed + row * words + word, mask=word < words, other=0)
    nibble = tl.arange(0, 8)
    column = word[:, None] * 8 + nibble[None, :]
    kept = column < columns
    values = ((loaded[:, None] >> (nibble[None, :] * 4)) & 0xF) - 8
    scales = tl.load(scale + row * groups + column // group, mask=kept, other=0.0)
    weights = values.to(tl.float32) * scales.to(tl.float32)
    tl.store(out + row * columns + column, weights.to(out.dtype.element_ty), mask=kept)


def dequantized(packed: torch.Tensor, scale: torch.Tensor, columns: int, dtype: torch.dtype) -> torch.Tensor:
    """A weight of `columns` input columns in `dtype`, from its packed int4 words (`packed`, a row's eight to an
    int32) and its scales (`scale`, a row's one for each group of `columns // groups` columns), as
    `rollout_lora.quantized.dequantize` makes it."""
    rows, words = packed.shape
    out = torch.empty(rows, columns, dtype=dtype, device=packed.device)
    groups = scale.shape[1]
    grid = (rows, triton.cdiv(words, WORDS))
    _dequantize[grid](
        packed.contiguous(), scale.contiguous(), out, words, columns, groups, columns // groups, WORDS=WORDS
    )
    return out
