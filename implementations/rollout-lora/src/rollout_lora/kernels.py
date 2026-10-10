# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportArgumentType=false, reportCallIssue=false
# (Triton has no type stubs, types a kernel's parameters as it compiles it, and takes launch options as keywords.)
"""Triton kernels of the 4-bit layers (`rollout_lora.quantized`), imported only where a weight is on a GPU.

`dequantized` writes a weight from its packed words in one pass. `half_matmul` multiplies on fp16 tensor cores
accumulating in fp16, which a consumer GPU (GeForce) runs at twice the rate of any accumulation in fp32, bf16's
included: each tile's fp16 sum of `BLOCK_K` products is added to an fp32 total, so no sum in fp16 is longer than
`BLOCK_K` terms. The rows of the left operand are scaled into fp16 first (`half_rows`: each row's largest magnitude
made 1, so no partial sum overflows), and its scale multiplies the row of the result back."""

import torch
import triton
import triton.language as tl

__all__ = ["BLOCK_K", "WORDS", "dequantized", "half_matmul", "half_rows"]

WORDS = 128
"""Packed words (eight weights each) one program of the dequantizing kernel takes."""


@triton.jit
def _dequantize(packed, scale, out, words, columns, groups, group, factor, WORDS: tl.constexpr):
    """One row's `WORDS` packed words, from the `program_id(1)`-th block on, as their weights in `out`: each int4
    value (lowest bits first, less 8) times its group's scale and `factor`, in float32, rounded once to `out`'s
    dtype."""
    row = tl.program_id(0).to(tl.int64)
    word = tl.program_id(1) * WORDS + tl.arange(0, WORDS)
    loaded = tl.load(packed + row * words + word, mask=word < words, other=0)
    nibble = tl.arange(0, 8)
    column = word[:, None] * 8 + nibble[None, :]
    kept = column < columns
    values = ((loaded[:, None] >> (nibble[None, :] * 4)) & 0xF) - 8
    scales = tl.load(scale + row * groups + column // group, mask=kept, other=0.0)
    weights = values.to(tl.float32) * scales.to(tl.float32) * factor
    tl.store(out + row * columns + column, weights.to(out.dtype.element_ty), mask=kept)


def dequantized(
    packed: torch.Tensor, scale: torch.Tensor, columns: int, dtype: torch.dtype, factor: float = 1.0
) -> torch.Tensor:
    """A weight of `columns` input columns in `dtype`, from its packed int4 words (`packed`, a row's eight to an
    int32) and its scales (`scale`, a row's one for each group of `columns // groups` columns), as
    `rollout_lora.quantized.dequantize` makes it; times `factor` (a power of two: exact), to keep its smallest
    weights out of fp16's subnormals."""
    rows, words = packed.shape
    out = torch.empty(rows, columns, dtype=dtype, device=packed.device)
    groups = scale.shape[1]
    grid = (rows, triton.cdiv(words, WORDS))
    _dequantize[grid](
        packed.contiguous(), scale.contiguous(), out, words, columns, groups, columns // groups, factor, WORDS=WORDS
    )
    return out


ROW_BLOCK = 1024
"""Columns of a row `half_rows` reads at a time."""
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
"""`half_matmul`'s tile: rows, columns, and the products each fp16 sum takes before the fp32 total does (the fastest
on an RTX 5080 of those whose pipeline fits a consumer GPU's 99 KiB of shared memory a block)."""
GROUP_M = 8
"""Row tiles of `half_matmul` that take their column tiles in turn, so that each column tile is read from L2."""


@triton.jit
def _half_rows(x, out, rows, columns, BLOCK: tl.constexpr):
    """Row `program_id(0)` of `x` divided by its largest magnitude, in fp16 (`out`), and that magnitude (`rows`);
    a row of zeros stays zeros."""
    row = tl.program_id(0).to(tl.int64)
    largest = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, columns, BLOCK):
        column = start + tl.arange(0, BLOCK)
        loaded = tl.load(x + row * columns + column, mask=column < columns, other=0.0)
        largest = tl.maximum(largest, tl.abs(loaded.to(tl.float32)))
    biggest = tl.max(largest, axis=0)
    inverse = tl.where(biggest > 0, 1.0 / biggest, 0.0)
    tl.store(rows + row, biggest)
    for start in range(0, columns, BLOCK):
        column = start + tl.arange(0, BLOCK)
        loaded = tl.load(x + row * columns + column, mask=column < columns, other=0.0)
        tl.store(out + row * columns + column, (loaded.to(tl.float32) * inverse).to(tl.float16), mask=column < columns)


def half_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A matrix in fp16 with each row divided by its largest magnitude, and those magnitudes (float32)."""
    x = x.contiguous()
    count, columns = x.shape
    out = torch.empty(count, columns, dtype=torch.float16, device=x.device)
    rows = torch.empty(count, dtype=torch.float32, device=x.device)
    if count:
        _half_rows[(count,)](x, out, rows, columns, BLOCK=ROW_BLOCK)
    return out, rows


@triton.jit
def _half_matmul(
    a, rows, b, c, M, N, K, factor, stride_bk, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, EVEN_K: tl.constexpr,
):  # fmt: skip
    """One `BLOCK_M` x `BLOCK_N` tile of `c = rows[:, None] * factor * (a @ b)` (`EVEN_K`: `K` a multiple of `BLOCK_K`,
    so no load is masked along it, which costs about a tenth)."""
    program = tl.program_id(0)
    tiles_m = tl.cdiv(M, BLOCK_M)
    tiles_n = tl.cdiv(N, BLOCK_N)
    band = GROUP_M * tiles_n
    first = (program // band) * GROUP_M
    height = min(tiles_m - first, GROUP_M)
    tile_m = first + (program % band) % height
    tile_n = (program % band) // height
    row = (tile_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    column = (tile_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    inner = tl.arange(0, BLOCK_K)
    left = a + row[:, None] * K + inner[None, :]
    right = b + inner[:, None] * stride_bk + column[None, :] * stride_bn
    total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start in range(0, K, BLOCK_K):
        if EVEN_K:
            x = tl.load(left, mask=row[:, None] < M, other=0.0)
            y = tl.load(right, mask=column[None, :] < N, other=0.0)
        else:
            x = tl.load(left, mask=(row[:, None] < M) & (start + inner[None, :] < K), other=0.0)
            y = tl.load(right, mask=(start + inner[:, None] < K) & (column[None, :] < N), other=0.0)
        part = tl.dot(x, y, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16), out_dtype=tl.float16)
        total += part.to(tl.float32)
        left += BLOCK_K
        right += BLOCK_K * stride_bk
    scales = tl.load(rows + row, mask=row < M, other=0.0) * factor
    tl.store(
        c + row[:, None] * N + column[None, :],
        (total * scales[:, None]).to(c.dtype.element_ty),
        mask=(row[:, None] < M) & (column[None, :] < N),
    )


def half_matmul(x: torch.Tensor, b: torch.Tensor, factor: float = 1.0) -> torch.Tensor:
    """`factor * (x @ b)` in `x`'s dtype, on fp16 tensor cores accumulating in fp16 a tile at a time (`half_rows`
    scales `x`'s rows into fp16 first); `b` in fp16, of any strides (a weight or its transpose)."""
    half, rows = half_rows(x)
    count, inner = half.shape
    columns = b.shape[1]
    out = torch.empty(count, columns, dtype=x.dtype, device=x.device)
    if count and columns:
        grid = (triton.cdiv(count, BLOCK_M) * triton.cdiv(columns, BLOCK_N),)
        _half_matmul[grid](
            half, rows, b, out, count, columns, inner, factor, b.stride(0), b.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M, EVEN_K=inner % BLOCK_K == 0,
            num_warps=8, num_stages=3,
        )  # fmt: skip
    return out
