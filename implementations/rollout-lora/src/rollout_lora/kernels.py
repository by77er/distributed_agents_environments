# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportArgumentType=false, reportCallIssue=false
# (Triton has no type stubs, types a kernel's parameters as it compiles it, and takes launch options as keywords.)
"""Triton kernels of the 4-bit layers (`rollout_lora.quantized`) and of the fused elementwise operations
(`rollout_lora.fused`), imported only where they run on a GPU.

`dequantized` writes a weight from its packed words in one pass. `half_matmul` multiplies on fp16 tensor cores
accumulating in fp16, which a consumer GPU (GeForce) runs at twice the rate of any accumulation in fp32, bf16's
included: each tile's fp16 sum of `BLOCK_K` products is added to an fp32 total, so no sum in fp16 is longer than
`BLOCK_K` terms. The rows of the left operand are scaled into fp16 first (`half_rows`: each row's largest magnitude
made 1, so no partial sum overflows), and its scale multiplies the row of the result back."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

__all__ = [
    "BLOCK_K", "WORDS", "dequantized", "gated_norm_backward", "gated_norm_forward", "half_matmul", "half_rows",
    "norm_backward", "norm_forward", "silu_product_backward", "silu_product_forward",
]  # fmt: skip

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


ROWS_ELEMENTS = 4096
"""Elements of a block of rows one program of a norm takes (a row of a 9B model's hidden state, or many rows of a
head's)."""
SILU_BLOCK = 2048
"""Elements one program of `silu_product_forward` and `silu_product_backward` takes."""


def _row_blocks(width: int) -> tuple[int, int, int]:
    """Rows a program takes, the row's block (a power of two), and warps, for rows of `width`."""
    block = int(triton.next_power_of_2(width))
    return max(1, min(64, ROWS_ELEMENTS // block)), block, (8 if block >= 2048 else 4)


@triton.jit
def _norm_forward(
    x, weight, out, scales, R, D, stride, eps, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr
):  # fmt: skip
    """`BLOCK_R` rows of `out = x * rsqrt(mean(x²) + eps) * (1 + weight)`, in float32, rounded once; and each row's
    rsqrt (`scales`)."""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    columns = tl.arange(0, BLOCK_D)
    kept = (rows[:, None] < R) & (columns[None, :] < D)
    values = tl.load(x + rows[:, None].to(tl.int64) * stride + columns[None, :], mask=kept, other=0.0).to(tl.float32)
    scale = libdevice.rsqrt(tl.sum(values * values, axis=1) / D + eps)
    weights = tl.load(weight + columns, mask=columns < D, other=0.0).to(tl.float32)
    normed = (values * scale[:, None]) * (1.0 + weights[None, :])
    tl.store(out + rows[:, None].to(tl.int64) * D + columns[None, :], normed.to(out.dtype.element_ty), mask=kept)
    tl.store(scales + rows, scale, mask=rows < R)


@triton.jit
def _norm_backward(
    gradient, x, weight, scales, dx, R, D, stride, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr
):  # fmt: skip
    """`BLOCK_R` rows of `_norm_forward`'s gradient with respect to `x`."""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    columns = tl.arange(0, BLOCK_D)
    kept = (rows[:, None] < R) & (columns[None, :] < D)
    values = tl.load(x + rows[:, None].to(tl.int64) * stride + columns[None, :], mask=kept, other=0.0).to(tl.float32)
    given = tl.load(gradient + rows[:, None].to(tl.int64) * D + columns[None, :], mask=kept, other=0.0)
    weights = tl.load(weight + columns, mask=columns < D, other=0.0).to(tl.float32)
    scale = tl.load(scales + rows, mask=rows < R, other=0.0)
    normed = given.to(tl.float32) * (1.0 + weights[None, :])
    along = tl.sum(normed * values, axis=1)
    found = scale[:, None] * normed - (scale * scale * scale * along / D)[:, None] * values
    tl.store(dx + rows[:, None].to(tl.int64) * D + columns[None, :], found.to(dx.dtype.element_ty), mask=kept)


def _as_rows(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """`x` as rows of its last dimension a fixed stride apart (itself where it already is, as a head's slice of a
    projection is), and that stride."""
    width = x.shape[-1]
    flat = all(x.stride(axis) == x.stride(axis + 1) * x.shape[axis + 1] for axis in range(x.dim() - 2))
    if x.dim() > 1 and x.stride(-1) == 1 and x.stride(-2) >= width and flat:
        return x, x.stride(-2)
    return x.contiguous(), width


def norm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor, int]:
    """A zero-centred RMS norm of `x`'s rows, as Qwen3.5's `RMSNorm` computes it; with each row's rsqrt and the
    rows (and their stride) the backward pass reads."""
    rows, stride = _as_rows(x)
    width = x.shape[-1]
    count = x.numel() // width
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    scales = torch.empty(count, dtype=torch.float32, device=x.device)
    block_rows, block, warps = _row_blocks(width)
    if count:
        _norm_forward[(triton.cdiv(count, block_rows),)](
            rows, weight, out, scales, count, width, stride, eps, BLOCK_R=block_rows, BLOCK_D=block, num_warps=warps
        )
    return out, scales, stride


def norm_backward(
    gradient: torch.Tensor, rows: torch.Tensor, stride: int, weight: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """`norm_forward`'s gradient with respect to its input (contiguous, in its shape)."""
    gradient = gradient.contiguous()
    width = rows.shape[-1]
    count = scales.shape[0]
    dx = torch.empty(rows.shape, dtype=rows.dtype, device=rows.device)
    block_rows, block, warps = _row_blocks(width)
    if count:
        _norm_backward[(triton.cdiv(count, block_rows),)](
            gradient, rows, weight, scales, dx, count, width, stride, BLOCK_R=block_rows, BLOCK_D=block,
            num_warps=warps,
        )  # fmt: skip
    return dx


@triton.jit
def _gated_norm_forward(
    x, gate, weight, out, scales, R, D, eps, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr
):  # fmt: skip
    """`BLOCK_R` rows of Qwen3.5's gated norm: `round(weight * round(x * rsqrt(mean(x²) + eps))) * silu(gate)`,
    rounded once more, as its eager operations round."""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    columns = tl.arange(0, BLOCK_D)
    kept = (rows[:, None] < R) & (columns[None, :] < D)
    at = rows[:, None].to(tl.int64) * D + columns[None, :]
    values = tl.load(x + at, mask=kept, other=0.0).to(tl.float32)
    gates = tl.load(gate + at, mask=kept, other=0.0).to(tl.float32)
    scale = libdevice.rsqrt(tl.sum(values * values, axis=1) / D + eps)
    weights = tl.load(weight + columns, mask=columns < D, other=0.0).to(tl.float32)
    normed = (values * scale[:, None]).to(out.dtype.element_ty).to(tl.float32)
    weighted = (weights[None, :] * normed).to(out.dtype.element_ty).to(tl.float32)
    activated = tl.math.div_rn(gates, 1.0 + libdevice.exp(-gates))
    tl.store(out + at, (weighted * activated).to(out.dtype.element_ty), mask=kept)
    tl.store(scales + rows, scale, mask=rows < R)


@triton.jit
def _gated_norm_backward(
    gradient, x, gate, weight, scales, dx, dgate, R, D, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr
):  # fmt: skip
    """`BLOCK_R` rows of `_gated_norm_forward`'s gradients with respect to `x` and `gate`, rounded where the eager
    operations' backward passes round."""
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    columns = tl.arange(0, BLOCK_D)
    kept = (rows[:, None] < R) & (columns[None, :] < D)
    at = rows[:, None].to(tl.int64) * D + columns[None, :]
    values = tl.load(x + at, mask=kept, other=0.0).to(tl.float32)
    gates = tl.load(gate + at, mask=kept, other=0.0).to(tl.float32)
    given = tl.load(gradient + at, mask=kept, other=0.0).to(tl.float32)
    weights = tl.load(weight + columns, mask=columns < D, other=0.0).to(tl.float32)
    scale = tl.load(scales + rows, mask=rows < R, other=0.0)
    normed = (values * scale[:, None]).to(dx.dtype.element_ty).to(tl.float32)
    weighted = (weights[None, :] * normed).to(dx.dtype.element_ty).to(tl.float32)
    sigmoid = tl.math.div_rn(1.0, 1.0 + libdevice.exp(-gates))
    activated = tl.math.div_rn(gates, 1.0 + libdevice.exp(-gates))
    through = (given * activated).to(dx.dtype.element_ty).to(tl.float32)
    found_gate = given * weighted * (sigmoid * (1.0 + gates * (1.0 - sigmoid)))
    tl.store(dgate + at, found_gate.to(dgate.dtype.element_ty), mask=kept)
    unweighted = (through * weights[None, :]).to(dx.dtype.element_ty).to(tl.float32)
    along = tl.sum(unweighted * values, axis=1)
    found = scale[:, None] * unweighted - (scale * scale * scale * along / D)[:, None] * values
    tl.store(dx + at, found.to(dx.dtype.element_ty), mask=kept)


def gated_norm_forward(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Qwen3.5's gated RMS norm of `x`'s rows by `gate`'s; with each row's rsqrt, and the contiguous `x` and `gate`
    the backward pass reads."""
    x, gate = x.contiguous(), gate.contiguous()
    width = x.shape[-1]
    count = x.numel() // width
    out = torch.empty_like(x)
    scales = torch.empty(count, dtype=torch.float32, device=x.device)
    block_rows, block, warps = _row_blocks(width)
    if count:
        _gated_norm_forward[(triton.cdiv(count, block_rows),)](
            x, gate, weight, out, scales, count, width, eps, BLOCK_R=block_rows, BLOCK_D=block, num_warps=warps
        )
    return out, scales, x, gate


def gated_norm_backward(
    gradient: torch.Tensor, x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`gated_norm_forward`'s gradients with respect to `x` and `gate`."""
    gradient = gradient.contiguous()
    width = x.shape[-1]
    count = scales.shape[0]
    dx, dgate = torch.empty_like(x), torch.empty_like(gate)
    block_rows, block, warps = _row_blocks(width)
    if count:
        _gated_norm_backward[(triton.cdiv(count, block_rows),)](
            gradient, x, gate, weight, scales, dx, dgate, count, width, BLOCK_R=block_rows, BLOCK_D=block,
            num_warps=warps,
        )  # fmt: skip
    return dx, dgate


@triton.jit
def _silu_product_forward(gate, up, out, N, BLOCK: tl.constexpr):
    """`BLOCK` elements of `round(round(silu(gate)) * up)`, as an MLP's `act_fn(gate) * up` rounds."""
    at = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    kept = at < N
    gates = tl.load(gate + at, mask=kept, other=0.0).to(tl.float32)
    ups = tl.load(up + at, mask=kept, other=0.0).to(tl.float32)
    activated = tl.math.div_rn(gates, 1.0 + libdevice.exp(-gates)).to(out.dtype.element_ty).to(tl.float32)
    tl.store(out + at, (activated * ups).to(out.dtype.element_ty), mask=kept)


@triton.jit
def _silu_product_backward(gradient, gate, up, dgate, dup, N, BLOCK: tl.constexpr):
    """`BLOCK` elements of `_silu_product_forward`'s gradients with respect to `gate` and `up`."""
    at = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    kept = at < N
    gates = tl.load(gate + at, mask=kept, other=0.0).to(tl.float32)
    ups = tl.load(up + at, mask=kept, other=0.0).to(tl.float32)
    given = tl.load(gradient + at, mask=kept, other=0.0).to(tl.float32)
    sigmoid = tl.math.div_rn(1.0, 1.0 + libdevice.exp(-gates))
    activated = tl.math.div_rn(gates, 1.0 + libdevice.exp(-gates)).to(dup.dtype.element_ty).to(tl.float32)
    through = (given * ups).to(dgate.dtype.element_ty).to(tl.float32)
    tl.store(dup + at, (given * activated).to(dup.dtype.element_ty), mask=kept)
    found = through * (sigmoid * (1.0 + gates * (1.0 - sigmoid)))
    tl.store(dgate + at, found.to(dgate.dtype.element_ty), mask=kept)


def silu_product_forward(gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`silu(gate) * up`, rounded as the eager operations round; with the contiguous `gate` and `up` the backward
    pass reads."""
    gate, up = gate.contiguous(), up.contiguous()
    out = torch.empty_like(gate)
    count = gate.numel()
    if count:
        _silu_product_forward[(triton.cdiv(count, SILU_BLOCK),)](gate, up, out, count, BLOCK=SILU_BLOCK, num_warps=8)
    return out, gate, up


def silu_product_backward(
    gradient: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`silu_product_forward`'s gradients with respect to `gate` and `up`."""
    gradient = gradient.contiguous()
    dgate, dup = torch.empty_like(gate), torch.empty_like(up)
    count = gate.numel()
    if count:
        _silu_product_backward[(triton.cdiv(count, SILU_BLOCK),)](
            gradient, gate, up, dgate, dup, count, BLOCK=SILU_BLOCK, num_warps=8
        )
    return dgate, dup
