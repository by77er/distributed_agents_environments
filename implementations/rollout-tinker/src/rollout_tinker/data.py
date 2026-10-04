"""Segments as Tinker's `Datum`s: one segment, one datum, however many turns it holds.

A datum's input is the segment's tokens but the last, and its targets the tokens but the first: row *i* predicts token
*i* + 1. So a sampled token at position *t* is row *t* - 1, which carries what the loss needs of it (its behaviour or
starting logprob, its advantage, its weight). Every other row carries zeros: a prompt token, a tool result, a token the
gateway forced (none of which is in a span) adds nothing to the loss, and no gradient.
"""

from collections.abc import Mapping, Sequence

from tinker import Datum, ModelInput, TensorData

from rollout_lora.step import sampled
from rollout_train.trainer import Weighted

__all__ = ["datum", "rows"]


def rows(weighted: Weighted) -> list[int]:
    """The rows of a segment's datum that predict the tokens the policy sampled, in order."""
    positions = sampled(weighted)
    if positions and positions[0] < 1:
        raise ValueError("a segment's first token cannot have been sampled: nothing came before it")
    return [position - 1 for position in positions]


def datum(tokens: Sequence[int], at: Sequence[int], values: Mapping[str, Sequence[float]]) -> Datum:
    """The datum of a segment's `tokens`, with each of `values` (one number per row of `at`) at its row of `at` and
    zero elsewhere, beside the targets."""
    width = len(tokens) - 1
    inputs: dict[str, TensorData] = {
        "target_tokens": TensorData(data=list(tokens[1:]), dtype="int64", shape=[width]),
    }
    for key, numbers in values.items():
        if len(numbers) != len(at):
            raise ValueError(f"{key}: {len(numbers)} numbers for {len(at)} rows")
        dense = [0.0] * width
        for row, number in zip(at, numbers, strict=True):
            dense[row] = float(number)
        inputs[key] = TensorData(data=dense, dtype="float32", shape=[width])
    return Datum(model_input=ModelInput.from_ints(list(tokens[:-1])), loss_fn_inputs=inputs)
