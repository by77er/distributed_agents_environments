"""A segment's datum: the shift by one, spans across several turns, and forced tokens left out of the loss."""

import pytest
import torch

from rollout_tinker.data import datum, rows
from rollout_tinker.testing import FakeService
from rollout_train.recorder.recorder import Segment, Span
from rollout_train.trainer import Weighted

# A prompt (0-2), a turn the policy sampled (3-4), a token the recorder forced (5), a tool's result (6), a second turn
# (7-8).
TOKENS = [10, 11, 12, 13, 14, 15, 16, 17, 18]
SEGMENT = Weighted(Segment(TOKENS, [Span(3, 5, 1), Span(7, 9, 2)], [-0.1, -0.2, -0.3, -0.4]), 0.5)


def test_row_t_minus_one_predicts_sampled_token_t() -> None:
    assert rows(SEGMENT) == [2, 3, 6, 7]
    made = datum(TOKENS, rows(SEGMENT), {"logprobs": [-0.1, -0.2, -0.3, -0.4], "advantages": [0.5] * 4})
    assert made.model_input.to_ints() == TOKENS[:-1]
    targets = made.loss_fn_inputs["target_tokens"].data
    assert targets == TOKENS[1:]
    assert [targets[row] for row in rows(SEGMENT)] == [13, 14, 17, 18]  # (the sampled tokens, and only them)
    assert made.loss_fn_inputs["logprobs"].data == pytest.approx([0, 0, -0.1, -0.2, 0, 0, -0.3, -0.4])  # (float32)
    assert made.loss_fn_inputs["advantages"].data == [0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5]  # (the forced 15: 0, no gradient)


def test_the_logprob_at_a_row_is_its_tokens_given_the_one_before() -> None:
    service = FakeService(vocabulary=24)
    made = datum(TOKENS, rows(SEGMENT), {"weights": [1.0] * 4})
    table = torch.zeros_like(service.base)
    found = service.logprobs(
        table, made.model_input.to_ints(), [int(each) for each in made.loss_fn_inputs["target_tokens"].data]
    )
    for position in (3, 4, 7, 8):
        expected = torch.log_softmax(service.base[TOKENS[position - 1]], dim=-1)[TOKENS[position]]
        assert float(found[position - 1]) == pytest.approx(float(expected))


def test_numbers_for_rows_must_match_them() -> None:
    with pytest.raises(ValueError, match="3 numbers for 4 rows"):
        datum(TOKENS, rows(SEGMENT), {"weights": [1.0] * 3})
    with pytest.raises(ValueError, match="first token"):
        rows(Weighted(Segment([1, 2], [Span(0, 1, 1)], [-1.0]), 1.0))
