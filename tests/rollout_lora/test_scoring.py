# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""The output layer's scores, a chunk of rows at a time: the logprobs of given tokens at each position (a teacher's
top-k) are those of the whole distribution there, with their gradient."""

import pytest
import torch
from torch import nn

from rollout_lora import policy
from rollout_lora.policy import scored_among, scored_with_entropy


class Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lm_head = nn.Linear(6, 11, bias=False)


def test_the_logprobs_of_given_tokens_are_the_distributions_there_a_chunk_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Head()
    hidden = torch.randn(9, 6, requires_grad=True)
    ids = torch.randint(0, 11, (1, 9))
    positions = [2, 3, 5, 8]
    candidates = torch.randint(0, 11, (4, 3))
    monkeypatch.setattr(policy, "LOGIT_ROWS", 3)  # (two chunks)
    sampled, among = scored_among(model, hidden, ids, positions, candidates)
    full = torch.log_softmax(model.lm_head(hidden[[p - 1 for p in positions]]), -1)
    torch.testing.assert_close(among, full.gather(-1, candidates))
    torch.testing.assert_close(sampled, scored_with_entropy(model, hidden, ids, positions, entropy=False)[0])
    among.sum().backward()
    assert hidden.grad is not None and bool(hidden.grad[[p - 1 for p in positions]].abs().sum() > 0)
