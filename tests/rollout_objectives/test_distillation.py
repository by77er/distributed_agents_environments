# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Distillation's terms against their definitions: each top-k divergence over the whole vocabulary is the full one
(GKD's JSD as TRL writes it, Hinton's softened KL), what the teacher did not score adds nothing, a teacher that gave
fewer than k tokens is a smaller k, the importance mask (icepop), a KL in the reward, a policy gradient's distillation
term; and the step on a toy policy, whose teacher gap falls."""

import math
from collections.abc import Sequence

import pytest
import torch
from torch import nn
from torch.nn import functional

from rollout_objectives.distillation import Taught, distillation, distilled, top_k_divergence
from rollout_objectives.settings import StepSettings
from rollout_objectives.step import PolicyStep
from rollout_objectives.terms import policy_gradient, reduced
from rollout_train.objectives import Objective, resolved
from rollout_train.recorder import Segment, Span, TeacherScores
from rollout_train.trainer import Distilled, Weighted

V = 12


def distributions(rows: int = 4, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """The student's logits (a leaf) and the teacher's logprobs, over the whole vocabulary."""
    generator = torch.Generator().manual_seed(seed)
    student = torch.randn(rows, V, generator=generator, dtype=torch.float64).requires_grad_(True)
    teacher = torch.log_softmax(torch.randn(rows, V, generator=generator, dtype=torch.float64) * 2, -1)
    return student, teacher


def over_everything(objective: Objective, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """The top-k divergence at k = the vocabulary."""
    every = torch.ones_like(teacher, dtype=torch.bool)
    return top_k_divergence(objective, torch.log_softmax(student, -1), teacher, every)


def with_divergence(divergence: str, **values: float) -> Objective:
    overrides = {"distillation.divergence": divergence, **{f"distillation.{key}": v for key, v in values.items()}}
    return resolved("distillation", {"distillation.top_k": V, **overrides})


@pytest.mark.parametrize("beta", [0.1, 0.5, 0.9])
def test_the_jsd_over_the_whole_vocabulary_is_gkds_as_trl_writes_it(beta: float) -> None:
    student, teacher = distributions()
    ours = over_everything(with_divergence("jsd", beta=beta), student, teacher)
    # TRL's `generalized_jsd_loss`: the mixture's logprobs, then beta KL(teacher || m) + (1 - beta) KL(student || m).
    logged = torch.log_softmax(student, -1)
    mixture = torch.logsumexp(torch.stack([logged + math.log(1 - beta), teacher + math.log(beta)]), dim=0)
    kl_teacher = functional.kl_div(mixture, teacher, reduction="none", log_target=True).sum(-1)
    kl_student = functional.kl_div(mixture, logged, reduction="none", log_target=True).sum(-1)
    torch.testing.assert_close(ours, beta * kl_teacher + (1 - beta) * kl_student)


@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_the_forward_kl_over_the_whole_vocabulary_is_hintons_softened_kl(temperature: float) -> None:
    student, teacher = distributions()
    ours = over_everything(with_divergence("forward_kl", temperature=temperature), student, teacher)
    soft_teacher = torch.log_softmax(teacher / temperature, -1)
    soft_student = torch.log_softmax(torch.log_softmax(student, -1) / temperature, -1)
    expected = temperature**2 * (soft_teacher.exp() * (soft_teacher - soft_student)).sum(-1)
    torch.testing.assert_close(ours, expected)


def test_the_reverse_kl_over_the_whole_vocabulary_is_the_full_reverse_kl_and_over_its_top_k_is_at_least_0() -> None:
    student, teacher = distributions()
    objective = resolved("mopd_top_k", {"distillation.top_k": V})
    logged = torch.log_softmax(student, -1)
    full = (logged.exp() * (logged - teacher)).sum(-1)
    torch.testing.assert_close(over_everything(objective, student, teacher), full)  # (sum p - q = 0 over everything)
    top = teacher.topk(3, -1).indices
    partial = top_k_divergence(objective, logged.gather(-1, top), teacher.gather(-1, top), torch.ones_like(top).bool())
    assert bool((partial >= 0).all())
    same = top_k_divergence(objective, teacher.gather(-1, top), teacher.gather(-1, top), torch.ones_like(top).bool())
    torch.testing.assert_close(same, torch.zeros_like(same))  # (0 where the two agree)


def segment_of(length: int) -> Segment:
    return Segment(list(range(length + 1)), [Span(1, length + 1, 0)], [0.0] * length)


def test_a_token_the_teacher_did_not_score_adds_nothing_and_counts_in_the_mean() -> None:
    now = torch.tensor([-1.0, -2.0, -0.5], dtype=torch.float64, requires_grad=True)
    old = torch.tensor([-1.1, -1.9, -0.6], dtype=torch.float64)
    scores = TeacherScores("t", [-0.5, None, -3.0])
    objective = resolved("on_policy_distillation")
    found = distilled(objective, Distilled(segment_of(3), scores), now, old)
    found.loss.backward()
    assert now.grad is not None and float(now.grad[1]) == 0.0
    expected = -((-0.5 + 1.1) * now[0] + (-3.0 + 0.6) * now[2])
    torch.testing.assert_close(found.loss, expected)  # (summed: the minibatch's units are its 3 tokens)
    assert (found.scored, found.distilled) == (2.0, 3.0)
    assert found.gap == pytest.approx((-1.0 + 0.5) + (-0.5 + 3.0))


def test_a_teacher_that_gave_fewer_than_k_tokens_is_a_smaller_k_there() -> None:
    student, teacher = distributions(rows=2)
    logged = torch.log_softmax(student, -1)
    top = teacher.topk(4, -1)
    short = TeacherScores("t", [-1.0, -1.0], [top.indices[0].tolist(), top.indices[1, :2].tolist()],
                          [top.values[0].tolist(), top.values[1, :2].tolist()])  # fmt: skip
    taught = Taught.of(short, 4, dtype=torch.float64)
    assert taught.kept is not None and taught.top_tokens is not None and taught.top_logprobs is not None
    assert taught.kept.tolist() == [[True] * 4, [True, True, False, False]]
    objective = resolved("distillation", {"distillation.top_k": 4})
    ours = top_k_divergence(objective, logged.gather(-1, taught.top_tokens), taught.top_logprobs, taught.kept)
    two = top_k_divergence(
        objective, logged[1:].gather(-1, top.indices[1:, :2]), top.values[1:, :2], torch.ones(1, 2).bool()
    )
    torch.testing.assert_close(ours[1:], two)


def test_the_importance_mask_zeroes_a_token_outside_its_bounds_as_icepop_does() -> None:
    now = torch.tensor([-1.0, -2.0, -0.5], dtype=torch.float64, requires_grad=True)
    old = torch.tensor([-1.0, -2.0, -0.5], dtype=torch.float64)
    behavior = torch.tensor([-1.0, -0.5, -0.6], dtype=torch.float64)  # weights 1, e^-1.5, e^0.1
    objective = resolved("mopd", {"importance.correction": "mask", "importance.floor": 0.5, "importance.cap": 2.0})
    taught = Taught(torch.tensor([-0.2, -0.2, -0.2], dtype=torch.float64))
    found = distillation(objective, now, old, taught, behavior)
    found.loss.backward()
    assert now.grad is not None and float(now.grad[1]) == 0.0 and float(now.grad[0]) != 0.0
    assert found.truncated == 1.0
    weights = torch.exp(old - behavior)
    expected = -(weights[0] * 0.8 * now[0] + weights[2] * 0.3 * now[2]) / 3
    torch.testing.assert_close(found.loss, expected)  # (a mean over the segment's 3 tokens)


def test_a_kl_in_the_reward_is_taken_from_the_advantage_and_one_in_the_loss_is_added() -> None:
    now = torch.tensor([-1.0, -2.0], dtype=torch.float64, requires_grad=True)
    old = now.detach().clone()
    reference = torch.tensor([-1.5, -1.0], dtype=torch.float64)
    taught = Taught(torch.tensor([-0.5, -2.5], dtype=torch.float64))
    k1 = old - reference  # (k1: -(reference - now))
    reward = resolved("on_policy_distillation", {"kl.target": "reference", "kl.estimator": "k1",
                                                  "kl.placement": "reward", "kl.coefficient": 0.1})  # fmt: skip
    found = distillation(reward, now, old, taught, reference=reference)
    expected = -(((taught.logprobs - old) - 0.1 * k1) * now).sum()
    torch.testing.assert_close(found.loss, expected)
    loss = resolved("on_policy_distillation", {"kl.target": "reference", "kl.estimator": "k1",
                                                "kl.coefficient": 0.1})  # fmt: skip
    found = distillation(loss, now, old, taught, reference=reference)
    torch.testing.assert_close(found.loss, (-(taught.logprobs - old) * now + 0.1 * (now - reference)).sum())


def test_a_policy_gradient_with_a_distillation_term_adds_the_term_times_its_coefficient() -> None:
    now = torch.tensor([-1.0, -2.0, -0.5], dtype=torch.float64, requires_grad=True)
    old = torch.tensor([-1.1, -1.9, -0.6], dtype=torch.float64)
    behavior = old.clone()
    scores = TeacherScores("t", [-0.5, -1.0, -3.0])
    objective = resolved("default", {"distillation.coefficient": 0.5, "distillation.advantage_clip": 1.0})
    assert objective.distills and not resolved("default").distills
    found = distilled(objective, Distilled(segment_of(3), scores, advantage=0.7), now, old, behavior)
    pg = policy_gradient(objective, now, 0.7, old, behavior)
    advantage = (torch.tensor(scores.logprobs, dtype=torch.float64) - old).clamp(-1.0, 1.0)
    torch.testing.assert_close(found.loss, pg.loss + 0.5 * reduced(objective, -advantage * now))
    assert found.advantage_clipped == 1.0 and found.scored == 3.0


class ToyPolicy:
    """A bigram model over a tiny vocabulary, which gives the logprobs of given tokens too."""

    def __init__(self, seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.model = nn.Sequential(nn.Embedding(8, 16), nn.Linear(16, 8))

    def parameters(self) -> list[nn.Parameter]:
        return list(self.model.parameters())

    def distribution(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        ids = torch.tensor(list(tokens))
        return torch.log_softmax(self.model(ids[[p - 1 for p in positions]]), -1)

    def logprobs(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        ids = torch.tensor(list(tokens))
        return self.distribution(tokens, positions).gather(-1, ids[list(positions)].unsqueeze(-1)).squeeze(-1)

    def logprobs_among(
        self, tokens: Sequence[int], positions: Sequence[int], candidates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.logprobs(tokens, positions), self.distribution(tokens, positions).gather(-1, candidates)


def taught_by(teacher: ToyPolicy, student: ToyPolicy, tokens: list[int], top_k: int) -> Distilled:
    """Every token after the first was sampled by the student; the teacher's scores of them."""
    positions = list(range(1, len(tokens)))
    with torch.no_grad():
        behavior = student.logprobs(tokens, positions).tolist()
        theirs = teacher.distribution(tokens, positions)
        top = theirs.topk(top_k, -1)
        sampled = teacher.logprobs(tokens, positions).tolist()
    scores = TeacherScores("teacher", sampled, top.indices.tolist(), top.values.tolist())
    return Distilled(Segment(tokens, [Span(1, len(tokens), 0)], behavior), scores)


def kl_to(teacher: ToyPolicy, student: ToyPolicy, sequences: Sequence[list[int]]) -> float:
    """The mean KL(student || teacher) over the sequences' positions, over the whole vocabulary."""
    with torch.no_grad():
        found: list[torch.Tensor] = []
        for tokens in sequences:
            mine = student.distribution(tokens, range(1, len(tokens)))
            theirs = teacher.distribution(tokens, range(1, len(tokens)))
            found.append((mine.exp() * (mine - theirs)).sum(-1).mean())
    return float(torch.stack(found).mean())


def sampled_from(policy: ToyPolicy, generator: torch.Generator, count: int = 24, length: int = 6) -> list[list[int]]:
    """Sequences the policy samples, each from a first token drawn uniformly."""
    made: list[list[int]] = []
    with torch.no_grad():
        for _ in range(count):
            tokens = [int(torch.randint(0, 8, (1,), generator=generator))]
            for _ in range(length - 1):
                logged = policy.distribution([*tokens, 0], [len(tokens)])[0]
                tokens.append(int(torch.multinomial(logged.exp(), 1, generator=generator)))
            made.append(tokens)
    return made


@pytest.mark.parametrize("preset", ["on_policy_distillation", "mopd", "mopd_top_k", "distillation"])
def test_a_distillation_step_on_the_students_samples_moves_it_toward_its_teacher(preset: str) -> None:
    teacher, student = ToyPolicy(seed=1), ToyPolicy(seed=0)
    generator = torch.Generator().manual_seed(0)
    every = [[first, token] for first in range(8) for token in range(8)]  # (every position the bigram has)
    objective = resolved(preset, {"distillation.top_k": 8} if "top_k" in preset or preset == "distillation" else {})
    step = PolicyStep(student, StepSettings(learning_rate=0.03, tokens_per_step=10**4, max_kl=None,  # type: ignore[arg-type]
                                            objective=objective))  # fmt: skip
    before = kl_to(teacher, student, every)
    gaps: list[float] = []
    for seed in range(40):
        items = [taught_by(teacher, student, tokens, 8) for tokens in sampled_from(student, generator)]
        metrics = step.step(items, seed=seed)
        gaps.append(metrics["teacher_gap"])
        assert metrics["unscored_fraction"] == 0.0
    after = kl_to(teacher, student, every)
    assert after < before * 0.6, (before, after)
    assert sum(gaps[-5:]) / 5 < sum(gaps[:5]) / 5, gaps  # (the student's logprob less the teacher's, falling)


def test_a_step_refuses_items_its_objective_does_not_take() -> None:
    student = ToyPolicy()
    weighted = Weighted(Segment([1, 2, 3], [Span(1, 3, 0)], [-1.0, -1.0]), 1.0)
    step = PolicyStep(student, StepSettings(objective="mopd", max_kl=None))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="distilled segments"):
        step.step([weighted])
