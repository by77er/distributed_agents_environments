"""The rubrics and the strict reading of a judge's verdict."""

import json

import pytest

from judging.rubric import EXPLAIN, RUBRICS, SUMMARIZE, MalformedVerdict, normalized


def verdict(score: int = 8, **changes: object) -> str:
    said: dict[str, object] = {
        "criteria": {each.name: 7 for each in EXPLAIN.criteria},
        "score": score,
        "reason": "Clear.",
    }
    return json.dumps({**said, **changes})


def test_a_well_formed_verdict_is_read() -> None:
    read = EXPLAIN.verdict(f"  {verdict(9)}\n")
    assert (read.score, read.reason) == (9, "Clear.") and set(read.criteria) == {each.name for each in EXPLAIN.criteria}
    assert read.normalized == pytest.approx(8 / 9) and normalized(1) == 0.0 and normalized(10) == 1.0


@pytest.mark.parametrize(
    ("reply", "why"),
    [
        ("Score: 8", "not one JSON object"),
        (f"Here it is: {verdict()}", "not one JSON object"),
        (f"```json\n{verdict()}\n```", "not one JSON object"),
        ("{score: 8}", "not valid JSON"),
        (verdict(11), "score is not a whole number from 1 to 10: 11"),
        (verdict(True), "score is not a whole number from 1 to 10: true"),  # type: ignore[arg-type]
        (verdict(7.5), "score is not a whole number"),  # type: ignore[arg-type]
        (verdict(reason=""), "reason is not a sentence"),
        (verdict(criteria={"accuracy": 7}), "criteria is not an object scoring exactly accuracy, clarity"),
        (verdict(criteria={**{each.name: 7 for each in EXPLAIN.criteria}, "length": 0}), "criteria.length is not"),
        (json.dumps({"score": 8, "reason": "x"}), "not criteria, score and reason"),
        (verdict(extra=1), "not criteria, score and reason"),
    ],
)
def test_a_malformed_verdict_says_why(reply: str, why: str) -> None:
    with pytest.raises(MalformedVerdict, match=why.replace("(", r"\(")):
        EXPLAIN.verdict(reply)


def test_each_rubric_is_versioned_and_tells_the_judge_its_criteria_and_the_form() -> None:
    assert {rubric.version for rubric in RUBRICS.values()} == {"explain-1", "summarize-1"}
    for rubric in (EXPLAIN, SUMMARIZE):
        told = rubric.instructions()
        assert all(each.name in told for each in rubric.criteria) and "from 1" in told and "JSON object" in told
        example = json.loads(told.splitlines()[-1])
        assert rubric.verdict(json.dumps(example)).score == 8  # (the form it is shown is one it reads)
