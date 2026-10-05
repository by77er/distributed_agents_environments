"""The episode: the policy answers an open-ended request, and a judge scores the answer against the request's rubric.

Two model slots: `policy`, which answers and is trained, and `judge`, which is not trained (`ModelSlot(trained=False,
judge=True)`): a run binds it to a channel by name, and to one serving the run's own checkpoints only with
`self_judging`. The judge is shown the rubric (`judging.rubric`), the request and the answer, and must reply with the
rubric's JSON verdict. A reply that cannot be read is answered once with what was wrong, and the judge asked again; a
second that cannot be read ends the episode with reward 0, and the result says why. With `judgements` of 2, the judge
is asked twice, each time afresh, and the scores averaged.

The reward is the average score on a scale of 0 to 1 (`judging.rubric.normalized`). The result says the kind of
request, the rubric's version, the answer's length in words, every reply the judge gave (`verdicts`, as it gave them),
the scores read from them and their average (`score`, from 1 to 10), and whether the answer was judged (`judged`, with
`why` where it was not).
"""

from collections.abc import Mapping
from typing import cast

from pydantic import JsonValue

from judging.rubric import RUBRICS, MalformedVerdict, Verdict, normalized
from rollout.contracts import Message
from rollout.harness import ModelSlot, Program, RunContext

__all__ = ["RETRIES", "SYSTEM", "JudgedAnswer"]

SYSTEM = "Answer the request. Reply with the answer only."
"""The policy's system prompt."""
RETRIES = 1
"""Times the judge is asked again after a verdict that cannot be read."""


class JudgedAnswer(Program):
    """Parameters: `kind` (a key of `judging.rubric.RUBRICS`), `request` (what the policy is asked), `judgements` (how
    many times the judge is asked: 1 or 2) and `seed` (unused by the episode: it tells starts apart)."""

    def __init__(self, parameters: Mapping[str, JsonValue] | None = None) -> None:
        parameters = parameters or {}
        self.kind = str(parameters.get("kind", "explain"))
        if self.kind not in RUBRICS:
            raise ValueError(f"the kind is one of {', '.join(RUBRICS)}, not {self.kind!r}")
        self.rubric = RUBRICS[self.kind]
        self.request = str(parameters.get("request", "Explain photosynthesis to a ten-year-old in under 80 words."))
        self.judgements = int(cast(int, parameters.get("judgements", 1)))
        if self.judgements not in (1, 2):
            raise ValueError(f"the judge is asked once or twice, not {self.judgements} times")

    def model_slots(self) -> Mapping[str, ModelSlot]:
        return {"policy": ModelSlot(), "judge": ModelSlot(trained=False, judge=True)}

    async def main(self, run: RunContext) -> None:
        reply = await run.models["policy"].sample([Message.system(SYSTEM), Message.user(self.request)])
        answer = reply.text.strip()
        verdicts: list[JsonValue] = []
        scores: list[JsonValue] = []
        why: str | None = None
        for _ in range(self.judgements):
            verdict, why = await self.judged(run, answer, verdicts)
            if verdict is None:
                break
            scores.append(verdict.score)
        judged = why is None
        score = sum(cast(list[int], scores)) / len(scores) if judged else None
        run.reward(normalized(score) if score is not None else 0.0, slot="policy")
        result: dict[str, JsonValue] = {
            "kind": self.kind,
            "rubric": self.rubric.version,
            "words": len(answer.split()),
            "verdicts": verdicts,
            "scores": scores,
            "score": score,
            "judged": judged,
        }
        if why is not None:
            result["why"] = f"the judge's verdict could not be read after {RETRIES + 1} replies: {why}"
        await run.emit("result", result)

    async def judged(
        self, run: RunContext, answer: str, verdicts: list[JsonValue]
    ) -> tuple[Verdict | None, str | None]:
        """One judgement of the answer, afresh: the verdict read, or none and why the last reply could not be read.
        Every reply the judge gives is added to `verdicts`, as it gave it."""
        messages = [Message.system(self.rubric.instructions()), Message.user(self.rubric.asked(self.request, answer))]
        why = ""
        for _ in range(RETRIES + 1):
            said = await run.models["judge"].sample(messages)
            verdicts.append(said.text)
            try:
                return self.rubric.verdict(said.text), None
            except MalformedVerdict as error:
                why = str(error)
            messages += [
                said,
                Message.user(f"Your verdict could not be read: {why}. Reply with the JSON verdict alone."),
            ]
        return None, why
