"""An environment's eval data is derived from rows and seeds, or listed; training never draws an eval start."""

import random
from collections.abc import Mapping, Sequence

import pytest
from pydantic import JsonValue

from rollout.environment import Description, Row, Start, drawn, held_out, start_key, train_start
from rollout.harness import ProgramReference


class Coins:
    """Rows whose starts are a coin's faces: heads is held out of the first two rows for evals, tails of the third."""

    program = ProgramReference(program="nothing:Nothing")
    version = "1"
    description = Description()

    def rows(self) -> Sequence[Row]:
        return [Row("heads-first", "heads first"), Row("tails-first", "tails first"), Row("hidden", "hidden")]

    def start(self, row: Row, rng: random.Random) -> JsonValue:
        return {"row": row.key, "face": rng.choice(["heads", "tails"])}

    def evals(self) -> Mapping[str, Sequence[Start]]:
        heads = [Start(row.key, row.title, 0, {"row": row.key, "face": "heads"}) for row in self.rows()[:2]]
        return {
            "coins-heads": heads,
            "coins-hidden": [Start("hidden", "hidden", 0, {"row": "hidden", "face": "tails"})],
        }


def test_eval_data_is_derived_from_rows_and_seeds() -> None:
    coins = Coins()
    starts = drawn(coins, seeds=[1, 2], rows=["tails-first"])
    assert [(start.task, start.seed) for start in starts] == [("tails-first", 1), ("tails-first", 2)]
    assert starts == drawn(coins, seeds=[1, 2], rows=["tails-first"])  # (one seed, one start)
    assert len(drawn(coins, seeds=[5])) == 3
    with pytest.raises(ValueError, match="no row nothing"):
        drawn(coins, seeds=[1], rows=["nothing"])
    with pytest.raises(ValueError, match="a seed at least"):
        drawn(coins, seeds=[])


def test_training_never_draws_an_eval_start() -> None:
    coins, held = Coins(), held_out(Coins())
    assert start_key({"face": "heads", "row": "heads-first"}) in held  # (keys are canonical: order does not matter)
    rows = coins.rows()
    drawn_faces = {
        str(train_start(coins, row, random.Random(seed), held)["face"])  # type: ignore[index]
        for row in rows[:2]
        for seed in range(50)
    }
    assert drawn_faces == {"tails"}
    raw = {str(coins.start(rows[0], random.Random(seed))["face"]) for seed in range(50)}  # type: ignore[index]
    assert raw == {"heads", "tails"}  # (left to itself, it draws both)
    assert train_start(coins, rows[2], random.Random(0), held) == {"row": "hidden", "face": "heads"}
    with pytest.raises(ValueError, match="every start of heads-first drawn was an eval start"):
        train_start(coins, rows[0], random.Random(0), held | {start_key({"row": "heads-first", "face": "tails"})})


def test_a_description_says_what_results_say_and_what_an_episode_samples() -> None:
    said = Description(rewards=(0.0, None), saturated=True, duration="turns").to_json()
    assert said == {
        "rewards": [0.0, None], "solved": True, "saturated": True, "duration": "turns", "observations": None,
        "turns": None, "samples_per_turn": 1.0, "prompt_tokens": None,
    }  # fmt: skip
    team = Description(turns=40.0, samples_per_turn=3.0, prompt_tokens=1500).to_json()
    assert (team["turns"], team["samples_per_turn"], team["prompt_tokens"]) == (40.0, 3.0, 1500)
