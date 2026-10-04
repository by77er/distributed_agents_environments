"""Every run of a ledger in figures: groups joined with their rows, steps with the trainer's statistics, what waited."""

from pydantic import JsonValue

from rollout_train.monitor.statistics import THROUGHPUT, statistics


def tables() -> dict[str, dict[str, JsonValue]]:
    """A run of three groups: one trained on, one skipped, one in flight; one step, its version made."""
    return {
        "runs/a/groups": {
            "1": {"task": "t1", "title": "first", "decided": 10.0, "parameters": {"names": ["ada", "bo"]}},
            "2": {"task": "t2", "title": "second", "decided": 12.0, "parameters": None},
            "3": {"task": "t1", "title": "first", "decided": 30.0},
        },
        "runs/a/results": {
            "1": {"time": 20.0, "rewards": [1.0, 0.0], "solved": [True, False], "segments": 6},
            "2": {"time": 25.0, "rewards": [2.0, 2.0], "solved": [True, True], "segments": 0, "skipped": "same"},
        },
        "runs/a/steps": {"1": {"policy": "p", "parent": None, "number": 1, "groups": [1], "decided": 40.0}},
        "policies/p/versions": {
            "1": {"policy": "p", "number": 1, "made": 50.0, "metrics": {"kl_moved": 0.01, "update_seconds": 9.0}}
        },
        "runs/b/starts": {"1": {"policy": "q", "started": 5.0}},
    }


def test_a_runs_groups_are_joined_with_their_rows_and_what_their_steps_did() -> None:
    figures = statistics(tables(), now=60.0)
    runs = {run["run"]: run for run in figures["runs"]}
    assert list(runs) == ["a", "b"] and runs["b"]["groups"] == [] and runs["b"]["wrote"] is None
    first, second, third = runs["a"]["groups"]
    assert first == {
        **first,
        "task": "t1",
        "title": "first",
        "rollout_seconds": 10.0,
        "step": 1,
        "trained": "committed",
    }
    assert first["names"] == 2 and second["names"] is None and second["skipped"] == "same" and second["step"] is None
    assert third["time"] is None and third["rewards"] == []  # (in flight)
    (step,) = runs["a"]["steps"]
    assert step == {**step, "version": "p@1", "made": 50.0, "state": "committed", "groups": 1}
    assert step["metrics"] == {"kl_moved": 0.01, "update_seconds": 9.0}
    assert runs["a"]["wrote"] == 50.0
    # In flight from each decision to its result; group 1 waited from its result to its step, group 2 never did.
    assert runs["a"]["flight"] == [[10.0, 1, 0], [12.0, 2, 0], [20.0, 1, 1], [25.0, 0, 1], [30.0, 1, 1], [40.0, 1, 0]]


def test_a_runs_engines_are_measured_from_its_own_feed_merged_down_to_a_few_hundred_points() -> None:
    notes = [
        {"kind": "inference", "at": float(at), "channel": "policy", "tokens_per_second": 10.0, "generated_tokens": 5}
        for at in range(1, THROUGHPUT * 2 + 2)
    ]
    figures = statistics(tables(), {"a": [*notes, {"kind": "published", "at": 3.0}]}, now=60.0)
    a, b = figures["runs"]
    (channel,) = a["inference"]
    assert channel["every"] == 3 and len(channel["points"]) == 481 <= THROUGHPUT  # (means of rates, sums of counts)
    assert channel["points"][0] == [3.0, 10.0, 0.0, 15.0, 0.0] and b["inference"] == []
