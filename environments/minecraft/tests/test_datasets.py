"""The turn filter of the team's play: a turn is kept when the action it called came back ok in the agent's next
observation."""

from datetime import UTC, datetime
from typing import Any

from minecraft_team.datasets import LEFT_OUT, worked
from rollout.contracts import RunEvent, RunEventType
from rollout_train.datasets import Turn


def events(*said: tuple[RunEventType, dict[str, Any]]) -> list[RunEvent]:
    at = datetime(2026, 10, 4, tzinfo=UTC)
    return [RunEvent(run_id="r", seq=seq, type=kind, recorded_at=at, payload=payload) for seq, (kind, payload) in
            enumerate(said)]  # fmt: skip


def sample(effect: str) -> tuple[RunEventType, dict[str, Any]]:
    return RunEventType.EFFECT_REQUESTED, {"effect_id": effect, "kind": "model.sample", "payload": {}}


def observed(effect: str, agent: str, last: dict[str, Any] | None) -> list[tuple[RunEventType, dict[str, Any]]]:
    call = {"tool": "observe", "arguments": {"episode": "e", "agent": agent}}
    structured = {"last_action": last} if last is not None else {}
    return [
        (RunEventType.EFFECT_REQUESTED, {"effect_id": effect, "kind": "tool.call", "payload": call}),
        (RunEventType.EFFECT_COMPLETED, {"effect_id": effect, "payload": {"structured": structured}}),
    ]


def turn(slot: str, index: int, effect: str) -> Turn:
    return Turn(slot, index, (effect,), 100, 10, 3)


def test_a_turn_is_kept_when_its_action_worked_and_left_out_when_it_failed_or_was_never_seen() -> None:
    turns = [turn("agent-1", 0, "s1"), turn("agent-2", 0, "s2"), turn("agent-1", 1, "s3"), turn("agent-1", 2, "s4")]
    played = events(
        *observed("o0", "ada", None),  # (the first observation reports on no turn)
        sample("s1"),
        sample("s2"),
        *observed("o1", "ada", {"action": {"name": "mine"}, "ok": True}),
        *observed("o2", "bo", {"action": {"name": "move"}, "ok": False, "error": "a wall"}),
        sample("s3"),  # (a summary: then the turn that acts)
        sample("s4"),
        *observed("o3", "ada", {"action": {"name": "craft"}, "ok": True}),
    )
    said = worked(turns, played, {"team": ["ada", "bo"]})
    assert said == [{"action": "mine"}, {LEFT_OUT: "action failed", "action": "move"}, {LEFT_OUT: "no action seen"},
                    {"action": "craft"}]  # fmt: skip


def test_slots_named_by_their_agents_are_matched_by_name() -> None:
    played = events(sample("s1"), *observed("o1", "ada", {"action": {"name": "chat"}, "ok": True}))
    assert worked([turn("ada", 0, "s1")], played, {}) == [{"action": "chat"}]
    assert worked([turn("ada", 0, "s1")], events(sample("s1")), {}) == [{LEFT_OUT: "no action seen"}]
