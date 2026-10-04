"""What a dataset of the team's play can keep of each turn (`rollout_train.datasets`): `worked`, the turns whose action
came back ok.

An agent's turn is one sample; what came of the action it called is in the agent's next observation
(`last_action`: the action, whether it was ok, and the error if not). A model sample names neither its agent nor its
slot, so a turn is matched to its outcome through the run's events: the sample a turn's span came from is requested
for one agent, and the next `observe` of that agent completes with the outcome. Model slots `agent-1` to `agent-4`
play under the names in the episode's result (`team`), in slot order.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

from pydantic import JsonValue

from rollout.contracts import RunEvent, RunEventType

LEFT_OUT = "left_out"
"""What a turn filter says of a turn that is no example (`rollout_train.datasets.LEFT_OUT`)."""


class Turn(Protocol):
    """A turn as `rollout_train.datasets.Turn` gives it: its slot and the samples its spans came from."""

    @property
    def slot(self) -> str: ...

    @property
    def samples(self) -> tuple[str, ...]: ...


def worked(
    turns: Sequence[Turn], events: Sequence[RunEvent], info: Mapping[str, JsonValue]
) -> list[dict[str, JsonValue]]:
    """The turns whose action came back ok in the agent's next observation, each with the action's name. A turn whose
    action failed is left out, and so is one whose outcome was never seen (an agent's summary of its old turns, a
    sample that called no action, the last turn)."""
    team = _listed(info.get("team"))

    def agent(slot: str) -> str:
        number = slot.removeprefix("agent-")
        named = slot.startswith("agent-") and number.isdigit() and 0 < int(number) <= len(team)
        return str(team[int(number) - 1]) if named else slot

    turn_of = {sample: position for position, turn in enumerate(turns) for sample in turn.samples}
    asked: dict[str, dict[str, Any]] = {}
    waiting: dict[str, int] = {}
    """The turn each agent's next observation reports on, by agent."""
    outcome: dict[int, dict[str, Any]] = {}
    for event in events:
        payload = _mapping(event.payload)
        effect = str(payload.get("effect_id", ""))
        if event.type is RunEventType.EFFECT_REQUESTED:
            asked[effect] = payload
            if payload.get("kind") == "model.sample" and effect in turn_of:
                waiting[agent(turns[turn_of[effect]].slot)] = turn_of[effect]
        elif event.type is RunEventType.EFFECT_COMPLETED:
            request = asked.get(effect, {})
            call = _mapping(request.get("payload")) if request.get("kind") == "tool.call" else {}
            if call.get("tool") != "observe":
                continue
            who = str(_mapping(call.get("arguments")).get("agent"))
            last = _mapping(_mapping(_mapping(payload.get("payload")).get("structured")).get("last_action"))
            if who in waiting and _mapping(last.get("action")):
                outcome[waiting.pop(who)] = last
    verdicts: list[dict[str, JsonValue]] = []
    for position in range(len(turns)):
        last = outcome.get(position, {})
        action = str(_mapping(last.get("action")).get("name"))
        if last.get("ok") is True:
            verdicts.append({"action": action})
        elif last.get("ok") is False:
            verdicts.append({LEFT_OUT: "action failed", "action": action})
        else:
            verdicts.append({LEFT_OUT: "no action seen"})
    return verdicts


def _mapping(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _listed(value: Any) -> list[Any]:
    return cast(list[Any], value) if isinstance(value, list) else []
