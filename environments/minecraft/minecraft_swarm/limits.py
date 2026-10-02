"""The limits that the prompts state and the harness keeps: `limits.json`, which the Node harness reads too
(harness/lib/limits.js), so that what agents are told of an action is what the harness does."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Limits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reach_blocks: float
    """How far from its eyes an agent mines, places and uses."""
    move_blocks: int
    """The most blocks one `move` walks."""
    walk_dig_seconds: int
    """Walking digs through a block only if the agent's tools break it within this."""
    scaffolding: tuple[str, ...]
    """The blocks walking bridges and pillars with."""
    wait_seconds: int
    """How long `wait` waits."""
    smelt_seconds: int
    """What a furnace takes for one item."""
    fuels: tuple[str, ...]
    """The fuels agents are told of, as they are named to them."""
    window_seconds: int
    """The most game time a turn's window runs: an action that takes longer is cut off there."""
    chat_characters: int
    """The length a chat message is cut to."""


LIMITS = Limits.model_validate_json((Path(__file__).resolve().parents[1] / "limits.json").read_text())

TICKS_PER_SECOND = 20
"""The game's pace."""
