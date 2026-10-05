"""The reward: half for solving the level, half for progress along its stages.

A level's stages are its doors that open for good, in the order they open (a door with plates, a door a lever opens),
then its final plates. Progress is the share of the stages done, with the final plates counted as the share of them
pressed at once at the end of a turn (`Game.most_pressed`); a solved level has made all its progress. Every agent of
the team gets the same reward:

    reward = 0.5 * solved + 0.5 * progress

So any solved episode scores more than any unsolved one, and an unsolved one scores less than 0.5. Only what lasts or
needs the team together counts: a door opens once, and the final plates count only as many as are pressed at the same
time, so stepping onto a plate and off again, or one agent touring every plate, earns no more than standing on one.
A team that never moves, or wanders without opening a door or pressing a final plate, scores 0. A gate held open is
no stage of its own: what it is for, the lever behind it, is. How fast the team was does not count (the platform's
`advantage.tiebreak` can prefer the shorter of episodes that all solved: their results say `saturated`).
"""

from collections.abc import Mapping
from dataclasses import dataclass

from gridworld.game import Game
from gridworld.level import DoorKind, Level

__all__ = ["FINAL", "SOLVED_SHARE", "Scored", "progress_of", "scored", "stages"]

SOLVED_SHARE = 0.5
"""What solving is worth; progress is worth the rest."""
FINAL = "final plates"
"""The last stage's name."""


@dataclass(frozen=True)
class Scored:
    reward: float
    solved: bool
    progress: float
    """From 0 to 1: the share of the level's stages done."""
    parts: Mapping[str, float]
    """What each part of the reward earned: `solved`, then each stage's share of the progress (`stages`)."""


def stages(level: Level) -> list[str]:
    """A level's stages, in order: `door a` for each door that opens for good, then the final plates (`FINAL`)."""
    return [f"door {door.label}" for door in level.doors if door.kind is not DoorKind.HELD] + [FINAL]


def scored(game: Game) -> Scored:
    """The reward of a game as it stands, and its parts."""
    level = game.level
    done = {f"door {door.label}": float(game.solved or door.label in game.opened)
            for door in level.doors if door.kind is not DoorKind.HELD}  # fmt: skip
    done[FINAL] = 1.0 if game.solved else game.most_pressed / len(level.final)
    progress = sum(done.values()) / len(done)
    parts = {"solved": SOLVED_SHARE if game.solved else 0.0}
    parts |= {name: round((1 - SOLVED_SHARE) * value / len(done), 4) for name, value in done.items()}
    reward = (SOLVED_SHARE if game.solved else 0.0) + (1 - SOLVED_SHARE) * progress
    return Scored(round(reward, 6), game.solved, round(progress, 6), parts)


def progress_of(reward: float, solved: bool) -> float:
    """An episode's progress, from its reward and whether it solved its level (what a group's result keeps)."""
    return 1.0 if solved else min(max(reward / (1 - SOLVED_SHARE), 0.0), 1.0)
