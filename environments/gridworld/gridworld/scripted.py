"""A scripted team: a policy that reads each observation as an agent does and solves every row.

Its members share one convention instead of a conversation: they rank themselves by the alphabetical order of the
team's names. While a door with plates is closed, the first ranks stand on its plates; while a lever behind a gate is
not pulled, the first holds the gate's plate and the second pulls the lever; then each goes to the final plate of its
rank (the first only once the second is back out of the gate). Each announces its job in chat the first time.
Everything it decides comes from the latest observation, so it shows that an observation alone says enough to solve a
level. The tests play it through the real program; `ScriptedTeam` serves it as a model endpoint.
"""

import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from gridworld.game import DIRECTIONS
from gridworld.level import Point
from rollout.contracts import (
    CapabilityContract,
    FinishReason,
    Message,
    Role,
    SampleRequest,
    SampleResult,
    ToolCall,
    Usage,
)

__all__ = ["ScriptedTeam", "decide", "reply"]

THING = re.compile(r"^- (plate (\d+)|(door|gate) (\w)|lever): (here|[\dNSEW ]+)\. (.*)$")
TEAMMATE = re.compile(r"^- (\w+): (here|[\dNSEW ]+)$")
ME = re.compile(r"You are (\w+)\.")
SAID = re.compile(r"^- you, \d+ turns? ago: (.*)$")
CLOSED, WALL = "+", "#"


@dataclass
class Seen:
    """An observation, read."""

    me: str
    at: Point
    grid: list[list[str]]
    team: dict[str, Point] = field(default_factory=dict[str, Point])
    plates: dict[int, tuple[Point, str]] = field(default_factory=dict[int, tuple[Point, str]])
    """By number: where, and what the observation says of it."""
    doors: dict[str, tuple[Point, str, bool]] = field(default_factory=dict[str, tuple[Point, str, bool]])
    """By label: where, `door` or `gate`, and whether it is open."""
    lever: tuple[Point, bool] | None = None
    """Where, and whether it is pulled."""
    said: list[str] = field(default_factory=list[str])

    def blocked(self, at: Point) -> bool:
        x, y = at
        return not (0 <= y < len(self.grid) and 0 <= x < len(self.grid[y])) or self.grid[y][x] in (WALL, CLOSED)


def _offset(origin: Point, text: str) -> Point:
    x, y = origin
    for part in text.split():
        if part == "here":
            continue
        count, way = int(part[:-1]), part[-1]
        x += count if way == "E" else -count if way == "W" else 0
        y += count if way == "S" else -count if way == "N" else 0
    return (x, y)


def read(observation: str) -> Seen:
    lines = observation.splitlines()
    found = ME.search(lines[0])
    assert found is not None, lines[0]
    start = lines.index("Map (north is up):") + 1
    end = next(index for index in range(start, len(lines)) if lines[index].startswith("@ you"))
    grid = [line.split(" ") for line in lines[start:end]]
    at = next((x, y) for y, row in enumerate(grid) for x, glyph in enumerate(row) if glyph == "@")
    seen = Seen(found.group(1), at, grid)
    for line in lines[end:]:
        if (thing := THING.match(line)) is not None:
            where, said = _offset(at, thing.group(5)), thing.group(6)
            if thing.group(2):
                seen.plates[int(thing.group(2))] = (where, said)
            elif thing.group(3):
                seen.doors[thing.group(4)] = (where, thing.group(3), said.endswith(" Open."))
            else:
                seen.lever = (where, said.endswith(" Pulled."))
        elif (mate := TEAMMATE.match(line)) is not None:
            seen.team[mate.group(1)] = _offset(at, mate.group(2))
        elif (mine := SAID.match(line)) is not None:
            seen.said.append(mine.group(1))
    return seen


@dataclass(frozen=True)
class Job:
    target: Point | None
    """Where to go; None: stay put."""
    say: str | None = None
    through: Point | None = None
    """A gate the way may pass, waiting before it while it is closed."""


def jobs(seen: Seen) -> dict[str, Job]:
    """What every member of the team does now, by the convention they share."""
    ranks = sorted([seen.me, *seen.team])
    idle = {name: Job(None) for name in ranks}
    doors_with_plates = sorted(
        label for label, (_, kind, is_open) in seen.doors.items() if kind == "door" and not is_open
    )
    for label in doors_with_plates:
        pressing = [where for _, (where, said) in sorted(seen.plates.items()) if f"opens door {label} " in said]
        if pressing and all(_path(seen, seen.at, where, set()) is not None for where in pressing):
            standing = {
                name: Job(where, f"I'll stand on plate {_number(seen, where)}.")
                for name, where in zip(ranks, pressing, strict=False)
            }
            return {**idle, **standing}
    final = [where for _, (where, said) in sorted(seen.plates.items()) if said.startswith("A final plate")]
    finishing = {
        name: Job(where, f"I'll take final plate {_number(seen, where)}.")
        for name, where in zip(ranks, final, strict=True)
    }
    gate = next(((label, where) for label, (where, kind, _) in seen.doors.items() if kind == "gate"), None)
    holding = next((where for where, said in seen.plates.values() if said.startswith("Holds gate")), None)
    if seen.lever is None or gate is None or holding is None:
        return finishing
    lever, pulled = seen.lever
    holder, puller = ranks[0], ranks[1]
    hold = Job(holding, f"I'll hold gate {gate[0]} open.")
    if not pulled:
        return {**idle, holder: hold, puller: Job(lever, "I'll pull the lever.", through=gate[1])}
    behind = _flood(seen, lever, lambda at: at != gate[1])
    where = seen.at if seen.me == puller else seen.team[puller]
    if where in behind or where == gate[1]:  # (the gate is held until the puller is back out)
        return {**finishing, holder: hold}
    return finishing


def decide(seen: Seen) -> tuple[str, str | None, str | None]:
    """The tool to call, the direction (for `move`) and anything to say. A member keeps clear of teammates that stay
    put, of moving teammates ranked before it and of the ways they take, and steps aside when it stands where one of
    them steps next; it goes straight through teammates ranked after it, who keep clear of it."""
    everyone = jobs(seen)
    mine = everyone[seen.me]
    say = _fresh(seen, mine.say)
    positions = {seen.me: seen.at, **seen.team}
    ranks = sorted(positions)
    still = {
        positions[name]
        for name, job in everyone.items()
        if name != seen.me and (job.target is None or job.target == positions[name])
    }
    ahead = {positions[name] for name in ranks[: ranks.index(seen.me)]} - still
    if mine.target is None or mine.target == seen.at:
        return ("wait", None, say)
    routes = [
        route
        for name in ranks[: ranks.index(seen.me)]
        if positions[name] not in still
        and (target := everyone[name].target) is not None
        and (route := _path(seen, positions[name], target, still, everyone[name].through))
    ]
    claimed = {at for route in routes for at in route}
    path = _path(seen, seen.at, mine.target, still | ahead | claimed, mine.through)
    path = path or _path(seen, seen.at, mine.target, still | ahead, mine.through)
    if any(route[0] == seen.at for route in routes):  # (in the way of someone ranked before: make room)
        return _aside(seen, path, positions, routes, say, mine.target)
    if not path or path[0] in still | ahead or (path[0] == mine.through and seen.blocked(path[0])):
        return ("wait", None, say)
    return _move(seen, path[0], say)


def _aside(
    seen: Seen,
    path: list[Point] | None,
    positions: dict[str, Point],
    routes: list[list[Point]],
    say: str | None,
    target: Point,
) -> tuple[str, str | None, str | None]:
    """Out of the way of the routes of teammates ranked before: on along its own way if that is off theirs, else to a
    side off them (never into a doorway), else on along its own way if that is not where one of them steps next, else
    to any side that is not."""
    claimed = {at for route in routes for at in route}
    nexts = {route[0] for route in routes}
    occupied = set(positions.values())
    doorways = {where for where, _, _ in seen.doors.values()}
    ahead = path[0] if path and not seen.blocked(path[0]) and path[0] not in occupied else None
    sides = [
        (seen.at[0] + dx, seen.at[1] + dy)
        for dx, dy in DIRECTIONS.values()
        if not seen.blocked((seen.at[0] + dx, seen.at[1] + dy))
        and (seen.at[0] + dx, seen.at[1] + dy) not in occupied | doorways | nexts
    ]
    sides.sort(key=lambda at: abs(at[0] - target[0]) + abs(at[1] - target[1]))
    choices = (
        [ahead] if ahead is not None and ahead not in claimed else [],
        [at for at in sides if at not in claimed],
        [ahead] if ahead is not None and ahead not in nexts else [],
        sides,
    )
    for choice in choices:
        if choice:
            return _move(seen, choice[0], say)
    return ("wait", None, say)


def _move(seen: Seen, to: Point, say: str | None) -> tuple[str, str | None, str | None]:
    step = (to[0] - seen.at[0], to[1] - seen.at[1])
    return ("move", next(name for name, delta in DIRECTIONS.items() if delta == step), say)


def _number(seen: Seen, where: Point) -> int:
    return next(number for number, (at, _) in seen.plates.items() if at == where)


def _fresh(seen: Seen, say: str | None) -> str | None:
    return say if say is not None and say not in seen.said else None


def _path(
    seen: Seen, start: Point, target: Point, avoid: set[Point], through: Point | None = None
) -> list[Point] | None:
    """The squares from `start` (not included) to `target`, the shortest way that avoids `avoid` and closed doors (but
    `through`)."""
    came: dict[Point, Point | None] = {start: None}
    queue = deque([start])
    while queue:
        at = queue.popleft()
        if at == target:
            path: list[Point] = []
            while at != start:
                path.append(at)
                previous = came[at]
                assert previous is not None
                at = previous
            return path[::-1]
        for dx, dy in DIRECTIONS.values():
            step = (at[0] + dx, at[1] + dy)
            if step in came or step in avoid:
                continue
            if seen.blocked(step) and not (step == through and seen.grid[step[1]][step[0]] == CLOSED):
                continue
            came[step] = at
            queue.append(step)
    return None


def _flood(seen: Seen, start: Point, passable: Callable[[Point], bool]) -> set[Point]:
    found = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS.values():
            step = (x + dx, y + dy)
            if step not in found and passable(step) and seen.grid[step[1]][step[0]] != WALL:
                found.add(step)
                queue.append(step)
    return found


def reply(observation: str) -> Message:
    """The scripted team's reply to an observation: one tool call."""
    tool, direction, say = decide(read(observation))
    arguments: dict[str, str] = {"direction": direction} if direction else {}
    if say:
        arguments["say"] = say
    return Message(role=Role.ASSISTANT, content=[ToolCall(call_id="scripted", name=tool, arguments=arguments)])


class ScriptedTeam:
    """A model endpoint that answers every slot with `reply` to the request's last message."""

    def describe(self, session_id: str) -> CapabilityContract:
        return CapabilityContract(context_limit=32_768, max_output_tokens=1_000)

    async def sample(self, request: SampleRequest) -> SampleResult:
        message = reply(request.context.append[-1].text)
        return SampleResult(
            message=message,
            finish_reason=FinishReason.TOOL_USE,
            usage=Usage(context_used=1, context_limit=32_768),
        )

    async def cancel(self, effect_id: str) -> None:
        pass
