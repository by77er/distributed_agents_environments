"""The tasks: each objective, from each setting it suits, against each budget of game time on the ladder.

A task is an objective (`minecraft_horizons.objectives`), a setting (where the team starts, with what, among what
dangers) and a budget of game time (`LADDER`; a speedrun's segments `SEGMENT_LADDER`, the whole game `GAME_LADDER`).
A speedrun comes whole, from nothing to the dragon, and in segments that start partway along the game with what that
stage takes (a portal kit, beside a fortress, near a stronghold with eyes of ender, on the end's platform).

The same objective from the same setting comes at every budget, so a policy meets it with little time and with a lot,
and is told which (`prompts.clock`): with five minutes the way to the most iron is the nearest ore; with two hours it
may be better tools first, or an iron farm. The ladder doubles, so what a team gets as its budget grows is a curve,
and a strategy that only pays given time shows where it bends.

A task's turns are capped as the team's tasks are (`TURNS_PER_MINUTE` a minute of its budget): turns are what cost
(each is a round of thinking), and game time passes only while actions happen, so a team can also let time pass
(`wait`) while a furnace or a farm works, at a turn's cost.

Settings reuse the team package's starts and kits (`minecraft_team.tasks`), whose builders lay them out in a live world.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from minecraft_horizons.objectives import OBJECTIVES, Measure, Objective
from minecraft_team.tasks import KILL, TURNS_PER_MINUTE, Coordination, Hazards, Kit, Start, Tier
from minecraft_team.tasks import Objective as TeamObjective
from minecraft_team.tasks import Task as TeamTask

__all__ = ["GAME_LADDER", "LADDER", "SEGMENT_LADDER", "SETTINGS", "TASKS", "Setting", "Task", "catalog", "ladder"]

LADDER = (5, 10, 20, 40, 80, 160)
"""Budgets of game time, in minutes: each twice the one before."""
SEGMENT_LADDER = (5, 10, 20, 40, 80)
"""Budgets of a segment of a speedrun, which starts partway along the game with what that stage takes."""
GAME_LADDER = (80, 160, 320, 640)
"""Budgets of the whole game, from nothing to the dragon."""


@dataclass(frozen=True)
class Setting:
    id: str
    title: str
    start: Start
    kit: Kit
    hazards: Hazards
    coordination: Coordination = Coordination.KITTED

    @property
    def keeps_inventory(self) -> bool:
        """Whether whoever dies keeps what they carried: only where nothing is out to hurt the team."""
        return self.hazards is Hazards.SAFE


FRESH = Setting("fresh", "a fresh start among trees", Start.WOODLAND, Kit.NOTHING, Hazards.EASY)
"""The game as it begins: nothing at all, on the surface beside trees, easy mobs at night and in the dark."""
UNDERGROUND = Setting("underground", "in a cave with stone tools", Start.CAVE, Kit.STONE, Hazards.EASY)
"""Underground in a natural cave with a stone pickaxe and sword, coal, a furnace, sticks and a table (and food and
torches): the tech tree's first steps done, the ores close."""

PORTAL_KIT = Setting("portal-kit", "on the surface with a portal kit", Start.SURFACE, Kit.NETHER_READY, Hazards.EASY)
"""Iron armor and tools, 14 obsidian and flint and steel: build and light a portal."""
FORTRESS = Setting("fortress", "beside a nether fortress, armed", Start.FORTRESS, Kit.FORTRESS_READY, Hazards.EASY)
"""Iron armor, sword, bow and arrows, food and blocks, in the nether beside a fortress: fight blazes."""
STRONGHOLD_AREA = Setting(
    "stronghold-area", "near a stronghold with eyes of ender", Start.STRONGHOLD_AREA, Kit.EYES_READY, Hazards.EASY
)
"""Iron armor and tools and twelve eyes of ender, a few hundred blocks from a stronghold."""
PORTAL_ROOM = Setting(
    "portal-room", "beside the end portal with eyes of ender", Start.PORTAL_ROOM, Kit.EYES_READY, Hazards.EASY
)
"""Inside a stronghold beside the end portal's frame, with eyes of ender to fill it."""
END = Setting("end", "on the end's platform, armed", Start.END, Kit.END_READY, Hazards.EASY)
"""Diamond armor and sword, a bow, arrows, blocks, food and a water bucket, the dragon alive."""

SETTINGS: dict[str, Sequence[Setting]] = {
    "wood": (FRESH,),
    "food": (FRESH,),
    "coal": (FRESH, UNDERGROUND),
    "iron": (FRESH, UNDERGROUND),
    "gold": (FRESH, UNDERGROUND),
    "diamonds": (FRESH, UNDERGROUND),
    "huts": (FRESH,),
    "advancements": (FRESH,),
    "iron-tools": (FRESH,),
    "nether": (FRESH, PORTAL_KIT),
    "blaze-rods": (FORTRESS,),
    "stronghold": (STRONGHOLD_AREA,),
    "end": (PORTAL_ROOM,),
    "dragon": (END, FRESH),
}
"""The settings each objective comes from."""


def ladder(objective: Objective, setting: Setting) -> tuple[int, ...]:
    """The budgets an objective comes at from a setting: the whole game's for the dragon from nothing, a segment's for
    a speedrun's other stages, else `LADDER`."""
    if objective.measure is Measure.PROGRESS and objective.goal == KILL and setting.kit is Kit.NOTHING:
        return GAME_LADDER
    if objective.measure is Measure.PROGRESS and objective.id != "iron-tools":
        return SEGMENT_LADDER
    return LADDER


@dataclass(frozen=True)
class Task:
    objective: Objective
    setting: Setting
    minutes: int
    """The budget of game time (it passes only while actions happen)."""

    @property
    def id(self) -> str:
        return f"{self.objective.id}-{self.setting.id}-{self.minutes}m"

    @property
    def title(self) -> str:
        return f"{self.objective.title}, {self.setting.title}, {self.minutes} minutes"

    @property
    def turns(self) -> int:
        """The budget of turns."""
        return round(self.minutes * TURNS_PER_MINUTE)

    def laid_out(self) -> TeamTask:
        """The setting as a task of the team package, which its builders lay out (`minecraft_team.tasks.build`):
        only its start, kit, coordination, hazards and tier are read. A speedrun's is a progress task toward its goal,
        whose path of milestones (`minecraft_team.tasks.path_of`) and progress along it (`scored`) are the team
        package's."""
        speedrun = self.objective.measure is Measure.PROGRESS
        return TeamTask(
            id=self.id,
            title=self.title,
            tier=Tier.SURVIVAL,  # (natural: a real day and night, weather)
            objective=TeamObjective.PROGRESS if speedrun else TeamObjective.CRAFT,
            goal=self.objective.goal if speedrun else None,
            start=self.setting.start,
            kit=self.setting.kit,
            coordination=self.setting.coordination,
            hazards=self.setting.hazards,
            minutes=self.minutes,
            guided=False,
            difficulty=0.0,
        )


def catalog() -> list[Task]:
    """Every task, shortest budgets first and, within a budget, the objectives in the order of `OBJECTIVES`: the
    order rows unlock in."""
    tasks = [
        Task(objective, setting, minutes)
        for objective in OBJECTIVES.values()
        for setting in SETTINGS[objective.id]
        for minutes in ladder(objective, setting)
    ]
    order = list(OBJECTIVES)
    return sorted(tasks, key=lambda task: (task.minutes, order.index(task.objective.id)))


TASKS = {task.id: task for task in catalog()}
"""Every task, by its id."""
