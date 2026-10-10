"""Building: copies of a blueprint on a grid of sites near the team, every block in its place counted.

A building task lays out `SITES` sites near where the team starts, each a flat square of smooth stone with the space
above it cleared (`prepare`), and asks for as many blocks of the blueprint (`HUT`) in their places as the team can
build on them: a whole hut is 71 blocks, and there are sites for sixteen. A block counts if it is one of the building
materials (`counts`: any planks, logs or wood, cobblestone, stone, bricks…), whoever placed it; what stood there before
the site was prepared was cleared. So the amount grows with every block, a finished hut is worth no more than its
blocks, and the materials (wood to cut, stone to mine) are the team's to gather: with little time, a few walls; with
more, tools first, and huts.

The sites and the blueprint reach agents in the system prompt (`describe`), with the coordinates of every site.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from minecraft_team.control import Control

__all__ = ["HUT", "SITES", "Site", "count", "counts", "describe", "positions", "prepare"]

SIDE = 5
"""A site's square, each way."""
WALLS = 3
"""Layers of wall, above the floor; the roof is the next."""
SPACING = 8
"""From one site's corner to the next's: a site and three blocks between."""
GRID = 4
"""Sites each way: `GRID * GRID` of them."""
SITES = GRID * GRID
FLOOR = "smooth_stone"
"""What a site's floor is made of: not a building material, so it never counts."""


@dataclass(frozen=True)
class Site:
    x: int
    y: int
    """The level a hut's first layer stands on (the floor is one below)."""
    z: int
    """The site's north-west corner (least x and z)."""


def _hut() -> list[tuple[int, int, int]]:
    """The hut's blocks, relative to its site's corner: walls three high around the square's edge but for a doorway one
    wide and two high in the middle of the north side, and a flat roof over the whole square."""
    blocks: list[tuple[int, int, int]] = []
    door = {(SIDE // 2, 0, 0), (SIDE // 2, 1, 0)}
    for dy in range(WALLS):
        for dx in range(SIDE):
            for dz in range(SIDE):
                edge = dx in (0, SIDE - 1) or dz in (0, SIDE - 1)
                if edge and (dx, dy, dz) not in door:
                    blocks.append((dx, dy, dz))
    blocks.extend((dx, WALLS, dz) for dx in range(SIDE) for dz in range(SIDE))
    return blocks


HUT = _hut()
"""The blueprint: 46 blocks of wall and 25 of roof."""

MATERIALS = {
    "cobblestone", "mossy_cobblestone", "stone", "stone_bricks", "mossy_stone_bricks", "cracked_stone_bricks",
    "bricks", "cobbled_deepslate", "deepslate", "deepslate_bricks", "polished_deepslate", "andesite",
    "polished_andesite", "diorite", "polished_diorite", "granite", "polished_granite", "sandstone", "smooth_sandstone",
    "mud_bricks", "tuff", "tuff_bricks", "polished_tuff", "blackstone", "polished_blackstone",
    "polished_blackstone_bricks", "bamboo_block", "bamboo_planks",
}  # fmt: skip
"""Building materials besides any planks, logs, wood, stems and hyphae (`counts`)."""


def counts(block: str) -> bool:
    """Whether a block counts toward a hut: a building material."""
    wood = block.endswith(("_planks", "_log", "_wood", "_stem", "_hyphae"))
    return wood or block in MATERIALS


def positions(sites: Sequence[Site]) -> list[tuple[int, int, int]]:
    """Every blueprint block of every site, absolute: site by site, in the blueprint's order."""
    return [(site.x + dx, site.y + dy, site.z + dz) for site in sites for dx, dy, dz in HUT]


async def prepare(control: Control, anchor: tuple[int, int, int], world: str = "world") -> list[Site]:
    """Lay out the sites near `anchor` (where the team stands): a grid of `GRID` by `GRID` from a few blocks south-east
    of it (clear of where the team stands), each flattened at the ground's height at its middle, its square floored
    with smooth stone and the space above cleared for the hut and a block more."""
    ax, _, az = anchor
    sites: list[Site] = []
    for row in range(GRID):
        for column in range(GRID):
            x, z = ax + 6 + column * SPACING, az + 6 + row * SPACING
            y = await control.surface(x + SIDE // 2, z + SIDE // 2, world) + 1
            await control.carve(
                x, y, z, width=SIDE, height=WALLS + 3, depth=SIDE, light=False, floor=FLOOR, world=world
            )
            for dx in range(SIDE):
                for dz in range(SIDE):
                    await control.set_block(x + dx, y - 1, z + dz, FLOOR)
            sites.append(Site(x, y, z))
    return sites


async def count(control: Control, sites: Sequence[Site], world: str = "world") -> tuple[int, dict[str, float]]:
    """How many blueprint blocks are in their places (a building material at each), and how many on each site."""
    found = await control.blocks(positions(sites), world=world)
    per_site: dict[str, float] = {}
    for index, site in enumerate(sites):
        blocks = found[index * len(HUT) : (index + 1) * len(HUT)]
        placed = sum(counts(block) for block in blocks)
        if placed:
            per_site[f"site {index + 1} at ({site.x}, {site.y}, {site.z})"] = float(placed)
    return int(sum(per_site.values())), per_site


def describe(sites: Sequence[Site]) -> str:
    """The sites and the blueprint, as agents read them."""
    listed = "; ".join(f"{index} at ({site.x}, {site.y}, {site.z})" for index, site in enumerate(sites, 1))
    return (
        f"The sites: {len(sites)} squares of smooth stone, {SIDE} by {SIDE} blocks, near where you start. Each "
        f"is given by its north-west corner, the lowest x and z of its square, at the height a hut's first layer "
        f"stands on (the smooth stone floor is one below): {listed}. A hut on a site: walls {WALLS} blocks high "
        f"around the square's edge, from x to x+{SIDE - 1} and z to z+{SIDE - 1}, leaving a doorway one block "
        f"wide and two high in the middle of the north wall (at x+{SIDE // 2}, z); and a flat roof over the whole "
        f"square, on top of the walls (at y+{WALLS}). That is {len(HUT)} blocks a hut."
    )
