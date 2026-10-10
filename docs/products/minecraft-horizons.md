# Minecraft horizons

Code: `environments/minecraft-horizons`

**Read first:** [Minecraft team](minecraft-team.md). **Next:** [Gridworld](gridworld.md).

One to four agents share a Minecraft world and race a budget of game time for as much of an objective as they can get:
iron, food, advancements. Nothing caps the count, so a team can always do better; the agents see the clock, so what
pays depends on how much time there is. With five minutes the way to the most iron is the nearest ore. With two hours
it may be better tools first, an enchanting table for Fortune, or an iron farm. Every objective comes at every budget
of a ladder that doubles from 5 to 160 minutes, so what a policy gets as its budget grows is a curve, and a strategy
that pays only given time shows where the curve bends.

The environment is the package `minecraft-horizons` (import `minecraft_horizons`). It depends on `rollout` and on
`minecraft-team`, whose Paper servers, bots, ground-truth plugin, observations, tools and starts' builders it uses as
they are. Its tasks, prompts, episode, scoring and sandbox kind are its own.

```bash
uv run rollout train minecraft_horizons.environment:environment --preset minecraft-one-gpu --name horizons-1
```

The cluster config needs a pool of its worlds, a sandbox kind of its own (`[sandboxes.minecraft-horizons]`, `provider =
"minecraft_horizons.worlds:worlds"`), and the environment among its `[environments]`.

## Objectives

An objective is an amount the team ends the game with, measured from the plugin's ground truth
(`minecraft_horizons.objectives`):

| Objective | What counts |
|---|---|
| Wood | Logs, wood and stems of any tree, one each (stripped too); planks a quarter each |
| Food | Food by the hunger it restores, as the game has it (bread 5, a cooked steak 8, a raw one 3): cooking pays. Food that hurts whoever eats it does not count |
| Coal | Coal and coal ore one each, a block nine (not charcoal) |
| Iron | Raw iron, ingots and ore one each, a block of iron or of raw iron nine, a nugget a ninth; not iron made into tools or armor |
| Gold | As iron |
| Diamonds | Diamonds and diamond ore one each, a block nine; not diamonds made into tools or armor |
| Advancements | Every advancement the team earns, once, whoever earns it (not recipes) |

An item counts when the team holds it at the end: in a member's inventory (with armor, offhand, cursor and crafting
grid; a member who left, what they held then) or in a container a member placed during the episode. A chest keeps
what a member who dies would drop, so stashing is a strategy. What the team held once the world was set up (its kit)
does not count: the amount is what it holds at the end beyond that, never below nothing. Ore blocks count, so mining
with Silk Touch to smelt or Fortune-mine later is not lost.

The reward is `log(1 + amount)`, the same for every agent of the team. Each doubling is worth as much at any scale,
so a team that ends with ten times its group's iron is clearly ahead, and one group's large numbers do not swamp every
other group's in an update. Results report the amount itself.

## Tasks

A task is an objective, a setting and a budget (`minecraft_horizons.tasks`):

| Setting | Where the team starts | Objectives |
|---|---|---|
| `fresh` | The game as it begins: on the surface beside trees, with nothing; easy mobs at night and in the dark; what you carry drops where you die | All |
| `underground` | In a natural cave with a stone pickaxe and sword, coal, a furnace, sticks and a table, food and torches | Coal, iron, gold, diamonds |

Each objective comes from each of its settings at each budget of `LADDER` (5, 10, 20, 40, 80 and 160 minutes of game
time): 66 tasks, with ids like `iron-underground-40m`. Settings are laid out by the team package's builders
(`minecraft_team.tasks.build`, as natural survival worlds: a real day and night and weather). As for the team's tasks,
turns are capped at 12 a minute of the budget. Game time passes only while actions happen, so a team can let it pass
(`wait`) while a furnace or a farm works, at a turn's cost.

The rows are the tasks, shortest budgets first. A start is a world seed of 12, a layout and the names of a team of one
to four, drawn at random; episodes given the same start begin identically, so a group compares teams in the same
world with the same time. The eval data (`horizons-held-out`) is one start of every task in 4 worlds training never
draws. A result says `solved` when the team ended with at least one of its objective: what unlocks rows in the
curriculum.

## What agents read

The system prompt is the team package's (how the game runs, what an agent knows, chat and teamwork) with a goal of its
own: what counts, what each item is worth, that what the team began with does not count, how long the game lasts in
game time and in turns, that every observation shows how much is left, and that nothing caps the count. It advises
planning for the time there is: with little, use what is at hand; with more, what takes time to set up can pay off.

Every observation, and every turn an agent remembers, begins with the clock (`prompts.clock`):

```text
Time left: 37.4 of 60 minutes of game time; 412 of 720 turns.
```

The rest of the observation, the tools and memory are the team package's.

## The episode

`HorizonEpisode` plays as the team's episode does: each turn every agent observes, thinks and calls one action tool
while the world is frozen, then the world runs one window while the actions happen. Nothing ends the game early: it
lasts until its game time or its turns are spent. Then the world is scored and the reward goes to every playing agent.

The result reports the amount, its unit and breakdown (`amount_parts`), the amount a minute of game time, what the
team holds (`held`, `stored` and how many containers it placed), what it held at the start, the advancements it
earned, what it got hold of, blocks mined and events, and the game time and turns spent (`duration` is turns).

## The worlds

`HorizonWorlds` (`minecraft_horizons.worlds`) is a [sandbox](../libraries/rollout/sandboxes.md) provider of the kind
`minecraft-horizons`. For each lease it starts a Paper server from a template, connects the team's bots, lays out the
setting, and notes what the team holds. Its operations are `observe`, `act` and `window`, as the team's worlds have
them, and `score`: the objective's amount, its reward, and the ground truth it is measured from (the plugin's
`/holdings` and `/state`).

## Tests

`environments/minecraft-horizons/tests`:

- `test_objectives.py`: what each objective counts, that what the team began with does not, and the reward;
- `test_tasks.py`: the catalog and its order, the settings as the builders lay them out, starts in training worlds
  and evals in held-out ones, the goal's wording and the clock;
- `test_episode.py`: an episode on a made-up world lasts its whole budget, shows the clock in every observation, and
  rewards every agent with `log(1 + amount)`;
- `test_worlds.py` (live: Java and Node): a setting laid out on a real server; iron picked up counts, and so does iron
  stored in a chest a bot placed, but not the kit or a chest the team did not place.
