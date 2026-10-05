# Minecraft rewards

**Status: built.** The reward from 0 to 1, its parts in each result, ore rows ending on solving, `duration` in turns
and `advantage.tiebreak` are built. The signals listed under [what results do not record](#what-results-do-not-record)
are proposed. A design note: see [Design notes](README.md) for the others.

Code: `minecraft_team.tasks` (`path_of`, `scored`), `rollout_train.algorithm` (`tiebreak`) · See
[the Minecraft team](../products/minecraft-team.md#rewards), [training](../libraries/rollout-train/training.md),
[curricula](curricula.md)

Every Minecraft task scores an episode from 0 to 1. Solving the task is worth half. Progress along the task's path is
worth the other half: the steps toward the objective that the start and the kit leave to the team, each weighted by
how far along the path it is. How long an episode took is not part of the reward. A group-relative update compares
episodes of one start, so the measurements below are made within groups.

## The data

| Source | What it holds |
|---|---|
| `curriculum-9` (qwencraft-1) | 72 groups: 67 played to the end, 5 never built. 267 episodes: 264 matched to their recorded results, 232 with a turn-by-turn record (17,006 windows of game time, 56,017 actions whose outcome the next observation reports) |
| `curriculum-1` to `curriculum-8` | No group results. `curriculum-6` has 2 groups (8 episodes) and `curriculum-3` 4 episodes; too few to use |
| The cluster's ledger | `curriculum-9` is its only Minecraft run |

The ledger has each group's rewards, `solved`, durations and what the algorithm did; the run directory has each
episode's events, from which each turn's window, actions and their outcomes, and the team's inventory come. The
scripts are outside the repository; they read both sources read-only.

## What the run's rewards were

The reward was the objective's raw count: diamonds held, or the weights of the steps of a chain done. Rewards ran from
0 to 20, and a row's scale was the diamonds there were to find.

| Rows (objective, start) | Groups | Episodes | Distinct rewards | Groups with every reward equal | Solved | Zero rewards |
|---|---|---|---|---|---|---|
| Crafting, woodland | 9 | 36 | 4 (4, 5, 8, 10) | 4 | 42% | 0 |
| Diamonds, chests | 17 | 68 | 16 | 1 | 56% | 10 |
| Diamonds, items on the floor | 6 | 24 | 8 | 3 | 96% | 1 |
| Diamonds, ore far | 3 | 12 | 6 | 0 | 50% | 6 |
| Diamonds, ore in sight | 24 | 96 | 15 | 11 | 50% | 34 |
| Diamonds, ore nearby | 8 | 31 | 9 | 4 | 29% | 21 |

23 of the 67 played groups had every reward equal: 10 in which no one scored (all from ore rows), 2 in which every
team got equally far without solving, and 11 in which every team solved and scored the same. Dynamic sampling skipped
18 of them. The other 5 trained on the speed bonus alone.

Coarse where the team fails: an ore row with a kit scored nothing until a diamond was held, whatever the team made on
the path (an iron pickaxe crafted counted for nothing). Steep where it succeeds: in an ore row, every diamond past one
each counted as much as the first.

## The speed bonus

The algorithm gave a point to the fastest of a group's saturated episodes (least game time). On diamonds a point is a
diamond.

| Measure | Value |
|---|---|
| Share of the pooled within-group variance of the scores | 0.4% (reward 98.6%, the cross term 1.0%) |
| Share of the total size of the advantages | 3.4% |
| Groups it acted in | 10 of 49 trained groups |
| Groups it alone trained | 5 (9, 22, 24, 34, 41): every episode solved and scored the same |
| Episodes whose advantage it gave a sign or flipped | 5 |

In the groups it alone trained, the winner's margin was 9 to 150 ticks (0.45 to 7.5 seconds of game time). In 2 of the
10 groups (41 and 62) the episode with the least game time was not the one with the fewest turns: in group 41 the
point went to an episode of 5 turns and 237 ticks over one of 4 turns and 330 ticks.

What decides game time, for the same outcome:

| Measure | Value |
|---|---|
| Episodes of one group with the same outcome and the same turns (41 sets, play cut off by the turn budget) | median coefficient of variation of game time 0.15 (75th percentile 0.24), with every decision the same in number |
| Saturated episodes of one group with the same outcome (10 sets) | coefficient of variation 0.35 for game time and 0.35 for turns |
| Saturated episodes by row | t001 0.59 game time against 0.60 turns, t003 0.49 against 0.15, t004 0.46 against 0.80, t005 0.24 against 0.11, t006 1.27 against 1.08, t008 0.52 against 0.45, t008u 0.32 against 0.43, t012 0.31 against 0.32 |
| Variance of log game time within saturated sets | 50% from the number of turns, 23% from game time per turn, 27% from their covariance |

Game time per turn is what the policy does not decide:

- **Pathing and the slowest agent.** A window runs until every agent's action has ended, so the slowest path sets it.
  Windows in which every agent moved took a median of 39 ticks and a 90th percentile of 241 (coefficient of variation
  1.25); mining, 13 and 51 (0.82). 3.1% of windows were cut off at 20 seconds, three in four with a move in them.
- **Server ticks.** The bots move and dig in real time while the server counts ticks. Inside windows of more than 3
  seconds the server ran at a median 0.85 of its 20 ticks a second (10th percentile 0.80, 90th 0.96): the same action
  took fewer ticks when the server lagged.
- **Failed actions** do not explain it: 60% of actions failed (mining 79%, mostly diamond ore or deepslate without the
  tool that drops it; moving 28%), but within sets of the same outcome, game time and the share of failed actions
  were uncorrelated (r = -0.07).

A turn is a decision of every agent, and it is what costs: each is a round of thinking. Turn counts are integers, so
two episodes that took as many turns tie.

## What results record

What an episode's result records of the path to its objective, from the plugin's ground truth:

| Field | What it is | Seen in |
|---|---|---|
| `team_obtained` | every item a member picked up, crafted or took from a furnace, with counts | 252 of 264 results (all but the oldest) |
| `team_advancements` | advancements earned after the start | every result |
| `mined` | blocks the team broke, by kind | every result |
| `events` | counts of pickups, crafts, smelts, mined blocks, deaths, hurts and refused moves | every result |
| `players`, `team_diamonds`, `available_diamonds` | each member's diamonds, the team's, and the diamonds laid out or ore counted near the start | every result |

By row: with a kit of ingots, 15 of 30 teams crafted an iron pickaxe and 14 held a diamond; with raw iron, 8 of 12
smelted an ingot and 4 crafted the pickaxe; with stone tools, 6 of 8 mined raw iron and none got further; of the
crafting teams, every one made a table and 17 of 36 a wooden pickaxe. Items of the kit appear in `team_obtained` when
they are picked up again (a crafting table placed and taken back, in 22 of 30 results of the ingots kit), so a step
whose item the kit holds cannot count.

### What results do not record

Proposed, for ties that remain:

- How close the team came to ore it never found (iron-kit rows with ore out of sight: 2 groups in which no one scored
  anything also tie under this reward). The plugin knows where the ore is.
- Who holds what. Where a kit is dealt in parts, bringing them to one player is a step of its own; `team_obtained`
  sums the team.

## The reward

`tasks.scored(task, state, available, players)` gives the reward and its parts:

- **Solved**, worth `SOLVED_SHARE` (0.5): the task's own test. More than half the diamonds laid out, or one
  diamond each from ore; the task's item; the task's milestone.
- **Progress**, worth the other 0.5: the share of the weight of the task's path (`path_of`) the team got done.

| Objective | The path | A step counts when |
|---|---|---|
| Diamonds laid out | the diamonds | as the share of those laid out that the team holds |
| Diamonds from ore | the chain to a diamond after the last step whose item the kit holds, then the diamonds: an iron pickaxe (6) and diamonds (8) for the ingots kit, raw iron (4), an iron ingot (5), an iron pickaxe (6) and diamonds (8) for stone tools, the diamonds alone for an iron pickaxe | the step's item is in `team_obtained`; the diamonds as the share of one each that the team holds |
| An item | every step of its chain (for a stone pickaxe: logs 1, planks 1, a table 2, sticks 1, a wooden pickaxe 3, cobblestone 2, the pickaxe 3) | the step's item is in `team_obtained` |
| A milestone | the milestones from the first that the kit and start leave to the task's own, after the first steps of the game for a team with nothing | the advancement was earned; killing the dragon counts `DRAGON_DAMAGE` times the most it was hurt, as a share of its weight |

Solving the task completes its path (however it was done: a furnace needs no stone pickaxe), except where diamonds are
laid out, whose path ends with holding every one. So:

- every episode that solved its task scores at least 0.5, and every one that did not, less;
- of those that did not, the one further along scores more;
- an episode scores 1 exactly when nothing is left to earn, which is `saturated`, and the episode ends. Ore rows end
  when every player holds a diamond.

The result records the parts: `reward_parts` (`solved`, then each step of the path by name, what it adds to the
reward; they sum to it) and `progress`. The monitor shows them with the rest of the result. The goal in the system
prompt says what counts: one diamond each and the steps toward it for an ore row, the milestones of the path for a
progress task.

`duration` is the turns the team took; game time is `game_minutes`.

Diamonds past one each do not count. They carried 31% of the pooled within-group variance of the run's scores,
the largest term after solving, and they were earned after the task was solved: in the run, 61% of the turns of
solved ore episodes (2,585 of 4,254) came after the team held one diamond each, 23% of every ore row's turns. Ending
there spends those turns elsewhere.

### Speed

How long an episode took does not change its score by default. `advantage.tiebreak` (an advantage component, 0 in
every preset) gives that much more to the shortest episodes of a group whose every episode saturated its task, by
`Episode.duration`: for Minecraft, turns. A group in which any episode fell short gets nothing from it: its rewards
already differ, or none did. A run that wants it sets it small against the reward's range (0.05 against 0 to 1).

Turns, not game time: game time per turn is a quarter of the variance of saturated episodes' game time and comes from
paths, the slowest agent and the server's pace; and in two of the ten groups the speed bonus acted in, it went to an
episode that took more turns.

### The curriculum

`Curriculum` reads `solved` to unlock rows and whether a group's rewards differed to weigh them; neither reads the
reward's scale. Progress makes more unsolved groups differ, and their rows weigh as much as rows with mixed groups
([curricula](curricula.md) proposes weighing groups with partial credit less).

## Replayed

This reward, recomputed from the results of 65 of the run's played groups (259 episodes; 2 groups have an episode
whose result was not found), with today's test of solved (3 episodes from the run's first groups were solved by an
older rule). The play is the run's: a policy trained on this reward would play otherwise.

| | The run (reward and speed bonus) | This reward | This reward, tiebreak 0.05 on turns |
|---|---|---|---|
| Groups skipped for equal scores | 17 of 65 (26%) | 24 (37%) | about 14 (22%) |
| Within-group variance from the speed bonus | 0.4% | 0 | only in groups where every episode saturated |
| Within-group variance from diamonds past one each | 31% | 0 | 0 |
| Within-group variance from solving / from the rest of the reward | 50% / 48% (counts within one outcome) | 37% / 18% (progress; the rest their covariance) | |
| Mean size of the advantages, groups with some solved / none / all | 2.88 / 1.53 / 1.81 | 0.33 / 0.047 / 0.058 | |

- **Groups that differ and did not in the run:** 3 in which no one scored, where some teams got raw iron (35, t021u) or an ingot (59,
  t020; 68, t017u) and others did not.
- **Groups that differed in the run and do not:** 5 in which every team solved an ore row and the diamond counts differed (6, 7, 19,
  27, 39), and the 5 the speed bonus alone trained. Under this reward 15 groups saturate in every episode; for 11
  the turns to saturation are known (from the turn-by-turn record of ore rows), and 10 of those differ: with a
  tiebreak those train.
- **Distinct rewards** by rows: crafting 4 to 7, chests 14 to 26, items 8 to 4, ore in sight 15 to 9, ore nearby 9
  to 4, ore far 6 to 2. Zero rewards in ore rows in sight 34 to 23, nearby 21 to 16.
- **Scale.** Rows do not weigh by their diamonds: a group with mixed outcomes carries six to seven times the
  advantage of a group where every team solved or none did; in the run it carried 1.6 to 1.9 times.

The share of skipped groups rises, by the groups whose only differences were speed or diamonds past the objective.
The remaining ties are equal outcomes: every team at the same step of the path (in t019, group 58, every team made a
wooden pickaxe and none mined stone; in t009u, group 30, every team held 2 of 4 diamonds), or nobody started (group
26).
