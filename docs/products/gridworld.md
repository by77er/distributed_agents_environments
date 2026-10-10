# Gridworld

Code: `environments/gridworld`

**Read first:** [Example environments](README.md). **Next:** [Judging](judging.md).

Two to four agents share a small grid level. To win they have to spread out, one agent on each final plate at the
same time. In the harder levels they first get through doors, a gate and a lever. Every agent can reach every plate,
so what they have to learn is to split up: they agree over chat who goes where, who holds the gate and who pulls the
lever. They don't race each other to the nearest plate. Every agent gets the team's reward: half for solving the
level, half for progress through its stages.

The environment is the package `gridworld` (import `gridworld`), which depends on `rollout` alone. Its parts are its
levels as the rows of an [environment](../guide/perspectives.md#building-an-environment), and a program that plays one
episode. It needs no sandbox, no tool set and no GPU of its own.

```bash
uv run rollout env check gridworld.environment:environment                       # without a model
uv run rollout env check gridworld.environment:environment --preset gridworld-qwen3-0.6b   # groups played by its model
uv run rollout train gridworld.environment:environment --preset gridworld-qwen3-0.6b
```

## Where it runs

The preset `gridworld-qwen3-0.6b` (`deploy/chart/rollout/files/presets/gridworld-qwen3-0.6b.toml`) trains it on one
16 GB GPU: Qwen/Qwen3-0.6B on the cluster's vLLM engines, with small budgets (thinking 384 tokens, answers 128, prompts
within 4,096), groups of eight episodes (`group_size = 8`), `episodes_at_once = 8` (a group's episodes at once, each
two to four agents sampled together every turn), and a LoRA trainer of rank 16 that shares the card (64 segments of up
to 4,096 tokens a step). [Training it](#training-it) says why. It needs no sandbox pool and no tool set; the cluster
config lists it under `[environments]`, in the platform's Python, and serves the model (`deploy/clusters/example.toml`
and the chart's config both do).

On the chart's Kubernetes cluster (`deploy/chart/rollout`), a run asked for from the monitor with this preset is a
RayJob of its own. Its project declares its environment for an import from git
(`[project.entry-points."rollout.environments"]`, [writing an environment others can import](../guide/publishing.md)):
imported from this repository with the subdirectory `environments/gridworld`, it is the version `gridworld@VERSION`,
which runs in the platform's Python too, since `rollout` is its only dependency.

## The game

A level is rooms of floor between walls. Every agent starts in the first room. The final plates are in the last
room, one for each agent.

- **Turns are simultaneous.** Each turn every agent sees its observation and calls one tool. Only its first call
  counts. Then all the actions resolve together.
- **Tools:** `move(direction, say?)` steps one square `north`, `south`, `east` or `west`. `wait(say?)` stays put,
  and standing on a plate keeps it pressed. `say(message)` speaks and stays put. Talking is free: `move` and
  `wait` can carry a message too. Everyone hears what is said from the next turn on. A message is cut to 200
  characters.
- **Two agents can't share a square.** A step into a wall or a closed door fails, and the observation says why.
  When several agents step onto the same square, one gets there and the others stay put. The level's seed and the
  turn decide who, so the same start always breaks ties the same way. An agent can't step onto a square whose
  occupant stays put, and two agents can't pass through each other. An agent can step into a square that its
  occupant is leaving.
- **Plates, doors, gates and levers:**
    - a *door with plates* opens for good once all its plates are pressed at the same time;
    - a *gate* is open only while its plate is pressed, or while someone stands in it, so it never shuts on anyone;
    - a *lever* opens its door for good when someone steps on it.
- **Winning:** the episode ends solved as soon as every final plate is pressed at the end of a turn. If the turn
  budget runs out first, it ends unsolved.

## The reward

Every agent gets the team's reward, from 0 to 1 (`gridworld.scoring.scored`): half for solving the level, half for
progress through its stages.

    reward = 0.5 × solved + 0.5 × progress

A level's stages are its doors that open for good, in the order they open, then its final plates. Progress is the
share of the stages done. The final plates count as the share of them pressed at once at the end of a turn: the most
the team managed in the episode. A solved level has made all its progress, so a solved episode scores 1 and an
unsolved one less than 0.5.

| Layout | Stages | Each stage's part of the reward |
|---|---|---|
| `open` | the final plates | 0.5 × the share pressed at once |
| `door` | door a, the final plates | 0.25; 0.25 × the share |
| `gate` | door b (the lever's), the final plates | 0.25; 0.25 × the share |
| `vault` | door a, door c (the lever's), the final plates | 0.167 each, the last × the share |

- **Only what lasts, or needs the team together, counts.** A door opens once. The final plates count only as many as
  are pressed at the same time, so stepping onto a plate and off again, or one agent touring every plate, earns no
  more than standing on one. A held gate is no stage of its own: the lever behind it is.
- **Doing nothing earns nothing.** A team that never moves scores 0, and so does one that wanders without opening a
  door or pressing a final plate. A team walking at random does step onto plates: on a hundred starts of each row it
  scored 0.22 to 0.28 in the open rooms and 0.01 to 0.17 elsewhere.
- **Speed isn't in the reward.** A result says `saturated` when the level was solved, so the platform's
  `advantage.tiebreak` (off by default) can score the shortest of a group's solved episodes more.

Replayed over the 74 recorded episodes of a run of Qwen3.5-9B (18 groups of four), the scores of 11 groups differ
within the group, where whether each episode solved differs in 2. The 16 door, gate and vault groups, none solved,
score 0 to 0.375:

| Row | Episodes | Mean | Scores (how many) |
|---|---|---|---|
| `open-2` | 18 | 0.958 | 1 (17), 0.25 (1) |
| `open-3` | 12 | 0.833 | 1 (9), 0.333 (3) |
| `door-2` | 16 | 0.258 | 0.375 (7), 0.25 (6), 0 (3) |
| `door-3` | 12 | 0.132 | 0.333 (4), 0.25 (1), 0 (7) |
| `gate-3` | 8 | 0.031 | 0.25 (1), 0 (7) |
| `vault-4` | 8 | 0.125 | 0.167 (6), 0 (2) |

Counting the final plates ever pressed instead of pressed at once would add one group of the 18, and a single agent
could earn it by touring the plates.

## The rows

Easiest first: by layout, each adding a step before the final plates, and within a layout by agents, more to share
out among the plates.

| Rows | Agents | Turns | The level |
|---|---|---|---|
| `open-2`, `open-3`, `open-4` | 2, 3, 4 | 20, 24, 28 | One open room with a final plate for each agent |
| `door-2`, `door-3`, `door-4` | 2, 3, 4 | 40, 40, 50 | Two plates in the first room open the door to the room with the final plates. With more agents, two of them stand on the pair |
| `gate-2`, `gate-3`, `gate-4` | 2, 3, 4 | 48, 50, 60 | A gate in the first room's north wall leads to a side room with a lever. One agent holds the gate's plate while another goes in, pulls the lever and comes back out. The lever opens the door to the room with the final plates |
| `vault-2`, `vault-3`, `vault-4` | 2, 3, 4 | 64, 72, 80 | Both: a pair of plates opens the door to a hall, the hall's gate leads to the lever, and the lever opens the door to the room with the final plates |

A start is a row's parameters plus a seed (`{"layout": "gate", "agents": 3, "turns": 50, "seed": …}`). The seed
draws the room sizes, the doorways, where everyone starts, where the plates and the lever go, and the agents'
names. The same row and seed always give the same level. Names are drawn from a list with a different initial for
each (`gridworld.game.NAMES`), so the policy learns no name's part.

The eval data, `gridworld-eval`, is the starts of seeds 1, 2 and 3 of every row: 36 starts. Training draws starts
through `train_start`, which draws again whenever a draw is one of them.

## The curriculum

The environment has a curriculum of its own (`gridworld.curriculum.GridCurriculum`). It is the platform's
`Curriculum` ([training](../libraries/rollout-train/training.md#the-curriculum)) deciding two things its own way, and
like any curriculum it is the fold of the run's group results: a run started again rebuilds it from the ledger.

- **Unlocking.** The three open rooms are unlocked at first. A row is learned once at least two of its groups are
  recorded and the moving average (the newest group counting half) of its success reaches 0.5, or that of its
  progress reaches 0.7. Progress is read back from each episode's reward and whether it solved. The rows up to two
  past the hardest one learned are unlocked.
- **Drawing.** A row not yet tried weighs 1. Otherwise it weighs 0.1 plus the moving average of whether its groups'
  rewards differed, scaled by 0.5 + 0.5 × 4p(1 − p), where p is its success average. A row solved half the time
  weighs 1.1, one with spread progress and no solve 0.6, and a settled row (every episode solved, or none making any
  progress) 0.1. Settled rows are still drawn now and then, so that forgetting them shows.

### Every level can be solved

`gridworld.level.generate` draws a level and checks it with `problems`. If the check finds a problem, it draws
again (up to 200 times). A level passes when:

- the agents start on different squares, and there is one final plate for each agent;
- no door needs more plates than there are agents, and a gate has a plate and at least two agents (one to hold it,
  one to pass);
- opening every door as soon as it can be opened reaches every final plate. A gate counts as open once its plate
  can be reached;
- with every door open, the floor that is neither a plate nor a start stays connected and touches every plate and
  every start. So an agent standing on a plate, or still at its start, never cuts anyone off.

Drawing also keeps plates, levers and starts off the squares next to a doorway, where someone standing would block
the way. The tests check this for a hundred seeds of every row. They also check that each step is needed: with
every door shut no final plate can be reached, the lever can only be reached through its gate, and in `vault-4` the
gate's plate can only be reached through the first door.

## What an agent sees

Each turn, an agent sees the system prompt and its latest observation. The observation holds everything the agent
needs to act: the whole level, where everything is from the agent, what each thing does and its state, the recent
chat and what happened last turn. The system prompt is the same for every agent in a game. It names the team, the
goal, the turn budget, the tools and the rules, and describes only the kinds of door the level has.

```text
Turn 7 of 50. You are Sal.

Map (north is up):
# # # # # # # # # # #
# . . . = # # # # # #
# . @ . . # # # # # #
# # / # # # # # # # #
# . . . . # . 2 . . #
# . T . . # . . . . #
# . . . . # . . 3 . #
# N . . . + . . 4 . #
# # # # # # # # # # #
@ you, T Tess, N Ned, 1-4 plates, + a closed door, / an open door, = a lever, # wall, . floor.

Teammates (where they are from you):
- Tess: 3S
- Ned: 5S 1W

Plates, doors and levers (where they are from you):
- plate 1: 5S 1W. Holds gate a open while pressed. Pressed by Ned.
- plate 2: 2S 5E. A final plate. Not pressed.
- plate 3: 4S 6E. A final plate. Not pressed.
- plate 4: 5S 6E. A final plate. Not pressed.
- gate a: 1S. Open only while plate 1 is pressed, or someone stands in it. Open.
- door b: 5S 3E. Opens for good when the lever is pulled. Closed.
- lever: 1N 2E. Step on it to pull it: it opens door b for good. Not pulled.

Chat, oldest first:
- you, 6 turns ago: I'll pull the lever.
- Ned, 6 turns ago: I'll hold gate a open.

Last turn: You moved north.
```

The chat shows the last ten lines, each with its age in turns. The last line says what the agent's own action did:
a move that failed says what was in the way. It also says what changed in the world, for example "Plates 1 and 2
were pressed at once: door a is open for good." or "Gate a closed.".

## Results

An episode's result reports:

- `solved`, and `saturated` (solved: nothing was left to earn);
- `duration`: the number of turns played;
- `ended`: `every final plate pressed` or `turns`;
- the layout, the number of agents, the seed and the team's names;
- `progress` and `reward_parts`: what each part of the reward earned (`solved`, each door, `final plates`);
- `most_pressed`: the most final plates pressed at once at the end of a turn;
- `opened`: each door that opened for good, with the turn it opened in;
- `actions`: the actions taken, counted by kind (`move`, `wait`, `say`, `none` for a reply with no tool call,
  `invalid`), plus the moves that failed (`blocked`);
- the chat;
- the level's walls.

The environment's description says the same: rewards fall in `[0, 1]`, results report `solved` and `saturated`,
and `duration` counts turns. It also says what an episode samples, for estimating a run's spend (`spend_of`): every
agent samples every turn, so a turn is as many samples as the episode has agents (3 on average over the rows); an
episode plays at most its turn budget (49.3 turns on average over the rows, weighted so that turns times agents is the
mean of each row's budget times its agents, 148 samples); and a sample's prompt is about 1,500 tokens (measured on
Qwen3.5-9B's chat template: 1,050 for two agents in an open room to 1,720 for four in the vault).

## The scripted team

`gridworld.scripted` is a scripted policy that plays from the observation text alone, as a model would, and solves
every row. Its members follow one convention instead of talking it over: they rank themselves by the alphabetical
order of the team's names.

- While a door with plates is closed, the first ranks stand on its plates.
- While the lever behind a gate hasn't been pulled, the first holds the gate's plate and the second pulls the lever.
- Then each goes to the final plate of its rank. The holder lets go only once the lever-puller is back out through
  the gate.

The members announce their jobs in chat. To stay out of each other's way, a member keeps clear of teammates who
stay put and of moving teammates ranked before it, and steps aside when it stands where one of them steps next.
`ScriptedTeam` serves it as a model endpoint. The tests play it through the real program on the local runner, and on
the game alone for many seeds. Within the budgets it solves every start it has been given: a thousand seeds of each
row took at most 10, 14 and 17 turns in the open rooms, 29, 30 and 42 through the door, 37, 36 and 45 through the
gate, and 45, 55 and 65 in the vault, with two, three and four agents.

## Training it

- **Group size.** With rewards of 0 or 1, a group of four of a row solved one time in ten holds both solved and
  unsolved episodes only 34% of the time; a group of eight, 57%. Progress credit makes a group's scores differ more
  often, but the open rooms are mostly all-or-nothing: set `group_size = 8`.
- **Thinking budget.** An action is one tool call with a short message: 256 to 512 thinking tokens and 128 for the
  answer are enough. Qwen3.5-9B with 2,048 thinking tokens sampled about 1,050 tokens a reply.
- **Models.** Qwen/Qwen3-0.6B on one 16 GB card to begin with, and Qwen/Qwen3.5-4B on the same card for a stronger
  team. Qwen3.5-9B solved 26 of 30 recorded open-room episodes and none of 44 door, gate and vault ones.
- **The preset** `gridworld-qwen3-0.6b` encodes these: groups of eight, thinking 384 and answers 128, on the
  cluster's own GPU, so a step costs nothing.
- **On Tinker** a step is estimated at up to $25.55 for Qwen3.5-9B and $12.85 for Qwen3.5-4B (four groups of eight,
  thinking 512 and answers 128, the trainer's and the sampler's parts together; the estimate takes every budget as
  spent). A recorded run of Qwen3.5-9B with groups of four and 2,048 thinking tokens spent about $5.60 on sampling
  for every four groups (1,816 samples of 1,050 sampled and 1,500 prompt tokens each); the estimate for its settings
  is $14.43 for sampling and $28.48 with training. No preset runs it on Tinker: a hundred steps would cost over a
  thousand dollars.

## The pieces

Paths are under `environments/gridworld/`.

| Piece | Where | What it does |
|---|---|---|
| Levels | `gridworld/level.py` | The four layouts (`open`, `door`, `gate`, `vault`), drawing a level from a layout, a number of agents and a seed, and the check that a level can be solved |
| Game | `gridworld/game.py` | The rules, one turn at a time: moves and ties, talk, doors, gates and levers, winning and the budget |
| Prompts | `gridworld/prompts.py` | The system prompt, observations as text, the three tools, and a reply read as an action |
| Episode | `gridworld/episode.py` | The program: a model slot for each agent (`agent-1` to `agent-4`, of which a start uses the first as many as it has agents), all agents sampled at once each turn (`run.gather`), the team's reward to each |
| Reward | `gridworld/scoring.py` | The reward and its parts: half for solving, half for progress through the level's stages |
| Curriculum | `gridworld/curriculum.py` | `GridCurriculum`: rows unlocked by success or progress, drawn most where outcomes are mixed |
| Environment | `gridworld/environment.py` | The twelve rows, a start of one (its parameters and a seed), the eval data `gridworld-eval`, the curriculum, and what its results say and what an episode samples |
| Scripted team | `gridworld/scripted.py` | The policy above, and `ScriptedTeam`, which serves it as a model endpoint |
| Tests | `tests/` | Levels (same seed, same level; every level sound; each step needed; the check catches unsound levels), the game (walls, ties, swaps, doors, gates, levers, chat, winning, the budget), observations from each agent's side, replies read as actions, the reward's parts (solved, partly done, idle, farming), the curriculum (unlocking, drawing, rebuilt from results), whole episodes through the runner, and `rollout env check` |
