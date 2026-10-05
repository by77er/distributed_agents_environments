# Gridworld

Code: `environments/gridworld`

Two to four agents share a small grid level. To win they have to spread out, one agent on each final plate at the
same time. In the harder levels they first get through doors, a gate and a lever. Every agent can reach every plate,
so what they have to learn is to split up: they agree over chat who goes where, who holds the gate and who pulls the
lever. They don't race each other to the nearest plate. Every agent gets the team's reward.

The environment is the package `gridworld` (import `gridworld`), which depends on `rollout` alone. Its parts are its
levels as the rows of an [environment](../guide/perspectives.md#building-an-environment), and a program that plays one
episode. It needs no sandbox, no tool set and no GPU of its own.

```bash
uv run rollout env check gridworld.environment:environment                       # without a model
uv run rollout env check gridworld.environment:environment --profile PROFILE    # groups played by the profile's model
uv run rollout train environments/gridworld/profiles/one-gpu.toml gridworld.environment:environment
```

## Where it runs

`environments/gridworld/profiles/one-gpu.toml` trains it on one 16 GB GPU: Qwen/Qwen3-0.6B on one vLLM engine, with
the budgets its check played it with (thinking 384 tokens, answers 128, prompts within 4,096), and a LoRA trainer of
rank 16 that shares the card. It needs no sandbox pool and no tool set.

In the K3s cluster (deploy/chart/rollout), the launcher `gridworld` offers it with the same profile over the cluster's
stores (`files/profiles/gridworld/qwen3-0.6b.toml`), submitting each run as a Ray job that asks for the GPU; the
cluster config lists it under `[environments]`, in the platform's Python. Its project declares its environment for an
import from git (`[project.entry-points."rollout.environments"]`, [writing an environment others can
import](../guide/publishing.md)): imported from this repository with the subdirectory `environments/gridworld`, it is
the version `gridworld@VERSION`, which runs in the platform's Python too, since `rollout` is its only dependency.

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
- **Winning:** the episode ends solved, with reward 1 for every agent, as soon as every final plate is pressed at
  the end of a turn. If the turn budget runs out first, it ends with reward 0. There is no shaping.

## The rows

Easiest first. Rows get harder in two ways: more agents to share out among the plates, and more steps before the
final plates.

| Row | Agents | Turns | The level |
|---|---|---|---|
| `open-2` | 2 | 20 | One open room with two final plates |
| `open-3` | 3 | 24 | One open room with three final plates |
| `door-2` | 2 | 40 | Two plates in the first room open the door to the room with the final plates |
| `door-3` | 3 | 40 | The same, with three agents and three final plates: two of them stand on the pair |
| `gate-3` | 3 | 50 | A gate in the first room's north wall leads to a side room with a lever. One agent holds the gate's plate while another goes in, pulls the lever and comes back out. The lever opens the door to the room with the final plates |
| `vault-4` | 4 | 80 | Both: a pair of plates opens the door to a hall, the hall's gate leads to the lever, and the lever opens the door to the room with the four final plates |

A start is a row's parameters plus a seed (`{"layout": "gate", "agents": 3, "turns": 50, "seed": …}`). The seed
draws the room sizes, the doorways, where everyone starts, where the plates and the lever go, and the agents'
names. The same row and seed always give the same level. Names are drawn from a list with a different initial for
each (`gridworld.game.NAMES`), so the policy learns no name's part.

The eval data, `gridworld-eval`, is the starts of seeds 1, 2 and 3 of every row: 18 starts. Training draws starts
through `train_start`, which draws again whenever a draw is one of them.

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

- `solved`;
- `duration`: the number of turns played;
- `ended`: `every final plate pressed` or `turns`;
- the layout, the number of agents, the seed and the team's names;
- `opened`: each door that opened for good, with the turn it opened in;
- `actions`: the actions taken, counted by kind (`move`, `wait`, `say`, `none` for a reply with no tool call,
  `invalid`), plus the moves that failed (`blocked`);
- the chat;
- the level's walls.

The environment's description says the same: rewards fall in `[0, 1]`, results report `solved`, and `duration`
counts turns.

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
row took at most 10, 14, 29, 30, 36 and 65 turns, row by row.

## The pieces

Paths are under `environments/gridworld/`.

| Piece | Where | What it does |
|---|---|---|
| Levels | `gridworld/level.py` | The four layouts (`open`, `door`, `gate`, `vault`), drawing a level from a layout, a number of agents and a seed, and the check that a level can be solved |
| Game | `gridworld/game.py` | The rules, one turn at a time: moves and ties, talk, doors, gates and levers, winning and the budget |
| Prompts | `gridworld/prompts.py` | The system prompt, observations as text, the three tools, and a reply read as an action |
| Episode | `gridworld/episode.py` | The program: a model slot for each agent (`agent-1` to `agent-4`, of which a start uses the first as many as it has agents), all agents sampled at once each turn (`run.gather`), the team's reward to each |
| Environment | `gridworld/environment.py` | The six rows, a start of one (its parameters and a seed), the eval data `gridworld-eval`, and what its results say |
| Scripted team | `gridworld/scripted.py` | The policy above, and `ScriptedTeam`, which serves it as a model endpoint |
| Profile | `profiles/one-gpu.toml` | Qwen/Qwen3-0.6B on one vLLM engine and a LoRA trainer sharing its card ([where it runs](#where-it-runs)) |
| Tests | `tests/` | Levels (same seed, same level; every level sound; each step needed; the check catches unsound levels), the game (walls, ties, swaps, doors, gates, levers, chat, winning, the budget), observations from each agent's side, replies read as actions, whole episodes through the runner, and `rollout env check` |
