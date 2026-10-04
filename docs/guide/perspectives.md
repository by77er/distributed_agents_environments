# Three ways in

Three people use this system, and each has one surface. They meet in two places only: the **session** (how anything
gets a model) and the **episode** (the labelled trajectory that comes out).

| You are | You write | You are given | You never see | You import |
|---|---|---|---|---|
| Building an environment | A world and its rulebook: a catalog of situations, what a player perceives, what it can do, how it went | A model per player: messages and tools in, a message out | Tokens, context limits, engines, trainers, where anything runs | `rollout` |
| Designing training | What to run, how to group it, what each episode counts for | A ledger to ask for episodes in, the finished episodes back, somewhere to publish weights | Worlds, servers, which machine ran what | `rollout_train` |
| Deploying | A profile: channels and their engines, the trainer, the runner, where tool sets live | The same protocols in process or over the network | Tasks and algorithms | Nothing: a profile names implementations |

## Building an environment

An environment says how a situation is set up, what a player perceives, what it can do, how the world moves, and how
it went. It is a package that depends on `rollout` and on nothing above it (`tests/test_layers.py` checks that), as
[`environments/minecraft`](../products/minecraft-team.md) does. It chooses how much of the platform's loop to use:

| Depth | You write | The platform provides |
|---|---|---|
| The platform's loop | A `Task`: tools, observations, a score | The agent loop, and memory that fits any model (`CompactingAgent`) |
| Your own loop | A `Program`: several players at once, lockstep, anything | A model per slot (`run.models[name]`), and `Memory` for long games |
| Your own harness, inside the environment | Launch it; say when it ended and how it went | An address per slot (`run.model.address()`): a base URL and a key for an OpenAI-compatible endpoint |

In all three, the reward goes through the run (`run.reward`, or an observation's `reward`), and the result says how it
went in the world's own terms:

```py
await run.emit("result", {"solved": True, "saturated": False, "duration": 3.5})
```

`solved` and `saturated` (nothing was left to earn) are booleans; `duration` is in whatever the world counts. A
curriculum and a tie-break read them; nothing else about the world is known to training.

The environment's own infrastructure (game servers, sandboxes) is a **tool set**, imported by name. It runs in the
process that runs episodes, or on machines of its own (`rollout tools module:factory`), and the program calls
`run.tools` the same way.

What there is to train on is a **catalog**:

```py
class Words:
    program = agent_program(Guess)

    def rows(self) -> Sequence[Row]:                      # every situation, easiest first
        return [Row("say-yes", "say yes", {"word": "yes"}), Row("say-no", "say no", {"word": "no"})]

    def start(self, row: Row, rng: random.Random) -> JsonValue:   # one start: what every run of a group is given
        return {**row.parameters, "seed": rng.randrange(1000)}
```

A model's limits reach an environment only as outcomes. A long game uses `Memory`: the environment says what a turn
looks like once it is no longer the current one, and what to ask when turns must go; the library decides when.

## Designing training

```py
await plan(ledger, "miner-1", Plan(catalog.program, binding), fence)
group = {"parameters": catalog.start(row, rng), "episodes": 4, "task": row.key}
await ledger.append(table("miner-1", GROUPS), "12", group, fence)      # runners play it from here
episodes = await episodes_of(ledger, blobs, "miner-1", 12, 4)          # when all four have ended
batch = Grpo().batch(episodes, trainer.budget, rng)         # weighted segments, or why there are none
makes = new_id()                                            # the version's id, chosen before the step
await trainer.step(batch.segments, seed=12, parent=checkpoint, into=Path(f"versions/{makes}"))
await publish("policy", makes, f"versions/{makes}/weights", 3)  # served under its id, at depth 3
```

- A run asks for a **group** of episodes in the ledger, and **runners**, wherever they are, claim them, play them
  and record the finished **episodes** there. Asking for more while others are in flight is all asynchronous
  training needs: every sampled token carries the weights version it was sampled at.
- An **episode** has its labels, its outcome, its result, and for each model slot its trajectory: the segments of
  tokens the policy saw and continued, with the logprobs it sampled them at.
- `rollout_train.train` is the loop most runs use: a curriculum over a catalog picks rows, each start is played as a
  group of episodes, and a step is taken over several groups at a time, while play goes on
  ([training](../libraries/rollout-train/training.md)).
- Watching: the ledger (each group, its claims and its episodes, the versions), the runners' heartbeats (their
  machines, and what each channel serves and how fast), and each run's feed of the episodes playing now, on one page
  (`rollout monitor RUN`).

The same code serves when the runners are in this process and when they are on other machines: they share only the
ledger and the blob store ([rollouts](../libraries/rollout-train/rollouts.md)).

## Deploying

A profile is a TOML file that names each of these: the channels (a model, its token format, what serves it, one
entry per replica), the trainer and the channel it trains, the runner, and where each tool set lives. Engines,
renderers and trainers are packages of their own, named in the file as `module:name`.

```bash
uv run rollout train profile.toml minecraft_team.catalog:catalog
```

The trainer's longest segment becomes its channel's longest turn, and the channel's limits reach environments only
as outcomes, such as a full [memory](../libraries/rollout/memory.md). Scaling is a change to this file
([deploying](deploying.md)).
