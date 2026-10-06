# Three ways in

The platform seen from where you stand: building an environment, designing training, or deploying. Each part says what
you write, what you are given and what you never see; read the one that matches your work.

**Read first:** [Start here](../start/README.md). **Next:** [Write an environment](README.md), [Train a
model](../train/README.md) or [Deploy the platform](../deploy/README.md).

Three people use this system, and each has one surface. They meet in two places only: the **session** (how anything
gets a model) and the **episode** (the labelled trajectory that comes out).

| You are | You write | You are given | You never see | You import |
|---|---|---|---|---|
| Building an environment | A world and its rulebook: its situations, what a player perceives, what it can do, how it went | A model per player: messages and tools in, a message out | Tokens, context limits, engines, trainers, where anything runs | `rollout` |
| Designing training | What to run, how to group it, what each [episode](../libraries/rollout-train/episodes.md) counts for | A [ledger](../libraries/rollout-train/checkpoints.md#the-ledger) to ask for episodes in, the finished episodes back, somewhere to publish weights | Worlds, servers, which machine ran what | `rollout_train` |
| Deploying | A [cluster config](cluster.md): inference providers, trainers, pools, environments; and a run's settings or a [preset](cluster.md#presets) | The same protocols in process or over the network | Tasks and algorithms | Nothing: the config names implementations |

## Building an environment

An environment says how a situation is set up, what a player perceives, what it can do, how the world moves, and how
it went. It is a package that depends on `rollout` and on nothing above it (`tests/test_layers.py` checks that), as
[`environments/minecraft`](../products/minecraft-team.md) does. It chooses how much of the platform's loop to use:

| Depth | You write | The platform provides |
|---|---|---|
| The platform's loop | A `Task`: tools, observations, a score | The agent loop, and memory that fits any model (`CompactingAgent`) |
| Your own loop | A `Program`: several players at once, lockstep, anything | A model per slot (`run.models[name]`), and `Memory` for long games |
| Your own harness, inside the environment | Launch it; say when it ended and how it went | An address per slot (`run.model.address()`): a base URL and a key for an endpoint that speaks OpenAI's and Anthropic's APIs |

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

What a run trains on and an eval measures is an **environment**:

```py
class Words:
    program = agent_program(Guess)
    version = "1"                                         # changed whenever rows, starts, evals or scoring change
    description = Description(rewards=(0.0, 1.0), saturated=True, duration="turns")  # what its results say

    def rows(self) -> Sequence[Row]:                      # every situation, easiest first
        return [Row("say-yes", "say yes", {"word": "yes"}), Row("say-no", "say no", {"word": "no"})]

    def start(self, row: Row, rng: random.Random) -> JsonValue:   # one start: what every run of a group is given
        return {**row.parameters, "seed": rng.randrange(1000)}

    def evals(self) -> Mapping[str, Sequence[Start]]:     # eval data: starts training never draws
        return {"words-held-out": drawn(self, seeds=[1, 2])}
```

It may also have a `curriculum()` of its own: what to train on next, and when to open harder rows (on an eval's
results, say). `rollout env check module:name` checks it before anything trains on it
([rollouts](../libraries/rollout-train/rollouts.md#environment)).

A model's limits reach an environment only as outcomes. A long game uses `Memory`: the environment says what a turn
looks like once it is no longer the current one, and what to ask when turns must go; the library decides when.

## Designing training

```py
await plan(ledger, "miner-1", Plan(environment.program, binding), fence)
group = {"parameters": environment.start(row, rng), "episodes": 4, "task": row.key}
await ledger.append(table("miner-1", GROUPS), "12", group, fence)      # runners play it from here
episodes = await episodes_of(ledger, blobs, "miner-1", 12, 4)          # when all four have ended
batch = Grpo().batch(episodes, trainer.budget, rng)         # weighted segments, or why there are none
makes = new_id()                                            # the checkpoint's id, chosen before the step
await trainer.step(batch.segments, seed=12, parent=files, into=Path(f"checkpoints/{makes}"))  # files: `Files`
await publish("policy", makes, fetched, 3)  # served under its id, at depth 3; `fetched` reads its files if needed
```

- A run asks for a **group** of episodes in the ledger, and **runners**, wherever they are, claim them, play them
  and record the finished **episodes** there. Asking for more while others are in flight is all asynchronous
  training needs: every sampled token carries the weights version it was sampled at.
- An **episode** has its labels, its outcome, its result, and for each model slot its trajectory: the segments of
  tokens the policy saw and continued, with the logprobs it sampled them at.
- `rollout_train.train` is the loop most runs use: a curriculum over an environment picks rows, each start is played as a
  group of episodes, and a step is taken over several groups at a time, while play goes on
  ([training](../libraries/rollout-train/training.md)).
- Watching: the ledger (each group, its claims and its episodes, the checkpoints), the runners' heartbeats (their
  machines, and what each channel serves and how fast), and each run's feed of the episodes playing now, on one page
  (`rollout monitor RUN`).

The same code serves when the runners are in this process and when they are on other machines: they share only the
ledger and the blob store ([rollouts](../libraries/rollout-train/rollouts.md)).

## Deploying

A cluster config is a TOML file, written once per cluster, that names what it offers: inference providers (what samples
a channel, and its models), trainers, sandbox pools, tool sets served elsewhere and environments
([the cluster config](cluster.md)). A run's settings pick from it: the environment, the trainer, each channel's provider,
model and token format, and the numbers each takes; a preset holds a set of them under a name.

```bash
uv run rollout train minecraft_team.environment:environment --preset minecraft-one-gpu
```

The run's driver builds the run from the two and asks Ray for what it needs. The trainer's longest segment becomes its
channel's longest turn, and the channel's limits reach environments only as outcomes, such as a full
[memory](../libraries/rollout/memory.md). Scaling is a change to the cluster config ([deploying](deploying.md)).
