# Three ways in

Three people use this system, and each has one surface. They meet in two places only: the **session** (how anything
gets a model) and the **episode** (the labelled trace that comes out).

| You are | You write | You are given | You never see |
|---|---|---|---|
| Building an environment | A world and its rulebook: a catalog of situations, what a player perceives, what it can do, how it went | A model per player: messages and tools in, a message out | Tokens, context limits, engines, trainers, where anything runs |
| Designing training | What to run, how to group it, what each episode counts for | Jobs to submit rows to, a stream of finished episodes, somewhere to publish weights | Worlds, servers, which machine ran what |
| Deploying | A profile: channels and their engines, the trainer, the runner, where tool sets live | The same protocols in process or over the network | Tasks and algorithms |

## Building an environment

An environment says how a situation is set up, what a player perceives, what it can do, how the world moves, and how
it went. It chooses how much of the platform's loop to use:

| Depth | You write | The platform provides |
|---|---|---|
| The platform's loop | A `Task`: tools, observations, a score | The agent loop, and memory that fits any model (`CompactingAgent`) |
| Your own loop | A `Program`: several players at once, lockstep, anything | A model per slot (`run.models[name]`), and `Memory` for long games |
| Your own harness, inside the environment | Launch it; say when it ended and how it went | An address per slot (`run.model.address()`): a base URL and a key for an OpenAI-compatible endpoint |

In all three, the reward goes through the run (`run.reward`, or an observation's `reward`), and the result says how it
went in the world's own terms:

```python fragment
await run.emit("result", {"solved": True, "saturated": False, "duration": 3.5})
```

`solved` and `saturated` (nothing was left to earn) are booleans; `duration` is in whatever the world counts. A
curriculum and a tie-break read them; nothing else about the world is known to training.

The environment's own infrastructure (game servers, sandboxes) is a **tool set**, imported by name. It runs in the
process that runs episodes, or on machines of its own (`rollout tools module:factory`), and the program calls
`run.tools` the same way.

What there is to train on is a **catalog**:

```python fragment
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

```python fragment
job = await jobs.start(program=catalog.program, binding=binding, in_flight=5)
ticket = await job.run(catalog.start(row, rng), labels={"group": "0012", "task": row.key}, count=4)
episodes = await ticket.episodes()                          # when all four have ended
step = await trainer.step(Grpo().batch(episodes, trainer.budget, rng), seed=12)
await job.publish("policy", step.adapter, step.path)
```

- A **job** runs rows and keeps a log of finished **episodes**, read with a cursor (`job.episodes(cursor)`) or per
  ticket. Reading while runs are in flight is all asynchronous training needs: every sampled token carries the
  weights version it was sampled at.
- An **episode** has its labels, its outcome, its result, and for each model slot the token sequences the policy saw
  and continued, with the logprobs it sampled them at.
- `rollout_train.train` is the loop most runs use: a curriculum over a catalog, groups, a step per group.
- Watching: `job.status()`, the job's own events (tickets, episodes, published weights, your notes), and each run's
  feed, on one page (`rollout monitor RUN/feed`).

The same code holds `RolloutJobs` (runs in this process) or `RolloutClient(url)` (runs elsewhere).

## Deploying

A profile is a TOML file that names each of these: the channels (a model, its token format, what serves it, one
entry per replica), the trainer and the channel it trains, the runner, and where each tool set lives.

```bash
rollout train profile.toml minecraft_swarm.catalog:catalog
```

The trainer's limits become its channel's limits; the channel's limits reach environments as "your memory is full".
Scaling is a change to this file ([deploying](deploying.md)).
