# Training

Code: `rollout_train` · See [rollouts](rollouts.md), [episodes](episodes.md),
[API reference](../../guide/reference.md#rollout_train)

The training loop, what it asks of an algorithm and of a trainer, and the curriculum. The loop is written against
[`Jobs`](rollouts.md), a [`Catalog`](rollouts.md#catalog), `Trainer`, `Algorithm` and [`Policies`](policies.md)
only: the same loop runs with everything in one process and with the runs, the engines and the trainer on machines
of their own.

```python
await train(jobs, catalog, trainer, policies, policy="swarm", channel="policy", directory=versions, groups=100)
```

`rollout train PROFILE CATALOG` runs this loop over what a profile describes: the profile opens into jobs, a
trainer and the policies, and names the policy to train and the channel that serves it
([deploying](../../guide/deploying.md)).

## The loop

[`train`](../../guide/reference.md#train) trains one [policy](policies.md) on a catalog and serves it on one
channel. Unless a `binding` says otherwise, every model slot of the catalog's program is served from that channel.

- **Groups.** A group is one ticket: `algorithm.group_size` runs of one start of one row, labelled `group`,
  `iteration`, `task` and `title`. The curriculum picks the row and the catalog draws the start.
- **Nothing waits for all episodes.** `OUTSTANDING` groups are kept asked for. The job starts the next group as
  soon as there is room beside what is left of the one before (`overlap`), and each group is trained on when its
  last episode ends, while the next group's episodes run on. Their tokens then carry two versions, which the
  trainer's objective corrects for.
- **For each group**: the curriculum records it, the algorithm says what to train on, the trainer steps from the
  policy's newest version, the new version is added to the policy and served on the channel, and the group's
  outcome is written.
- **A group with nothing to train on** is logged with the algorithm's reason (`skipped`), and the policy stays as
  it was.
- **A step that fails** (`StepFailed`) is logged with its `error`, and the policy stays as it was.
  `FAILED_UPDATES` in a row stop the loop.
- **A ticket the job refuses** stops the loop with [`Refused`](rollouts.md#guarantees).

## Dying and starting again

The loop can be killed at any moment and started again. It keeps nothing it cannot read back: what it decides and
what happens are appended to three tables in the [ledger](policies.md#the-ledger), each under the group's number,
and every action is one that can be taken twice.

| Table | Written | Holds |
|---|---|---|
| `runs/RUN/groups` | before a group is asked for | the row, and the start every episode of the group is given |
| `runs/RUN/steps` | before the trainer is called | the policy, the version the step starts from, the number of the one it will make, the batch (a blob) and how many sequences it has, the seed, when it was decided |
| `runs/RUN/iterations` | last | how the group went and what was done with it: an [`Iteration`](../../guide/reference.md#iteration) |

| It died | Started again, it |
|---|---|
| after deciding a group | asks for that group again under the same key, and the job gives back the ticket it has |
| while a group played | waits for the episodes the job still owes; the others are in the job's [log](rollouts.md#the-log) |
| after a group ended | finds the group has no outcome, and takes it from there |
| during a step | finds the step decided and no version made, and takes the step again from the same parent |
| after the step | finds the version, serves it, and writes the group's outcome |

- **A step that was decided is finished before another is decided.** A decision names the version it will make,
  and only one group may make it.
- **The version is the commit.** A step's files are kept in the blob store and then the version is appended to the
  policy's table. A step that died before the append made nothing.
- **One loop at a time.** Starting takes the run's fence and the policy's. A loop that was replaced, and does not
  know it yet, has its next write refused.
- **`groups` counts groups done with**, those a stopped loop left unfinished among them.

What is redone: a step that was in progress, and the runs that were in flight if they ran in the loop's own process
(the job runs them again from the same start).

## The algorithm (`Grpo`)

The loop asks two things of an [`Algorithm`](../../guide/reference.md#algorithm): how many episodes of one start it
compares (`group_size`), and what to train on from a group's episodes (`batch`). `batch` is given the episodes of
every outcome, the trainer's budget and a random number generator seeded by the group's number, and returns a
[`Batch`](../../guide/reference.md#batch): the weighted sequences, or why there are none, and notes to log with the
group. Another algorithm is passed as `train(..., algorithm=...)`.

[`Grpo`](../../guide/reference.md#grpo), the default, is group-relative policy optimisation:

| | |
|---|---|
| Advantages | An episode's score minus its group's mean, with no division by the group's spread (Dr. GRPO). Every token the policy sampled in the episode gets it; with several model slots, every slot's, so a team is rewarded together. |
| Dynamic sampling | A group whose scores are all equal has nothing to teach and is skipped (DAPO). So is a group with fewer than two episodes fit to train on. |
| The fastest of the saturated | Episodes that reached everything their task has to give earned the same; the one that took the least scores a point more, and episodes that tie for fastest all do. The task says what saturated means and how long it took ([result conventions](episodes.md#result-conventions)); comparing across the group is done here. An episode that does not say its duration is not compared. `tie_break` turns this off. |
| What is trained on | Every sequence of the episodes whose advantage is not zero, up to what the trainer can afford in a step (`Budget.sequences`). Beyond that, sequences are taken at even steps through the group, so that each episode and slot keeps its share, spread over its whole game. |

`complete_groups` gathers the episodes of a job's stream by a label, for a loop of your own that reads the stream
rather than tickets.

## The curriculum

[`Curriculum`](../../guide/reference.md#curriculum) says which row to train on next.

- **Weight.** A group-relative update learns from the differences between a group's episodes, so a row's weight is
  the share of its recent groups whose rewards differed (a moving average), plus a little for every unlocked row so
  that none is forgotten. Untried rows come first.
- **Pending rows.** A row whose group is still running is not picked again until that group is recorded: choosing
  it would be choosing on what was known before it.
- **Unlocking.** Rows unlock in the catalog's order: the first `start` of them, and `reach` past the hardest one
  solved at least half the time. Whether a row was solved decides only what unlocks.
- **Rows that cannot be set up.** A group none of whose episodes completed counts for nothing at first: the next
  start may be one the row can be set up from. After `FAILED_GROUPS` such groups in a row, the row counts as tried,
  and as having taught nothing.
- **What it sees.** The task's own rewards, whatever the algorithm adds to them.
- **It is a fold.** A curriculum is rebuilt from the run's iterations when the loop starts: each goes to the row of
  its title, so records stay with their rows when a catalog changes. A choice is made with a random number
  generator seeded by the group's number, and is written down before it is acted on.

## The trainer

A [`Trainer`](../../guide/reference.md#trainer) takes a batch and makes new weights from given ones. It keeps
nothing between steps that it cannot be given again, so any trainer can take any step of any policy.

- **`budget`** ([`Budget`](../../guide/reference.md#budget)) is what the trainer can take: the longest sequence, and
  how many sequences a step can afford. It comes from the trainer's hardware, and nothing above the trainer chooses
  it. The algorithm selects within it, and a deployment makes the longest sequence its channel's longest turn
  ([limits](channels.md#limits)).
- **`step(batch, seed=..., parent=..., into=...)`** trains on [`Weighted`](../../guide/reference.md#weighted)
  sequences, starting from a [`Checkpoint`](../../guide/reference.md#checkpoint) (a version's files on this
  machine; none means the base model). It leaves the new weights in `into/weights`, as engines load them, and what
  a later step starts from (an optimizer's state, say) in `into/state`. It returns its metrics.
- **`StepFailed`** means the step produced no weights: the policy is as it was, and a later step may succeed.

[`Colocated`](../../guide/reference.md#colocated) wraps a trainer that shares an accelerator with the engines of
some channels. For each step it holds new requests back, waits for those in flight, puts the engines to sleep,
steps, wakes the engines and lets requests go on. A `guard` is called once the engines are asleep and raises if the
step should not start. It adds `waited_for_requests_seconds` and `update_seconds` to the step's metrics.

`LoraTrainer` is the trainer this repository gives: [LoRA trainer](../../implementations/rollout-lora.md).

## The record

For each group the run appends an [`Iteration`](../../guide/reference.md#iteration) to its `iterations` table. The
same line goes to the job as an `iteration` note, where the [monitor](monitor.md) reads it; the report reads the
table (`iterations(ledger, run)`).

| A group that was | Has |
|---|---|
| trained on | `update` (the trainer's metrics), `adapter` (the version the step made, by name) and `version` (its number) |
| not trained on | `skipped`: the algorithm's reason |
| given to a step that failed | `error`: the last line of what the step raised |

Every line has the row, the rewards, `solved` and durations of the episodes fit to train on, how many episodes
failed and why, how many sequences were recorded and how many trained on, the algorithm's notes, and how many rows
are unlocked.

With the job's [log](rollouts.md#the-log) and the policy's [versions](policies.md), that is the whole run: every
episode, what each step was trained on, and the weights and the trainer's state after it.

## Reporting

`rollout report RUN CATALOG` writes `progress.png` and `progress.md` into the run's directory: the climb through the
curriculum, every group's rewards, and what each update did. With `--watch` it does so after every group; with a
Discord webhook (`--webhook`, or `DISCORD_WEBHOOK_URL`) it posts both there. It needs the `report` extra.

## Trying it without a GPU

`rollout_train.testing` has public test doubles, so that a catalog, an algorithm or a whole profile can be tried
with no model and no accelerator.

| Double | Stands for |
|---|---|
| [`ScriptedEngine`](../../guide/reference.md#scriptedengine) | an engine: it answers each request from a script, and keeps what it was asked and told |
| [`PlainRenderer`](../../guide/reference.md#plainrenderer) | a model family's token format: one token per character, readable in a failing test |
| `plain_channel(script)` | a channel over both |
| `scripted_engine`, `plain_renderer` | what a profile names as `engine` and `renderer`: `rollout_train.testing:scripted_engine`, `rollout_train.testing:plain_renderer` |

`tests/rollout_train/test_profile.py` opens a profile made of these, with a trainer that trains nothing, and runs
the loop over it. For tasks and agents alone, see [guide: testing](../../guide/testing.md).
