# Training

Code: `rollout_train` · See [rollouts](rollouts.md), [episodes](episodes.md),
[API reference](../../guide/reference.md#rollout_train)

The training loop, what it asks of an algorithm and of a trainer, and the curriculum. The loop is written against
[`Jobs`](rollouts.md), a [`Catalog`](rollouts.md#catalog), `Trainer`, `Algorithm` and `Store` only: the same loop
runs with everything in one process and with the runs, the engines and the trainer on machines of their own.

```python
await train(jobs, catalog, trainer, Directory(run), channel="policy", groups=100)
```

`rollout train PROFILE CATALOG` runs this loop over what a profile describes: the profile opens into jobs, a
trainer and a store, and names the channel to train ([deploying](../../guide/deploying.md)).

## The loop

[`train`](../../guide/reference.md#train) trains one channel's policy on a catalog. `channel` names it: new weights
are published to it, and unless a `binding` says otherwise every model slot of the catalog's program is served from
it.

- **Groups.** A group is one ticket: `algorithm.group_size` runs of one start of one row, labelled `group`,
  `iteration`, `task` and `title`. The curriculum picks the row and the catalog draws the start.
- **Nothing waits for all episodes.** `OUTSTANDING` groups are kept submitted. The job starts the next group as
  soon as there is room beside what is left of the one before (`overlap`), and each group is trained on when its
  last episode ends, while the next group's episodes run on. Their tokens then carry two weights versions, which
  the trainer's objective corrects for.
- **For each group**, in order: the curriculum records it, the algorithm says what to train on, the trainer steps,
  the new weights are published to the channel, one line is written, and the group's episodes are acknowledged.
- **A group with nothing to train on** is logged with the algorithm's reason (`skipped`), and the weights stay as
  they were.
- **A step that fails** (`StepFailed`) is logged with its `error`, and the weights stay as they were.
  `FAILED_UPDATES` in a row stop the loop.
- **A ticket the job refuses** stops the loop with [`Refused`](rollouts.md#guarantees).
- **Started again** over the same store, the loop goes on after the last group logged, with the curriculum it
  saved and on starts drawn anew. Episodes that a stopped loop left in the job's log are set aside: they are groups
  no more.

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
- **Saved** as `curriculum.json` in the run's store after every group, each record with its row's title. Records
  go back to the row of the same title, so they stay with their rows when a catalog changes.

## The trainer

A [`Trainer`](../../guide/reference.md#trainer) takes a batch, moves the policy and says where the new weights
are. Nothing in the protocol says where it runs or what it trains.

- **`budget`** ([`Budget`](../../guide/reference.md#budget)) is what the trainer can take: the longest sequence, and
  how many sequences a step can afford. It comes from the trainer's hardware, and nothing above the trainer chooses
  it. The algorithm selects within it, and a deployment makes the longest sequence its channel's longest turn
  ([limits](channels.md#limits)).
- **`step(batch, seed=...)`** trains on [`Weighted`](../../guide/reference.md#weighted) sequences and returns a
  [`Step`](../../guide/reference.md#step): the name of the new weights, where engines read them, and metrics.
- **`StepFailed`** means the step produced no weights: the policy is as it was, and a later step may succeed.
- **`latest`** is the name and path of the newest weights, if a step has been taken by this trainer or by one
  before it over the same state. A deployment publishes them when it starts, so a run that is started again serves
  its newest weights.

[`Colocated`](../../guide/reference.md#colocated) wraps a trainer that shares an accelerator with the engines of
some channels. For each step it holds new requests back, waits for those in flight, puts the engines to sleep,
steps, wakes the engines and lets requests go on. A `guard` is called once the engines are asleep and raises if the
step should not start. It adds `waited_for_requests_seconds` and `update_seconds` to the step's metrics.

`LoraTrainer` is the trainer this repository gives: [LoRA trainer](../../implementations/rollout-lora.md).

## The record

A run writes one line per group to `metrics.jsonl` in its store: an
[`Iteration`](../../guide/reference.md#iteration). The same line goes to the job as an `iteration` note, where the
[monitor](monitor.md) reads it; the report reads the file.

| A group that was | Has |
|---|---|
| trained on | `update` (the trainer's metrics), `adapter` and `version` (the weights and the channel's version once published) |
| not trained on | `skipped`: the algorithm's reason |
| given to a step that failed | `error`: the last line of what the step raised |

Every line has the row, the rewards, `solved` and durations of the episodes fit to train on, how many episodes
failed and why, how many sequences were recorded and how many trained on, the algorithm's notes, and how many rows
are unlocked. `iterations(store)` reads them back.

Given a blob store (`train(..., blobs=...)`), a group that was trained on also has:

| Field | Names |
|---|---|
| `batch` | A blob: every sequence the step trained on, as its place in the job's log (`cursor/slot/index`) and its advantage |
| `checkpoint` | Blobs by name: what the step left behind ([`Step.artifacts`](../../guide/reference.md#step)). A directory is kept as a tar archive |

With the job's [log](rollouts.md#the-log), that is the whole run: every episode, what each step was trained on, and
the weights and trainer state after it.

A [`Store`](../../guide/reference.md#store) holds the run's small state as named texts.
[`Directory`](../../guide/reference.md#directory) is one in a directory.

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
