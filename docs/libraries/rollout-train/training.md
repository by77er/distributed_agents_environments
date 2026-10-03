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
- **Play and training go their own ways.** `OUTSTANDING` groups are kept asked for. The job starts the next group
  as soon as there is room beside what is left of the one before (`overlap`).
- **When a group's last episode ends**, its result is written at once: the curriculum records it, and the algorithm
  says what in it to train on. A group with nothing to train on is done with, with the algorithm's reason
  (`skipped`); the others join a queue.
- **A step is taken over the queue** once at least `groups_per_step` groups are in it (4 by default, so that no
  step leans toward one task), over every group queued by then, while play goes on; at the end of the run, over
  whatever is left. The trainer's segment budget is spread over the groups. The step starts from the policy's
  newest version; the version it makes is added to the policy and served on the channel. Tokens sampled under an
  older version are corrected for by the trainer's objective. One step is taken at a time.
- **A step that fails** (`StepFailed`) is written down with its `error`, its groups are done with, and the policy
  stays as it was. `FAILED_UPDATES` in a row stop the loop.
- **A ticket the job refuses** stops the loop with [`Refused`](rollouts.md#guarantees).

## Dying and starting again

The loop can be killed at any moment and started again. It keeps nothing it cannot read back: what it decides and
what happens are appended to four tables in the [ledger](policies.md#the-ledger), and every action is one that can
be taken twice.

| Table | Keyed by | Written | Holds |
|---|---|---|---|
| `runs/RUN/groups` | group | before a group is asked for | the row, and the start every episode of the group is given |
| `runs/RUN/results` | group | when its last episode ends | how it went: a [`Result`](../../guide/reference.md#result) |
| `runs/RUN/steps` | step | before the trainer is called | the groups it covers, the policy, the version it starts from, the number of the one it will make, the batch (a blob) and how many segments it has, the seed, when it was decided |
| `runs/RUN/failures` | step | when a step's trainer fails | its error |

A step's outcome is the version it makes, in the policy's table. A group is done with once its result trains on
nothing, or the step that covers it has made its version or failed.

| It died | Started again, it |
|---|---|
| after deciding a group | asks for that group again under the same key, and the job gives back the ticket it has |
| while a group played | waits for the episodes the job still owes; the others are in the job's [log](rollouts.md#the-log) |
| after a group ended | finds no result, and writes it |
| with groups queued | finds results to train on that no step covers, and queues them again |
| during a step | finds the step decided and no version made, and takes it again over the same groups, from the same parent |
| after the step | finds the version, serves it, and goes on |

- **A step that was decided is finished before another is decided.** A decision names the version it will make,
  and only one step may make it.
- **Episodes are acknowledged once their group is done with**, so that a loop started again finds in the job every
  episode a queued group or an unfinished step still needs. What a step trains on is the algorithm's batch of
  those episodes, the same each time it is computed.
- **The version is the commit.** A step's files are kept in the blob store and then the version is appended to the
  policy's table. A step that died before the append made nothing.
- **Saves thin out.** Once a version is served, the policy is thinned to `retention` (by default the trainer state
  of the newest three versions and of every tenth; [policies](policies.md#versions)).
- **One loop at a time.** Starting takes the run's fence and the policy's. A loop that was replaced, and does not
  know it yet, has its next write refused.
- **`groups` counts groups played**, those a stopped loop left unplayed among them; the loop ends once they are
  played and every one with something to train on has been in a step.

What is redone: a step that was in progress, and the runs that were in flight if they ran in the loop's own process
(the job runs them again from the same start).

## The algorithm (`Grpo`)

The loop asks two things of an [`Algorithm`](../../guide/reference.md#algorithm): how many episodes of one start it
compares (`group_size`), and what to train on from a group's episodes (`batch`). `batch` is given the episodes of
every outcome, the trainer's budget and a random number generator seeded by the group's number, and returns a
[`Batch`](../../guide/reference.md#batch): the weighted segments, or why there are none, and notes to log with the
group. Another algorithm is passed as `train(..., algorithm=...)`.

[`Grpo`](../../guide/reference.md#grpo), the default, is group-relative policy optimisation:

| | |
|---|---|
| Advantages | An episode's score minus its group's mean, with no division by the group's spread (Dr. GRPO). Every token the policy sampled in the episode gets it; with several model slots, every slot's, so a team is rewarded together. |
| Dynamic sampling | A group whose scores are all equal has nothing to teach and is skipped (DAPO). So is a group with fewer than two episodes fit to train on. |
| The fastest of the saturated | Episodes that reached everything their task has to give earned the same; the one that took the least scores a point more, and episodes that tie for fastest all do. The task says what saturated means and how long it took ([result conventions](episodes.md#result-conventions)); comparing across the group is done here. An episode that does not say its duration is not compared. `tie_break` turns this off. |
| What is trained on | Every segment of the episodes whose advantage is not zero, up to what the trainer can afford in a step (`Budget.segments`). Beyond that, segments are taken at even steps through the group, so that each episode and slot keeps its share, spread over its whole game. |

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
- **It is a fold.** A curriculum is rebuilt from the run's results when the loop starts: each goes to the row of
  its title, so records stay with their rows when a catalog changes. A choice is made with a random number
  generator seeded by the group's number, and is written down before it is acted on.

## The trainer

A [`Trainer`](../../guide/reference.md#trainer) takes a batch and makes new weights from given ones. It keeps
nothing between steps that it cannot be given again, so any trainer can take any step of any policy.

- **`budget`** ([`Budget`](../../guide/reference.md#budget)) is what the trainer can take: the longest segment, and
  how many segments a step can afford. It comes from the trainer's hardware, and nothing above the trainer chooses
  it. The algorithm selects within it, and a deployment makes the longest segment its channel's longest turn
  ([limits](channels.md#limits)).
- **`step(batch, seed=..., parent=..., into=...)`** trains on [`Weighted`](../../guide/reference.md#weighted)
  segments, starting from a [`Checkpoint`](../../guide/reference.md#checkpoint) (a version's files on this
  machine; none means the base model). It leaves the new weights in `into/weights`, as engines load them, and what
  a later step starts from (an optimizer's state, say) in `into/state`. It returns its metrics.
- **`StepFailed`** means the step produced no weights: the policy is as it was, and a later step may succeed.

[`Colocated`](../../guide/reference.md#colocated) wraps a trainer that shares an accelerator with the engines of
some channels. For each step it holds new requests back, waits for those in flight, puts the engines to sleep,
steps, wakes the engines and lets requests go on. A `guard` is called once the engines are asleep and raises if the
step should not start. It adds `waited_for_requests_seconds` and `update_seconds` to the step's metrics.

`LoraTrainer` is the trainer this repository gives: [LoRA trainer](../../implementations/rollout-lora.md).

## The record

For each group the run appends a [`Result`](../../guide/reference.md#result) to its `results` table as soon as its
episodes have ended: the row, the rewards, `solved` and durations of the episodes fit to train on, how many episodes
failed and why, how many segments were recorded and how many the algorithm found to train on (`segments`; none, and
`skipped` with its reason, if it found none), its notes, and how many rows are unlocked. The same line goes to the
job as a `result` note; each step goes as a `step` note with the version it made and its metrics, or its error.

What was done with a group is read by joining: `trained(ledger, run)` gives, for each group a step covers, that step
and the version it made or why it failed (a [`Trained`](../../guide/reference.md#trained)); the version's record
has the trainer's metrics. The report and the [monitor](monitor.md) read `results(ledger, run)` and that join.

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

## Imitation

`rollout_train.imitation` trains a policy to do, without being told how, what it did when it was told. An
environment that guides its agents reports, in each episode's result, the guidance its prompts carried, word for
word and by kind (`info["guidance"]`, for example `way` and `teamwork`).

- **`without(segment, texts, renderer)`** cuts guidance out of a segment: the fewest tokens before its first sampled
  token whose text holds it, and which encode back to themselves, are decoded, the guidance taken out, and the rest
  encoded again; the sampled tokens stay as they were, and their spans move with them. (A segment's tokens were
  joined from pieces encoded apart, so a whole prompt need not encode back to itself; a stretch of plain text
  does.) A segment where no such stretch is found is left out.
- **`examples(log, blobs, renderer, kinds=...)`** reads a job's log for the episodes that carried guidance of those
  kinds and solved their task, and gives their segments, cut, each weighted 1.
- **`imitate(policies, trainer, examples, ...)`** takes one step of a trainer whose objective is likelihood
  ([LoRA trainer](../../implementations/rollout-lora.md)) from the policy's newest version, and commits the next.

```bash
rollout imitate PROFILE --without way [--limit N]    # with the run stopped: it takes the policy's writer
```

Started again, the training loop serves the version imitation made and trains on from it.
