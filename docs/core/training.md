# Training

Status: **Working** (2026-10-02) · Code: `rollout.training` · See [rollouts](rollouts/README.md), [episodes](trajectories/README.md)

The loop, the group algorithm and the curriculum, written against `Jobs`, `Trainer` and `Store` only, and a reference
trainer for 4-bit checkpoints with LoRA.

```python
await train(jobs, catalog, trainer, Directory(run), channel="policy", groups=100)
```

## The loop

`train` keeps two groups of episodes outstanding. The job starts the second as soon as there is room beside what is
left of the first, and each group is trained on when its last episode ends, while the next group's episodes run on.
For each group: record it in the curriculum, compute what to train on, step the trainer, publish the weights, write
a line to `metrics.jsonl` and the same line to the job as an `iteration` note.

A failed update is logged and the weights stay as they were; three in a row stop the loop. A loop started again over
the same store goes on after the last group logged, on starts drawn anew.

## The algorithm (`Grpo`)

| | |
|---|---|
| Advantages | An episode's score minus its group's mean, with no division by the group's spread (Dr. GRPO). Every token the policy sampled in the episode gets it; with several model slots, every slot's. |
| Dynamic sampling | A group whose scores are all equal has nothing to teach and is skipped (DAPO). |
| The fastest of the saturated | Episodes that reached everything their task has to give earned the same; the one that took the least scores a point more. The task says what saturated means and how long it took (`info`); comparing across the group is done here. |
| What is trained on | Every sequence of the episodes whose advantage is not zero, up to what the trainer can afford in a step (`Trainer.budget`); beyond that, sequences are taken at even steps through the group, so that each episode and slot keeps its share. |

## The curriculum

A row's weight is the share of its recent groups whose rewards differed (a moving average), plus a little for every
unlocked row; untried rows come first, and a row whose group is still running is not picked again until that group
is recorded. Rows unlock in the catalog's order: the first three, and four past the hardest one solved at least half
the time. A row that could not be set up counts as tried only after three groups of it failed: the next start may be
one it can be set up from. Records are kept with the rows' titles, so that they stay with their rows when a catalog
changes.

## The trainer

```python
class Trainer(Protocol):
    budget: Budget                                                    # the longest sequence, and how many a step can afford
    async def step(self, batch: Sequence[Weighted], *, seed: int) -> Step: ...   # Step(adapter, path, metrics)
```

The budget comes from the trainer's hardware; nothing above it chooses it, and each line of metrics says how many
sequences were recorded and how many were trained on.

`LoraTrainer` trains a LoRA adapter over the checkpoint the engines serve, each step in a fresh process that loads
the previous adapter and optimizer state, makes one pass, saves and exits. `Colocated` wraps a trainer that shares
its accelerator with the engines.

The step is PPO's clipped objective against the logprobs recorded while sampling, with DAPO's wider upper clip
(0.8 to 1.28), a token-level mean over each minibatch of 4,096 sampled tokens, and no KL term. The pass stops if a
minibatch finds the policy more than 0.02 nats a token from where the first one found it. It reports `kl_floor` (the
first minibatch: the engine's and the trainer's numerical difference, and how stale the turns are) and `kl_moved`
(the last minibatch less the floor: how far the step moved the policy).

## Reporting

`rollout report RUN CATALOG` writes `progress.png` and `progress.md` into the run's directory: the climb through the
curriculum, every group's rewards, and what each update did. With `--watch` it does so after every group; with a
Discord webhook (`--webhook`, or `DISCORD_WEBHOOK_URL`) it posts both there. Needs the `report` extra.
