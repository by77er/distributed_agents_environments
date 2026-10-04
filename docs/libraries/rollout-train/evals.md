# Evals

An **eval** plays a frozen list of starts with one checkpoint (or the base model), trains on nothing, and records
how each episode went. Every checkpoint that plays the same list plays the same starts, so checkpoints compare start
for start. The code is `rollout_train.evals`; the commands are `rollout suite` and `rollout eval`; the monitor's
**Evals** page shows them and asks for new ones.

## Suites

A **suite** is a named list of starts of an environment's rows: for each row it names (every row, by default) and each
seed, the row's start drawn with `random.Random(seed)`. It is made once and never changed. To play other starts,
make another suite.

```bash
uv run rollout suite make words-v1 --environment minecraft_team.environment:environment --rows chests,diamonds --seeds 1,2,3 --ledger L
uv run rollout suite list --ledger L
```

`make_suite(ledger, name, environment_name, environment, rows=…, seeds=…)` writes it under the fence `suites/NAME`. A name
that is taken, a row the environment does not have, or no seeds is refused. `suite_of(ledger, name)` reads it back, and
`suites_in(ledger)` lists every suite's name. The suite is kept in two tables:

| Table | Key | Holds |
|---|---|---|
| `evaluations/SUITE/suite` | `suite` | its environment (`module:name`), when it was made, its rows and its seeds |
| `evaluations/SUITE/starts` | `1` to `N` | each start: the row's key (`task`) and title, the seed, and the start's `parameters` |

## An eval

```bash
uv run rollout eval profile.toml words-v1 --checkpoint diamonds --episodes 4 --directory EVAL
```

An eval is a run of its own, with its own id and name in the registry ([runs](checkpoints.md#runs)) and its own
fence. Its start record says `kind: eval`, the suite, and the checkpoint. `--checkpoint` takes any
[reference](checkpoints.md#references): a bookmark, `RUN:STEP`, `RUN`, or a checkpoint's id. A bookmark is read
once, when the eval starts. Without `--checkpoint`, the base model of the profile's trained channel (or its first
channel) plays.

`rollout eval` opens the profile without its trainer (`Profile.open(training=False)`): no trainer is made, and the
engines never sleep for one. What the trainer's `start` would be is the checkpoint, so the trained channel's engines
load what it is served over: the model; a full checkpoint's files; or, for an adapter over a full checkpoint, that
checkpoint's files, with the adapter loaded over them. The engines keep the profile's sizes (`--set` changes them).
Without a trainer, the channel's longest turn is what its engines accept, not the trainer's longest segment. When it
ends, however it ends, it deletes what it fetched to serve the checkpoint (its directory's `bases/`, `checkpoints/`
and `resharding/`): a full checkpoint's files are a whole model's.

`evaluate(environment, checkpoints, run=…, suite=…, subject=…, …)` does the work:

1. It writes the run's plan (its program and binding, as a training run does), its start, and a record of who plays.
2. If a checkpoint plays, it fetches the checkpoint's files into the run's `checkpoints/` (resharded into the
   channel's layout when the profile has one, [resharding](checkpoints.md#resharding)) and publishes them on the
   channel. A checkpoint whose weights were released cannot play. With `publish=None`, the channel serves the
   checkpoint already (a training run's newest), and nothing is fetched.
3. It asks for one group per start, `--episodes` episodes each, in the run's `groups` table. Episode runners claim and
   play them as they play any run's ([rollouts](rollouts.md)), starting with the runner the eval's profile opens.
4. As each group's episodes end, it records each episode's outcome, and the group's result in the run's `results`.
   The result's `skipped` is "an evaluation trains on nothing", so the run's page reads like any run's.
5. It returns how many episodes were played and solved, the mean reward, and each start's `Result` (`results`), as
   first recorded.

| Table | Key | Holds |
|---|---|---|
| `evaluations/SUITE/EVAL/subject` | `subject` | who played (`kind`: `checkpoint` or `model`), the checkpoint's id, the base model, episodes a start, who asked, the eval's run |
| `evaluations/SUITE/EVAL/results` | `START-EPISODE` | each episode: its `run_id`, reward, whether it solved the start, its duration, outcome and detail |

`EVAL` is the eval's run id, so two evals of one checkpoint are kept apart. An eval started again, in the same
directory, goes on from where it was: groups it asked for and outcomes it recorded are not done twice. It prints how
many episodes were solved of how many were played, and the mean reward.

## Asked for from the page

The monitor's **Evals** page (`#/evals`) lists every suite and every eval. A suite's page (`#/evals/SUITE`) shows:

- each subject's episodes at each start, with totals;
- two subjects compared at the starts both played;
- a **Run this suite** form, which asks a launcher to play the suite.

The form takes the checkpoint (the base model, a bookmark, or any checkpoint whose weights are kept, by where it came
from and its short id), the episodes a start, the profile, and the eval's name. It posts a launch of kind `eval`
(`POST /api/launches`, `{"kind": "eval", "suite", "profile", "name", "start", "episodes"}`). The monitor fills in the
suite's environment, checks the launch as it checks a run's, and refuses an unknown suite (404) or fewer than one episode
a start (409). A launcher that offers the profile and the environment claims it and starts

```bash
python -m rollout_train.cli eval PROFILE SUITE --directory RUNS/NAME-ID --name NAME --episodes N --checkpoint REF
```

as a process of its own, or as a Ray job with `--ray` ([launchers](../../guide/deploying.md#launchers)). The suite's
page shows the launch and how it goes. A checkpoint's page lists the suites it played, each with a link to the suite.

## Evals during training

A profile's `[evals]` table has a training run evaluate its own checkpoints as it makes them
([deploying](../../guide/deploying.md#the-file)):

```toml
[evals]
suite = "words-v1"      # made beforehand with `rollout suite make`, in the run's ledger
every = 2               # the checkpoint of every second step (1 unless it says otherwise)
episodes = 1            # episodes of each start (1)
```

`rollout train` reads the suite (a suite the ledger does not have stops it before it starts) and passes the loop a
[`Schedule`](../../guide/reference.md#schedule): the suite, its environment, how often, how many episodes, and the binding
of the suite's program to the trained channel. After a step whose number is a multiple of `every` makes its checkpoint
and serves it, the loop:

1. Registers the eval's run, `NAME-eval-STEP` (its id kept in `directory/evals/RUN-eval-STEP/run.json`, so the same
   run each time it is asked for), and adds it to the runs the profile's episode runner plays.
2. Calls `evaluate` with that checkpoint and `publish=None`: the channel already serves it. The eval's start record
   says `by` (the training run) and `step`, and `from` is empty, so whether the checkpoint keeps its files is the
   run's retention's ([saves thin out](training.md#dying-and-starting-again)), not the eval's.
3. Waits until every start has been played. The next step is not taken until then, so the channel serves that
   checkpoint all that time. Training groups go on being played meanwhile, under that same checkpoint, and queue for
   the next step.
4. Writes the eval to the run's `evals` table, under the step's number: the suite, the checkpoint, the eval's run,
   episodes played and solved, the mean reward. Then it calls `curriculum.evaluated(suite, checkpoint, results)` with
   each start's `Result`.

| Table | Key | Holds |
|---|---|---|
| `runs/RUN/evals` | step | the suite, the checkpoint, the eval's run (`run`), `played`, `solved`, `reward`, and when it ended (`at`) |

The eval is an eval like any other: it is listed on the **Evals** page and the suite's grid, with its own run's page.
Started again, the loop folds each eval in `evals` into its curriculum (reading the eval's `results`), and if it died
during an eval, it finishes that eval (the checkpoint is the run's newest, served when the loop starts) before it
decides anything else.

[`Curriculum.evaluated`](../../guide/reference.md#curriculum) keeps the newest eval of each suite in
`curriculum.evaluations`, by the suite's name: the checkpoint, and its result at each start. Sampling does not read
it; a curriculum that gates rows on how a checkpoint did reads it there.

### Why in the loop, not as a launch

A run could instead ask for an eval launch of each checkpoint, which a launcher would start as `rollout eval`. That
fits a cluster: the eval runs on other engines while training goes on, and Ray places it. On one GPU, with the
trainer colocated, it does not: a second process needs engines of its own beside the run's, in memory sized for the
run's engines and the trainer, and a launcher must be running. The loop already serves the new checkpoint on engines
that are awake between steps, and its runner can play the eval's episodes beside the run's, so playing the suite
there needs no memory and no engines of its own. The cost is time: the next step waits for the eval (training groups
are still played meanwhile, so the engines stay busy). A run whose evals should not hold up its steps asks for
launches instead, from the page or with `rollout eval`.

