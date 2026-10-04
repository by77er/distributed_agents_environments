# Evals

An **eval** plays a frozen list of starts with one checkpoint (or the base model), trains on nothing, and records
how each episode went. Every checkpoint that plays the same list plays the same starts, so checkpoints compare start
for start. The code is `rollout_train.evals`; the commands are `rollout suite` and `rollout eval`; the monitor's
**Evals** page shows them and asks for new ones.

## Suites

A **suite** is a named list of starts of an environment's rows, made once and never changed. To play other starts,
make another suite. Most suites are an environment's **eval data** (`Environment.evals()`, [train and
eval](rollouts.md#train-and-eval)): each of its named lists of starts is frozen as a suite of that name the first time
it is played (or made), and training never draws those starts. A suite can be made by hand too, of rows and seeds:
for each row it names (every row, by default) and each seed, the row's start drawn with `random.Random(seed)`. Nothing
keeps a hand-made suite's starts out of training.

```bash
uv run rollout suite list --environment minecraft_team.environment:environment --ledger L  # its eval data too
uv run rollout suite make teams-every-task --environment minecraft_team.environment:environment --ledger L
uv run rollout suite make words-v1 --environment E --rows chests,diamonds --seeds 1,2,3 --ledger L   # by hand
```

`suite_for(ledger, name, environment_name, environment)` is the suite of that name in the ledger, or else the
environment's eval data of that name, frozen now; a suite in the ledger of another environment, or a name neither has,
is refused. `make_suite(ledger, name, environment_name, environment, rows=…, seeds=…)` makes one by hand (or, with
`starts=`, of given starts) under the fence `suites/NAME`. A name that is taken, a row the environment does not have,
or no seeds is refused. `suite_of(ledger, name)` reads one back, and `suites_in(ledger)` lists every suite's name. The
suite is kept as one record:

| Table | Key | Holds |
|---|---|---|
| `evaluations/SUITE/suite` | `suite` | its environment (`module:name`) and its `version` then, when it was made, its rows and seeds, whether it is `held_out` (the environment's eval data), and its `starts` in order: each the row's key (`task`) and title, the seed, and the start's `parameters` |

Two makers of one suite at once leave one of their suites whole. The second to take the fence shuts the first out
(`Fenced`); a maker whose record finds the suite's already there plays the one there, not its own. `suite_for` makes
the environment's eval data again after a `Fenced` until it finds a suite of the name, and plays that one. A suite made
before suites were one record keeps its starts in a table of their own (`evaluations/SUITE/starts`, by number from 1),
and reads the same.

## An eval

```bash
uv run rollout eval profile.toml words-v1 --checkpoint diamonds --episodes 4 --directory EVAL
uv run rollout eval profile.toml teams-every-task --environment minecraft_team.environment:environment --directory EVAL
```

With `--environment`, a suite not made yet is that environment's eval data of the name, frozen as the eval starts.
An eval is a run of its own, with its own id and name in the registry ([runs](checkpoints.md#runs)) and its own
fence. Its start record says `kind: eval`, the suite, the checkpoint, and the environment's version and description. `--checkpoint` takes any
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
suite's environment (a suite not made yet is an environment's eval data, and the launch says the `environment`),
checks the launch as it checks a run's, and refuses an unknown suite (404) or fewer than one episode a start (409). A
launcher that offers the profile and the environment claims it and starts

```bash
python -m rollout_train.cli eval PROFILE SUITE --directory RUNS/NAME-ID --name NAME --episodes N --environment E \
    --checkpoint REF
```

as a process of its own, or as a Ray job with `--ray` ([launchers](../../guide/deploying.md#launchers)). The suite's
page shows the launch and how it goes.

A checkpoint's page (`#/checkpoint/ID`) has the same form the other way round, **Run an eval**: it takes the suite
(by default the first a launcher alive can play), the episodes per start, the profile and the name, and posts the same
launch with the checkpoint as `start`; the launches of evals of it follow, with how each goes. The page lists every
eval the checkpoint had, by hand or by its run's schedule (the suite, the share solved where its episodes say, the
mean reward, the episodes, who asked for it and when), each opening the eval's own page (`#/eval/RUN`: who played,
the suite, who asked, its score, and how it did at each start), and charts each suite's score along its line from the
base model ([scores along a line](monitor.md#scores-along-a-line)).

On a suite's page, a column stands for each subject: the base model first, then each run's checkpoints under the run's
name, runs in the order their checkpoints grew and within a run by depth.

## Evals during training

A profile's `[evals]` table has a training run evaluate its own checkpoints as it makes them
([deploying](../../guide/deploying.md#the-file)):

```toml
[evals]
suite = "words-held-out"  # the environment's eval data of that name, frozen on first use; or a suite made by hand
every = 2               # the checkpoint of every second step (1 unless it says otherwise)
episodes = 1            # episodes of each start (1)
```

`rollout train` finds the suite with `suite_for`: the ledger's suite of that name, or else the environment's eval data
of that name, frozen now (a name neither has stops it before it starts). It passes the loop a
[`Schedule`](../../guide/reference.md#schedule): the suite, its environment, how often, how many episodes, and the
binding of the suite's program to the trained channel. After a step whose number is a multiple of `every` makes its
checkpoint and serves it, the loop:

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

The eval is an eval like any other: it is listed on the **Evals** page, the suite's grid (under its run's name, with
its step) and the checkpoint's page, with a page of its own.

The evals are among a run's changeable settings ([changing a running run's
settings](training.md#changing-a-running-runs-settings)): set when the run is launched (the **New run** form's suite,
every N steps and episodes per start), and changed while it runs from its page. Each step's record says the evals it
was decided with, and those decide whether its checkpoint is evaluated: a change made while a step is taken applies
from the next. `rollout train` gives the loop `scheduled`, which resolves a suite by name as the profile's is
(`suite_for`: the ledger's, or the environment's eval data of that name, frozen on first use).
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

