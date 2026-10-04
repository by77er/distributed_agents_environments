# Evals

An **eval** plays a frozen list of starts with one checkpoint (or the base model), trains on nothing, and records
how each episode went. Every checkpoint that plays the same list plays the same starts, so checkpoints compare start
for start. The code is `rollout_train.evals`; the commands are `rollout suite` and `rollout eval`; the monitor's
**Evals** page shows them and asks for new ones.

## Suites

A **suite** is a named list of starts of a catalog's rows: for each row it names (every row, by default) and each
seed, the row's start drawn with `random.Random(seed)`. It is made once and never changed. To play other starts,
make another suite.

```bash
uv run rollout suite make words-v1 --catalog minecraft_team.catalog:catalog --rows chests,diamonds --seeds 1,2,3 --ledger L
uv run rollout suite list --ledger L
```

`make_suite(ledger, name, catalog_name, catalog, rows=…, seeds=…)` writes it under the fence `suites/NAME`. A name
that is taken, a row the catalog does not have, or no seeds is refused. `suite_of(ledger, name)` reads it back, and
`suites_in(ledger)` lists every suite's name. The suite is kept in two tables:

| Table | Key | Holds |
|---|---|---|
| `evaluations/SUITE/suite` | `suite` | its catalog (`module:name`), when it was made, its rows and its seeds |
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

`evaluate(catalog, checkpoints, run=…, suite=…, subject=…, …)` does the work:

1. It writes the run's plan (its program and binding, as a training run does), its start, and a record of who plays.
2. If a checkpoint plays, it fetches the checkpoint's files into the run's `checkpoints/` (resharded into the
   channel's layout when the profile has one, [resharding](checkpoints.md#resharding)) and publishes them on the
   channel. A checkpoint whose weights were released cannot play.
3. It asks for one group per start, `--episodes` episodes each, in the run's `groups` table. Episode runners claim and
   play them as they play any run's ([rollouts](rollouts.md)), starting with the runner the eval's profile opens.
4. As each group's episodes end, it records each episode's outcome, and the group's result in the run's `results`.
   The result's `skipped` is "an evaluation trains on nothing", so the run's page reads like any run's.

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
suite's catalog, checks the launch as it checks a run's, and refuses an unknown suite (404) or fewer than one episode
a start (409). A launcher that offers the profile and the catalog claims it and starts

```bash
python -m rollout_train.cli eval PROFILE SUITE --directory RUNS/NAME-ID --name NAME --episodes N --checkpoint REF
```

as a process of its own, or as a Ray job with `--ray` ([launchers](../../guide/deploying.md#launchers)). The suite's
page shows the launch and how it goes. A checkpoint's page lists the suites it played, each with a link to the suite.
