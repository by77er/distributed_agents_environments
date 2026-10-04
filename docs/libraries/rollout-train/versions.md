# Versions, runs and the ledger

Code: `rollout_train.versions`, `rollout_train.registry`, `rollout_train.ledger` · See [training](training.md),
[`Versions`](../../guide/reference.md#versions), [`Version`](../../guide/reference.md#version),
[`Manifest`](../../guide/reference.md#manifest), [`Retention`](../../guide/reference.md#retention),
[`Ledger`](../../guide/reference.md#ledger)

What is trained is a graph of versions. Every version grows from a base model, and says where it came from.

- A **version** has an id of its own: sixteen random letters (`kpqxwlmrtsnvoyzu`), shown by the shortest start of
  it that no other id shares, at least four letters (`kpqx`). The id never changes and means the same thing in every
  process and on every machine. It is what a channel serves, the name of the adapter an engine loads, and what a
  request to an engine names.
- A version says what it was made from: its **parents** (the version it was trained from first, then any others it
  learned from, such as a distillation's teachers; none, the base model), the **base** model it adapts
  (`Qwen/Qwen3.5-9B`, its first parent's, or the one its line began from), and the **run** and **step** that made it.
- Its **depth** counts the steps from the base model along its first parents. It is the number stamped on the tokens
  it samples, and it grows along any line of training.
- Nothing about a version is a name. A version is found by its id, by where it came from, or by a
  [bookmark](#bookmarks): a name a version can be given, and that a run can carry forward as it trains.

```python
versions = Versions(FileLedger(directory / "ledger"), blobs)
version = await versions.add(fence, new_id(), weights=step / "weights", state=step / "state",
                             run=run, step=18, parents=[parent.id], base="Qwen/Qwen3.5-9B")
head = await versions.head(run)                                     # the newest version a run made
files = await versions.files(head.weights, cache / head.id)         # on any machine: read from the blob store
```

## Versions

- **A version is made by its append.** `add` keeps the checkpoint's files in the blob store and then appends the
  version to the ledger's `versions` table, under the fence of the run that makes it. A writer that dies before the
  append has made nothing. The training loop decides a step's version id before the trainer is called (the step's
  `makes`), so a step taken again after a crash adds the same id and gets the version that is there.
- **A run trains from a version and goes on from the newest it made.** Its first step goes on from where it starts
  (`start`: any version, of this run or another; the base model if none); each step after from the version the one
  before made. Started again, a run goes on from its own newest version (`head`).
- **A fork is a run started from any version.** It shares its parent's blobs and costs nothing until it differs. Its
  versions continue its parent's depth and base.
- **A run serves its newest version** on its channel, as the adapter named by its id; the version before stays
  loaded until the turns that began under it finish. Runs on other channels serve their own.
- **Saves thin out with age.** `thin(fence, run, Retention(recent=2, every=20), keep)` deletes the files, weights
  and trainer state, of the versions a run made, but the newest `recent` and every `every`-th by depth. Whatever
  retention says, a version keeps its files while it is served (and its parent, for a turn in progress), while any
  run starts from it, and while a bookmark names it (`keep`). A kept version can be served, compared, forked and
  trained on from where it was; a released one keeps its record (where it came from, what it was trained on, its
  metrics). A release is appended to `versions/released` before its blobs are deleted, and a blob is deleted only if
  no version still names it. A released version reads with no `weights`, no `state` and the time it was `released`.

## Manifests

A version's weights and state are each a [`Manifest`](../../guide/reference.md#manifest): a map from the files of a
checkpoint, by their paths within it, to blobs.

- **Nothing is packed.** Each file is a blob of its own, as the trainer wrote it, so whoever needs a checkpoint
  reads the files it needs and loads them as they are.
- **`layout`** says how the weights are divided among the files, where they are divided. A reader with the same
  division reads its own files and no others.
- **`files(manifest, directory)`** puts a manifest's files under a directory, from the blob store, if they are not
  there. The directory appears whole or not at all.

## The ledger

A [`Ledger`](../../guide/reference.md#ledger) is append-only tables and fences. It is the only state an
orchestrator has: something that must survive its own death appends what it decided and what happened, and on
starting again reads it back.

| | |
|---|---|
| Keys | A record is appended under a key, and a table has each key once. Appending under a key that is there changes nothing and says so. An action that is written down before it is taken can be taken again after a crash without being done twice. |
| Fences | Whoever means to write takes the fence of a scope: a number higher than any taken before. An append that carries an older fence is refused (`Fenced`). |

Two ledgers are provided:

| Ledger | Keeps | Shared by |
|---|---|---|
| `FileLedger(directory)` | each table as a file of JSON lines | the processes of one machine |
| `DatabaseLedger(url)` (`rollout_train.database`, with the `durable` extra) | every table in two SQL tables, `ledger_records` and `ledger_fences` (`sqlite:///path`, `~` allowed, or `postgresql://…`) | every run and machine using the database |

A profile says which (`ledger`), and an open profile writes where it is into the run's directory (`ledger.json`), so
that the monitor, the report and imitation open the same one from the directory alone (`of_run`). Each run is kept
under its own id ([runs](#runs)), so the runs sharing a ledger keep apart, and they share one graph of versions.

Moving a ledger to Postgres (or from files to SQLite) is a copy and a changed URL, with nothing writing to it:

```bash
rollout ledger copy ~/.cache/rollout/runs/curriculum-9 postgresql://trainer@db-1/rollout --point
```

then the profile's `[ledger]` names the same `url`. `copy` (`rollout_train.database.copy`) takes any ledger, a run's
directory, files or a database, into a database that has none of its tables yet: every record under its key, in the
order it was appended, and every fence, so a writer from before the move is still shut out. `--point` makes the run's
directory name the copy at once (an open profile writes the same when it starts). The runs and bookmarks registered
beside the source are registered beside the copy.

A ledger also lists its tables and its scopes' fences: `runs_in` reads from the tables' names which runs it has, and
`versions_in` reads every version; the [monitor](monitor.md) shows them. The training loop's tables are described
under [dying and starting again](training.md#dying-and-starting-again).

## The registry

What runs and versions are called is kept beside the ledger, in the registry (`rollout_train.registry`):
`registry.json` beside a ledger of files, the `runs` and `bookmarks` tables in a database ledger's database
(`DatabaseRegistry`). It is ordinary state, changed in place, not part of the ledger's append-only record; nothing
the ledger keeps is under a name, so naming anything again moves nothing. Two more kinds of ordinary state are kept
beside the ledger the same way: the runners' and launchers' heartbeats (`presence.json`, the `presence` table;
[heartbeats](rollouts.md#heartbeats)) and the runs asked for (`launches.json`, the `launches` table;
[launchers](../../guide/deploying.md#launchers)). A name says none of `/`, `@` and `:` (they
are what a reference is made of), and is not `base`.

### Runs

Every run has an id that never changes and a name that can be chosen and changed. Everything kept of it is under its
id (`runs/ID/...`, and the `run` of each version it made). A run's name is one no other run has, as its name or as
its id, so that either finds one run.

| | |
|---|---|
| A new run | `rollout train` registers it the first time it starts in a directory, under `--name` (by default the directory's name), with a new id (`run_` and a ULID) that the directory's `run.json` keeps (`run_of`) |
| Renaming | `rollout rename WHO NAME --ledger WHERE`, where `WHO` is the run's name or its id and `WHERE` a run's directory, a ledger's directory or a database's URL; or `Registry.rename` |
| From before the registry | a run recorded under a key before there was a registry keeps that key as its id, and is registered under it as its name the first time it is asked for |

### Bookmarks

A bookmark is a name for a version. It is made, moved and taken away by hand (`rollout bookmark`), or carried by a
run: a profile's `[trainer] bookmark` moves it to each version the run makes, once the version is served. A version
a bookmark names keeps its files. A version needs no bookmark: it is shown by where it came from.

## References

Wherever a version is asked for (a profile's `[trainer] start`, `rollout bookmark`), it is named by a reference,
read in this order (`resolved`):

| Reference | The version |
|---|---|
| `base` | none: the base model |
| a bookmark's name | the version it names |
| `RUN:STEP` | the version the run (by its name or its id) made at that step |
| `RUN` | the newest version the run made |
| an id, or the start of one | that version (a start of at least four letters that no other id begins with) |

## The command line

```bash
rollout versions --ledger RUN                        # every version, newest first: id, depth, run:step, parents, bookmarks
rollout bookmark diamonds curriculum-9:20 --ledger RUN    # name a version, or move the bookmark there
rollout bookmark diamonds --delete --ledger RUN      # take the bookmark away (the version stays)
rollout rename curriculum-9 "diamonds, guided" --ledger RUN   # call a run something else
```

A new run started from a version is a fork: its profile's `[trainer] start = "diamonds"` (any reference), and
`rollout train … --directory NEW --name "diamonds, unguided"`.
