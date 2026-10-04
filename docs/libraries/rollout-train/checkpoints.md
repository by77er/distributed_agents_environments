# Checkpoints, runs and the ledger

Code: `rollout_train.checkpoints`, `rollout_train.registry`, `rollout_train.ledger` · See [training](training.md),
[datasets](datasets.md),
[`Checkpoints`](../../guide/reference.md#checkpoints), [`Checkpoint`](../../guide/reference.md#checkpoint),
[`Manifest`](../../guide/reference.md#manifest), [`Retention`](../../guide/reference.md#retention),
[`Ledger`](../../guide/reference.md#ledger)

What is trained is a graph of checkpoints. Every checkpoint grows from a base model, and says where it came from.

- A **checkpoint** has an id of its own: sixteen random letters (`kpqxwlmrtsnvoyzu`), shown by the shortest start of
  it that no other id shares, at least four letters (`kpqx`). The id never changes and means the same thing in every
  process and on every machine. It is what a channel serves, the name of the adapter an engine loads, and what a
  request to an engine names.
- A checkpoint says what it was made from: its **parents** (the checkpoint it was trained from first, then any others it
  learned from, such as a distillation's teachers, or the checkpoints that sampled a dataset it was trained on; none,
  the base model), the **base** model it adapts (`Qwen/Qwen3.5-9B`, its first parent's, or the one its line began
  from), and the **run** and **step** that made it. A checkpoint a supervised step on a [dataset](datasets.md) made
  names the dataset (`dataset`).
- Its **kind** says what its weights are: `lora`, an adapter over its base; or `full`, every weight of a model
  ([full weights](#full-weights-and-merges)).
- Its **depth** counts the steps from the base model along its first parents. It is the number stamped on the tokens
  it samples, and it grows along any line of training.
- Nothing about a checkpoint is a name. A checkpoint is found by its id, by where it came from, or by a
  [bookmark](#bookmarks): a name a checkpoint can be given, and that a run can carry forward as it trains.

```python
checkpoints = Checkpoints(FileLedger(directory / "ledger"), blobs)
checkpoint = await checkpoints.add(fence, new_id(), weights=step / "weights", state=step / "state",
                             run=run, step=18, parents=[parent.id], base="Qwen/Qwen3.5-9B")
head = await checkpoints.head(run)                                     # the newest checkpoint a run made
files = await checkpoints.files(head.weights, cache / head.id)         # on any machine: read from the blob store
```

## Checkpoints

- **A checkpoint is made by its append.** `add` keeps the checkpoint's files in the blob store and then appends the
  checkpoint to the ledger's `checkpoints` table, under the fence of the run that makes it. A writer that dies before the
  append has made nothing. The training loop decides a step's checkpoint id before the trainer is called (the step's
  `makes`), so a step taken again after a crash adds the same id and gets the checkpoint that is there.
- **A run trains from a checkpoint and goes on from the newest it made.** Its first step goes on from where it starts
  (`start`: any checkpoint, of this run or another; the base model if none); each step after from the checkpoint the one
  before made. Started again, a run goes on from its own newest checkpoint (`head`).
- **A fork is a run started from any checkpoint.** It shares its parent's blobs and costs nothing until it differs. Its
  checkpoints continue its parent's depth and base.
- **A run serves its newest checkpoint** on its channel, as the adapter named by its id (a full checkpoint, in place
  of the engines' weights: [full weights](#full-weights-and-merges)); the checkpoint before stays
  loaded until the turns that began under it finish. It writes down that its channel serves it
  ([what a channel should serve](channels.md#what-a-channel-should-serve)), so that engines on other machines follow. Runs on other channels serve their own. When its channel names
  a `reshard`, it serves the checkpoint's [resharded](#resharding) files, and waits for them.
- **Saves thin out with age.** `thin(fence, run, Retention(recent=2, every=20), keep)` deletes the files, weights
  and trainer state, of the checkpoints a run made, but the newest `recent` and every `every`-th by depth. Whatever
  retention says, a checkpoint keeps its files while it is served (and its parent, for a turn in progress), while any
  run starts from it, and while a bookmark names it (`keep`). A kept checkpoint can be served, compared, forked and
  trained on from where it was; a released one keeps its record (where it came from, what it was trained on, its
  metrics). A release is appended to `checkpoints/released` before its blobs are deleted. A released checkpoint reads
  with no `weights`, no `state` and the time it was `released`.
- **A blob is deleted only if nothing names it, and nothing put it lately.** What names a blob: a checkpoint that was
  not released (its weights and state), or what one was [resharded](#resharding) into. A checkpoint being added at the
  same moment may have found a blob already stored (content addressing: it is not written again) and not have appended
  itself yet, so `thin` also spares every blob put in the last `Retention.grace` seconds (an hour by default). Every
  put of a blob, one that writes it or one that finds it, sets the blob's time to now: a store of files sets the file's
  modification time (with a hard-linked file, the working copy's time too, since they are one file), and an S3 store
  copies the object onto itself, at most once a minute. A blob spared so is deleted by a later `thin`, of this run or
  any other. The grace must be longer than a checkpoint takes from its first file's put to its append, and a `thin`
  from reading what is named to its last delete, together.
  - A store of files moves a blob's file aside before deleting it and looks at its time again there: a put that found
    it in between is seen (the file is put back), and a put just after finds no file and writes it again.
  - An S3 store looks at an object's time and deletes it in two requests: a put that finds the object between the two
    is not seen.
  - Datasets', episodes' and batches' blobs are never deleted, and do not count as names: only checkpoints' files are
    deleted. One of them would be lost only if a released checkpoint had a file of exactly its bytes (a trajectory, a
    dataset's examples, a batch's list of segments).

## Full weights and merges

A full checkpoint holds every weight of a model, in the model's own layout (`config.json`, safetensors files, the
tokenizer), as a trainer of every weight writes it or a merge makes it.

- **A merge folds a LoRA checkpoint into the weights it was trained over** (`rollout_train.merging.merge`; `rollout
  merge`): each adapted layer's weight becomes W + (alpha / rank) · B · A, in a full checkpoint of its own. Its parent
  is the LoRA checkpoint, its base the model the merged weights came from, and no run made it. What folds an adapter
  in is named as `module:name` (`rollout_lora.merge:merge`). An adapter trained over a quantized model merges into
  the same model's unquantized weights (`--base Qwen/Qwen3.5-9B` for one trained over `…-AWQ-4bit`): the layers have
  the same names.
- **A run started from a full checkpoint trains over it.** Its trained channel's engines and its trainer load that
  checkpoint's files as their model (fetched to `directory/bases/ID`). With a LoRA trainer, the run's first step
  begins a new adapter over those weights, and its checkpoints' base is the full checkpoint's id; with a trainer of
  every weight, each step goes on from the weights before. A run started from an adapter over a full checkpoint
  serves that full checkpoint's files too.
- **Every weight is trained from full weights only.** A trainer of every weight refuses to start from an adapter:
  merge it first.
- **A full checkpoint is served in place**: the channel tells its engines to read the checkpoint's files into the
  model they hold ([publishing weights](channels.md#publishing-weights)), and the adapters loaded before are dropped.
- **Merging an adapter over a full checkpoint** folds it into that checkpoint's files; the merged checkpoint's base
  is still the model the line began from.

## Manifests

A checkpoint's weights and state are each a [`Manifest`](../../guide/reference.md#manifest): a map from the files of a
checkpoint, by their paths within it, to blobs.

- **Nothing is packed.** Each file is a blob of its own, as the trainer wrote it, so whoever needs a checkpoint
  reads the files it needs and loads them as they are.
- **Nothing is stored twice on one machine.** With a blob store of files (`FileBlobStore`) on the same filesystem, a
  checkpoint's files are kept as hard links to what the trainer wrote, and fetched as hard links to the blobs: the
  run's working copy and the blob are one file. A kept file is read-only from then on, since changing it would change
  the blob, and nothing writes a kept file in place: a trainer writes each step into a directory of its own. A blob
  is checked as it is linked out (`FileBlobStore.link`): its size always, and its hash too up to 64 MiB, so a blob
  changed in place by a writer that made it writable again (or wrote as root) is refused rather than served. Files are
  read and written one at a time, never a whole checkpoint in memory.
- **`layout`** says how the weights are divided among the files, where they are divided. A reader with the same
  division reads its own files and no others.
- **`files(manifest, directory)`** puts a manifest's files under a directory, from the blob store, if they are not
  there. The directory appears whole or not at all. A file the store lacks is read from the store of any run that
  has it, as each run's start says where its store is: a run can start from a checkpoint another run kept in its own
  blob store, or from a merge of one.

## Resharding

A trainer writes a checkpoint's weights in its own layout; the engines may load another (their division across devices,
a merged checkpoint, a format of their own). `rollout_train.resharding` rewrites a checkpoint's files into its engines'
layout, as a task of its own.

- **A layout is a function**, named as `module:name`: `layout(weights, into)` writes the engines' files under `into`
  from the trainer's under `weights`, and returns what it says about them. `verbatim` is the layout of engines that
  load the trainer's files as they are, such as vLLM with a LoRA adapter: each file is linked (or copied) as it is,
  so its blobs are the same.
- **`reshard(checkpoints, fence, checkpoint, layout, scratch)`** appends to `checkpoints/resharding` when it begins (the
  layout and the host), reads the checkpoint's weights to `scratch`, runs the layout, keeps the files it wrote in the
  blob store, and appends what it made to `checkpoints/resharded` (the layout, what it said, and the manifest of the
  files), both under the fence of the run that made the checkpoint. A checkpoint resharded before is not resharded again:
  `resharded(ledger, checkpoint)` is what it was resharded into. A released checkpoint cannot be resharded: its weights
  were deleted. The scratch files are removed once the files are kept.
- **As a Ray task.** `on_ray(ledger_at, blobs_at, fence, checkpoint, layout)` runs `reshard` as a Ray task of one CPU on
  the cluster the process is connected to (`connect(address)`, `disconnect()`): the worker opens the ledger and the
  blob store from where they are, and keeps its scratch files on disk under `~/.cache/rollout/resharding`. `connect`
  tells Ray not to start workers through `uv run`, so they run in the cluster's own environment.

A profile turns it on for a channel (`[channels.NAME] reshard = "rollout_train.resharding:verbatim"`); with `ray`, the
run's reshards are Ray tasks, else they run in its process, with scratch under `directory/resharding`
([deploying](../../guide/deploying.md#ray)). The [monitor](monitor.md)'s checkpoints graph shows a checkpoint resharding and
resharded.

## The ledger

A [`Ledger`](../../guide/reference.md#ledger) is append-only tables and fences. It is the only state an
orchestrator has: something that must survive its own death appends what it decided and what happened, and on
starting again reads it back.

| | |
|---|---|
| Keys | A record is appended under a key, and a table has each key once. Appending under a key that is there changes nothing and says so. An action that is written down before it is taken can be taken again after a crash without being done twice. |
| Fences | Whoever means to write takes the fence of a scope: a number higher than any taken before. An append that carries an older fence is refused (`Fenced`). |
| Order | A table reads in the order its appends took effect, whichever scopes made them (every runner appends to a run's claims, every run to `checkpoints`). |

An append answers whether it wrote its record. `appended(ledger, table, key, record, fence)` says also what the table
holds under the key when it did not (`Appended(wrote, record)`): both ledgers answer that from the append itself
(`append_returning`), and of any other ledger the table is read back. A maker that loses uses the winner's record, as a
suite's second maker does ([evals](evals.md)).

Two ledgers are provided:

| Ledger | Keeps | Shared by |
|---|---|---|
| `FileLedger(directory)` | each table as a file of JSON lines | the processes of one machine |
| `DatabaseLedger(url)` (`rollout_train.database`, with the `durable` extra) | every table in two SQL tables, `ledger_records` and `ledger_fences` (`sqlite:///path`, `~` allowed, or `postgresql://…`) | every run and machine using the database |

- **`FileLedger`** holds a lock on its directory for every operation, in a thread, so the event loop goes on while it
  waits. An append is on disk (`fsync`) before it is acknowledged. A last line left unfinished, by a writer that died
  mid-record or a full disk, was never acknowledged: the next append removes it first (or ends it, if it holds a whole
  record), so nothing is glued to it. `fences.json` is replaced whole: written beside it, put on disk, and renamed over
  it, so a crash while taking a fence leaves the fences as they were. Each process keeps which keys a table has, and
  reads a table's file again only as far as it grew since (or whole, if the file was replaced).
- **`DatabaseLedger`** takes a fence and appends in transactions that hold the scope's lock. An append on Postgres
  holds its table's lock too, while it numbers its record after the table's last: records are numbered in the order
  they commit, one number each. SQLite runs one write at a time.

A profile says which (`ledger`), and an open profile writes where it is into the run's directory (`ledger.json`), so
that the monitor, the report and imitation open the same one from the directory alone (`of_run`). Each run is kept
under its own id ([runs](#runs)), so the runs sharing a ledger keep apart, and they share one graph of checkpoints.

Moving a ledger to Postgres (or from files to SQLite) is a copy and a changed URL, with nothing writing to it:

```bash
rollout ledger copy ~/.cache/rollout/runs/curriculum-9 postgresql://trainer@db-1/rollout --point
```

then the profile's `[ledger]` names the same `url`. `copy` (`rollout_train.database.copy`) takes any ledger, a run's
directory, files or a database, into a database that has none of its tables yet: every record under its key, in the
order it was appended, and every fence, so a writer from before the move is still shut out. `--point` makes the run's
directory name the copy at once (an open profile writes the same when it starts). The runs, bookmarks and dataset
names registered beside the source are registered beside the copy.

A ledger also lists its tables and its scopes' fences: `runs_in` reads from the tables' names which runs it has, and
`checkpoints_in` reads every checkpoint; the [monitor](monitor.md) shows them. The training loop's tables are described
under [dying and starting again](training.md#dying-and-starting-again).

## The registry

What runs, checkpoints, datasets and suites are called is kept beside the ledger, in the registry
(`rollout_train.registry`): `registry.json` beside a ledger of files, the `runs`, `bookmarks`, `dataset_names` and
`suite_names` tables in a database ledger's database (`DatabaseRegistry`). It is ordinary state, changed in place, not part of the ledger's append-only record; nothing
the ledger keeps is under a name, so naming anything again moves nothing. Two more kinds of ordinary state are kept
beside the ledger the same way: the runners' and launchers' heartbeats (`presence.json`, the `presence` table;
[heartbeats](rollouts.md#heartbeats)) and the runs asked for (`launches.json`, the `launches` table;
[launchers](../../guide/deploying.md#launchers)). A name says none of `/`, `@` and `:` (they
are what a reference is made of), and is not `base`.

### Runs

Every run has an id that never changes and a name that can be chosen and changed. Everything kept of it is under its
id (`runs/ID/...`, and the `run` of each checkpoint it made). A run's name is one no other run has, as its name or as
its id, so that either finds one run.

| | |
|---|---|
| A new run | `rollout train` registers it the first time it starts in a directory, under `--name` (by default the directory's name), with a new id (`run_` and a ULID) that the directory's `run.json` keeps (`run_of`) |
| Renaming | `rollout rename WHO NAME --ledger WHERE`, where `WHO` is the run's name or its id and `WHERE` a run's directory, a ledger's directory or a database's URL; or `Registry.rename` |

### Bookmarks

A bookmark is a name for a checkpoint. It is made, moved and taken away by hand (`rollout bookmark`), or carried by a
run: a profile's `[trainer] bookmark` moves it to each checkpoint the run makes, once the checkpoint is served. A checkpoint
a bookmark names keeps its files. A checkpoint needs no bookmark: it is shown by where it came from.

### Dataset names

A [dataset](datasets.md) can be given a name when it is made (`rollout dataset make … --name NAME`, or
`Registry.name_dataset`). A name says one dataset for good, as the dataset never changes; another dataset cannot be
given it.

### Suite names

A [suite](evals.md#versions)'s name points to one of its versions, by id (`NAME@NUMBER`): the newest, moved there by
each edit (`Registry.point_suite(name, version, forward=True)`, which moves it only to a later version, so two edits at
once leave it at the newer). `Registry.suites()` lists where each points. A version never changes; the name moves. A
suite never edited needs none: its name is its version 1.

## References

Wherever a checkpoint is asked for (a profile's `[trainer] start`, `rollout bookmark`), it is named by a reference,
read in this order (`resolved`):

| Reference | The checkpoint |
|---|---|
| `base` | none: the base model |
| a bookmark's name | the checkpoint it names |
| `RUN:STEP` | the checkpoint the run (by its name or its id) made at that step |
| `RUN` | the newest checkpoint the run made |
| an id, or the start of one | that checkpoint (a start of at least four letters that no other id begins with) |

## The command line

```bash
rollout checkpoints --ledger RUN                        # every checkpoint, newest first: id, depth, run:step, parents, bookmarks
rollout bookmark diamonds curriculum-9:20 --ledger RUN    # name a checkpoint, or move the bookmark there
rollout bookmark diamonds --delete --ledger RUN      # take the bookmark away (the checkpoint stays)
rollout rename curriculum-9 "diamonds, guided" --ledger RUN   # call a run something else
rollout merge diamonds --base Qwen/Qwen3.5-9B --bookmark diamonds-merged   # fold an adapter in: a full checkpoint
```

A new run started from a checkpoint is a fork: its profile's `[trainer] start = "diamonds"` (any reference), and
`rollout train … --directory NEW --name "diamonds, unguided"`.
