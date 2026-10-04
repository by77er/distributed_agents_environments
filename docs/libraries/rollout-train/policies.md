# Policies, versions and the ledger

Code: `rollout_train.policies`, `rollout_train.ledger` · See [training](training.md),
[`Policies`](../../guide/reference.md#policies), [`Version`](../../guide/reference.md#version),
[`Manifest`](../../guide/reference.md#manifest), [`Ledger`](../../guide/reference.md#ledger)

What is being trained has an identity of its own, apart from the run that trains it.

- A **policy** is one line of training, by name.
- Its **versions** are an append-only table. A version says which version it came from, where its weights are, what
  a trainer goes on from, and what it was trained on.
- A version's **name**, `policy@number`, means the same in every process and on every machine. It is what a channel
  serves, what a request to an engine names, and what the numbers stamped on sampled tokens count.

```python
policies = Policies(FileLedger(directory / "ledger"), blobs)
writer = await policies.writer("miner")                              # the one that may add versions to it
version = await policies.add(writer, "miner", 18, weights=step / "weights", state=step / "state", parent="miner@17")
head = await policies.head("miner")
files = await policies.files(head.weights, cache / head.name)       # on any machine: read from the blob store
```

## Versions

- **A version is made by its append.** `add` keeps the checkpoint's files in the blob store and then appends the
  version. A writer that dies before the append has made nothing. Adding a number that is there gives back the
  version that is there.
- **A parent may be another policy's version.** That is a fork: it shares its parent's blobs and costs nothing
  until it differs.
- **Several policies can be served at once**, each on its own channel: a student that follows its newest version,
  and teachers that stay at the versions they were given.
- **One writer.** `writer(policy)` takes the policy's fence. Whoever held it before can add no more versions.
- **Saves thin out with age.** `thin(fence, policy, Retention(recent=2, every=20))` lets go of the files, weights
  and trainer state, of every version but the newest `recent` and every `every`-th by number. The kept ones can be
  served, compared, forked and trained on from where they were; a released one keeps its record (its parent, what it
  was trained on, its metrics). A release is appended to the table `policies/NAME/released` before its blobs are
  deleted, and a blob is deleted only if no version still names it. A released version reads with no `weights`, no
  `state` and the time it was `released`.

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
that the monitor, the report and imitation open the same one from the directory alone (`of_run`). A run is named
after its directory, so the runs sharing a ledger keep apart.

Moving a ledger to Postgres (or from files to SQLite) is a copy and a changed URL, with nothing writing to it:

```bash
rollout ledger copy ~/.cache/rollout/runs/curriculum-9 postgresql://trainer@db-1/rollout --point
```

then the profile's `[ledger]` names the same `url`. `copy` (`rollout_train.database.copy`) takes any ledger, a run's
directory, files or a database, into a database that has none of its tables yet: every record under its key, in the
order it was appended, and every fence, so a writer from before the move is still shut out. `--point` makes the run's
directory name the copy at once (an open profile writes the same when it starts).

A ledger also lists its tables and its scopes' fences: `runs_in` and `policies_in` read from the tables' names
which runs and which policies it has, and the [monitor](monitor.md) shows them.
The training loop's tables are described under [dying and starting again](training.md#dying-and-starting-again); a
policy's versions are the table `policies/NAME/versions`.
