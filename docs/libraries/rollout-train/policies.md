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
writer = await policies.writer("swarm")                              # the one that may add versions to it
version = await policies.add(writer, "swarm", 18, weights=step / "weights", state=step / "state", parent="swarm@17")
head = await policies.head("swarm")
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

`FileLedger` keeps each table as a file of JSON lines in a directory, which processes on one machine may share.
The training loop's tables are described under [dying and starting again](training.md#dying-and-starting-again); a
policy's versions are the table `policies/NAME/versions`.
