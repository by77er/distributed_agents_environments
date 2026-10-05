# Postgres and S3

The platform keeps its lasting state in two stores: the ledger in Postgres and blobs in S3 or an S3-compatible
store. This page says how a cluster config names them, and how to move an existing ledger and blob store into them.
It is for whoever deploys the platform or moves it from one machine to a cluster.

**Read first:** [What runs where](roles.md). **Next:** [Volumes and backups](backups.md).

## The ledger in Postgres

The [ledger](../libraries/rollout-train/checkpoints.md#the-ledger) is append-only tables with fences, kept in one
database. SQLite serves one machine; Postgres serves every process on every node, which a cluster needs.

The cluster config's `[ledger]` names it. A URL that holds a password is refused, so name the password apart:

```toml
[ledger]
url = "postgresql://rollout@postgres.rollout:5432/rollout"  # the password comes from PGPASSWORD
```

Or name the whole URL by an environment variable or a file that holds it:

```toml
[ledger]
url_env = "ROLLOUT_LEDGER_URL"
```

- The chart runs Postgres 17 as the StatefulSet `postgres`, with `max_connections=400`, since each
  [episode](../libraries/rollout-train/episodes.md) runner holds a pool of connections. A managed Postgres needs as
  many.
- Expose Postgres only inside the cluster. Processes outside it (GPU pods elsewhere) are given scoped access through
  the ledger service the [runtime design](../research/runtime-design.md#decisions-after-review) describes, never
  the database itself.

## Blobs in S3

The blob store holds checkpoints' files, episodes' trajectories, datasets and imported environments, each named by its
SHA-256. Any S3-compatible service works: AWS S3, Cloudflare R2, MinIO, or versitygw, which the chart runs as the
StatefulSet `s3` on a volume of its own.

```toml
[blobs]
kind = "rollout_s3:S3BlobStore"
bucket = "rollout-blobs"
prefix = "blobs/"
# endpoint_url = "https://ACCOUNT.r2.cloudflarestorage.com"   # optional; else AWS_ENDPOINT_URL
# region = "auto"                                              # optional
```

The credentials never appear in the cluster config: the store reads boto3's usual sources, such as
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, and the endpoint from `AWS_ENDPOINT_URL` (or `AWS_ENDPOINT_URL_S3`)
when the config gives none. The chart sets these from the Secret `stores`
([what every pod is given](helm.md#what-every-pod-is-given)).

To use a managed Postgres or a cloud bucket with the chart, edit `[ledger]` and `[blobs]` in the chart's
`files/cluster.toml`, and set `AWS_ENDPOINT_URL` and the credentials in `templates/_helpers.tpl`.

## Move an existing ledger and blob store

These steps move a ledger and a store of files (a machine's `~/.cache/rollout`) into Postgres and S3. Blobs are named
by their SHA-256, so stores from several machines merge into one bucket without a clash. Copying the blobs and
rewriting the records can each be run again; the ledger is copied once, into a database that has none of its tables.

1. **Stop everything that writes.** Let runs finish or pause them, then stop the runs, the
   [gateway](../libraries/rollout-train/gateway.md) and the monitor.

2. **Reach the target stores.** On Kubernetes, forward their ports to your machine and read their credentials:

    ```bash
    kubectl -n rollout port-forward svc/postgres 5432:5432 &
    kubectl -n rollout port-forward svc/s3 7070:7070 &
    export PGPASSWORD="$(kubectl -n rollout get secret stores -o jsonpath='{.data.POSTGRES_PASSWORD}' | base64 -d)"
    export AWS_ENDPOINT_URL=http://127.0.0.1:7070 AWS_DEFAULT_REGION=us-east-1
    export AWS_ACCESS_KEY_ID="$(kubectl -n rollout get secret stores -o jsonpath='{.data.ROOT_ACCESS_KEY_ID}' | base64 -d)"
    export AWS_SECRET_ACCESS_KEY="$(kubectl -n rollout get secret stores -o jsonpath='{.data.ROOT_SECRET_ACCESS_KEY}' | base64 -d)"
    ```

3. **Copy the blobs.** Say what would be copied first, then copy. Each file is uploaded in parts, a blob the bucket
   already has is skipped, and a file that is not the blob its name says is reported and not copied:

    ```bash
    uv run python -m rollout_s3.copying ~/.cache/rollout/blobs --to s3://rollout-blobs/blobs/ --dry-run
    uv run python -m rollout_s3.copying ~/.cache/rollout/blobs --to s3://rollout-blobs/blobs/
    ```

    Name several directories to merge several stores of files.

4. **Take a copy of the ledger.** Work on a copy, never on the live file. For a SQLite ledger, use SQLite's backup:

    ```bash
    sqlite3 ~/.cache/rollout/ledger.db ".backup 'ledger-copy.db'"
    ```

    A ledger kept as files (a run's directory) becomes a SQLite copy with
    `uv run rollout ledger copy RUN_DIRECTORY "sqlite:///$PWD/ledger-copy.db"`.

5. **Rewrite the copy's records for what moved.** Records name where each blob and directory was. `--into` is the new
   blob store as a location, `--uris` how its blobs' URIs begin, `--store` each store of files that was copied, and
   `--path` each directory that moves (run directories onto the state volume, say):

    ```bash
    uv run python -m rollout_train.relocating "sqlite:///$PWD/ledger-copy.db" \
      --into '{"kind": "rollout_s3:S3BlobStore", "bucket": "rollout-blobs", "prefix": "blobs/"}' \
      --uris s3://rollout-blobs/blobs/ \
      --store "$HOME/.cache/rollout/blobs" \
      --path "$HOME/.cache/rollout/runs=/root/.cache/rollout/runs"
    ```

    `--home` says what `~` stands for in the records, when you run it on another machine than the one that wrote them.

6. **Copy the ledger into Postgres.** The target must not have any of the copy's tables yet. Records keep their keys and
   order, and the registry (runs' names, bookmarks, [suites](../libraries/rollout-train/evals.md#suites)' and datasets'
   names) comes along:

    ```bash
    uv run rollout ledger copy "sqlite:///$PWD/ledger-copy.db" postgresql://rollout@127.0.0.1:5432/rollout
    ```

7. **Copy the run directories** to where `--path` said, if runs will go on from them: for example with `kubectl cp`
   into a pod that mounts the state volume.

8. **Point the cluster config at the new stores** (`[ledger]` and `[blobs]`, as above), start the platform, and check
   in the monitor that runs, checkpoints and episodes open.

Keep the old ledger and blob directory until the new stores are backed up ([volumes and backups](backups.md)).
