# Local Postgres and S3 for development

For developers and single-machine setups: Postgres and an S3-compatible store in Docker Compose, for the tests and for a
cluster config.

**Read first:** [Choose a setup](../deploy/setups.md). **Next:** [Postgres and S3](../deploy/stores.md).

Code: `deploy/local`

Runners that share state need two services: Postgres for the
[ledger](../libraries/rollout-train/checkpoints.md#the-ledger) and object storage for blobs. `deploy/local/compose.yaml`
runs both, and the tests can use them too:

| Service | Image | Port | Credentials |
|---|---|---|---|
| Postgres | `postgres:17`, with `max_connections=400` (each runner holds a pool) | 5432 | `rollout` / `rollout`, database `rollout` |
| S3-compatible storage | `versity/versitygw:v1.8.0`, the POSIX backend (buckets are directories) | 7070 | `rollout` / `rollout-secret` |
| (one-shot) | `amazon/aws-cli:2.27.0` creates the bucket `rollout-blobs` | | |

```bash
docker compose -f deploy/local/compose.yaml up -d
set -a; . deploy/local/services.env; set +a
```

`deploy/local/services.env` sets:

| Variable | Value | Read by |
|---|---|---|
| `ROLLOUT_TEST_POSTGRES` | `postgresql://rollout:rollout@localhost:5432/rollout` | the tests |
| `ROLLOUT_TEST_S3` | `http://localhost:7070` | the tests |
| `ROLLOUT_LEDGER_URL` | `postgresql://rollout:rollout@localhost:5432/rollout` | a cluster config's `[ledger] url_env` |
| `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` | the storage service and its credentials | boto3 |

## Run the tests against them

The tests that need Postgres or S3 start their own by default (`pgembed`, `moto`). With `ROLLOUT_TEST_POSTGRES` and
`ROLLOUT_TEST_S3` set, they use the services instead: every Postgres test creates a database of its own on the
server, and every S3 test a bucket of its own.

```bash
uv run pytest
```

## Point a cluster config at them

A [cluster config](../guide/cluster.md) whose runners share these services names them so:

```toml
[ledger]
url_env = "ROLLOUT_LEDGER_URL"

[blobs]
kind = "rollout_s3:S3BlobStore"
bucket = "rollout-blobs"
```

`rollout_s3.S3BlobStore` keeps each blob as one object named by its SHA-256 under its prefix
([content](../guide/content.md#media-and-blobs)). The endpoint and credentials come from the usual `AWS_*` variables,
so the same config works on AWS S3 and on S3-compatible services.
