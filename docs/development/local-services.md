# Local services

Code: `deploy/local`

Runners that share state need two services: Postgres (DBOS's journal and the run and coordination stores) and object
storage for blobs. `deploy/local/compose.yaml` runs both:

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
| `ROLLOUT_DATABASE` | `postgresql://rollout:rollout@localhost:5432/rollout` | your shell, for `--database` |
| `ROLLOUT_BLOBS` | `s3://rollout-blobs/blobs` | your shell, for `--blobs` |
| `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` | the storage service and its credentials | boto3 |

## Tests

The tests that need Postgres or S3 start their own by default (`pgembed`, `moto`). With `ROLLOUT_TEST_POSTGRES` and
`ROLLOUT_TEST_S3` set, they use the services instead: every Postgres test creates a database of its own on the
server, and every S3 test a bucket of its own.

```bash
uv run pytest
```

## Servers

```bash
uv run agents serve --database "$ROLLOUT_DATABASE" --blobs "$ROLLOUT_BLOBS" --runner-id server-0 --state /shared
```

Running several such servers is described in [several runners](../implementations/rollout-durable/runners.md).

`--blobs s3://bucket/prefix` keeps images in object storage through `rollout_s3.S3BlobStore`
([content](../guide/content.md#media-and-blobs)). Each blob is one object named by its SHA-256 under the prefix. The
endpoint and credentials come from the usual `AWS_*` variables, so the same flag works on AWS S3 and on S3-compatible
services. Environments stay directories under `--state`.
