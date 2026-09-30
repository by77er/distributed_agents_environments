# Local services

Status: **Working** (2026-09-29) · See [several runners](../durability/runners.md)

Runners that share state need two services: Postgres (DBOS's journal and the run and coordination stores) and object
storage for blobs. `deploy/local/compose.yaml` runs both:

| Service | Image | Port | Credentials |
|---|---|---|---|
| Postgres | `postgres:17`, `max_connections=400` (each runner holds a pool) | 5432 | `rollout` / `rollout`, database `rollout` |
| S3-compatible storage | `versity/versitygw:v1.8.0`, the POSIX backend (buckets are directories) | 7070 | `rollout` / `rollout-secret` |
| (one-shot) | `amazon/aws-cli` creates the bucket `rollout-blobs` | | |

```bash
docker compose -f deploy/local/compose.yaml up -d
set -a; . deploy/local/services.env; set +a
```

## Tests

The tests that need Postgres or S3 start their own by default (`pgembed`, `moto`). With `ROLLOUT_TEST_POSTGRES` and
`ROLLOUT_TEST_S3` set, as `services.env` does, they use the services instead: every Postgres test creates a database of
its own on the server, and every S3 test a bucket of its own.

```bash
uv run pytest
```

## Servers

```bash
uv run agents serve --database "$ROLLOUT_DATABASE" --blobs "$ROLLOUT_BLOBS" --runner-id server-0 --state /shared --port 8421
uv run agents serve --database "$ROLLOUT_DATABASE" --blobs "$ROLLOUT_BLOBS" --runner-id server-1 --state /shared --port 8422
```

`--blobs s3://bucket/prefix` keeps images in object storage (`rollout.adapters.s3.S3BlobStore`, the `s3` extra); the
endpoint and credentials come from the usual `AWS_*` variables, so the same flag works on AWS S3, R2, SeaweedFS or
MinIO. Environments are still directories under `--state`.

## Verified (2026-09-29)

Docker was not available on the development machine, so the same software ran natively on the same ports and
credentials: Postgres 17.9 (pgembed's binaries, over TCP) and versitygw 1.8.0 (from its release package), with the
bucket created as the compose file does.

- The full test suite with `services.env`: 173 passed, 3 skipped; it created 25 databases and 6 buckets on the
  services.
- Live, two `agents serve` processes on the services with `--blobs s3://rollout-blobs/blobs` and the Codex model:
  a session looked at a 2400×1600 BMP through `server-0` (a green triangle and "17", both named correctly); the
  scaled image became one 35 KB object in the bucket and nothing was stored locally. `server-0` was killed and the
  image file deleted; through `server-1`, which took the session over, the session described the image from its
  history (white background, apex at about 38% across and 19% down; drawn at 37.5% and 18.75%), read from S3.
