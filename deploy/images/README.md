# Images

`platform/` is the image the K3s cluster runs (Ray's pods, each run's RayJob, the gateway, the monitors), built in the
cluster ([deploy/k3s](../k3s/README.md#images)). The other three are for GPU pods rented elsewhere (RunPod), each reached
at a public TCP port over mutual TLS:

| Image | Directory | What runs in it |
|---|---|---|
| `ghcr.io/by77er/rollout-inference` | [inference](inference/README.md) | A stock vLLM server (the workspace's version), the follower that keeps it serving what one run's channel should (`rollout_train.pods.inference`), Envoy in front |
| `ghcr.io/by77er/rollout-trainer` | [trainer](trainer/README.md) | The training service over rollout-lora's trainers (`rollout_train.pods.training`), Envoy in front |
| `ghcr.io/by77er/rollout-host` | [host](host/README.md) | Both on one GPU: vLLM with its share of the GPU's memory and the follower, the training service beside it, one Envoy in front |

They are built by `.github/workflows/images.yml` on a version tag (`v*`) or when it is run by hand, after
`envoy --mode validate` has checked each Envoy configuration. Each image is tagged with the repository's version
(the tag without its `v`, or `sha-COMMIT` for a run by hand); the run's summary has each pushed image's digest, which is
what a cluster's configuration should name.

## What is in them

Each is a widely used public image, as it is published, with a thin layer of the platform's on top:

| Image | Base | Compressed | The platform's layers |
|---|---|---|---|
| inference | `vllm/vllm-openai:vVLLM_VERSION` (8.7 GB) | 8.8 GB | 85 MB: Envoy and step (51 MB), the follower's dependencies (31 MB), the platform's packages (2 MB) |
| host | `vllm/vllm-openai:vVLLM_VERSION` | 8.8 GB | 110 MB: Envoy and step, the follower's and rollout-lora's dependencies (57 MB), the packages (2 MB) |
| trainer | `pytorch/pytorch:TORCH-cuda13.0-cudnn9-runtime` (3.0 GB) | 3.2 GB | 170 MB: Envoy and step, rollout-lora's dependencies (transformers, accelerate, flash-linear-attention, …: 116 MB), the packages (2 MB) |

- **One PyTorch.** The platform's packages run on the base image's Python (3.12), in `/opt/rollout/venv`: a virtual
  environment that sees the image's packages. A package the image has at `uv.lock`'s version is not installed again,
  and the image's PyTorch, Triton and CUDA libraries are the only ones. `/opt/rollout/venv/added.txt` lists what the
  virtual environment adds.
- **The base's layers are the public image's**, byte for byte, so a machine that has pulled that image pulls only
  the platform's layers.
- **What changes often comes last.** Envoy and step first, then the dependencies, installed from a list exported from
  `uv.lock` alone (so that layer is built again only when the list changes), then the scripts and Envoy's
  configuration, and the platform's packages last: with the build's cache (the workflow keeps one for each image), a
  change to the sources builds and pushes 2 MB.
- **The cluster's versions.** In `/opt/rollout/venv` every package is the version `uv.lock` pins, as the cluster runs
  it, but for PyTorch's own build (`+cu130`) and the CUDA libraries and Triton built with it, which are the image's;
  the build fails unless the image's torch is `uv.lock`'s release (the trainer image's tag names it). So the adapters
  and weights a pod's trainer writes are made by the transformers, safetensors and torch the cluster's bridges and vLLM
  read them with, and the follower speaks to the ledger service and the blob store with the cluster's pydantic, httpx
  and boto3. The build also checks that every requirement of what is in the virtual environment is met there or by
  the image (`pip check`). Ray and uv's own package, which the pods do not run, are left out.
- **Python 3.12.** The packages the pods run (rollout, rollout-train, rollout-s3, rollout-objectives, rollout-lora)
  require Python 3.12 or later, and ruff and pyright check them for 3.12 (`pyproject.toml`); the rest of the workspace
  requires 3.13.
- vLLM's own processes see the image's packages alone. The trainer image has no vLLM. Both bases have the C compiler
  Triton builds its kernels' launchers with.

What they share is in `common/`:

| File | Does |
|---|---|
| `install.sh` | Exports the locked dependencies, and installs them and the workspace packages into `/opt/rollout/venv` on the image's Python ([What is in them](#what-is-in-them)) |
| `pki.sh` | The pod's certificate from step-ca: the first with a one-time token, then renewed by the pod itself, each new one published for Envoy |
| `supervise.sh` | Starts the pod's processes and ends them all when one ends, so the container exits and RunPod starts it again |
| `sds/certificate.yaml`, `sds/ca.yaml` | Where Envoy reads the pod's certificate and the cluster's root: `/certs/current`, which `pki.sh` swaps in one rename |

## Certificates

Every pod has a certificate from the cluster's step-ca, good for 24 hours, whose one URI SAN is
`spiffe://rollout/pod/NAME` (NAME: `ROLLOUT_POD_NAME`, the pod's name):

1. The pod is given a one-time token for that identity as `STEP_TOKEN` (`rollout_runpod.StepCa.pod_token` mints one,
   good for 15 minutes).
2. The pod fetches the root, checking it against `STEP_FINGERPRINT`, makes its key itself, and asks for its
   certificate with the token (`step ca certificate`). step-ca takes a token once.
3. The pod renews its certificate at about two thirds of its life, over mutual TLS with the one it has
   (`step ca renew --daemon`), and publishes each new one; Envoy reads it without a restart.
4. A revoked certificate (`rollout_runpod.StepCa.revoke`, for a pod stopped or deleted) is not renewed, and lapses
   within a day.

The certificates are kept on the pod's volume (`ROLLOUT_CERTS`, default `/workspace/certs`): a pod started again renews
the one it has while it is valid, and needs a new token only when it has lapsed.

Envoy takes requests only from a client whose certificate chains to the same root and carries
`spiffe://rollout/gateway`. The gateway's client certificate comes from cert-manager on Kubernetes (an `Issuer` for
step-ca through its step-issuer, and a `Certificate` with that URI SAN), or, on one machine, from `step ca certificate`
and a `step ca renew --daemon` beside the gateway.

## The variables both pods read

| Variable | Says |
|---|---|
| `ROLLOUT_POD_NAME` | The pod's name, as its starter gave it (lowercase letters, digits, hyphens): its identity is `spiffe://rollout/pod/NAME` |
| `ROLLOUT_ROLE` | What the pod does: `inference`, `trainer`, or `host` (both) |
| `ROLLOUT_LEDGER` | Where the ledger is, as JSON: the ledger service, `{"kind": "rollout_train.ledger_service:HttpLedger", "url": "https://…", "token_env": "ROLLOUT_LEDGER_TOKEN"}` |
| `ROLLOUT_LEDGER_TOKEN` | The pod's token for the ledger service: it reads its run's serving records, starts and checkpoints, and writes the pod's beats and reads its lease, nothing else. A run that takes the pod later gives a token for itself in the lease |
| `ROLLOUT_BLOBS` | Where the blob store is, as JSON: `{"kind": "rollout_s3:S3BlobStore", "bucket": "…", "endpoint_url": "…", "access_key_id_env": "…", "secret_access_key_env": "…"}`, the store's key in the two variables it names: a read-only key for an inference pod, a key that writes for a trainer or host pod |
| `STEP_CA_URL` | The cluster's step-ca (`https://ca.example.com`) |
| `STEP_FINGERPRINT` | The SHA-256 fingerprint of step-ca's root certificate (`rollout_runpod.fingerprint`, or `step certificate fingerprint root_ca.crt`) |
| `STEP_ROOT` | The cluster's root certificate itself (PEM), pinned when the pod is leased and checked against `STEP_FINGERPRINT`; without it the pod fetches it from step-ca |
| `STEP_CA_TRUST` | `root` (default): step-ca is reached directly, its TLS checked by the cluster's root; `system`: behind a proxy that ends TLS with a public certificate (a Cloudflare Tunnel), checked by the system's roots, and renewals use a token signed by the certificate's key (`--mtls=false`) |
| `STEP_TOKEN` | The one-time token for the pod's first certificate; unset before any other process starts |
| `ROLLOUT_CERTS` | Where the certificates are kept (default `/workspace/certs`, on the volume) |
| `ROLLOUT_CERT_SERIAL_FILE` | The file the certificate's serial is read from for the pod's beats (default `/certs/current/serial`) |
| `ROLLOUT_BLOB_CACHE` | A directory on the pod's disk that keeps a copy of every blob the pod's processes put or read (a host pod's, `/workspace/blobs`): none by default |
| `SYSTEM_ROOTS` | The system's root certificates, which step-ca's TLS is checked by with `STEP_CA_TRUST=system` (default `/etc/ssl/certs/ca-certificates.crt`) |
| `ENVOY_CONFIG` | Another Envoy configuration (default `/etc/envoy/envoy.yaml`) |
| `ENVOY_LOG_LEVEL` | Envoy's log level (default `warn`); its access log is always on |
| `ENVOY_CONCURRENCY` | Envoy's worker threads (default 4): left to itself, Envoy runs one per hardware thread of the machine, not of the pod |
| `HF_HOME` | Where models are downloaded (default `/workspace/huggingface`); `HF_TOKEN` for a gated model |
