# Images

`platform/` is the image the K3s cluster runs (Ray's pods, each run's RayJob, the gateway, the monitors), built in the
cluster ([deploy/k3s](../k3s/README.md#images)). The other two are for GPU pods rented elsewhere (RunPod), each reached
at a public TCP port over mutual TLS:

| Image | Directory | What runs in it |
|---|---|---|
| `ghcr.io/by77er/rollout-inference` | [inference](inference/README.md) | A stock vLLM server (the workspace's version), the follower that keeps it serving what one run's channel should (`rollout_train.pods.inference`), Envoy in front |
| `ghcr.io/by77er/rollout-trainer` | [trainer](trainer/README.md) | The training service over rollout-lora's trainers (`rollout_train.pods.training`), Envoy in front |

Both are built by `.github/workflows/images.yml` on a version tag (`v*`) or when it is run by hand, after
`envoy --mode validate` has checked both Envoy configurations. Each image is tagged with the repository's version
(the tag without its `v`, or `sha-COMMIT` for a run by hand); the run's summary has each pushed image's digest, which is
what a cluster's configuration should name.

What both share is in `common/`:

| File | Does |
|---|---|
| `install.sh` | Installs workspace packages, with their dependencies exactly as `uv.lock` pins them, into `/opt/rollout/venv` on a Python 3.13 of its own |
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
| `STEP_TOKEN` | The one-time token for the pod's first certificate; unset before any other process starts |
| `ROLLOUT_CERTS` | Where the certificates are kept (default `/workspace/certs`, on the volume) |
| `ROLLOUT_ADDRESS` | The pod's public address, `https://IP:PORT`; by default from RunPod's `RUNPOD_PUBLIC_IP` and `RUNPOD_TCP_PORT_8443` |
| `ROLLOUT_CERT_SERIAL_FILE` | The file the certificate's serial is read from for the pod's beats (default `/certs/current/serial`) |
| `ENVOY_CONFIG` | Another Envoy configuration (default `/etc/envoy/envoy.yaml`) |
| `ENVOY_LOG_LEVEL` | Envoy's log level (default `warn`); its access log is always on |
| `HF_HOME` | Where models are downloaded (default `/workspace/huggingface`); `HF_TOKEN` for a gated model |
