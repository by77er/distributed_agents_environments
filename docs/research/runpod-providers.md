# RunPod pods as inference and training providers

**Status: in progress.** The images, the code on the pods and the clients of RunPod and step-ca are built; the provider
kinds are declared but do not start pods yet. A design note: see [Design notes](README.md) for the others.

GPU pods rented by the hour on RunPod, serving a run's channel or taking its training steps, reached over mutual TLS.
This page says what is built (the images, the code on the pods, the launcher's clients of RunPod and step-ca, and how
a gateway reaches a pod by its identity), the two provider kinds the [runtime design](runtime-design.md)'s provider
framework declares for it and what is left to wire them, and the security model.

Code: `deploy/images`, `.github/workflows/images.yml`, `rollout_train.pods`, `rollout_train.inference.remote.Connection`
(`identity`), `rollout_runpod`.

## What is built

| Piece | Where | What it does |
|---|---|---|
| The inference image | `deploy/images/inference` | A stock vLLM server (the version `uv.lock` pins) on `127.0.0.1:8000`, adapters loaded while it runs; the follower beside it; Envoy in front on the pod's exposed TCP port; the certificate's issue and renewal |
| The trainer image | `deploy/images/trainer` | The training service over rollout-lora's `LoraTrainer` and `FullTrainer` (each step in a fresh process) on `127.0.0.1:8001`; Envoy in front; the certificate's issue and renewal |
| The images' build | `.github/workflows/images.yml` | On a `v*` tag or by hand: `envoy --mode validate` on both configurations, then both images built and pushed to `ghcr.io/by77er/rollout-inference` and `rollout-trainer`, tagged with the version, their digests in the run's summary. |
| The pod's follower | `rollout_train.pods.inference.InferencePod` | The engine hosts' `Follower` over one channel whose one engine is the pod's vLLM server (`RemoteEngine`): it fetches what the run says the channel should serve from the blob store and loads it by the checkpoint's id. After each look it asks the server what it has: ready once it has what the channel should serve. It beats with the pod's name, identity, public address, readiness and certificate serial; `/healthz` and `/readyz` on the pod's loopback interface |
| The training service | `rollout_train.pods.training.TrainerService` | One step at a time, asked for by the checkpoint it makes: the batch (a blob of weighted segments) and the parent's files (manifests) from the blob store, the changeable settings, the seed; the new weights and state back into the blob store, their manifests and the metrics in the answer. Idempotent by the checkpoint: the same step while it runs, the kept answer once made (on the pod's volume), or the ledger's checkpoint if it has it; a failed step is taken again when asked again. Another step while one runs is refused (409) |
| The training client | `rollout_train.pods.RemoteTrainer` | A `Trainer` over the service, as `TinkerTrainer` is over Tinker: puts the batch and the parent's files in the blob store, asks for the step by `into`'s name, asks after it until it is made, fetches the files into `into`. Changeable settings are kept on the client and sent with every step. Every failure is a `StepFailed`; the pod's are `TrainerUnreachable` (no answer for `patience` seconds), `TrainerRefused` (an HTTP refusal, the proxy's 404 among them) and `TrainerBusy` |
| Identities and live pods | `rollout_train.pods.identity` | `pod_identity(name)` is `spiffe://rollout/pod/NAME`; `GATEWAY_IDENTITY` is `spiffe://rollout/gateway`; `live(beats, role)` is the pods whose newest beat is fresh and names the identity named for the pod, with their addresses |
| A server known by its identity | `Connection.identity` | The URI SAN a server's certificate must carry, checked in the TLS handshake (in place of the host name, which a pod reached by IP has not), before anything is sent. `requiring(context, identity)` does the same for a server's context, of its clients |
| RunPod's pods API | `rollout_runpod.RunPod` | Create (image, GPU types, environment, console secrets by reference, an exposed TCP port, a volume), start, stop, delete, list; a pod's public address from its port mappings. The key is read from `RUNPOD_API_KEY` at each request and never kept, logged or put in an error |
| step-ca, for pods' certificates | `rollout_runpod.StepCa` | A one-time token for a pod's first certificate (what `step ca token` makes, signed with a JWK provisioner's key), and revoking a certificate by its serial (passive: it is not renewed) |

How it is tested, with no network, Docker or step-ca (`tests/rollout_train/pods`, `tests/rollout_runpod`,
`tests/deploy`):

- **Mutual TLS end to end**, with certificates made by a test authority (`cryptography`): the gateway reaches a fake
  vLLM server by its SPIFFE identity at 127.0.0.1, which its certificate does not name; a client with another pod's
  identity, no identity, another root's certificate, or none, is refused before its request reaches the server; a
  server with another identity, another root, or only the right IP address is refused before anything is sent.
- **The follower** against a fake vLLM server, a scratch SQLite ledger and a file blob store: ready with the base model,
  then with each checkpoint once loaded (the one before stays), not ready while the server does not answer (and why),
  its beat read back by `live`.
- **The training service** with a fake trainer, through `RemoteTrainer`: the batch and parent through the blob store
  (files and moto S3), the weights back; the same step asked again (while running, after a restart, when the ledger has
  the checkpoint) is not stepped again; failures as `StepFailed` and taken again; busy, refused and unreachable pods.
- **The RunPod client** against a fake API served here: every call, secrets by reference, the key and a sensitive value
  in no log line, `repr` or error.
- **The token helper** against a fake step-ca over TLS that checks tokens as a JWK provisioner does: one pod's
  certificate for its identity alone, once, within 15 minutes; that certificate is what the gateway reaches the pod by;
  revocation by serial.
- **Envoy's configurations**, read as YAML: the routes each pod passes on, everything else 404 (loading adapters,
  sleeping, metrics, health); only the gateway's SAN; limits, timeouts, access logs without bodies; the certificate
  paths agree with `pki.sh`; every variable the pods read is in the images' READMEs. CI runs `envoy --mode validate`.

## The security model

**Identities.** Every certificate of the cluster carries one URI SAN in the trust domain `rollout`. A pod's is
`spiffe://rollout/pod/NAME`, where NAME is the launcher's name for the pod, chosen before the pod is asked for (RunPod's
id for it exists only after, and the token must be in the pod's environment when it is created). The gateway's is
`spiffe://rollout/gateway`.

**Who talks to whom.**

| From | To | Over | Checked |
|---|---|---|---|
| The gateway (and a run's `RemoteTrainer`) | A pod's Envoy, at `https://PUBLIC_IP:PORT` | Mutual TLS on a raw TCP port (RunPod's HTTPS proxy would end TLS and drop the client's certificate) | By the gateway: the pod's certificate chains to the cluster's root and carries the identity named for the pod in its fresh beat (`live`, `Connection.identity`). By Envoy: the client's certificate chains to the root and carries the gateway's identity |
| Envoy | vLLM, or the training service | `127.0.0.1` inside the pod | Only `POST /v1/completions` and `GET /v1/models` (inference), or `POST /v1/steps`, `GET /v1/steps/ID`, `GET /v1/trainer` (training); everything else 404; requests at most 8 MiB (1 MiB for training), rate-limited, timed out |
| The pod | The ledger and the blob store | Their own clients, with credentials given as RunPod console secrets | As those services check |
| The pod | step-ca | HTTPS, the root checked by `STEP_FINGERPRINT` | The first certificate by the one-time token; renewals by the pod's current certificate |
| The launcher | RunPod's API, step-ca | HTTPS | The API key; the provisioner's key |

**Certificates' lives.** Leaves last 24 hours and are renewed at about two thirds of their life by the pod
(`step ca renew --daemon`), each new one swapped in for Envoy without a restart. The first is issued with a token good
for 15 minutes and once, bound to the pod's identity: the pod makes its key itself, and the key never leaves it.

**Revocation is passive.** step-ca renews any certificate it has not revoked; it consults nothing else on renewal. So
the launcher's account of its live pods reaches the CA as revocations: when it stops or deletes a pod, or finds a pod
it no longer counts as its own, it revokes the certificate whose serial the pod's beats say (`StepCa.revoke`). That
certificate is not renewed and lapses within a day; a pod that is gone cannot renew, so its certificate lapses all the
same. Within that day a revoked certificate is still valid for TLS: the gateway's own check (only pods live in the
heartbeats, with the identity named for them) is what stops it being reached.

**The token in the pod's environment.** RunPod's REST API keeps console secrets but has no way to create one, so a
token minted for one pod cannot be a secret: it is sent as an ordinary variable (`PodSpec.sensitive`: never logged,
not in the spec's `repr`). Anyone who can read the account's pods through RunPod's API or console can read it for its
15 minutes, and use it if the pod has not yet. Once used it is worth nothing (step-ca keeps its id), and the entrypoint
unsets it before any other process starts. Long-lived secrets (the blob store's credentials, the ledger's) are console
secrets, referenced as `{{ RUNPOD_SECRET_name }}` (`PodSpec.secrets`).

**What this does not protect against.** RunPod itself (the host sees the pod's memory and disk, its volume, its
environment); anyone with the RunPod account's key or console; a compromised gateway (its certificate is what every pod
trusts). Rotating the gateway's certificate is cert-manager's (on Kubernetes) or a `step ca renew --daemon` beside it.

## How it plugs into the provider framework

The [runtime design](runtime-design.md) declares providers with capabilities in the cluster config and validates runs
against them. Both kinds are declared (`rollout_train.providers`): `runpod-inference` beside `vllm`, `vllm-servers`,
`tinker` and `api`, and `runpod-trainer` beside `lora`, `full` and `tinker`; the cluster config reads them
([the cluster config](../guide/cluster.md)). Starting pods and reaching them wait for commits 11 (the gateway) and 14
(the launcher):

```toml
[tls]                                                 # the cluster's CA, and the client certificate the gateway presents
ca = "~/.config/rollout/root_ca.crt"
certificate = "~/.config/rollout/gateway.crt"
key = "~/.config/rollout/gateway.key"

[inference.runpod-4090]
kind = "runpod-inference"                             # auth mtls: each pod's identity from its heartbeat
image = "ghcr.io/by77er/rollout-inference@sha256:…"   # a digest from the images workflow's summary
gpu_types = ["NVIDIA GeForce RTX 4090"]
pods = 2                                              # at most this many at once, across runs
idle_stop = 600                                       # seconds a pod no run's channel serves on runs before it is stopped
volume_gb = 50
api_key_env = "RUNPOD_API_KEY"
secrets = { AWS_ACCESS_KEY_ID = "r2_key_id", AWS_SECRET_ACCESS_KEY = "r2_secret", HF_TOKEN = "hf_token" }
step_ca = { url = "https://ca.example.com", provisioner = "launcher", key_file = "~/.config/rollout/provisioner.jwk", root = "~/.config/rollout/root_ca.crt" }
[inference.runpod-4090.models."Qwen/Qwen3.5-4B"]
context = 8192
options = { max_lora_rank = 96, args = "--max-model-len 8192 --gpu-memory-utilization 0.9" }

[trainers.runpod-lora]
kind = "runpod-trainer"
trainer = "lora"                                      # the trainer its pods run: lora or full
image = "ghcr.io/by77er/rollout-trainer@sha256:…"
gpu_types = ["NVIDIA H100 80GB HBM3"]
pods = 1
models = ["Qwen/Qwen3.5-4B"]
segment_tokens = 16000
# api_key_env, secrets, step_ca as above
```

| | `runpod-inference` | `runpod-trainer` |
|---|---|---|
| Capabilities | As `vllm-servers`: token-exact, sampled logprobs (`--logprobs-mode processed_logprobs`), prompt and top-k logprobs up to the server's, honours sampling, adapters by name; no full-weight reload; `loads = {"peft"}`; cost per hour from RunPod's `costPerHr`, not per token | `produces` and `format` as the trainer it names (`lora` in `peft` for `lora`, `full` in `full` for `full`), its objectives, `scores`; `segment_tokens` from the config |
| Bridges | A `peft` checkpoint (a local LoRA (low-rank adaptation) trainer's, or a `runpod-trainer`'s) is served verbatim; a Tinker checkpoint through `tinker → peft` | Its checkpoints are ordinary `peft` or `full` files in the blob store: every bridge from those formats applies |
| The launcher starts | Per run channel, `replicas` pods: a name (`inference-RUN-N`), a token for its identity, `RunPod.create` with the run, channel, model, stores and step-ca's URL and fingerprint in its environment | Per run, one pod (`trainer-RUN`), the same way, with the trainer, model and settings |
| The launcher stops | When the run ends, or after `idle_stop` with nothing to serve: `RunPod.stop` (or `terminate` when the run is done), then `StepCa.revoke` of the serial its beats said. A run is refused at validation when it would need more than `pods` | The same |
| What reaches the pod | The gateway's sampler for the kind: a `RemoteChannel` whose servers are `live(beats, "inference")` for the run's channel, each a `RemoteEngine` with its own `Connection` (the gateway's certificate, the root, `identity` from the beat) | The run job's trainer: `RemoteTrainer(address, checkpoints, connection=…)` with the address and identity from `live(beats, "trainer")` for the run's pod, `budget` and `changeable` from the config and the run's settings |
| Readiness | The beat's `ready`: the run's loop waits for its trained channel's pods to be ready before its first step's turns | `describe()` answers |

What the integration adds beyond the declarations:

1. The gateway's server list for a `runpod-inference` channel taken from `live(beats)` for the run and channel, each
   pod a `RemoteEngine` of its own (a `RemoteChannel` takes any `CheckpointServer`) with the `Connection` that
   `Auth.connection(tls, identity=…)` gives for the identity its beat names, so it connects only to pods live in
   heartbeats, each checked against its own identity.
2. The launcher's pod lifecycle: names, tokens, `create`, waiting for the beat, the pod-count limit and idle stop, stop or
   delete and revoke. The pod-count limit and the idle rule are the launcher's: `RunPod` does what it is asked.
3. The pods' access to the ledger. They open it with `rollout_train.ledger.opened` from `ROLLOUT_LEDGER`, which names
   the database ledger (Postgres reachable from RunPod). The design has every role reach the ledger through its HTTP
   service ([decisions after review](runtime-design.md#decisions-after-review)), which is not built: a pod will name it
   the same way and hold a token that can read its run's serving records and write its own beats, nothing else.
4. The blob store in S3 or R2 (`rollout_s3:S3BlobStore`), its credentials as console secrets: the pods have no shared
   disk with the cluster.

## What the user provides

- A RunPod API key, as `RUNPOD_API_KEY` in the launcher's environment; console secrets for the blob store's credentials
  (and `HF_TOKEN` for a gated model), named in the cluster config.
- A bucket in R2 or S3 for the blob store, reachable from RunPod and from the cluster.
- A step-ca deployment reachable from RunPod (`step ca init`, a JWK provisioner for the launcher with 24-hour default
  and maximum leaf lifetimes, `allowRenewalAfterExpiry` off), its root's fingerprint, and the provisioner's decrypted
  key as a file on the launcher's machine.
- The gateway's client certificate: on Kubernetes, cert-manager with step-issuer and a `Certificate` whose URI SAN is
  `spiffe://rollout/gateway`; on one machine, `step ca certificate spiffe://rollout/gateway` and a renewal daemon.
- The ledger reachable from the pods (today Postgres with a password, as a console secret).
- A first tag (`v0.1.0`) or a manual run of the images workflow, and the pushed digests in the cluster config.
