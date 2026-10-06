# RunPod pods as inference and training providers

**Status: built.** Runs lease GPU pods on RunPod for their channels, their training steps, or both on one GPU; pods
stay warm briefly for the next run and are reaped when no run holds them. What is designed and not built is under
[later](#later). A design note: see [Design notes](README.md) for the others.

GPU pods rented by the hour on RunPod, serving a run's channel or taking its training steps, reached over mutual TLS.
This page says how the pieces fit: the images and the code on the pods, the leases runs hold, the reaper, how the
gateway reaches a pod by its identity, and the security model. What a deployment provides, step by step, is in
[GPU pods on RunPod](../deploy/providers.md#gpu-pods-on-runpod).

Code: `deploy/images`, `.github/workflows/images.yml`, `rollout_train.pods` (`leases`, `leasing`, `routing`,
`inference`, `training`, `trainer`, `identity`), `rollout_train.ledger_service`, `rollout_train.pki`, `rollout_runpod`.

## Decisions

| Question | Decision |
|---|---|
| Who starts pods | The run that needs them: its driver claims them when it starts (after its placement group), as it claims everything it needs. There is no launcher |
| Who may hold a pod | One run at a time. A pod has a lease beside the ledger naming the run that holds it; every change is a compare-and-set |
| What a run pays for | A pod's time at its hourly price, from when the run asked for it or took it until another run takes it or it is deleted: its warm time after the run released it is charged to that run |
| Scale to zero | A run releases its pods when it ends, however it ends. A released pod stays warm for its provider's `idle_stop` (600 seconds unless said), then the reaper deletes it. The reaper also deletes the pods of stale leases (a run whose driver or cluster went away) and every pod with the cluster's tag that no lease names. No runs, no pods |
| Warm reuse | The next run with the same provider, image and model takes a warm pod, with no cold start. The pod is reset to it: its lease names the new run, channel, trainer settings and a ledger token for the new run, which the pod's processes follow |
| A cap on spend | Each provider's `max_pods` slots, across runs; a run with every slot taken waits; a warm pod of another model gives up its slot to it |
| Startup | Bounded by the provider's `start_timeout`: a pod not ready in time is deleted, and the run fails saying why |
| What the pods reach | The ledger service (scoped token), step-ca, the bucket. The gateway reaches the pods, at their public IP and raw TCP port, never the reverse |
| One GPU for both | A `runpod-host` pod runs vLLM with its share of the GPU's memory and the training service beside it; one lease for the run's trained channel and its steps |

## What is built

| Piece | Where | What it does |
|---|---|---|
| The images | `deploy/images/inference`, `trainer`, `host` | A stock vLLM server (the version `uv.lock` pins) and the follower; the training service over rollout-lora's trainers; both on one GPU. Envoy in front of each on the pod's exposed TCP port; the certificate's issue and renewal |
| The images' build | `.github/workflows/images.yml` | On a `v*` tag or by hand: `envoy --mode validate` on each configuration, then each image built and pushed to `ghcr.io/by77er/rollout-ROLE`, tagged with the version, its digest in the run's summary |
| Leases | `rollout_train.pods.leases` | A pod's lease (its name, RunPod's id, provider, slot, GPU type, price, image, model, the run that holds it, its channel, a ledger token for the run, the trainer's settings, when it was asked for, taken, renewed, released), by compare-and-set; a run's time on each pod. In the ledger's database (`pod_leases`, `pod_time`), beside a ledger of files, or through the ledger service |
| A run's pods | `rollout_train.pods.leasing.Pods` | Claims what the run's settings need (`needs_of`): a warm pod taken, else one started in a free slot (its lease written first, then RunPod asked for it), freeing a slot its own earlier start held where that lease is stale or its pod gone; writes where RunPod says each is reached into its lease; waits until each has that address and beats ready for the run; renews every 30 seconds, counting what the pods cost, reading a lease again and renewing it at its version then where it changed meanwhile and is still the run's; releases them warm |
| The reaper | `rollout_train.pods.leasing.reap`, `rollout pods reap` | Deletes idle leases past their `idle_stop`, leases not renewed for 5 minutes, and their pods, each lease first by compare-and-set (read again and deleted at its version then where it changed meanwhile and is still the same stale or idle lease; left where it was renewed or taken), and tagged pods no lease names; revokes each one's certificate and closes its time; logs a lease it could not delete as an error |
| The run's job | `rollout_train.jobs.Run` | Claims its pods after its placement group, releases them on the way out; a `runpod-trainer`'s steps go to its pod's training service; its blobs go to the store its providers name; `limits.spend` counts its pods' time and `limits.hours` bounds it |
| The pod's follower | `rollout_train.pods.inference.InferencePod` | Reads its lease, and keeps vLLM serving what the channel of the run that holds the pod should, with the ledger token the lease gives for that run; drops a released run's adapters; ready once vLLM has it (and, on a host pod, once the training service holds the run's trainer); beats with the pod's name, identity, readiness, run and certificate serial |
| The training service | `rollout_train.pods.training.TrainerService` | One step at a time, asked for by the checkpoint it makes, idempotent by it; its trainer made anew for the run that holds the pod (`following`) |
| The training client | `rollout_train.pods.RemoteTrainer` | A `Trainer` over the service, as `TinkerTrainer` is over Tinker |
| Routing | `rollout_train.pods.routing.LeasedServers` | What a run's channel asks at each look for its servers: the pods its leases name for it that beat ready, each a `RemoteEngine` at the `https` address its lease holds (RunPod's, never the beat's) with the gateway's certificate, checked by the identity named for the pod. Both a run's own gateway and the cluster's use it |
| The ledger over HTTP | `rollout_train.ledger_service` | The ledger and the stores beside it for whoever holds a token: the platform's does everything; a pod's reads its run's serving records, starts and checkpoints, writes its beat and reads its lease, while its lease names that run |
| Blob stores | `[stores.NAME]`, `rollout_train.stores.for_pods` | A bucket RunPod reaches beside the cluster's own; a pod gets its location and a key: read-only for inference, the writer's for a trainer or host |
| The platform's certificates | `rollout_train.pki`, `rollout pki publish` | From the chart's step-ca: its root and the provisioner's key (Secret `step-ca`), and a new gateway certificate every six hours (Secret `gateway-tls`) |
| RunPod's pods API | `rollout_runpod.RunPod` | Create, start, stop, delete, list; a pod's public address, GPU type and price. Its key is read at each request and never kept, logged or put in an error; each request says its User-Agent |
| step-ca | `rollout_runpod.StepCa` | One-time tokens for pods' first certificates, revocation by serial, certificates for the platform's own clients, a provisioner's key from step-ca's configuration |

How it is tested, with no network, Docker, RunPod or step-ca (`tests/rollout_train/pods`, `tests/rollout_runpod`,
`tests/rollout_train/test_ledger_service.py`, `tests/rollout_train/test_limits.py`, `tests/deploy`):

- **A fake of RunPod's API** (`tests/rollout_runpod/fake.py`): the pods API the client uses, in this process, refusing
  a request without the key or a User-Agent it accepts. With processes standing in for what runs on each pod: a run
  starts a pod, renews and releases it; the next run takes it warm, reset to it, and the first is charged its warm
  time; the reaper deletes idle pods past their idle stop, stale leases' pods and tagged orphans; a pod not ready in
  time is deleted and the run told why; `max_pods` bounds a provider's pods and a run waits for one; a lease whose
  version moved under its run is still renewed; a stale lease whose pod is gone frees its slot even after a version
  change, and the resumed run leases a new pod; the reaper never deletes a lease renewed meanwhile; a host pod is one
  lease for trainer and channel with the writer's key.
- **The gateway over mutual TLS**: a leased pod whose stand-in runs the real follower (reading its lease through the
  ledger service with its own token) beside a fake vLLM behind a TLS server that takes only the gateway's certificate;
  the run's channel finds it by its lease and beat and samples it; released, it is no server.
- **The ledger service's scopes**: a pod's token reads its run's records and checkpoints only, writes only its own
  beat, and reads nothing once its lease names another run; retries take one fence; `Fenced` comes back as itself.
- **Mutual TLS end to end**, the follower and the training service against fakes, the RunPod client, the token helper
  against a fake step-ca, and Envoy's configurations read as YAML, as before.
- **Live, opt-in** (`tests/rollout_runpod/test_live.py`, `ROLLOUT_RUNPOD=1`): a real pod leased and deleted past its
  start timeout, an orphan reaped, a bucket read and written; with a deployment, a provider's pod leased, sampled and
  released.

## The security model

**Identities.** Every certificate of the cluster carries one URI SAN in the trust domain `rollout`. A pod's is
`spiffe://rollout/pod/NAME`, where NAME is the pod's name, chosen when its lease is written (before RunPod is asked for
it: the token must be in the pod's environment when it is created): `rollout-CLUSTER-PROVIDER-SLOT-RANDOM`. A pod keeps
its name, and its certificate, across the runs that hold it. The gateway's is `spiffe://rollout/gateway`.

**Who talks to whom.**

| From | To | Over | Checked |
|---|---|---|---|
| The gateway (and a run's `RemoteTrainer`) | A pod's Envoy, at `https://PUBLIC_IP:PORT` | Mutual TLS on a raw TCP port (RunPod's HTTPS proxy would end TLS and drop the client's certificate) | By the gateway: the pod's certificate chains to the cluster's root and carries the identity named for the pod, which beats fresh, at the address its lease holds, and the pod's lease names the run. By Envoy: the client's certificate chains to the root and carries the gateway's identity |
| Envoy | vLLM, the training service | `127.0.0.1` inside the pod | Only `POST /v1/completions` and `GET /v1/models` (inference), `POST /v1/steps`, `GET /v1/steps/ID`, `GET /v1/trainer` (training), both on a host; everything else 404; requests limited in size, rate-limited, timed out |
| The pod | The ledger service | HTTPS, the pod's token | The token's signature (the platform's token signs it), its pod and run, and the pod's lease |
| The pod | The bucket | S3, the key it was given | As the store checks: read-only for inference pods |
| The pod | step-ca | HTTPS: by the cluster's root (`STEP_FINGERPRINT`), or by the system's roots behind a Cloudflare Tunnel (`trust = "system"`) | The first certificate by the one-time token; renewals by the pod's current certificate (over mutual TLS, or a token it signs, behind a proxy) |
| A run's driver | RunPod's API, step-ca | HTTPS | The API key; the provisioner's key |

**Certificates' lives.** Leaves last 24 hours and are renewed at about two thirds of their life by the pod
(`step ca renew --daemon`), each new one swapped in for Envoy without a restart. The first is issued with a token good
for 15 minutes and once, bound to the pod's identity: the pod makes its key itself, and the key never leaves it. The
cluster's root is pinned when the pod is leased (`STEP_ROOT`, checked against `STEP_FINGERPRINT`).

**Revocation is passive.** step-ca renews any certificate it has not revoked. When a pod is deleted (released past its
idle stop, reaped, or past its start timeout), the certificate whose serial its beats said is revoked: it is not
renewed and lapses within a day. Within that day the gateway's own check (only pods live in the heartbeats, with the
identity named for them and leased to the run) is what stops it being reached.

**The tokens in the pod's environment.** RunPod's REST API keeps console secrets but has no way to create one, so what
is minted for one pod (its one-time token, its ledger token) and the bucket's key are sent as ordinary variables
(`PodSpec.sensitive`: never logged, not in the spec's `repr`). Anyone who can read the account's pods through RunPod's
API or console can read them. The one-time token is worth nothing once used (the entrypoint unsets it first); the
ledger token reads only the run's records while the pod is held by it; a run that takes a warm pod gives it a token of
its own in the lease. Long-lived secrets a deployment wants kept in RunPod's console (`HF_TOKEN`) are referenced by
name (`PodSpec.secrets`).

**What this does not protect against.** RunPod itself (the host sees the pod's memory and disk, its volume, its
environment); anyone with the RunPod account's key or console; a compromised gateway (its certificate is what every
pod trusts). The gateway's certificate is renewed every six hours by the chart's `rollout pki publish`.

## Later

Designed, not built:

- **Copying a checkpoint to the pods' store.** A run whose trainer or servers are RunPod's writes its blobs to the store
  its RunPod providers name, and readers find each blob in the store its reference names. A checkpoint made in the
  cluster's own store and served on RunPod (a run that starts from a local checkpoint, an eval of one on pods) needs
  its files copied to the pods' store first: a bridge-like step that copies a manifest's blobs and records the copy as
  a second location of the same checkpoint, which the serving record then names.
- **Location-aware stores.** Where both the trainer and the servers are on RunPod, a checkpoint stays in RunPod's region
  or on a network volume they share, and goes to the bucket only for readers elsewhere. A host pod already loads its own
  checkpoints from its disk.
- **Per-run keys to the bucket.** R2 scopes a key to a bucket, so a writer's key may write anywhere in it; S3 can scope
  a key to a prefix with a session policy, and pods could get short-lived keys per run.
- **An elastic pool and multi-pod trainers** ([runtime design](runtime-design.md#later-directions)).
