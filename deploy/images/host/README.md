# The host image

`ghcr.io/by77er/rollout-host`: a pod that trains and samples for the run that holds it, on one GPU (`runpod-host`): the
one-GPU runs of a local machine, rented. vLLM takes its share of the GPU's memory (`--gpu-memory-utilization`, 0.42
unless the provider's `memory_fraction` says), and the trainer has the rest. Where the provider says so, the pod also
serves sandbox pools (Minecraft worlds) on the CPUs and memory its engine and trainer leave.

| Process | Listens on | Does |
|---|---|---|
| `pki.sh renew` | | Renews the pod's certificate at about two thirds of its life, and publishes each new one for Envoy |
| Envoy | `0.0.0.0:8443` (the pod's exposed TCP port), admin on `127.0.0.1:9901` | Ends mutual TLS; takes only the gateway's certificate; passes on `POST /v1/completions` and `GET /v1/models` to vLLM, `POST /v1/steps`, `GET /v1/steps/CHECKPOINT` and `GET /v1/trainer` to the training service, the sandbox pools' routes (below) to their process, and answers everything else 404 |
| vLLM (`vllm serve`, the workspace's version) | `127.0.0.1:8000` | Samples, as on an inference pod |
| The training service (`python -m rollout_train.pods.training`) | `127.0.0.1:8001` | Takes the run's steps, as on a training pod, its trainer made for the run that holds the pod |
| The follower (`python -m rollout_train.pods.inference`) | `127.0.0.1:8081` (`/healthz`, `/readyz`) | Loads what the run's channel should serve into vLLM, and beats for the pod: ready once vLLM serves it and the training service holds the run's trainer |
| The sandbox pools (`python -m rollout_train.pods.sandboxes`), where `ROLLOUT_SANDBOXES` names any | `127.0.0.1:8710` | Serves a pool of each kind it names, for the run that holds the pod (below) |

A checkpoint the training service makes is kept in the bucket (so evals, the monitor and a run started again find it)
and on the pod's disk (`ROLLOUT_BLOB_CACHE`, default `/workspace/blobs`), where the follower loads it from, with no
round trip through the bucket. With `ROLLOUT_SLEEP_VLLM=1` (the provider's `sleep`), vLLM sleeps while a step is
taken, as local engines do beside a trainer on one card (`rollout_train.colocated`), for a GPU too small for both.

When any process but the sandbox pools ends, the others are ended and the container exits, and RunPod starts it again.
The sandbox pools' process is started again alone, 5 seconds after it ends (`restarting` in `supervise.sh`): vLLM and
the trainer never go down with it.

## Sandbox pools

The provider's table lists the kinds its pods serve (`sandboxes = ["minecraft"]`), and the cluster config's
`[sandboxes.KIND] on_pods` says how ([sandboxes](../../../docs/libraries/rollout/sandboxes.md#on-a-runs-pods)); the
platform gives the pod both as `ROLLOUT_SANDBOXES` when it starts it.

| Route, through Envoy | To the pools' process | Timeout |
|---|---|---|
| `GET /v1/sandboxes/KIND/operations`, `GET /v1/sandboxes/KIND/capacity` | `/KIND/operations`, `/KIND/capacity` | 30 s |
| `POST /v1/sandboxes/KIND/acquire` | `/KIND/acquire` | 600 s (a world's Paper server, bots and task take a minute or more to start) |
| `POST /v1/sandboxes/KIND/release`, `POST /v1/sandboxes/KIND/call` | `/KIND/release`, `/KIND/call` | 120 s (an operation runs up to a 20-second window) |

They share the listener's rate limit with sampling and steps (a burst of 400 requests, then 200 a second): a pod's 12
worlds make a few requests a second.

- **Size.** One sandbox per spare vCPU (the pod's vCPUs less `ROLLOUT_SANDBOX_RESERVED_CPUS`, at the kind's `cpus` each),
  no more than its spare memory holds (less `ROLLOUT_SANDBOX_RESERVED_GIB`, at the kind's `memory_gib` each, 2.4 GiB
  for a Minecraft world), and no more than the kind's `size`. The pod's vCPUs and memory are those its lease records
  from RunPod's API; else the container's cgroup limits; else the machine's. A pod of 16 vCPUs and 188 GB holds 12
  worlds; one of 8 vCPUs and 125 GB, 4.
- **Whose.** The pools follow the pod's lease: they admit only keys of the run that holds the pod (and its evals'), and
  when another run takes the pod, or it is released, they release every lease and delete every sandbox. A lease no
  acquire or operation used for `ROLLOUT_SANDBOX_IDLE` seconds is released. The run's driver releases the leases of
  lapsed claims itself, through Envoy.
- **On the volume.** Each kind keeps its leases and logs in `ROLLOUT_SANDBOX_DIRECTORY/KIND`, so a restarted pool knows
  which keys' sandboxes it lost and answers them `SandboxLost`. Minecraft's cache (the Paper jar, the plugin, the
  world templates; a JDK to compile the plugin with, beside it) goes where the provider's `cache` setting says: on the
  volume (`/workspace/minecraft`), so a container started again finds it. The first world on a new volume builds it, which
  takes minutes.
- **What the image holds for it.** A Java 21 runtime (`/opt/java/openjdk`), Node 22, and the Minecraft environment in
  `/opt/rollout/minecraft`, installed editable into `/opt/rollout/venv`, its harness's packages beside it.

## Variables

Those of the [inference image](../inference/README.md#variables) and the [trainer image](../trainer/README.md#variables)
(`ROLLOUT_TRAINER` and `ROLLOUT_TRAINER_MODEL` are needed), those every pod reads
([deploy/images](../README.md#the-variables-both-pods-read)), and:

| Variable | Says |
|---|---|
| `ROLLOUT_BLOB_CACHE` | Where the pod keeps a copy of every blob its processes put or read (default `/workspace/blobs`) |
| `ROLLOUT_TRAINER_URL` | Where the follower asks the training service which run's trainer it holds (default `http://127.0.0.1:8001`) |
| `ROLLOUT_SLEEP_VLLM` | `1`: vLLM sleeps while a step is taken |
| `ROLLOUT_SANDBOXES` | The sandbox pools the pod serves, as JSON, by kind: each one's `provider`, its `settings`, `size`, `cpus` and `memory_gib` (set by the platform; none: the pools' process is not started) |
| `ROLLOUT_SANDBOX_ADDRESS` | Where the pools are served (default `127.0.0.1:8710`, where Envoy sends their routes) |
| `ROLLOUT_SANDBOX_DIRECTORY` | Each kind's directory is in it (default `/workspace/sandboxes`) |
| `ROLLOUT_SANDBOX_RESERVED_CPUS` | vCPUs kept for vLLM, the trainer, the follower and Envoy (default 4) |
| `ROLLOUT_SANDBOX_RESERVED_GIB` | GiB of memory kept for them (default 64) |
| `ROLLOUT_SANDBOX_IDLE` | Seconds a lease may go unused before it is released (default 1800: longer than a step, while a run's episodes wait) |

## Building

From the repository's root (CI builds it; this is the same command):

```bash
docker build -f deploy/images/host/Dockerfile --build-arg VLLM_VERSION=0.30.0 -t rollout-host .
```
