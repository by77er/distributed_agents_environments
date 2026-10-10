# The host image

`ghcr.io/by77er/rollout-host`: a pod that trains and samples for the run that holds it, on one GPU (`runpod-host`): the
one-GPU runs of a local machine, rented. vLLM takes its share of the GPU's memory (`--gpu-memory-utilization`, 0.42
unless the provider's `memory_fraction` says), and the trainer has the rest. The pod also serves whatever kinds of
sandbox the run that holds it asks for, on the CPUs and memory its engine and trainer leave; the image holds no
environment's code for that, and nothing an environment needs of its system (no Java, no Node): only uv.

| Process | Listens on | Does |
|---|---|---|
| `pki.sh renew` | | Renews the pod's certificate at about two thirds of its life, and publishes each new one for Envoy |
| Envoy | `0.0.0.0:8443` (the pod's exposed TCP port), admin on the Unix socket `/run/rollout/envoy-admin.sock` (root's alone) | Ends mutual TLS; takes only the gateway's certificate; passes on `POST /v1/completions` and `GET /v1/models` to vLLM, `POST /v1/steps`, `GET /v1/steps/CHECKPOINT` and `GET /v1/trainer` to the training service, the sandboxes' routes (below) to the sandbox host, and answers everything else 404 |
| vLLM (`vllm serve`, the workspace's version) | `127.0.0.1:8000` | Samples, as on an inference pod |
| The training service (`python -m rollout_train.pods.training`) | `127.0.0.1:8001` | Takes the run's steps, as on a training pod, its trainer made for the run that holds the pod |
| The follower (`python -m rollout_train.pods.inference`) | `127.0.0.1:8081` (`/healthz`, `/readyz`) | Loads what the run's channel should serve into vLLM, and beats for the pod: ready once vLLM serves it and the training service holds the run's trainer |
| The sandbox host (`python -m rollout_train.pods.sandboxes`) | The Unix socket `/run/rollout/sandboxes.sock` (root's alone: Envoy reaches it) | Serves the kinds of sandbox the pod's lease gives sources for, each kind's pool in a process of its own (below) |

A checkpoint the training service makes is kept in the bucket (so evals, the monitor and a run started again find it)
and on the pod's disk (`ROLLOUT_BLOB_CACHE`, default `/workspace/blobs`), where the follower loads it from, with no
round trip through the bucket. With `ROLLOUT_SLEEP_VLLM=1` (the provider's `sleep`), vLLM sleeps while a step is
taken, as local engines do beside a trainer on one card (`rollout_train.colocated`), for a GPU too small for both.

When any process but the sandbox host ends, the others are ended and the container exits, and RunPod starts it again.
The sandbox host is started again alone (`restarting` in `supervise.sh`): 5 seconds after it ends, the wait doubling up
to 5 minutes while it keeps ending within a minute, and not again after 10 such ends running. vLLM and the trainer
never go down with it.

## Sandboxes

What the pod serves is its lease's: the run's driver writes a source for each kind its environment needs that the
pod's provider lists (`sandboxes = ["KIND"]`, with `[sandboxes.KIND] on_pods`) into the lease's settings when it takes
the pod ([on a run's pods](../../../docs/libraries/rollout/sandboxes.md#on-a-runs-pods)). A source is data: the
provider (`module:name`) and its settings, its projects' zips in the pods' blob store, pins of everything else they
need, a Python version, and what a pool of it is sized by. One that is not well formed (a kind not named as a route
names it, a zip not named by its SHA-256) is left out.

- **Each kind's Python.** The host fetches the source's zips into `ROLLOUT_SANDBOX_DIRECTORY/projects/SHA256` and has
  uv make a virtual environment, `ROLLOUT_SANDBOX_DIRECTORY/venvs/DIGEST`, with a uv-managed Python of the source's
  version (3.13; uv keeps it in `ROLLOUT_SANDBOX_DIRECTORY/uv`), the projects installed editable (`--no-sources`) and
  the pins as they are (`--no-deps`: nothing is resolved, so nothing is fetched by a name alone). The digest is of the
  zips' contents, the pins and the Python: an environment is made once, under a file lock, and found again by later
  runs and after restarts; one no kind has used for 14 days is deleted, with the zips no kept environment uses.
- **First use downloads** a Python (python-build-standalone's releases on GitHub), the pinned wheels (PyPI), and the
  build backend each project's `[build-system]` names for its editable install (hatchling, from PyPI, at whatever
  version that `requires` allows: the platform holds no build backend to pin it from); an
  environment's own caches download what it needs at first use (Minecraft: Paper, a JDK, Node, the harness's npm
  packages). A failure to make a Python is waited out (5 seconds, doubling to 5 minutes) and tried again, never given
  up on.
- **Each kind's process.** `python -P -m rollout.harness.pool_server`, in that Python (nothing of its working directory
  on its path), as a user of its own (`ROLLOUT_SANDBOX_USERS`, `sandbox1` to `sandbox8`, which the image makes; one per
  kind served, kept in `ROLLOUT_SANDBOX_DIRECTORY/accounts.json` and given back once the lease asks for the kind no
  more, its directory then made root's until the kind is served again). It serves on a Unix socket the host makes and
  hands it, root's and readable by no one else, so no other process of the pod (another kind's included) reaches it or
  its control routes. It dies with the host, is ended with its process group, and one a host before left (its pid in
  `ROLLOUT_SANDBOX_DIRECTORY/pids`) is ended when the host starts. One that ends is started again after 5 seconds, then
  10, 20, up to 5 minutes; after it ends within a minute 5 times running, the kind is given up on until its source or
  its run changes, or 15 minutes pass. One kind failing (its code does not import, its settings are wrong) leaves the
  others served. The host logs a line with every kind's state whenever one changes.
- **What it may touch.** Its environment is made for it: `PATH`, `HOME` and `TMPDIR` in its own directory
  (`ROLLOUT_SANDBOX_DIRECTORY/kinds/KIND`, its user's alone: its leases, logs and caches, kept across restarts), a
  locale. The host, the follower, the training service, Envoy and the container's first process run as root, so their
  environments (the ledger token, the store's keys, the certificate's token) are not its to read, and the
  certificates' directory is root's alone. The environments and zips it runs from are root's, read-only to it. It
  cannot reach the sandbox host, nor through it another kind's pool, nor Envoy's admin interface: both serve on Unix
  sockets root's alone, in `/run/rollout` (0700). It can still reach what every process of the pod can: vLLM
  (`127.0.0.1:8000`) and the training service (`127.0.0.1:8001`), which take requests without a key, and Envoy's
  public port, which takes the gateway's certificate alone. A run's sandbox code must be code the cluster trusts
  beside its trainer.
- **Whose.** A kind's process admits only keys of the run that holds the pod (and its evals'). When another run takes
  the pod, it forgets the other runs' leases. When the lease's `renewed` is older than 5 minutes (the run's driver
  renews it every 30 seconds, but may stall), it takes no new keys (`PoolUnavailable`) and goes on serving those it
  holds. When the pod is released, or `renewed` is older than 30 minutes (the driver is gone), it marks every lease
  lost. A lost lease's key gets `SandboxLost` (410), so its episode is played again, never in a fresh sandbox; a lease
  ends when its run releases it, or when its claim lapses (the run's driver releases it). Leases are kept across
  restarts, and those whose sandboxes are gone are lost.
- **How many.** One sandbox per spare vCPU (the pod's vCPUs less `ROLLOUT_SANDBOX_RESERVED_CPUS`, at the source's
  `cpus` each), no more than the spare memory holds (less `ROLLOUT_SANDBOX_RESERVED_GIB`, at its `memory_gib` each),
  nor than its `size`; with several kinds, those that say only a `size` take it first, and those with a `share` take
  their part of what is left. The pod's vCPUs and memory are the lesser of what its lease records from RunPod and its
  cgroup's limits, read again at each look; a pool whose size changes is told so in place (or, where its provider's
  size is fixed, started again once it holds nothing). A pod of 16 vCPUs and 188 GB holds 12 Minecraft worlds (2.4 GiB
  each); one of 8 vCPUs and 125 GB, 4.

| Route, through Envoy | To the sandbox host | Timeout |
|---|---|---|
| `GET /v1/sandboxes/KIND/operations`, `GET /v1/sandboxes/KIND/capacity` | `/KIND/operations`, `/KIND/capacity` | 30 s |
| `POST /v1/sandboxes/KIND/acquire` | `/KIND/acquire` | 600 s (a sandbox may take minutes to start: a world's Paper server, bots and task) |
| `POST /v1/sandboxes/KIND/release`, `POST /v1/sandboxes/KIND/call` | `/KIND/release`, `/KIND/call` | 120 s (a Minecraft operation runs up to a 20-second window) |

The host answers 503 without saying the pool is full while a kind's process is not up (`PoolUnavailable`), and 409 for
a key of another run. The routes share the listener's rate limit with sampling and steps (a burst of 400 requests, then
200 a second).

## Variables

Those of the [inference image](../inference/README.md#variables) and the [trainer image](../trainer/README.md#variables)
(`ROLLOUT_TRAINER` and `ROLLOUT_TRAINER_MODEL` are needed), those every pod reads
([deploy/images](../README.md#the-variables-both-pods-read)), and:

| Variable | Says |
|---|---|
| `ROLLOUT_BLOB_CACHE` | Where the pod keeps a copy of every blob its processes put or read (default `/workspace/blobs`) |
| `ROLLOUT_TRAINER_URL` | Where the follower asks the training service which run's trainer it holds (default `http://127.0.0.1:8001`) |
| `ROLLOUT_SLEEP_VLLM` | `1`: vLLM sleeps while a step is taken |
| `ROLLOUT_SANDBOX_SOCKET` | The Unix socket the sandbox host serves on, root's alone (default `/run/rollout/sandboxes.sock`, where Envoy sends the sandboxes' routes) |
| `ROLLOUT_SANDBOX_DIRECTORY` | The sandbox host's state: projects, Python environments, uv's cache and Pythons, each kind's leases, the home its processes are given (default `/workspace/sandboxes`) |
| `ROLLOUT_SANDBOX_RESERVED_CPUS` | vCPUs kept for vLLM, the trainer, the follower and Envoy (default 4) |
| `ROLLOUT_SANDBOX_RESERVED_GIB` | GiB of memory kept for them (default 64) |
| `ROLLOUT_SANDBOX_USERS` | The users the kinds' processes run as, one per kind, by name (default `sandbox1` to `sandbox8`); a host that does not run as root runs them as itself |
| `RESTART_QUICK`, `RESTART_LIMIT` | How `restarting` counts quick ends: within how many seconds of a start (60), and after how many running it stops (10) |

## Building

From the repository's root (CI builds it; this is the same command):

```bash
docker build -f deploy/images/host/Dockerfile --build-arg VLLM_VERSION=0.30.0 -t rollout-host .
```
