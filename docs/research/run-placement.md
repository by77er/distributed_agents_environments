# Where a run runs: in the cluster, or on its pod

**Status: proposed.** Nothing here is built. It is read against main `e8e06d8`, and its figures are measured from the
run `run_01M4745DD5B1NHNDGEA9RGT030` (`gridworld-9b-pro6000`). A design note: see [Design notes](README.md) for the
others.

A run that rents a GPU pod on RunPod plays its episodes in the platform's Kubernetes cluster and samples every turn
across the internet on the pod. A pod comes with much more CPU and memory than its GPU's work uses (an RTX PRO 6000
pod: 16 vCPUs and 188 GB). This note asks which of a run's roles should run on the pod beside its engine and trainer,
what that takes, and what it gains. The roles of [the architecture](../architecture/overview.md) and what they exchange
do not change for any of it: where each runs is placement, said by configuration.

Code read for this note: `rollout_train.jobs`, `.loop`, `.ledger`, `.gateway` (`service`, `turns`), `.inference.remote`,
`.recorder.sampling`, `.rollouts.scheduler`, `.sandboxes`, `.demand`, `.submitting`, `.published`, `.pods` (`leasing`,
`routing`, `trainer`, `training`, `environment`), `.ledger_service` (`scopes`, `service`); `rollout.harness.remote`;
`rollout_runpod.api`; `rollout_s3.store`; `deploy/images/host`; `deploy/chart/rollout` (`sandboxes`, `tunnel`,
`_helpers.tpl`, `values.yaml`). Figures from outside are dated and their sources listed at the end.

## The answer in brief

| Question | Recommendation |
|---|---|
| What costs most on the internet path now | Not latency. 41% of the run's episodes failed on it (27 of 66), every one because a turn was not served in three attempts, and 16% of turns were sampled more than once. A turn took 19.7 s at the median, of which about 1 s was spent outside the GPU's work on network round trips and recording |
| What to do first | Make the remote path sound where it is (phase 0): a server that misses one look stays a server (main already lets a turn wait out a missed look instead of failing), the gateway's connection pool is as large as the turns it has in flight, the driver stops downloading a step's files, and the gateway stops reading a session's whole turn table for every turn. It is needed under any placement, since several pods will always sample each other's engines |
| What to move | The whole run: its driver (the loop), its gateway, its runners, the environment's code and its sandbox pools, in a Ray cluster of the run's own on its pod, as KubeRay gives it one at home today. Moving one role alone moves the hairpin elsewhere: a gateway on the pod with the ledger at home crosses the tunnel twice a turn instead of the internet twice a turn |
| What stays in the cluster | Kueue's admission, the launch and a thin job that leases the pod and follows the run, the monitor, the ledger of record (Postgres), step-ca, the reaper, the cluster's gateway for clients elsewhere, and runs that rent no pod |
| Ray | A Ray head inside the pod, one node, for the run's driver and its actors; the cluster's RayJob shrinks to the run's keeper. Not a Ray worker on the pod joined to a head at home: Ray wants every node to reach every other on many ports, which a pod behind NAT with one mapped port per declared port cannot give |
| The ledger from the pod | Through the ledger service over the tunnel, as pods reach it now, with a token scoped to the run. Turns' records go in batches (a group commit every quarter second), each record still appended under its own fence before its reply; the gateway keeps its own index of its sessions' turns instead of reading it back each turn. Turns stay one small record each beside their blobs |
| The trust boundary | Phase 1 hosts only environments the cluster trusts on a pod. Imported environments wait for their programs to run in a process of their own under a user that holds no secrets |
| Sandboxes on the pod | Processes only: a pod is a container that cannot run containers of its own. Minecraft's worlds (a JVM, Node and Python) fit; container sandboxes go to a sandbox service or stay in the cluster |
| Several pods | Each pod that serves a replica hosts runners and a gateway sampling its own engine; one of them hosts the loop. Pods coordinate through the ledger and the blob store, as runners on several machines do now, not through one Ray cluster across pods |

## What happens now

### Where each role runs

A run with a RunPod provider is a RayJob in the cluster. Its driver's process holds the training loop, the run's
gateway (served on `127.0.0.1` for the runner) and the runner (`LocalRunner` under an `EpisodeRunner`), which imports
the environment from its published version's Ray runtime environment and plays `episodes_at_once` episodes at once.
The driver leases the pod (`rollout_train.pods.leasing`) and renews the lease every 30 seconds.

| Role | Where it runs | What crosses the boundary |
|---|---|---|
| Training loop | The run's RayJob in the cluster | Steps' batches up to the bucket; each step's weights and state back down from it |
| Gateway | The driver's process | Each turn's generations to the pod (`https://PUBLIC_IP:PORT`, mutual TLS); each turn's blob to the bucket |
| Runners, the environment's code | The driver's process | Nothing directly: they reach the gateway on `127.0.0.1` |
| Sandbox pools | A pod of their own in the cluster (`sandboxes-KIND`, [where sandboxes run](sandbox-placement.md)) | Nothing |
| Engine (vLLM) and its follower | The pod | The ledger, through the Cloudflare Tunnel; checkpoints from its disk (`ROLLOUT_BLOB_CACHE`) |
| Training service | The pod | The batch from the bucket; the step's weights and state to the bucket |
| Ledger | Postgres in the cluster | Pods reach it through the ledger service behind the tunnel |
| Blob store | R2 (`[stores]`) | Every role that reads or writes blobs |

### A turn, measured

The run trained Qwen3.5-9B (LoRA rank 32) on the gridworld with 48 episodes at once, its channel and its trainer on
one RTX PRO 6000 host pod in RunPod's secure cloud at $2.09 an hour. It was stopped after 15.3 minutes, before its
first step. From its ledger and from the headers of half of its turns' blobs (1,678 of 3,356, every second one; each
blob holds `timings`: when the turn started, how long each generation took, and the whole turn):

| Figure | Value |
|---|---|
| Turns | 3,356 in 919 s: 3.65 a second; 66 episodes ended, of 2 to 4 agents and 20 to 28 rounds (a 4-agent episode: 112 turns) |
| Tokens per turn | 1,245 in the prompt and 584 sampled (means); the engine generated 2,120 tokens a second; one of the driver's beats counted 104 requests in flight on average |
| A turn | 19.7 s at the median (14.8 s at the 10th percentile, 26.5 s at the 90th) |
| Its two generations | Every turn is two requests to the engine (the thinking budget, then the answer): 15.6 s and 2.9 s at the median |
| Recording, after the generations | 0.68 s at the median, 1.09 s at the 90th percentile: the turn's blob (4.8 KB) put in R2 from the cluster (a `HEAD`, then a `PUT`), the session's previous turn read back from R2 to find what the new prompt shares with it, and the ledger record appended |
| Before the generations | 4 ms at the median for a turn sampled once; a turn that waited for a server to take it waited up to 40 s |
| Turns sampled again | 216 twice and 60 three times (13% and 4% of turns), in bursts: in some half-minutes most turns were sampled again, in others none |
| Episodes failed | 27 of 66, each "the turn was not served in 3 attempts": 26 "no server of policy would take a turn", one "ConnectError: All connection attempts failed" |

The records carry no network timing, and the pod was not contacted for this note, so the round trip from the cluster
to the pod is not measured. From the workstation, a ping to the nearest Cloudflare edge (San Jose) takes 28 to 62 ms
and a new TLS connection to R2 0.19 to 0.41 s. The pod's address is in a block registered to Boost Run LLC, in a
region the records do not say. A round trip of 50 to 150 ms is assumed below; it is
the figure to measure first (`ss -ti` on the driver's node shows the kernel's smoothed round trip per connection).

### Why turns failed

The run was played by `97de833`, in which a turn that found no server failed that attempt at once. What that code
did, read against the failures (a reading, not yet a test):

- Every two seconds a routed channel looks at its servers (`RemoteChannel.refresh`): it asks each `GET /v1/models`
  with a timeout of 2 seconds, and a server that does not answer in time is given no turn until the next look. With
  one pod, one slow answer left the channel no server, and every turn that began a generation in those two seconds
  failed with `Unserved` and was sampled again from the start; three in a row failed the turn, and the turn its
  episode.
- The look and the generations share one `httpx.AsyncClient` per server, with httpx's default pool: 100 connections,
  20 kept alive. The run had about 104 requests in flight, so a look can wait for a connection behind generations
  that take 15 seconds, and the pool's wait counts toward the look's 2 seconds. Connections beyond 20 are closed when
  idle, and opened again (TCP, then TLS with client certificates) across the internet at the next burst.
- Agents of an episode move in rounds, so turns start in bursts, which matches the bursts of failures.

Main now lets a turn with no server wait for one while the channel looks again (`RemoteChannel._asked`, for as long
as a turn waits for a replica), and waits before each attempt after the first (`Gateway._sampled`): a missed look no
longer fails the turns in flight. A missed look still takes the only server from every turn for a look or more, and
the pool is still smaller than the turns in flight; phase 0 is what remains of that. Placement does not cure it: a run
with several pods samples engines on other pods.

### What crosses the boundary

Per turn, per episode, per group and per step, as the code does it, with sizes measured where the records hold them
and estimated (marked) where they do not:

| When | Between | What | Size |
|---|---|---|---|
| Each turn | Gateway → pod | Two `POST /v1/completions`: the prompt's token ids, then the prompt and the thinking so far | About 21 KB up (estimated: about 7 bytes a token id in JSON) |
| | Pod → gateway | Two answers: the token ids, each sampled token's logprob, its text and offset, and the prompt's ids echoed (`return_token_ids`) | About 65 KB down (estimated from vLLM's answer: about 75 bytes a sampled token) |
| | Gateway → R2 | The turn's blob: `HEAD`, `PUT` | 4.8 KB (measured) |
| | R2 → gateway | The session's previous turn, read once to find the shared prefix | 4.8 KB (measured) |
| | Gateway ↔ ledger (in the cluster) | The whole table of the program's run's turns, read once (to find a recorded reply and the session's latest turns); one append | 393 bytes a record; the table grows to 112 records in a 4-agent episode, so 2.4 MB read over that episode's turns |
| Each episode | Runner ↔ ledger | A claim (122 bytes), the episode's fence, its record (4.2 KB on average) | |
| | Runner → R2 | Trajectories and events, compressed | 7 KB and 1.9 KB |
| Each group | Loop → ledger | The group (190 bytes) and its result (0.8 KB) | |
| Each step | Loop → R2 | The batch, as JSON: every segment's tokens and logprobs | About 20 MB for four groups of this run (estimated: 140 to 267 segments a group, about 25 KB a segment) |
| | Pod → R2 | The new adapter and the optimizer's state | 321 MB of adapter (rank 32 on Qwen3.5-9B, in FP32, as another run's checkpoints measure) and about 640 MB of Adam's two moments (estimated) |
| | R2 → driver | Both again: `RemoteTrainer.step` fetches them into the step's directory, and `Checkpoints.add` hashes and puts them once more | About 1 GB down to the cluster, every step |
| Every 2 s | Gateway → pod | `GET /v1/models`, and the pods' leases and beats read from the ledger | Small |
| Every 15 s | Pod → ledger, through the tunnel | The follower's and the training service's beats; the serving record read | Small |

At 3.65 turns a second that is about 0.3 MB a second between the cluster and the pod: bandwidth is not the cost. The
costs are the round trips (two generations and three R2 requests a turn, each from the cluster), the failures above,
and the gigabyte a step brought down to the cluster for no reader there.

### What it costs, in this run's terms

- **Latency.** Network round trips and recording took about 1 s of a 19.7 s turn: two round trips to the pod (0.1 to
  0.3 s at the assumed figure) and 0.68 s recording. That is 5%, and it would be 5% of the GPU's work too if
  `episodes_at_once` were not raised to cover it. A workload whose turns are short (no thinking budget, 50 to 100
  tokens a reply: 1 to 3 s a turn) would spend a third to a half of each turn on the same second.
- **Failures.** 41% of episodes, each a share of the GPU's time spent on turns thrown away.
- **Capacity.** The run's RayJob asked for 1.25 CPUs and 5 GiB; the cluster's node has 20 CPUs and 23 GB, shared with
  Postgres, the monitor, the sandbox pool and everything else. The Minecraft pool holds 4 worlds (4 CPUs, 7.5 GiB
  requested, 10 GiB limit), each 1.1 to 2.4 GiB and about one CPU while it plays ([Minecraft memory](minecraft-memory.md)).
  The PRO 6000 pod's 16 vCPUs and 188 GB were paid for and used by vLLM, the trainer and the follower: by estimate 2
  to 3 CPUs and a few tens of GB.

### What other systems do

| System | Agent loop | Environments, sandboxes |
|---|---|---|
| verl | `AgentLoopWorker` Ray actors, spread round-robin over every node with CPUs (GPU nodes included); many trajectories as coroutines in each; token in, token out to the engine as a Ray actor call, a session kept on one replica | Tools in the worker; Sandbox Fusion as a remote HTTP service, rate-limited |
| SkyRL | The generator reaches vLLM through a router over HTTP, with session affinity ("a large throughput win for agentic / multi-turn generation") | SkyRL-Gym environments in the generator's process (a thread pool); SWE sandboxes as local Podman containers or remote providers (Daytona, Modal, E2B), rate-limited |
| prime-rl | One orchestrator, a light CPU process, drives every rollout over HTTP to the inference servers; environments in subprocess pools | Prime's sandboxes, remote microVMs, "for production, especially for training"; their start overlapped with the first turn |
| AReaL | One rollout worker per inference engine, holding that engine's resources: the agent loop on the engine's node, in process or in a subprocess pool, through an HTTP proxy beside it | Remote reward services |
| slime | One `RolloutManager` actor, placed wherever Ray chooses, over HTTP to sglang-router | Its SWE example uses a remote E2B-compatible sandbox cluster |
| OpenRLHF | The agent executor inside the vLLM engine's own Ray actor: the environment's `step` runs in the engine's process | Remote reward models over HTTP |
| NeMo-RL | Environments as Ray actors; on one node every part shares it | NeMo Gym, CPU-only, as subprocesses of an actor, its model server an HTTP proxy to vLLM |
| Tinker's cookbook | The whole loop and the environment on the user's machine, sampling over the internet; Tinker is "optimized for throughput rather than latency", and clients are told to keep every request in flight and add no timeouts or retries | A local Sandbox Fusion container, or Modal |
| ROLL | Environment workers hold many environment managers, a thread each, asking a shared generation scheduler | ROCK, a client-server sandbox manager; RollArt sends environments to CPU clusters and measured resets of hundreds of seconds when image pulls saturate the network |

The common pattern: the agent loop is a light, highly concurrent process kept on or next to the engine's node, talking
token in, token out, with sessions pinned to a replica. Cheap, trusted environments run beside it. Heavy or untrusted
ones (repositories, terminals, browsers, VMs) run remotely on CPU capacity of their own, for isolation and because GPU
nodes are short of CPU; their cost (0.3 to 2 s an operation, measured by others) is hidden by concurrency, not
avoided. Tinker is the opposite end, workable only because it is tuned for throughput. A pod is a GPU node with CPU
to spare (16 vCPUs on a PRO 6000, but 8 on the cheapest H100 SXM pod), and a container that cannot run containers.

## The options

From the smallest move to the largest. "Crossings a turn" counts round trips that leave the machine, for a turn of a
run like the one measured.

| Option | What runs on the pod | Crossings a turn | For | Against |
|---|---|---|---|---|
| Now | Engine, follower, training service | 2 to the pod, 3 to R2, from the cluster | Built; the cluster holds everything but the GPU | The failures; the cluster's CPU and memory bound how many episodes play; a gigabyte a step down |
| 0. The remote path made sound | As now | As now, with fewer resampled turns | Small; needed under any placement | The cluster's capacity still bounds play |
| 1. Sandbox pool on the pod | Also the pool (`rollout pool`), served on a second mapped port behind Envoy | As now, plus every sandbox operation from the cluster's runners to the pod | Worlds get the pod's CPU and memory | The hairpin the user named: runner at home, world on the pod, model on the pod. The pool's leases need the ledger, so new scopes too. A second public endpoint |
| 2. A gateway on the pod | Also a gateway: the runner's turns go to it over the internet | 1 to the pod (the runner's request), 2 to the ledger through the tunnel, 3 to R2 from the pod | Generations stay on the pod; the internet carries one request a turn instead of two | The ledger is now two tunnel round trips a turn; play is still bounded by the cluster |
| 3. Play on the pod | Also runners, the environment's code, the gateway and the pools; the loop stays in the cluster | Ledger records through the tunnel; R2 from the pod | Every turn and sandbox operation stays on the pod | Runners are no Ray actors of the run's Ray cluster at home (a pod cannot join it, below), so they need a supervisor of their own; each step's segments still go to the cluster and back as the batch |
| **4. The run hosted on its pod** | The whole run: driver, loop, gateway, runners, environment, pools, trainer client, in a Ray cluster of its own on the pod | Ledger records through the tunnel, batched; R2 from the pod | Turns, sandbox operations and steps never leave the pod; the run's roles keep their Ray forms; the pod's spare CPU and memory are the run's | The pod holds more credentials and runs the environment's code beside the trainer; the monitor's live feed and the keeper need new paths; a pod's restart restarts the run's driver |
| Rejected: a Ray worker on the pod | The pod joins the run's Ray cluster in the cluster as a node | Ray's own protocols across the internet | One Ray cluster as now | Ray wants every node to reach every other node, on the GCS port, each raylet's and object manager's ports and 10002 to 19999 for workers. The pod is behind NAT with one public port per declared port, mapped to a port RunPod picks, and the cluster is reachable only through a Cloudflare Tunnel that carries HTTP |

Option 4 is the recommendation, reached in phases. Options 1 to 3 each keep one leg of the hairpin; 4 removes the legs
and leaves the records, which batch well.

## The run hosted on its pod

### Roles and where they go

The roles and their interfaces stay what they are. What changes is where each runs, and so how it reaches what it
exchanges.

| Role | `placement = "cluster"` | `placement = "pod"` | Interface |
|---|---|---|---|
| Launch, admission | The monitor or the CLI records the launch; Kueue admits the RayJob | The same; Kueue admits the run's keeper (a RayJob of a fraction of a CPU), and the pod's slot is the provider's `max_pods`, as now | Unchanged |
| The run's keeper | Part of the driver (`Run._leased`: claim, renew, release) | The cluster's RayJob: it claims the pod, writes the launch to its lease, renews it, counts its spend, follows the driver's beats and releases the pod when the run ends | `Pods`, the lease (one field more: the launch) |
| Driver, loop | The RayJob's head pod | A Ray job on the pod's own Ray head, started by the pod from its lease | `python -m rollout_train.jobs LAUNCH`, unchanged |
| Gateway | In the driver's process | In the driver's process, sampling the pod's vLLM on `127.0.0.1:8000` | Unchanged; the leased server's address is the loopback one when the gateway is on that pod |
| Runners, environment | In the driver's process, the environment from its version's runtime environment | The same, in the pod's Ray | Unchanged |
| Sandbox pools | The cluster's pool, by `url` | A pool in the run's Ray on the pod (`SandboxPool` over the provider), with its keeper | The claiming interface, unchanged |
| Trainer | `RemoteTrainer` to the pod's Envoy, over mutual TLS | `RemoteTrainer` to `127.0.0.1:8001`; its files through the pod's blob cache | Unchanged |
| Engine, follower | The pod | The pod | Unchanged |
| Evals | Played by the run's runner on its channel | The same, on the pod | Unchanged |
| Monitor | Reads the ledger, the beats and the feed's files on the shared volume | Reads the ledger and the beats; the feed goes beside the ledger (below) | The feed's transport |
| Ledger, blobs | Postgres; R2 | Postgres through the ledger service and the tunnel; R2, with the pod's disk as a cache | `HttpLedger`, `CachedBlobs`, unchanged |
| The cluster's gateway | Serves clients elsewhere | The same, reaching the pod over mutual TLS as now | Unchanged |

### Ray on the pod

The host image already holds Ray (`ray[default]` is a dependency of `rollout-train`). Its entrypoint starts a head,
bound to the loopback interface, with its temporary directory on the volume:

```bash
ray start --head --node-ip-address 127.0.0.1 --dashboard-host 127.0.0.1 --disable-usage-stats \
    --num-gpus 1 --num-cpus "$ROLLOUT_RAY_CPUS" --memory "$ROLLOUT_RAY_MEMORY" --temp-dir /workspace/ray
```

with CPUs and memory left over for vLLM, the follower, the training service and Envoy (on a PRO 6000 pod: 12 of 16
vCPUs, 160 of 188 GB). None of Ray's ports is mapped, so none is reachable from outside the pod.

A new process on the pod, the **run host** (`rollout_train.pods.hosting`), follows the pod's lease as the follower
and the training service do. When the lease names a run and its launch, it submits `python -m rollout_train.jobs
LAUNCH` to the local head through Ray's job API (`127.0.0.1:8265`), in the version's runtime environment, with the
cluster config a pod-hosted run needs (below); it beats with the job's state; when the lease is released or names
another run, it stops the job. The pod pulls its work, so nothing reaches the pod that does not today.

The driver then does what it does in a RayJob: reserves its placement group (here the trainer and the bridge need no
bundle: both are the pod's services), starts its gateway, runners and pools, and runs the loop. A run whose engine
hosts are `vllm` actors on the pod (instead of the image's vLLM server and follower) is a later choice; the image's
processes already serve what the run's serving records say.

### The keeper in the cluster

Kueue, the launch, `followed`, the monitor's queue and Stop all work against a RayJob, so the run keeps one. With
`placement = "pod"`, `rollout_train.jobs` runs as the run's keeper: it claims the pod (`Pods.claim`, warm or new, as
now), writes the launch into the lease, renews every 30 seconds (counting the pod's time toward `limits.spend`, and
ending the run at `limits.hours`), and follows the run host's beats. The run ends when the driver on the pod records
its end; the keeper then releases the pod warm. If the driver's beats stop for longer than `STALE` while the pod beats,
the run host started it again or will; if the pod's beats stop, the keeper ends the run failed, and the reaper deletes
the pod. A Stop from the monitor reaches the keeper as now, which writes it to the lease, and the run host stops the
job (which stops cleanly, as a RayJob's driver does).

The keeper's demand is a tenth of a CPU and 256 MiB: what Kueue admits is the run's place in the queue, and the pod's
slot, not the cluster's CPUs.

### The ledger from the pod

**Per turn, now.** The gateway reads the program's run's whole turn table once and appends one record. In the
cluster those are local Postgres queries of a few milliseconds. Through the ledger service and the tunnel each is a
round trip from the pod to the nearest Cloudflare edge, through the tunnel to the cluster, and back: tens to a few
hundred milliseconds, and the table grows with the episode.

**What the pod's gateway does instead:**

1. **Its own index.** The run's gateway is the only process that records its sessions' turns (a run's harnesses reach
   their run's gateway). It keeps each program's run's index in memory, fed by its own appends, and reads the table
   only on a miss (the first turn of a program's run it has not seen: after its driver started again). That removes
   the read.
2. **Group commit.** Appends go to a writer that sends what has accumulated every 250 ms (or at 64 records) in one
   request, `ledger/append_many`, which applies each record as its own append: under its own fence, first append
   wins, its own answer. The gateway still replies only once its turn's append has an answer. A turn waits up to the
   window plus one round trip, about 0.3 to 0.4 s, which every turn in the window shares; today's recording takes
   0.68 s.

Every guarantee of [the ledger](ledger-guarantees.md) holds unchanged, because nothing is appended later than the
reply or under a different fence: a batch is several appends in one request. A record refused as `Fenced` is refused
alone, and its turn's request is refused as now. Order holds: the runner's episode record is appended after its turns'
answers came back, as now. The service applies a batch in one transaction with a savepoint per record, so one refusal
does not undo the others.

**A durable buffer on the pod** (reply once a turn is in a journal on the pod's volume, flushed later) would take the
round trip off every turn. It changes the guarantees: a reply could be given for a turn whose fence was taken, and the
journal must be flushed before a runner started again adopts its claims, or the adopted runs' turns are refused under
the fence the adoption took. Group commit gets most of the gain without that; the journal waits until the round trip is
measured as a real share of a turn.

**Turns only as blobs, with a summary per episode.** The per-turn records are small (393 bytes, 1.4 KB a second at
this run's rate, 65,000 records in a five-hour run), and they are what dedupes a retried request, what the monitor
shows of an episode while it plays, and what segments are built from. With group commit their latency is shared; their
volume is nothing to Postgres. They stay.

**Other writes** go straight through the service as now: fences, claims, episodes, groups, results, steps, checkpoints,
serving records, beats. A few a second at most.

**Crash recovery** is the loop's own ([the training loop](../libraries/rollout-train/training.md)): a driver that died
with the pod (a host restart, a container that exited) is started again by the run host once the container is back; it
takes the run's fence anew and goes on from the ledger. Episodes playing on the pod are lost with their runner, their
claims lapse, and they are played again. A pod deleted outright ends the run failed; `rollout resume` starts it again
on another pod. A run hosted on a pod recovers exactly as one in the cluster does, at the granularity of its episodes.

### Blobs

From the pod, R2 is near (Cloudflare's edge), and the host image's blob cache (`ROLLOUT_BLOB_CACHE`) keeps a copy of
every blob the pod puts or reads, so:

- a turn's blob is put once and read back from the pod's disk;
- the loop builds each step's segments from turns on the pod's disk, and puts the batch, which the training service
  reads from the same disk;
- the step's weights and state are put once by the training service, and the driver adds the checkpoint from their
  manifests (phase 0), so nothing is downloaded anywhere for a step.

The bucket holds everything durable, as now; the pod's disk is a cache.

### The monitor's live view

A run's feed (`RunFeed`: events, samples, the loop's notes) is files in the driver's directory, which the monitor reads
from the cluster's shared volume. On the pod the driver writes them to the pod's disk. The feed goes beside the ledger
instead: the driver sends its lines in batches to the ledger service (`feeds/RUN`, a bounded table: the newest
`monitor.feed_episodes` episodes' lines), which the monitor reads as it reads the ledger. The same path then serves a
run in the cluster, and the shared volume stops being how the monitor reaches a run.

### Sandboxes on the pod

A RunPod pod is a container run by RunPod's Docker: it cannot run Docker of its own (RunPod: "you cannot spin up your
own Docker instance"), so no nested containers, and no KVM for microVMs (E2B's single-host build needs KVM). What it can
run is processes:

| Sandbox | On a pod |
|---|---|
| Minecraft worlds | Yes: a Paper server (Java), a Node process holding the team's bots, and the provider's Python, all on the loopback interface. The image needs Java 21, Node and the harness's `node_modules` |
| Toy games, text tools, a Python REPL | Yes, as processes |
| Repositories with their own images (SWE tasks), browsers in containers, VMs | No: a sandbox service (Modal, E2B, Daytona, Prime's), or a pool in the cluster, reached through the claiming interface |

A pool on the pod is a `SandboxPool` in the run's Ray, sized from what the pod has: the run's demand on a pod is
checked against the pod's CPUs and memory, which the run owns whole, so the objection that kept pools out of runs'
drivers in the cluster (memory Kubernetes does not account) does not apply. Its leases sit beside the ledger, through
the service, under the run's token.

How many Minecraft worlds a pod holds, by CPU (about one a world while it plays) after vLLM, the trainer and the
driver take about 4: 10 to 12 on a PRO 6000 pod (16 vCPUs, 188 GB), 3 or 4 on the cheapest H100 SXM pod (8 vCPUs,
125 GB), 12 on an H100 PCIe pod (16 vCPUs, 251 GB). Memory is never the limit on a pod (2.4 GiB a world at most); CPU
is. The cluster holds 4 now, bounded by its 23 GB.

### Security: the least a hosted run's pod needs

The pod now runs the run's driver, so it needs what the driver needs for one run, and nothing of the platform's:

| It needs | As | Scope |
|---|---|---|
| The ledger | A run's token (`rlr1.`), signed with the platform's secret as pods' tokens are, given in the lease and replaced when another run takes the pod | Read and append the run's own tables and those of its evals' runs (`runs/RUN*`); take fences of the run's scopes (the run, its runners, its episodes, its pools); append checkpoints of the run and read those it starts from and their bases; serving records; beats under the run's and the pod's names; its own lease; its launch's state; the run's desired settings; environment versions and suites, read. Nothing of other runs, presets, pod leases of other pods, launches of other runs |
| The bucket | Temporary R2 credentials, minted by the keeper with an account token only the cluster holds, given in the lease, good for a few hours and renewed with it | Object read and write on the bucket. R2 can scope them to prefixes, but blobs are content-addressed under one prefix shared by every run, so a run's key can be bounded in time, not in what it writes |
| Its certificate | The one-time token from step-ca, as now | The pod's own identity |
| Hugging Face | `HF_TOKEN` by name, kept in RunPod's console | Downloads |

It does not get: RunPod's API key, step-ca's provisioner key, the Postgres password, the cluster's own S3 keys,
Tinker's or hosted APIs' keys, the platform's ledger token, the cluster's Ray token, the tunnel's token. A pod-hosted
run whose channels include a hosted API or Tinker is refused in phase 1; later its turns on them go through the
cluster's gateway, which holds those keys, reached through the tunnel with a key the run mints.

**The environment's code beside the trainer.** A published environment's code runs in the driver's process today, in
the cluster, with whatever the RayJob's pod holds (and every role still gets every Secret there, which the security
work under way narrows). On the pod it would run beside the run's ledger token and bucket key, in a container whose
processes run as root. The steps:

1. Phase 1 hosts on pods only environments the cluster offers or versions published from sources it lists as trusted
   (`[pods] trusted_sources`); validation refuses the rest with `placement = "pod"`.
2. Later, programs run in a process of their own (the environment worker of the [runtime design](runtime-design.md),
   which the runner talks to), under a user of their own with no secrets in its environment and none readable: a second
   Ray node in the pod's container, started as that user with an emptied environment and a resource of its own
   (`play`), on which runner actors are placed. The runner's ledger writes then go through the driver's process, so
   the user that runs the environment's code holds no token at all. GPU device files are usually readable by every user
   in a container, so this keeps secrets from the code, not the GPU.

### Several pods

| Shape | Where the loop and runners sit | What crosses between pods |
|---|---|---|
| One host pod (trainer and engine on one GPU) | On it | Nothing |
| A trainer pod and an inference pod | The loop, runners, gateway and pools on the inference pod, where turns are made | The batch to the trainer and the checkpoint back, through R2 (or Global Networking, below); steps' requests to the training service over mutual TLS |
| Several inference replicas | Each replica's pod hosts runners and a gateway that samples its own engine, sessions kept on it; one pod also hosts the loop | Ledger records and blobs only: each pod's runners claim episodes from the ledger, as runners on several machines do now |
| Mixed: pods and the cluster | Runners in the cluster too, claiming the same run's episodes, sampling through the cluster's gateway | As now, for those episodes |

Pods coordinate through the ledger and the blob store, not through one Ray cluster spanning them. RunPod's Global
Networking gives an account's pods private addresses that reach each other on any port, in 17 data centres, at
100 Mbit/s between pods: enough for a Ray cluster across pods, too slow to move weights (321 MB takes 26 s at
100 Mbit/s) and absent outside those data centres. Each pod a Ray cluster of its own, every extra pod a run host
following its lease ("play run R"), works everywhere. A lease then names a pod's part in the run: `host` (the loop
and runners), `play` (runners and gateway only), `trainer`.

## What it gains and what it costs

| Change | Gain, in the measured run's terms |
|---|---|
| Phase 0 | With main's wait for a server, what remains of the 27 failed episodes of 66 and the 16% of turns sampled again; a gigabyte a step not brought down to the cluster (80 s a step at 100 Mbit/s, 8 s at a gigabit) |
| The run on its pod: latency | About 0.4 to 0.7 s of a 19.7 s turn (the two round trips to the pod and R2 from the cluster, less group commit's wait): 2 to 4% of a turn here, 15 to 40% of a 1 to 3 s turn of a short-reply workload |
| The run on its pod: capacity | Episodes and worlds bounded by the pod's 16 vCPUs and 188 GB instead of the cluster's share of 20 CPUs and 23 GB: 10 to 12 Minecraft worlds on a PRO 6000 pod against 4 now. Keeping vLLM's batch full (this run held 104 requests in flight) is what turns the pod's hourly price into tokens; more episodes at once is how |
| The run on its pod: the cluster | The cluster's CPU and memory free for runs on its own GPU, evals and the monitor; the cluster could be away, briefly, without stopping a pod's run (its ledger writes wait and retry) |
| The run on its pod: cost | The pod's CPU and memory are in its price already ($2.09 an hour for the PRO 6000 pod); nothing more is rented |

What it costs to build, in days of work for one person who knows the code (rough):

| Phase | Work | Days |
|---|---|---|
| 0 | Keep a server across a missed look (drop it only after several, or only for `generate`'s own failures; a turn already waits one out); size the connection pool from the turns in flight (`episodes_at_once` × slots × 2) and keep those connections alive; add a checkpoint from a step's manifests without downloading them; the gateway's own index of its sessions' turns | 2 to 3 |
| 1 | `placement` and its validation; the keeper (`Run._leased` as a job of its own); the launch in the lease; the run host on the pod and Ray's head in the host image's entrypoint; a run's token scope in the ledger service; loopback routing to the pod's own engine and training service; the cluster config a pod-hosted run is handed; the feed beside the ledger; tests with the fake RunPod and processes standing in for the pod | 10 to 15 |
| 2 | `ledger/append_many` and the group-commit writer; temporary R2 credentials minted by the keeper | 4 to 6 |
| 3 | Sandbox pools on the pod: Java, Node and the harness in the host image (or an image per environment), a pool sized from the pod, the pool's leases through the service | 4 to 6 |
| 4 | Imported environments on pods: programs in a process of their own under a user with no secrets (with the environment worker) | 10 or more, mostly the environment worker's |
| 5 | Several pods: parts in the lease (`host`, `play`, `trainer`), runners and a gateway per replica's pod, a trainer pod reached by the host | 10 to 15 |

## Phases

Most value first. Each later phase waits for its trigger.

| Phase | Build | Trigger |
|---|---|---|
| 0. The remote path made sound | As above | Now: 41% of episodes failed on it before main's wait for a server, the pool is still smaller than the turns in flight, and every multi-pod shape keeps a remote path |
| 1. The run hosted on its pod, trusted environments, no sandboxes | `placement = "pod"` for runs whose trainer and trained channel are one `runpod-host` provider and whose environment needs no sandbox | Phase 0 done and the round trip to the pod measured: start when a run is bounded by the cluster's CPU or memory (more episodes at once than its RayJob holds), or when a short-reply workload spends a fifth of a turn on the path |
| 2. Group commit, temporary bucket keys | As above | With or right after phase 1, once the ledger's round trip from the pod is measured: if recording a turn from the pod takes more than about 0.3 s |
| 3. Minecraft on the pod | As above | The first Minecraft run on a pod, or when the cluster's 4 worlds bound a run |
| 4. Imported environments on pods | As above | The first run on a pod of an environment from a source the cluster does not trust |
| 5. Several pods | As above | A run that needs more than one GPU's engine, or a trainer on a GPU of its own ([scaling](scaling-models-and-topologies.md) phases 2 and 4) |

## What changes

**Run settings.** `placement = "cluster" | "pod"`, a fixed setting, recorded in the run's start, `"cluster"` unless
said. With `"pod"`, validation requires (phase 1): the trainer and the trained channel on one provider of kind
`runpod-host`; every other channel on that pod or refused; no sandboxes (phase 3: kinds the pod's image can run);
an environment the cluster trusts on pods; `episodes_at_once` and the pools' sizes within the pod's CPUs and memory
less what its own processes take.

**Cluster config.**

```toml
[inference.pro6000]
kind = "runpod-host"
hosts_runs = true            # runs may be hosted on its pods
run_cpus = 12                # of the pod's vCPUs, what the run's Ray may use
run_memory_gib = 160

[pods]
trusted_sources = ["https://github.com/by77er/distributed_agents_environments"]

[sandboxes.minecraft]
url = "http://sandboxes-minecraft.rollout:8710"   # runs in the cluster
on_pods = { size = 10 }                           # a run on a pod makes a pool of its own of this size
```

The pod-hosted run's driver is handed a cluster config of its own, made by the keeper from the cluster's: the ledger
as the ledger service (`HttpLedger`, the run's token), the blobs as the pods' store (temporary keys), the pod's
loopback addresses for its engine and training service, and nothing it may not use.

**The chart.** The RayJob template gains the keeper's shape (a head of a tenth of a CPU and no worker group), chosen
by `submitting.rendered` from the run's placement. The tunnel's ledger hostname, step-ca and the reaper are as now.
A Secret for the Cloudflare account token that mints R2's temporary credentials, mounted only in runs' keepers. The
`sandboxes` Deployments are as now, for runs in the cluster.

**The images.** The host image's entrypoint starts Ray's head and the run host (`rollout_train.pods.hosting`) beside
its processes; `supervise.sh` ends the container when any ends, as now. Ray's temporary directory on the volume. For
phase 3, Java 21, Node and the Minecraft harness's `node_modules` in the host image (about 400 MB more), or an image
per environment (an open decision below). The inference and trainer images are unchanged.

**The code.** `rollout_train.jobs` runs as the keeper or as the driver by placement; `pods.leases` gains the launch and
the run's token and keys; `pods.hosting` is new; `pods.routing` resolves the pod the gateway runs on to its loopback
address; `ledger_service.scopes` gains the run's scope; `ledger_service` gains `append_many` and feeds; the gateway's
`TurnStore` keeps its index; `RemoteTrainer.step` returns manifests and `Checkpoints.add` takes them.

## Open decisions

1. **The setting's name and values.** `placement = "home" | "pod"` names a deployment. *Recommendation:*
   `placement = "cluster" | "pod"`: the run's Ray cluster in the platform's cluster, or on the pod it leases. Several
   pods later stay `"pod"`, with each pod's part in its lease.
2. **Who leases the pod.** The driver (as now) cannot, when it runs on the pod. *Recommendation:* the run's keeper,
   a RayJob of its own admitted by Kueue, so the queue, Stop and `followed` stay as they are.
3. **Push or pull the driver onto the pod.** The keeper could submit the driver through Ray's job API on the pod (a
   route behind Envoy). *Recommendation:* pull: the run host reads its lease. Ray's job API runs arbitrary code, and
   behind Envoy it would make the gateway's certificate a key to every pod.
4. **The pod's vLLM server, or engine hosts as actors.** On the pod the run's Ray could start a `vllm` engine host
   actor instead of the image's server and follower. *Recommendation:* keep the image's processes in phase 1 (they
   follow the serving record, are tested, and serve the cluster's gateway too); revisit when the same pod should serve
   a run's other channels.
5. **Group commit, or a journal on the pod.** *Recommendation:* group commit (every guarantee unchanged); a journal only
   if the round trip through the tunnel is measured as a real share of a turn after it.
6. **Turns' records.** *Recommendation:* keep one record per turn beside its blob; batch their appends. Summaries per
   episode only if the ledger's rows become a cost.
7. **The feed's transport.** Files on a shared volume do not reach a pod. *Recommendation:* the feed in batches beside
   the ledger, for every run, so the monitor reads every run one way.
8. **Untrusted environments on pods.** *Recommendation:* refuse them with `placement = "pod"` until programs run under a
   user with no secrets; do not rely on RunPod for isolation between the run's own processes.
9. **Channels on hosted APIs and Tinker, from a pod.** *Recommendation:* refused in phase 1; later sampled through the
   cluster's gateway over the tunnel, so their keys stay in the cluster.
10. **One host image, or an image per environment.** Minecraft needs Java and Node. *Recommendation:* one host image
    with them while Minecraft is the one heavy environment; an environment declares what it needs of its image, and a
    second image when a second environment needs something else.
11. **Several pods: one Ray cluster, or one each.** *Recommendation:* one each, coordinated through the ledger, which
    works without Global Networking and is what runners on several machines already do.
12. **Which GPU types host runs.** CPU per GPU varies: 16 vCPUs on a PRO 6000 or H100 PCIe pod, 8 on the cheapest H100
    SXM. *Recommendation:* `hosts_runs` per provider, with `run_cpus` from what RunPod gave the pod (its lease records
    the pod's vCPUs and memory from RunPod's API) rather than from a fixed figure.

## Sources

All read on 2026-10-05.

**RL frameworks**

- verl: [agent loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html),
  [agentic RL](https://verl.readthedocs.io/en/latest/start/agentic_rl.html),
  [`agent_loop.py`](https://github.com/volcengine/verl/blob/main/verl/experimental/agent_loop/agent_loop.py),
  [`llm_server.py`](https://github.com/volcengine/verl/blob/main/verl/workers/rollout/llm_server.py),
  [Sandbox Fusion](https://verl.readthedocs.io/en/latest/sglang_multiturn/sandbox_fusion.html)
- SkyRL: [inference architecture](https://docs.skyrl.ai/docs/getting-started/inference_architecture),
  [`skyrl_gym_generator.py`](https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl/train/generators/skyrl_gym_generator.py),
  [Mini-SWE-Agent](https://github.com/NovaSky-AI/SkyRL/blob/main/docs/content/docs/examples/mini_swe_agent.mdx),
  [Harbor](https://github.com/NovaSky-AI/SkyRL/blob/main/docs/content/docs/harbor/index.mdx),
  [SkyRL-Agent](https://arxiv.org/abs/2511.16108)
- prime-rl: [overview](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/overview.md),
  [scaling](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/scaling.md);
  [verifiers' architecture](https://docs.primeintellect.ai/verifiers/v1/architecture.md),
  [Prime Sandboxes](https://docs.primeintellect.ai/sandboxes/overview.md),
  [INTELLECT-3](https://www.primeintellect.ai/blog/intellect-3)
- AReaL: [paper](https://arxiv.org/abs/2505.24298),
  [`rollout_controller.py`](https://github.com/areal-project/AReaL/blob/main/areal/infra/controller/rollout_controller.py),
  [agent workflow](https://github.com/areal-project/AReaL/blob/main/docs/en/reference/agent_workflow.md)
- slime: [`placement_group.py`](https://github.com/THUDM/slime/blob/main/slime/ray/placement_group.py),
  [`rollout.py`](https://github.com/THUDM/slime/blob/main/slime/ray/rollout.py),
  [fully async](https://github.com/THUDM/slime/blob/main/examples/fully_async/README.md),
  [coding agent](https://github.com/THUDM/slime/blob/main/examples/coding_agent_rl/README.md)
- OpenRLHF: [`vllm_engine.py`](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/ray/vllm_engine.py),
  [`agent.py`](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/agent.py),
  [agent training](https://openrlhf.readthedocs.io/en/latest/agent_training.html)
- NeMo-RL: [environments](https://docs.nvidia.com/nemo/rl/latest/guides/environments.html),
  [design](https://docs.nvidia.com/nemo/rl/latest/design-docs/design-and-philosophy.html),
  [NeMo Gym](https://docs.nvidia.com/nemo/rl/latest/design-docs/nemo-gym-integration.html)
- Tinker: [documentation](https://tinker-docs.thinkingmachines.ai),
  [under the hood](https://tinker-docs.thinkingmachines.ai/tinker/under-the-hood/index.md)
- ROLL: [`environment_worker.py`](https://github.com/alibaba/ROLL/blob/main/roll/pipeline/agentic/environment_worker.py),
  [agentic pipeline](https://github.com/alibaba/ROLL/blob/main/docs_roll/docs/Development/Architecture/AgenticPipeline.md),
  [ROCK](https://arxiv.org/abs/2512.24873), [RollArt](https://arxiv.org/abs/2512.22560)

**Sandbox services**

- E2B: [infrastructure](https://github.com/e2b-dev/infra), [BYOC](https://docs.e2b.dev/byoc.md),
  [pricing](https://e2b.dev/pricing)
- [Modal sandboxes](https://modal.com/docs/guide/sandbox)
- Daytona: [architecture](https://www.daytona.io/docs/en/architecture), [limits](https://www.daytona.io/docs/en/limits)
- Measurements: [LogRocket's comparison](https://blog.logrocket.com/comparing-ai-agent-sandbox-platforms-e2b-modal-daytona),
  [Orchard](https://arxiv.org/abs/2605.15040)

**RunPod, Ray, R2**

- RunPod: [pods](https://docs.runpod.io/pods/overview), [exposing ports](https://docs.runpod.io/pods/configuration/expose-ports),
  [storage](https://docs.runpod.io/pods/storage/types), [Global Networking](https://docs.runpod.io/pods/networking);
  the vCPUs, memory and prices of secure-cloud pods from RunPod's public GraphQL `gpuTypes` listing (the cheapest offer
  of each type on the day, which changes with stock)
- [Ray's ports](https://docs.ray.io/en/latest/ray-core/configure.html)
- [R2's temporary credentials](https://developers.cloudflare.com/r2/api/s3/temporary-credentials/)
