# Where sandboxes run

**Status: built.** A pool served from a pod of its own on Kubernetes, reached through the claiming interface. A design
note: see [Design notes](README.md) for the others.

A run and its sandboxes are separate systems. A run reaches a sandbox only through the claiming interface
([sandboxes](../libraries/rollout/sandboxes.md)): it acquires a lease under its episode's claim
(`RUN/GROUP/EPISODE/ATTEMPT/NAME`), performs operations on it, and releases it; a lease ends with its claim. What a
sandbox runs, where, and what it costs in CPU and memory is its pool's business, scheduled and accounted by whatever
runs the pool. A run's demand ([what a run needs](../libraries/rollout-train/launching.md#what-a-run-needs)) counts
nothing of its sandboxes.

Code: `rollout.harness.remote` (`serve_pool`, `RemotePool`), `rollout train`'s `rollout pool`,
`rollout_train.jobs.Run._pools`, `deploy/chart/rollout/templates/sandboxes.yaml`.

## What a Minecraft world is

A Minecraft world is a sandbox of the kind `minecraft`, made by `minecraft_team.worlds:worlds`: a Paper server (Java,
its heap capped), a Node process that holds the team's bots and talks to the server on the loopback interface, and the
Python of the provider that drives both. An episode never reaches the server or the bots: its program performs the
pool's operations (`observe`, `act`, `window`, `score`), and the provider turns each into calls to the bots' process
and the server's plugin. So everything a world needs (the server, the bots, the Python between them) lives on one
machine, and the pool's HTTP interface is all a run needs to play in it. A world holds 1.1 to 1.85 GiB, up to 2.4 GiB
when the bots roam far ([Minecraft memory](minecraft-memory.md)); during play its server ticks and generates chunks
on about one CPU.

## The options

| Option | What runs where | For | Against |
|---|---|---|---|
| A pool in the run's driver | The provider in the driver's process, its worlds beside it in the run's pod | No service to run; the run's own pool | Kubernetes accounts nothing of what the worlds hold unless the run's demand models them, which puts the provider's internals in the platform |
| **A pool served from a pod of its own** | A Deployment of one pod running `rollout pool --kind KIND`, its worlds inside; a Service runs reach | The claiming interface, served over HTTP, is the only coupling; Kubernetes accounts the worlds by the pod's requests; built from what exists (`serve_pool`, `RemotePool`, leases beside the ledger) | The pod's requests are held while it is idle; one pod is the most a pool has, so its size is one node's worth |
| One world per pod (a StatefulSet) | Each pod a pool of one world, and a pool in front that spreads leases over them | A world over its memory limit is killed alone | A pool of pools to route leases and operations; the same requests held while idle |
| A pod for each lease | The provider creates a pod per world through the Kubernetes API, and deletes it with the lease | Memory is requested only while a world exists; an unschedulable world waits like a full pool | The provider needs the API (a service account, a pod template, waiting for readiness), and each world waits for a pod to start |
| Ray actors in a resource pool of their own | A Ray cluster for sandboxes, each world an actor | Ray schedules worlds by declared memory | A second Ray cluster beside the runs' RayJobs, and the provider's footprint declared to Ray again |

## The choice

A pool served from a pod of its own. It is the smallest change that keeps the claiming interface the only coupling:
a run's binding names the pool's URL (`PoolBinding(url=...)`), its runner asks the pool for room before it claims an
episode, and acquires, operates and releases over HTTP exactly as it does in process.

- The cluster config's `[sandboxes.KIND]` describes the pool: its `provider`, `size` and the provider's settings. With
  `url`, runs reach the pool there and make none of their own; `rollout pool --kind KIND --cluster` serves the pool the
  section describes, its provider made with `[scratch]/sandboxes/KIND`, its leases beside the cluster's ledger
  (ending with their claims, swept by its keeper, which beats as `pools/KIND`).
- The chart runs one Deployment per enabled kind (`sandboxes` in its values): one pod, `Recreate` (two pods of one pool
  would each delete the other's sandboxes), with requests for `size` worlds and a memory limit for `size` roaming
  ones, behind the Service `sandboxes-KIND`, and writes `url` into the cluster config. A pod started again finds its
  old leases lost; their episodes are played again.
- On Kubernetes, validation refuses a run whose environment needs sandboxes whose pool has no `url` and that no pod
  of the run's serves: a pool in the run's pod would hold memory Kubernetes does not account.
- A kind may also be served from the RunPod host pods a run leases (`on_pods`), on the CPUs and memory their engine and
  trainer leave, with the pool at `url` taking what they have no room for
  ([sandbox pools on a host pod](run-placement.md#sandbox-pools-on-a-host-pod-built)).
- On one machine, without `[kubernetes]`, the pool is made in each run's driver as before; `[guards] runs_gib` keeps a
  runner from claiming episodes while the machine is short of memory.

A pool served elsewhere runs the provider code its pod has (the platform's image), not that of a published version a
run plays: the pool is the sandbox system's, at its own version.

A pod for each lease is the step after this one, where idle requests matter: on a node shared with runs, the pool's
requests are held while no run plays.
