# Core systems review

**Status: in progress.** A review of every core system, taken at `f4de7f3`, with the fixes it led to. The findings
marked **fixed** are fixed on the branch `claude/core-systems-fanout-review-mn59p0`, each change to code with a test
that failed before it; the others are proposed, each with the change it needs. A design note: see
[Design notes](README.md) for the others.

Ten reviews ran side by side, one for each system: the harness and sandboxes, pods, the ledger, the gateway and
inference, the training loop and episode runners, the cluster config and run settings, launching and the driver, the
monitor, the trainers and their objectives, and the deployment. Each finding was traced from a real caller, and most
were reproduced with a script before they were written down; the reproductions were run again, apart from the review
that wrote them, before a finding was kept. An earlier review's claims were checked the same way, and each is answered
in [a section of its own](#the-earlier-reviews-claims).

Paths are relative to the repository root, except that `RT/` stands for `libraries/rollout-train/src/rollout_train/`,
`R/` for `libraries/rollout/src/rollout/` and `OBJ/` for `implementations/rollout-objectives/src/rollout_objectives/`.
Severity is how much harm a finding does where it can happen: **high** loses work, hangs or kills runs, corrupts
training, or opens a hole an outsider can use; **medium** does so under rarer conditions or less harmfully; **low** is
wrong but cheap.

## What holds

The core designs are sound, and the review found no way around them:

- **Fencing.** `FileLedger` and `DatabaseLedger` (SQLite and Postgres) compare a writer's fence with the newest
  inside the same lock or transaction as the append; Postgres appends take the scope's lock before the table's, so
  they cannot deadlock; the first append of a key wins atomically. A stale writer cannot append.
- **The loss math.** Every objective family (PPO, dual clip, CISPO, GSPO; the k1, k2 and k3 KL estimators; DPO, IPO,
  SimPO, ORPO, KTO; distillation) matches `RT/objectives.py`'s declaration in sign, direction and units; logprobs are
  gathered shifted by one; only sampled tokens are trained; packed sequences do not see each other. Advantages,
  preference pairs and the group algorithm are right, with a zero-variance guard.
- **The gateway's keys.** Signed with HMAC-SHA256, compared in constant time, the signature over the whole payload,
  expiry enforced, scoped to a run, a slot and a session. A turn is recorded before its reply is sent, under its
  episode's fence.
- **The monitor's access.** Every `/api` route needs the token (compared in constant time); a request that changes
  anything must be JSON from the monitor's own origin; the cookie is `HttpOnly` and `SameSite=Strict`; a Host
  allowlist stops DNS rebinding. No route that changes state or returns ledger data is reachable without the token.
- **The deployment's secrets.** Each role is given Secrets by name, and a test pins which; ingress is denied by
  default, with tests; an admission policy keeps the monitor's right to make RayJobs from making privileged pods.
  Envoy pins the gateway's SPIFFE identity inside the TLS handshake.
- **Pods.** Leases are compare-and-set; pod tokens are honoured only while the lease names their run.

What fails is mostly at the edges of these: what happens when an operation raises instead of returning, when an
answer is lost, when a value is out of range, or when imported code is hostile.

## Where it stands

| System | Findings | Fixed here | Proposed |
|---|---|---|---|
| [Episode runners and the driver](#1-episode-runners-and-the-driver) | 14 | 9 | 5 |
| [The ledger and its service](#2-the-ledger-and-its-service) | 9 | 6 | 3 |
| [Launching and job status](#3-launching-and-job-status) | 11 | 9 | 2 |
| [The gateway and inference](#4-the-gateway-and-inference) | 12 | 10 (one in part) | 2, and the rest of one |
| [The harness and sandboxes](#5-the-harness-and-sandboxes) | 7 | 7 | |
| [Pods](#6-pods) | 10 | 9 | 1 |
| [Run settings and validation](#7-run-settings-and-validation) | 12 | 11 | 1 |
| [The trainers](#8-the-trainers) | 7 | 3 | 4 |
| [The monitor](#9-the-monitor) | 10 | 7 | 3 |
| [The deployment](#10-the-deployment) | 10 | 2 (one in part) | 8, and the rest of one |
| [Environments and the process](#11-environments-and-the-process) | 7 | 4 | 1 (and two observations) |
| **All** | **109** | **77** | **30** |

Every change to code has a test that fails without it. With them, on Linux without the GPU extras (a plain `uv sync`),
ruff and its formatter pass, `helm lint` passes, the monitor page's checks pass (`tsc`, eslint, 117 vitest tests), and
the test suite passes, the tests that need an extra skipping by name ([X2](#11-environments-and-the-process)). The fixes
that need torch were run with a CPU build of it beside the workspace. The order to take the rest in is the last section,
[what to do next](#what-to-do-next).

## 1. Episode runners and the driver

The runner and the driver do not survive an operation that raises. The README's "no work is lost" holds when a
process is killed; until these fixes it did not hold when an exception was raised.

**T1 · high · fixed.** `RT/rollouts/scheduler.py` `EpisodeRunner.serve`: nothing guarded one look for work, so one
ledger, presence or recorder error (a Postgres failover, the ledger service restarting) ended `serve`, which cancelled
every episode in flight. Reproduced: `ledger.tables` failing once ends the runner, and an episode asked for later is
never played. *Fix:* each look is guarded; an error is logged and the runner looks again; `Fenced` (another runner
took its name) still ends it.

**J2 · high · fixed.** `RT/jobs.py` `_background`: the driver started the runner's `serve`, the gateway and its keeper
and never looked at them again; at the end `gather(..., return_exceptions=True)` dropped their errors. A dead runner
left the loop waiting for episodes forever, holding its GPUs and pods until `limits.hours`. *Fix:* background work
that ends with an error ends the run failed, saying why.

**T2 · high · fixed.** `scheduler.py` `_play`: only cancellation was handled. Any other error after a claim (taking
the episode's fence, reading the plan, the run's events, the gateway's turns, storing the episode) ended the task
silently; the claim stayed its episode's latest, held while the runner beat, so the episode was never offered again
and its group never ended. Reproduced: one `ConnectionError` from the recorder strands `1/1/1` for good. *Fix:* the
attempt is noted cut short (`interrupted`) and played again; an episode cut short this way `LOST_AT_MOST` times is
recorded failed. A runner's own claim holds for it only while it plays it, so a claim whose note could not be written
either is claimed again.

**T3 · medium · fixed.** `scheduler.py` `_playing` was keyed by `GROUP/EPISODE/ATTEMPT` alone. The driver's runner
plays the training run and its evals, whose keys collide (every part starts at group 1): a second episode replaced
the first in `_playing`, so the runner played more than `places` at once (3 with `places=2`) and, closing, left one
uncancelled. *Fix:* keyed by run as well.

**T4 · medium · fixed.** `RT/jobs.py` and `RT/loop.py`: when KubeRay submits a lost driver's job again, the new
driver started the loop with the launch's `groups`, and the loop counts `groups` from its own start; only
`rollout resume` subtracted what was played. Each lost driver could overrun the run's budget by up to `groups`, and
J4's fix makes a lost driver's retry run whether or not anyone watches. *Fix:* the driver gives the loop the groups its
launch has left (`owed`: `groups`, less those played since the launch first started).

**T5 · medium · proposed.** `scheduler.py` `ended`: every outstanding group reads the run's whole `episodes` table
every 0.5 s, and an eval does so for each of its starts. At 50,000 episodes that is about 10 reads a second of 75 MB.
*Proposed:* one reader per run that every waiting group shares, and a keyed read (`WHERE key = ANY(...)`) on the
`Ledger` protocol.

**T6 · low · fixed.** `RT/algorithm.py`: a group of fewer than two completed episodes was always skipped, though an
advantage with no baseline, scale or filter needs no comparison; REINFORCE with `group_size = 1` trained on nothing
and said "1 of 1 episodes completed". *Fix:* two are needed only where a baseline, scale or filter compares them,
and the reason says so.

**T7 · low · fixed.** `RT/loop.py`: a run started again wrote a serving record with the launch's `max_lag` before it
read the run's desired settings, undoing a change made mid-run until the next step. *Fix:* the desired settings are
read first.

**T8 · low · fixed.** `RT/colocated.py`: the channels were paused outside the `try` that resumes them, so a cancel
while pausing the second left the first closed for good.

**T9 · low · proposed.** `RT/evals.py`: an eval's per-entry share solved counts only trainable episodes, so a
checkpoint that crashes the harness or overflows its context scores higher than one that plays and fails.
*Proposed:* count failed episodes that are not excluded as unsolved, and report how many failed.

**T10 · low · fixed.** `RT/imitation.py` `examples` took untrained slots' segments (judges, fixed opponents) and
excluded episodes. (Only tests call it; datasets use `turns_of`, which filters.)

**T11 · design.** Failed and cancelled episodes are dropped before a group's baseline. Dropping infrastructure
failures is right, and dropping overlong samples matches DAPO's overlong filtering; but a failure the policy causes
(a context overflow, a task error from its own output) then carries no gradient at all. *Proposed:* an optional floor
score (`advantage.failed`) given to failed, not excluded episodes that have trained segments, never to a lost sandbox
or a start that failed.

**T13 · medium · proposed.** `RT/loop.py`, `OBJ/terms.py`: `max_lag` bounds the checkpoint a turn begins on, not
how many steps it waits before it trains, so a long episode's early turns can be many checkpoints old by the step that
trains them, with truncated importance weights as the only guard (from the second earlier review). *Proposed:* mask a
token sampled more than `importance.max_age` checkpoints before the step's start, or mask (`icepop`) by default.

**T12 · scale · proposed.** Each runner's look, at least every 0.5 s, reads every fence ever taken (one per episode),
every beat, `SELECT DISTINCT name` over all records, every run's plans and pause state, and six whole tables of each
run it serves. There is no `LISTEN`/`NOTIFY` and no retention. This serves tens of concurrent episodes, not thousands.
*Proposed:* a server-side claim (`SELECT … FOR UPDATE SKIP LOCKED` over open episodes), keyed reads, notifications
for new groups, and retention for finished runs' tables.

## 2. The ledger and its service

**L1 · high · fixed.** `RT/ledger_service/service.py`, `client.py`: any database error became `500 "failed"`, and the
client retried only 502, 503 and 504, so a dropped connection or a deadlock was a hard failure for every caller (and,
in the runner, a stranded episode: T2). *Fix:* errors that rolled their transaction back (serialization failures,
deadlocks, lock and pool timeouts, too many connections, an admin shutdown) are `503 unavailable` for every
operation, which the client retries with the same request id; a lost connection, which may have committed, is 503
only for operations that are safe to repeat.

**L2 · high · fixed.** `RT/ledger_service/*`, `RT/launches.py`, `RT/registry.py`: answers were kept per replica, in
memory, once an operation finished, so a retry that reached another replica, a restarted one, or the same one while
the first attempt still ran did the operation again. Reproduced: one `launches.ask` whose answer was lost made two
launches (two jobs, and an `asked` launch that makes `rollout resume` refuse the run for good); `registry.create`
refused its own run's name as taken. *Fix:* the client chooses the launch's id and the run's id, and the stores
return what that id already made; a request that arrives while the same request runs waits for its answer.

**L3 · medium · fixed.** `service.py`: every answer, reads included, was kept (up to 4,096 whole bodies). Reproduced:
300 reads of a 0.8 MB table held 236 MiB, against the chart's 1 GiB limit. *Fix:* answers of operations that change
nothing are not kept.

**L4 · medium · fixed.** The ledgers store records with Python's JSON, which writes `NaN`; the service answered and
the client asked in strict JSON. One `NaN` metric (a gradient norm that overflowed) made the `checkpoints` table
unreadable through the service, for every pod. *Fix:* both sides read and write what the stores do.

**L5 · medium · proposed.** `RT/sql.py`: no `idle_in_transaction_session_timeout`, `lock_timeout` or TCP keepalives.
A client lost mid-transaction (a node gone, no FIN) holds its advisory locks until the server's keepalive, about two
hours: every run's checkpoint appends, or every runner of a run, wait behind it. Reproduced on Postgres. *Proposed:*
`SELECT set_config('idle_in_transaction_session_timeout', '60000', true)` in each shared transaction (it works behind
pgbouncer's transaction mode), keepalives in `connect_args`, and a `lock_timeout` whose error L1 retries.

**L6 · medium · proposed.** `RT/database.py` `copy` (`rollout ledger copy`) copies records, fences, the registry and
desired settings, and leaves presets, published environment versions and launches behind; `RT/relocating.py` misses
`environment_versions` too. After the migration `docs/deploy/stores.md` describes, runs on published environments
cannot be resumed. *Proposed:* copy them, and list the tables each tool moves in one place, checked by a test against
the schema.

**L7 · low · fixed.** `RT/checkpoints.py` `files`: two fetches of one checkpoint into one directory (the driver's
`bases/`, a follower) raced, and the second raised `Directory not empty`. *Fix:* a fetch that finds the directory
made whole by another uses it.

**L8 · low · proposed.** `RT/resuming.py`: the check for an open launch and the launch's record are separate
transactions with seconds between them, so two resumes at once launch the run twice (the ledger stays safe: the
second driver fences the first out). *Proposed:* refuse under the launches' lock, in `ask`.

**L9 · low · fixed.** `RT/registry.py`, `RT/launches.py`, `RT/presence.py`: the stores beside a `FileLedger` replaced
whole files without `fsync`, so a power loss could leave `registry.json` empty. *Fix:* the ledger's durable replace.
(`FilePodLeases`, `settings.json` and `sandboxes.json` still replace theirs without it: proposed.)

What L2 leaves: `registry/rename`, `registry/unbookmark` and `presets/save` and `delete` can still be done twice when a
retry reaches another replica (on one replica, the kept answers and the wait for a running attempt cover them).

## 3. Launching and job status

`RT/submitting.py` `followed` made a launch's state final from a single observation: a transport error, a
controller's intermediate state, or a job deleted on purpose.

**J1 · high · fixed.** Ray's job client raises `RuntimeError` for every answer but 200, and `followed` took any of them
to mean the job server had forgotten the job: one 503 from Ray's dashboard marked every running launch failed for
good (the monitor follows every open launch when its launches page loads). The run then could not be stopped, and its
driver's last note was refused. *Fix:* only a 404 means the job is unknown; other errors are asked about again later,
and the SDK's blocking calls have a timeout.

**J4 · high · fixed.** KubeRay sets `jobDeploymentStatus: Retrying` in the same update that records the failed
attempt's `jobStatus: FAILED`; `status` read `FAILED` first, so a run being watched while KubeRay retried its lost
driver was recorded failed, and the retried driver exited 1. The documented recovery failed whenever anyone watched.
*Fix:* `Retrying` is submitted.

**J8 · medium · fixed.** A stop on Kubernetes deletes the RayJob, and the next follow found it gone and recorded the
run failed. *Fix:* a stopping launch whose job ended or is gone is stopped.

**J5 · medium · fixed.** The job was made before its name was recorded on the launch, so a Ctrl-C between the two left
a job the platform could not stop. *Fix:* the job's name (it is fixed by the launch) is recorded first.

**J9 · medium · fixed.** Ray gives a stopped job's driver 3 seconds before `SIGKILL`; the driver's way out (cancel the
episodes, close the pools, release the pods warm, note the run stopped) takes longer, so pods were left to the reaper.
*Fix:* `RAY_JOB_STOP_WAIT_TIME_S` in the job's environment. (The head pod's `terminationGracePeriodSeconds` should be
at least as long: proposed.)

**J6 · medium · fixed.** `RT/cli.py`: `rollout train --here` recorded the run's settings without filling in the
renderer and weights as `submit` does, so a run that left them to be filled in waited forever for its channel.

**J7 · medium · fixed.** `RT/publishing.py`: a version's id was the hash of the project's archive alone, so importing
the second environment of a project returned the first. *Fix:* the id covers the entry point too.

**J3 · high · proposed.** Environments imported from git run in the run's driver, beside every secret a run holds: the
Postgres password, the blob store's writing keys, the step-ca provisioner key (which can mint any pod's or the
gateway's certificate), the gateway's signing keys, RunPod's key. `runtime_env.uv` also installs what the imported
`pyproject.toml` names. Separating the pod-token key (an earlier review's suggestion) does not shrink this.
*Proposed:* play a published environment in a process of its own started without secrets, given only the gateway's
URL, a key for the run, the pools' URLs and a ledger token scoped to the run's episode tables; read the version's
recorded description in the driver instead of importing it; until then, refuse published environments unless the
cluster config opts in. Give step-ca's provisioner an x509 policy that allows only `spiffe://rollout/pod/*`.

**J10 · low · proposed.** `RT/bridges.py` `_on_worker` opens a ledger for every bridged checkpoint and never closes it
(file descriptors and pooled connections grow with each step).

**J11 · low · fixed.** `rollout resume` exited 0, without the reason, when the relaunch failed.

**J12 · low · fixed.** `RT/report.py`: a webhook's error put its secret URL in the traceback, and `--watch` stopped at
the first 429.

Also proposed: `RT/publishing.py` `fetched` accepts `file://`, local paths and any host, with the monitor's own
network and git credentials; allow `https` (and named hosts) only.

## 4. The gateway and inference

**G1 · high · fixed.** `implementations/rollout-gemma`: the tool-call parser looped forever on a `}` inside a list
(`a:[1},b:2`), growing memory; it ran on the gateway's event loop, so one sampled turn froze the gateway. *Fix:* a `}`
ends the list; a fuzz test bounds every parse.

**G3 · medium · fixed.** `RT/recorder/segments.py` `segments_of`: a turn continued by two later turns had its sampled
tokens trained in both segments, twice the gradient, and an abandoned compaction attempt (which `untrained` is meant
to keep out) brought its parent's tokens in again. The test asserted the double count. *Fix:* each sampled span is
trained once, in the first segment that holds it, and is context elsewhere.

**G2 · medium · fixed.** `RT/inference/channel.py`: the in-process channel (Tinker channels, tests) switched its
adapter before its version, with an await between, so a turn in that window recorded the new checkpoint at the old
depth; a later phase of a turn was not checked against the weights it began on (an engine `KeyError`, or phase two on
new weights recorded at the old depth). *Fix:* the three change together, and a turn whose weights went is sampled
again.

**P2 · high · fixed.** `RT/inference/remote.py` `Connection.client`: the gateway's client certificate was read once,
and the clients to a run's pods live as long as the run, while the 24-hour certificates are rotated every 6 hours. A
run longer than about a day lost every pod: sampling failed, and three failed steps failed the run. *Fix:* the
certificate is read again when its files change, before each new connection.

**G4 · medium · fixed.** `RT/inference/remote.py`: a routed channel that had not refreshed yet (or had just lost a
server) reported a context limit of 0, so the first request's `max_tokens` was clamped to 0 and vLLM refused it.

**G8 · low · fixed.** `RT/recorder/renderers.py` (Qwen3.5's XML arguments): `int(float(text))` raised on `inf` and
truncated `1.9` to 1.

**G5 · medium · fixed.** `R/local/context.py`, `R/harness/sandboxes.py`: a run's six-hour gateway key, in a sandbox
lease's environment, was recorded in the `sandboxes.acquired` event (the monitor's feed, the episode's events blob)
and kept in the leases table. Whoever read either could sample the run's channel and add turns to a live attempt.

**G6 · medium · partly fixed.** `RT/gateway/service.py`, `directory.py`: `GET /v1/models?run=` needs no key, kept a
lock for every run name ever asked (2,000 made-up names left 2,001), and never forgot a run. *Fixed:* the growth.
*Proposed:* require a key for the run.

**G7 · low · fixed (refused).** Turns do not record the temperature or top-p they were sampled at, and the trainers
score at 1, so a trained binding with either set gives wrong ratios. No shipped path sets them. *Fix:* the gateway
refuses a key for a trained binding sampled other than at 1 and 1. *Proposed:* record both on the turn and score at
the temperature in the trainers (R4).

**G9 · low · fixed.** Errors outside the model-endpoint contract (an engine's 400, a `KeyError`) left the OpenAI and
Anthropic handlers as plain-text 500s the official clients cannot read.

**G11 · low · proposed.** A channel keeps `max_lag + 1` adapters loaded, and an engine's LoRA slots
(`VLLM_MAX_LORAS`, `VllmEngine(max_loras=2)`) are set apart from it, with nothing checking that they cover the windows
of the runs it serves; raising `max_lag` past them makes vLLM swap adapters in and out. Size a local engine's slots from
its runs' windows, and have an engine host say in its beat when its windows exceed them.

**G10 · low · proposed.** The spending cap is checked before a turn and counted after, so turns in flight overshoot
it; reserve an estimate under a lock. A spending cap reached is a 503, which the client retries six times; it should
be a refusal that is not retried.

Also proposed: the official SDKs send no idempotency key, so a retry after a timeout samples and records a second turn
(`segments_of` drops it, but its GPU time or dollars are spent): honour an `Idempotency-Key` where a harness sends one.

## 5. The harness and sandboxes

**H1 · high (latent) · fixed.** `R/harness/tools.py` `execute_tool` and `R/harness/task.py` `_call`: a `SandboxLost`
raised in a `@tool` body became an error result the model saw; the run completed and was trained on, rewarded, in a
dead world, instead of being played again. No shipped environment uses `@tool`, so it was latent. *Fix:* it
propagates, and sibling calls are cancelled.

**H2 · medium · fixed.** `R/local/runner.py`: a run cancelled before its first step never finished (`events()` hung
for good), and a second cancel during the release of its sandboxes (as at shutdown) left them leased. *Fix:* such a
run ends cancelled, and the release is shielded.

**H3 · medium · fixed.** `R/harness/sandboxes.py`, `pool_server.py`: acquiring a lost lease forgot it, so the key got
a fresh sandbox next time, contrary to "a sandbox is never made again under a key whose lease it was". The test
asserted the forgetting. *Fix:* a lost lease is kept, lost, until its run releases it.

**H4 · medium · fixed.** `R/harness/remote.py`: a run bound to a remote tool set asked for its tools with a blocking
request on the event loop, up to 30 s, freezing every run, sample and beat of the runner (three in a row exceeded the
90 s after which a runner's claims lapse). *Fix:* asked asynchronously, once.

**H5 · medium · fixed.** `R/local/context.py` `gather`: when one branch raised, the others ran on after the run had
failed and released its sandboxes. *Fix:* they are cancelled and awaited.

**H6 · low · fixed.** `R/harness/blobs.py`: a blob's path was built from an unchecked digest, so a reference with an
absolute "digest" read or deleted a file outside the store.

**H7 · low · fixed.** `R/harness/sandboxes.py` `_load`: two first loads raced, and the later dropped a lease made
between them.

What H4 leaves: a synchronous request remains for callers outside an event loop, and `RemotePool.operations()` (a
sandbox's specifications, which nothing shipped asks for) can still reach it on one.

Also proposed: `SandboxLost` is recognized by its name in a failure's text, so any wrapping hides it; carry the kind of
failure on the outcome. The pool and tool-set servers have no authentication and read bodies of any size;
`docs/guide/deploying.md` shows `--host 0.0.0.0`, where anyone who reaches the port can release a run's sandboxes.

## 6. Pods

**P1 · high · fixed.** `RT/pods/leasing.py`: the reaper took every pod whose name began `rollout-CLUSTER-` for its
own, and a cluster's name may hold `-`: cluster `prod` reaped cluster `prod-eu`'s live pods on a shared RunPod
account. *Fix:* the tag carries a hash of the cluster's name; pods named the old way are still reaped through their
leases.

**P3 · high (security) · fixed.** `RT/pods/sandboxes.py` `_own`, `_started`: on a host pod, root changed the mode and
owner of paths in a kind's directory, and wrote its config there, following symlinks. A kind's process could plant
`home → /etc` and be given it, then read the pod's ledger token, blob keys and TLS key. *Fix:* nothing there is
followed: symlinks are removed, and modes and owners are changed through descriptors opened with `O_NOFOLLOW`.

**P4 · medium · fixed.** `RT/pods/leasing.py` `renewed`: each pod was described by RunPod (a 60 s timeout) one after
another before its lease was stamped; with RunPod's API hanging, a run of five pods let its last leases go stale and
the reaper deleted a live run's pods. *Fix:* each description is bounded, and a slow one does not hold back stamping.

**P5 · medium · fixed.** The orphan sweep read the leases before listing the pods, and a run writes its lease before
it makes its pod, so a pod made in between was deleted.

**P6 · medium · fixed.** A `runpod-host` pod asked for in the inference role (an eval) was given neither the trainer's
variables, without which its entrypoint exits, nor the writing key.

**P7 · medium · fixed.** `RT/pods/pools.py`: when an acquire's answer from one pod was lost and the release there
failed, the placement moved to another pod and the first pod's sandbox was never released.

**P9 · low · fixed.** `RT/pods/training.py`: the training pod's loop that follows its lease swallowed every error, so
a trainer that could not be built failed silently every 2 s, and the run failed later with no reason.

**P10 · low · fixed.** Pod spend was marked told before it was told, so a failed report dropped it from `limits.spend`.

**P11 · low · fixed.** Routing did not check that a pod's ready beat was for this run, so a warm pod taken by another
run was routed to while its beat still said it was ready for the first.

**P8 · medium · proposed.** `RT/pods/trainer.py` `LeasedTrainer.aclose` is never called, so a trainer pod still being
claimed when the run ends can be made after the run released its pods, and runs until the reaper deletes it.

Also proposed: on a host pod, a kind's process that leaves its process group (`setsid`) outlives `killpg`, and
`Venvs._made` installs a run's projects as root (`uv pip install --editable`), so their build backends run as root.

Pod tokens (an earlier review's claim): they are `HMAC(platform token, "pod-token:" + pod and run)`, with no expiry;
a token minted for a pod reads that pod's lease, which carries the current run's token, for as long as the pod lives,
across warm reuse. *Proposed:* derive the key under its own label, add an expiry, and split a pod's own long-lived
credential from a run's, minted again at each renewal.

## 7. Run settings and validation

Many checks were of type only: a number of the right type passed, however large, small or undefined.

**C1 · high · fixed.** The trainer's own settings changed mid-run reached the live trainer unchecked (a learning
rate of `Infinity`, a negative gradient norm cap, `true` for a token count), and at launch `rank = 0`, a negative
learning rate, `segment_tokens = 0` or `passes = 0` passed (`rank = 0` divides by zero at the first step). *Fix:*
`StepSettings` refuses them, which also keeps the old values when a change is refused mid-run, and validation builds
the trainer's settings so that launch refuses them too.

**C2 · medium · fixed.** `NaN` and infinities passed: `limits.spend = NaN` turned the spend limit off
(`total >= NaN` is never true), and an infinite `max_lag` raised `OverflowError` where only `ValueError` was caught.

**C3 · medium · fixed.** `8.0` passed for a whole-number setting (and `--set groups=1e2`), then failed as
`int("8.0")` in the loop or was dropped for a replica count. *Fix:* whole floats become ints.

**C4 · medium · fixed.** An answer budget at or above the model's context (or the trained channel's segment) passed,
and every turn then overflowed.

**C6 · medium · fixed.** `rollout env check` raised on the faults it exists to report: a `start()` or `evals()` that
raises.

**C7 · low · fixed.** A URL with a password was refused only as the ledger's; elsewhere in a cluster config it was
passed to every job.

**C8 · low · fixed.** A TOML date (`--set name=2026-10-15`) failed validation with a `TypeError`.

**C9 · low · fixed.** `secrets_of` left out the ledger's token for roles that open the ledger service.

**C10 · low · fixed.** A run's `name` given in its settings was replaced by a generated one.

**C11 · low · fixed.** A preset named `.` or `..` was written into the ledger's directory.

**C12 · low · fixed.** `http://127.evil.example` counted as this machine, so `auth = "none"` passed for it.

**C5 · medium · proposed.** Validation reads a trainer's settings from its implementation package, so without the
`gpu` or `tinker` extras its rules skip silently and valid runs are refused (the monitor and the command line on a
machine without GPUs). *Proposed:* the torch-free settings dataclasses move into `rollout-train`, which the
implementations import.

Also proposed: the monitor's `System.want` stores a `trainer.*` change that is out of range (the loop refuses it and
keeps the old values; the page should refuse it as it is asked); `RemoteTrainer.change` and `LeasedTrainer.change`
check nothing; `Objective.changed` takes `NaN` components from a direct caller; a run may be named `.` or `..`
(`registry.valid`).

## 8. The trainers

**R2 · high · fixed.** `OBJ/step.py`: the `max_kl` stop was checked before the first update. With
`old_logprobs = "sampler"` (the default) the distance then measured the engine's disagreement with the trainer, not
any movement, and with `max_kl = 0.02` (the default) a disagreement of 0.2 nats a token stopped every step before its
first update: the run went on, training nothing. Reproduced. *Fix:* the stop is checked after an update, as Tinker
does.

**R1 · medium · fixed.** For a step of one update, the LoRA step took the first minibatch's `old` from the sampler
(folding the engine's disagreement into the clipped ratio), while Tinker weighed it by the declared truncated
importance weight; the two backends differed by 0.1 in every weight. *Fix:* the first minibatch's `old` is the
trainer's own, which its gradient pass computes anyway, in both; they now agree to 1e-10.

**R3 · medium · fixed.** `rollout_tinker`: a step of one update ignored `importance.level = "segment"`.

**R4 · medium · proposed.** Behaviour logprobs come from vLLM after temperature and top-p; the trainers score at 1.
*Proposed:* record both on turns and segments; scale logits by the temperature in the LoRA trainer; refuse top-p
below 1, and Tinker at any temperature but 1. (Refused at the gateway for now: G7.)

**R7 · low · proposed.** With `old_logprobs = "sampler"` (the default), a segment of a later minibatch sampled on the
step's weights still starts from the engine's logprobs, so its clipped ratio holds the engine's disagreement (about
0.016 nats a token, with a long tail) and its importance weight is 1. Measure what `trainer` costs (a pass without a
gradient) against what it buys, and make it the default if the pass is cheap enough.

**R5 · low · proposed.** Warmup counts updates from each step's start, so it stops at the first step's end.

**R6 · low · proposed.** The out-of-memory recovery uses FSDP2 internals that `torch>=2.5` does not guarantee (they
are missing in 2.9.1); raise the floor to the locked version.

## 9. The monitor

**M1 · high · fixed.** `RT/monitor/system.py` `_once`: readers within one beat shared one task and awaited it
directly, so a page that went (a tab closed, or any navigation, which reopens the stream) cancelled the shared reading,
and with it the hub: every page's live updates stopped until the monitor restarted. Reproduced on a real server.

**M2 · medium · fixed.** `POST /login` read a body of any size before checking anything (400 MiB took the monitor
from 51 to 851 MiB, with no token), and a deeply nested body raised `RecursionError`.

**M3, M4 · medium · fixed.** Each beat read every run's desired settings and claims, and a live episode's page read
every run's episodes table; now they read the tables the beat already read.

**M5 · low · fixed.** Caches keyed by strings from URLs never shrank.

**M6 · low · fixed.** Behind a proxy the monitor's cookie (the raw token) was never `Secure`, because uvicorn trusts
`X-Forwarded-Proto` from loopback alone.

**M8 · low · fixed.** An unknown topic or a malformed `after` was a 500 after the stream had begun.

**M7, M9, M10 · low · proposed.** `GET /api/environments/{module:attr}` imports any installed module; the monitor's
bearer token is sent to any address a run's start names; the page reads an error's body as JSON before its status.

## 10. The deployment

**D1 · high · proposed.** `deploy/chart/rollout/templates/ray.yaml`: with in-tree autoscaling, KubeRay gives the
long-lived Ray head the autoscaler's service account, which may patch every pod and RayCluster in the namespace, and
mounts its token into the `ray-head` container, where imported environments are checked. Their code can change the
gateway's image or mount any Secret into a RayCluster. *Proposed:* `automountServiceAccountToken: false` on the head
and workers, and the token projected into the autoscaler sidecar alone; the admission policy to cover RayClusters.

**D2 · high · proposed.** Run pods (which run imported environments: J3) have no egress policy: they reach the cloud's
metadata service, the nodes, other namespaces, and every run's unauthenticated Ray head. *Proposed:* a `runs-egress`
policy that allows the platform's services, DNS and the internet's public addresses, and token authentication on
runs' Ray clusters.

**D3 · high on a shared network · fixed in part.** `deploy/k3s/build.yaml`: the build registry takes pushes and
deletes from anyone on port 30500 of every node, and BuildKit (privileged) listened on the pod network. A pushed
`rollout-platform:dev` runs with every Secret at the next restart. *Fixed:* BuildKit listens on its pod's loopback
alone (`kubectl port-forward` reaches it there). *Proposed:* a NetworkPolicy for the registry, or authentication.

**D4 · medium · proposed.** Every role connects to Postgres as its superuser, so code execution anywhere that reaches
Postgres (the sandbox pool's egress is limited to it) becomes a shell in the Postgres pod (`COPY … TO PROGRAM`).
Create a non-superuser owner for the platform.

**D5 · medium · fixed.** `deploy/local/compose.yaml` published Postgres and S3, with well-known credentials, on every
interface (Docker's rules bypass host firewalls). They are published on loopback alone.

**D6 · medium · proposed.** The ledger service has no TLS of its own and can be exposed by a `LoadBalancer`, a
`NodePort` or an Ingress without `tls`, and pods send their bearer tokens to it: refuse those in the chart.

**D7 · low · proposed.** No container sets a `securityContext` (`allowPrivilegeEscalation: false`, `RuntimeDefault`
seccomp), and every pod mounts its service account's token.

**D8 · low · proposed.** `.github/workflows/images.yml` grants `packages: write` to every job and uses actions by
mutable tag.

**D9, D10 · low · proposed.** The default storage class deletes the ledger's volumes with their claims; the K3s
install script runs an unpinned installer and makes the admin kubeconfig world-readable.

## 11. Environments and the process

**E1 · medium · fixed.** `environments/minecraft`: `teams-every-task` drew every task's start with `Random(1)`, so all
100 eval starts were one world, one layout and one team of three (and that world is one training draws from);
`horizons-held-out` likewise played every task in one held-out world. *Fix:* each row's eval start is drawn with a seed
of its own, from its key; the environments' versions are bumped (a suite already frozen keeps its starts).

**E2 · medium · fixed.** `environments/judging/judging/rubric.py`: the policy's answer was pasted into the judge's
prompt as it was, so the policy could learn to write instructions to its judge. *Fix:* the request and the answer are
each between tags, a tag inside either is broken up, and the judge is told what is between them is never an
instruction; the rubrics' versions are bumped.

**E3 · observation.** No shipped environment uses `Task` and `@tool`: all four subclass `Program` with loops of their
own, so the main authoring API is exercised only by the documentation and unit tests (which is why H1 was latent).

**X1 · high · fixed.** No workflow ran the tests, the linters or the type checker; the only one builds images on a
version tag. *Fix:* `.github/workflows/ci.yml` runs ruff, the tests on CPU (the chart's included, with Helm, without
which they skipped) and the monitor page's checks (`tsc`, eslint, vitest) on every pull request and push to `main`,
with actions pinned by commit and a token that reads.

**X2 · medium · fixed.** `uv sync && uv run pytest`, as the README says, stopped at collection
(`tests/rollout_train/gateway/test_storage.py` needs `rollout_qwen`), and 20 tests failed without the GPU extras.
*Fix:* a test marked `extras(MODULE, …)` is skipped, saying which, where a module it needs is not installed; the
docs' examples and the API reference's check skip the same way. One more failed only as root (a read-only blob is
writable by root); it now checks the file's mode.

**X4 · low · observed.** `test_a_stopped_run_resumed_is_submitted_again_and_goes_on_from_the_ledger`
(`tests/rollout_train/test_pausing.py`) failed once under load (ten processes running tests at once) and passed every
time alone: the groups it counts as played are read while the stopped run may still be writing one.

**X3 · low · proposed.** `pyright` reports hundreds of errors without the GPU extras (missing `torch`, `vllm`), so it
can only run where they are installed; a CI job for it needs `uv sync --all-extras` on Linux.

## The earlier review's claims

| Claim | Verdict |
|---|---|
| Shared history is trained twice where a conversation branches | Confirmed and reproduced; fixed (G3) |
| Temperature and top-p are not passed to the trainer | Confirmed, latent: no run setting sets them and bindings default to 1. Refused at the gateway now (G7); proposed in full (R4) |
| Failed and cancelled episodes are dropped before the baseline | Confirmed; a defensible design for infrastructure failures, incomplete for failures the policy causes (T11) |
| LoRA and Tinker use different estimators for the same setting | Confirmed, for steps of one update; fixed (R1, with R2) |
| Per-step rewards and truncation do nothing | Confirmed, and documented (`episodes.md`: rewards are summed by slot and key); `docs/guide/tasks.md` overstates it. No critic, so `TRUNCATED` would only matter for one |
| `_play` and `_watch` catch only `CancelledError`; `_done` drops the claim | Confirmed and reproduced; fixed (T2) |
| The ledger service's 500s are not retried | Confirmed and reproduced; fixed (L1) |
| The runner's serve loop is unguarded and started fire-and-forget | Confirmed and reproduced; fixed (T1, J2) |
| Polling scales with history; no retention; no `SKIP LOCKED`; no timeouts | Confirmed (T5, T12, L5); the blob store is thinned of released checkpoints, but episodes and datasets are kept |
| Environment code runs in the driver, whose token is also the pod tokens' key | Confirmed, and wider than stated (J3): every secret of a run is there |
| Six-hour gateway keys are written into leases and events | Confirmed; fixed (G5) |
| `GET /v1/models?run=` is unauthenticated and its cache never evicts | Confirmed; the growth is fixed, the key proposed (G6) |
| Run pods get every secret, have no egress policy; no `securityContext` | Confirmed (D2, D7). Nothing internet-facing is unauthenticated; inside the cluster runs' own Ray clusters and the K3s build registry are (D2, D3) |
| Minecraft eval starts barely vary | Confirmed, worse than stated: one world, layout and team for all 100; fixed (E1) |
| The judge prompt can be hijacked | Confirmed; fixed (E2) |
| No shipped environment uses `Task` and `@tool` | Confirmed (E3) |
| `execute_tool` turns a lost sandbox into training signal | Confirmed and reproduced; fixed (H1) |
| CI runs no tests | Confirmed; fixed (X1) |
| The README's quick start fails | Confirmed; fixed (X2) |

## The second earlier review's claims

A second review (of Rollout, Iso and Inframe, taken at the same commit) made five claims about Rollout:

| Claim | Verdict |
|---|---|
| Engine logprobs stand in for the trainer's at a step's start (`old_logprobs = "sampler"`), the clip spending its range on the engine's disagreement, `mean_mismatch` leaving those segments out, and the docstring saying first-minibatch ratios are 1 | Confirmed; fixed for the first minibatch (R1): its segments take the trainer's own logprobs, so their ratios are exactly 1 again, as the docstring says, and they count in `mean_mismatch`. A later minibatch's segment sampled on the step's weights still takes the engine's, by the setting's design (it saves a pass without a gradient); whether `trainer` should be the default is proposed, to be measured (R7) |
| No staleness cap when a turn is trained | Confirmed (T13): `max_lag` bounds the checkpoint a turn begins on, not how many steps it waits before it trains, so a long episode's early turns can be many checkpoints old; truncated importance weights are the only guard. The README said `max_lag` bounds how stale a turn can be; it now says what it bounds. *Proposed:* a training-time bound (`importance.max_age`: mask a token sampled more than N checkpoints before the step's start), or masking (`icepop`) by default |
| The ledger can record work twice in narrow windows | Known and documented (`ledger-guarantees.md`), each window pinned by a strict-xfail test; not counted here |
| Temperature and top-p are not tied to the trainer | Confirmed, latent; refused at the gateway now (G7), proposed in full (R4) |
| Adapters kept can exceed the engine's LoRA slots | Confirmed (G11, low): a channel keeps `max_lag + 1` adapters and `VLLM_MAX_LORAS` (or `VllmEngine(max_loras=2)`) is set apart from it; the docs ask an operator to keep it at least the sum of the windows, and nothing checks it, so raising `max_lag` past it makes vLLM swap adapters (throughput, not correctness). *Proposed:* size a local engine's `max_loras` from its runs' windows, and have an engine host say in its beat when the windows it serves exceed its slots |

## What to do next

In order of payoff over effort:

1. **Isolate imported code** (J3, D1, D2): play published environments in a process without secrets, take the
   autoscaler's token out of the Ray head, and give runs an egress policy. Until then, a published environment is as
   trusted as the platform's own code; a cluster config should have to say so.
2. **Make the ledger's hot paths keyed** (T5, T12): a server-side claim with `SKIP LOCKED`, keyed reads of episodes,
   notifications for new groups, and retention for finished runs. This is what lets runners scale past tens of
   episodes.
3. **Bound Postgres sessions** (L5): idle-in-transaction and lock timeouts, and keepalives.
4. **Run pyright in CI**, on a runner with `uv sync --all-extras`, and the reference's check with it (X3).
5. **Record the sampling distribution, and bound a turn's age** (R4, G7, T13): temperature and top-p on turns,
   scored by the trainers; a turn sampled too many checkpoints before its step masked.
6. **Move the trainers' settings into `rollout-train`** (C5), so that validation is whole on any machine.
7. **Migration tools** (L6): copy every table, checked against the schema.
8. **A failure floor** (T11): an option to give failures the policy causes a score.
