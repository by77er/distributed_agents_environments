# Sandboxes

For environment authors whose programs need something outside their own code, such as a game server or a container:
declaring sandboxes, pools and leases, and serving pools over HTTP.

**Read first:** [the harness](README.md). **Next:** [Determinism](determinism.md).

Code: `rollout.harness.sandboxes`, `rollout.harness.pool_server`, `rollout_train.sandboxes`, `rollout_train.pods` (`sources`, `sandboxes`, `pools`) · See [tools](../../guide/tools.md),
[rollouts](../rollout-train/rollouts.md), [API reference](../../guide/reference.md#sandboxspec)

A sandbox is something a program runs against for the length of one run, outside its own code: a Minecraft world, a
container a coding agent works in, a computer, the worker process of an environment. A program declares the sandboxes
it needs; the runner acquires them from pools before the program starts and releases them when it ends; the program
reaches each one by name. A pool hands out sandboxes under leases, one per key, as many as it can hold.

```python
class TeamEpisode(Program):
    def sandboxes(self) -> Mapping[str, SandboxSpec]:
        world = {"task": "t001", "world_seed": 12345, "layout_seed": 3, "names": ["ada", "ben"]}
        return {"world": SandboxSpec(kind="minecraft", parameters=world)}

    async def main(self, run: RunContext) -> None:
        world = run.sandbox("world")                                   # acquired before main, released after it
        seen = await world.call("observe", {"agent": "ada"})          # a recorded effect
        await world.call("act", {"agent": "ada", "action": {"name": "wait"}})
```

The parts and what they exchange:

| Part | Gives | Gets |
|---|---|---|
| A program | the sandboxes it needs, by name (`sandboxes()`) | each one, as `run.sandbox(name)` |
| A runner | a key per sandbox (the run's lease and the sandbox's name), the slots' model addresses | a `Lease`: a handle, addresses, environment variables |
| A pool | leases, how full it is, the operations on its sandboxes | specs and keys; releases |
| A provider | sandboxes of one kind, made and deleted; operations on them | what a pool asks |
| The leases table | where each lease is kept | what a pool writes |
| A keeper (training) | releases of leases whose claim no longer holds | the [ledger](../rollout-train/checkpoints.md#the-ledger)'s claims and the runners' beats |

Where each part runs is the deployment's business: a pool in the runner's process, or on a machine of its own served
over HTTP, is the same pool to everything else.

## Declaring sandboxes

A program declares its sandboxes in `Program.sandboxes()`, by name; a task declares them in its `sandboxes` class
attribute, as it declares `imports`. Each is a [`SandboxSpec`](../../guide/reference.md#sandboxspec), and the program's
parameters may decide it (the Minecraft [episode](../rollout-train/episodes.md)'s world is made from its task, seeds and
team). A sandbox for each slot is one entry per slot:

```py
def sandboxes(self) -> Mapping[str, SandboxSpec]:
    return {f"box-{slot}": SandboxSpec(kind="code", slots=(slot,)) for slot in ("agent-1", "agent-2")}
```

| `SandboxSpec` field | Says |
|---|---|
| `kind` | the kind of sandbox (`minecraft`): the binding names the pool that serves each kind |
| `parameters` | what the pool makes it from: a task and its seeds, an image |
| `slots` | the model slots a harness inside it samples: each slot's address is put in its environment ([harnesses inside a sandbox](#harnesses-inside-a-sandbox)) |
| `process` | a process it runs from its start (`Process`: `command`, `environment`, `directory`), launched by the pool with the lease's environment added to its own |
| `mounts` | files it sees, read-only (`Mount`: `source`, as the pool finds it, and `target`, inside): an environment version's files, its virtual environment |
| `scratch` | a directory it may write, empty at the start, and the most it may hold (`Scratch`: `path`, `mib`). Without one it writes nowhere |
| `network` | what it may reach besides its connection to the runner (`Network`: `allow`, a list of hosts): nothing by default |
| `limits` | `SandboxLimits`: `cpus`, `memory_mib`, `processes`, and `seconds` from its start, past which its lease ends and the pool deletes it |

A world for an episode needs only a kind and parameters. A worker for an environment is the same kind of thing with
more said:

```py
SandboxSpec(
    kind="worker",
    process=Process(command=["python", "-m", "worker"], directory="/environment"),
    mounts=[Mount(source="minecraft-team/1.4.0", target="/environment"), Mount(source="venvs/9f2c", target="/venv")],
    scratch=Scratch(mib=512),
    limits=SandboxLimits(cpus=2, memory_mib=4096, processes=64, seconds=4 * 3600),
    slots=("agent-1", "agent-2"),
)
```

A provider enforces what its spec says as far as it can; the pool ends a lease past its time limit at its next sweep.
The lease keeps the limit as a duration (`Lease.seconds`, from `Lease.at`), and the pool measures it by its process's
monotonic clock, which a change of the machine's wall clock (a resync, a resumed virtual machine) does not move. A pool
started again counts the time its leases have lasted by the wall clock, once, as it reads them.

## What a program sees

`run.sandbox(name)` is a [`Sandbox`](../../guide/reference.md#sandbox); a name the program did not declare raises
`KeyError`.

| Member | Is |
|---|---|
| `name`, `lease` | its name in the program, and the [`Lease`](../../guide/reference.md#lease) the runner acquired |
| `addresses` | where its services listen, by name (a world's `game` and `control`; a worker's connection to the runner) |
| `environment` | the environment variables for what runs inside it: its slots' model addresses, and the pool's own |
| `specifications()` | its operations, as tool specifications |
| `await call(operation, arguments)` | performs an operation and returns its `ToolResult` |

Every call is a `tool.call` effect whose payload is `{"sandbox", "tool", "arguments"}`: the sandbox by its name, so
a run never carries a handle. An operation's `retry_class` says whether it is safe to perform again, as for
[imported tools](../../guide/tools.md#retry-classes): `PURE` and `IDEMPOTENT` operations are, and `SIDE_EFFECTING` and
`UNKNOWN` ones are only when the pool deduplicates. The run records its leases, once
acquired, in a [`sandboxes.acquired`](contracts/run-events.md#lifecycle) event.

## Harnesses inside a sandbox

A harness that brings its own loop (a coding agent in a container, an environment's worker) needs no wrapper program
to reach its model. For each slot a spec names, the runner puts the slot's
[address](../rollout-train/harness-endpoint.md) in the sandbox's environment, as OpenAI's clients read it and as
Anthropic's do (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`,
`ANTHROPIC_MODEL`: [the variables](../rollout-train/gateway.md#claude-code-and-codex)). Each is suffixed with the
slot's name in capitals (`OPENAI_API_KEY_AGENT_1`), and unsuffixed too when the spec names one slot. Claude Code
needs nothing more; Codex is given a provider whose base URL is `OPENAI_BASE_URL` and whose key
is read from `OPENAI_API_KEY` ([Claude Code and Codex](../rollout-train/gateway.md#claude-code-and-codex)). What the
harness samples there is recorded for the slot like any other sample. A key names
the run's session of its slot only, and stops working once it expires or a newer attempt takes its episode's fence
([keys](../rollout-train/gateway.md#keys)). A slot whose model is not served over HTTP has no address, and the run
fails.

## The runner

| | |
|---|---|
| Binding | `RunBinding.pools` maps each kind to a [`PoolBinding`](../../guide/reference.md#poolbinding): `local`, a pool registered with the runner (`pools={"name": pool}`), or `url`, a pool served over HTTP. `bind` and `binding_for` bind each kind to the pool registered under the kind's name unless `pools` says otherwise |
| Key | each sandbox is acquired under `{lease}/{name}`. `Runner.start(..., lease=...)` gives the lease, by default the run's id; an [episode runner](../rollout-train/rollouts.md#a-runner) gives its claim's key, `RUN/GROUP/EPISODE/ATTEMPT` |
| When | before the program's `main`, in the order they are declared; each slot's address is taken first |
| A full pool | the runner tries again every second, for up to `ACQUIRE_SECONDS` (300); then the run fails, releasing what it had |
| Release | when the program ends, however it ends (completed, failed, cancelled), each lease the run acquired or began to. A release that fails is left to the lease's end |

## Pools

A [`Pool`](../../guide/reference.md#pool) hands out sandboxes of one kind under leases.

| Member | Does |
|---|---|
| `await acquire(spec, key, environment)` | the lease of `key`: the one there is, or a new sandbox given `environment`. Raises `NoCapacity` when the pool is full (`PoolUnavailable`, a kind of it, when the pool does not answer now), `LeaseRefused` for a key that may hold no lease now (its claim lapsed), and `SandboxLost` for a key whose sandbox is gone |
| `await release(key)` | ends the lease and deletes its sandbox; nothing when there is no such lease |
| `await capacity()` | a `Capacity`: `size`, `leased`, and `free` |
| `operations()`, `deduplicates` | what can be done to its sandboxes, and whether it performs each `effect_id` at most once |
| `await call(key, name, arguments, *, effect_id, arguments_digest)` | an operation on the sandbox leased under `key`; `SandboxLost` where the key has no live lease |

[`SandboxPool(provider, name=..., leases=..., admits=...)`](../../guide/reference.md#sandboxpool) is a pool over a
provider:

- **One sandbox per key.** Acquires of a key, at once or after a crash, get one sandbox: its handle is derived from
  the key, and the provider makes a handle once.
- **At most `provider.size`** leases at once, counting those being made.
- **Its leases are kept** in `leases` (in memory by default) under its `name`, the provider's kind unless it is
  given one: pools that share a table need names of their own. A key leased from another pool is refused.
- **`admits(key)`**, when given, says whether a key may hold a lease now. A key it refuses raises `LeaseRefused`, and
  the lease the key had is released: nothing is made for it, even for a moment. Beside a ledger it says whether the
  key's claim holds ([in training, a lease ends with its claim](#in-training-a-lease-ends-with-its-claim)); without it
  every key is admitted.
- **A sandbox is never made again under a key whose lease it was.** A lease whose sandbox is gone (it ended with
  the pool's process, say) is lost, and its key gets `SandboxLost`: a new sandbox would not be the one its run was
  playing in.
- **`sweep(ended)`** releases the leases `ended` says have ended and those past their time limit, marks lost those
  whose sandbox is gone, and deletes the sandboxes no lease names. A lost lease holds no room.
- **`lose(key)`** deletes a lease's sandbox and keeps the lease, marked lost: its key gets `SandboxLost`, from an
  operation as from an acquire, until its run releases it.
- **`close()`** releases every lease it holds and closes the provider.

A [`Provider`](../../guide/reference.md#provider) makes, deletes and operates sandboxes of one kind:

| Member | Does |
|---|---|
| `kind`, `size` | the kind it makes, and how many it can hold at once |
| `operations()` | what can be done to one of its sandboxes; none when a harness inside reaches it by its addresses |
| `await create(handle, spec, environment)` | makes the sandbox, or says how to reach it if it is there already: a `Reach`, its `addresses` and any `environment` of its own |
| `await delete(handle)` | deletes it, or does nothing if it is gone |
| `await held()` | the handles it has now |
| `await call(handle, name, arguments, *, effect_id, arguments_digest)` | performs an operation. An error the operation reports is a result with `is_error`; an exception is a platform failure |

A provider may also have `deduplicates` and an async `close`.

```py
class Boxes:
    kind, size = "box", 8

    def operations(self) -> Sequence[ToolSpecification]:
        return [ToolSpecification(name="run", input_schema={...}, retry_class=RetryClass.SIDE_EFFECTING)]

    async def create(self, handle: str, spec: SandboxSpec, environment: Mapping[str, str]) -> Reach:
        await start_container(handle, image=spec.parameters["image"], environment=environment)
        return Reach(addresses={"shell": f"unix:///run/boxes/{handle}.sock"})

    async def delete(self, handle: str) -> None: ...
    async def held(self) -> Sequence[str]: ...
    async def call(self, handle, name, arguments, *, effect_id, arguments_digest) -> ToolResult: ...


runner = LocalRunner(providers=providers, pools={"box": SandboxPool(Boxes())})
```

| Kind | Provider | Operations |
|---|---|---|
| `minecraft` | `MinecraftWorlds` ([Minecraft team](../../products/minecraft-team.md#the-worlds)): a Paper server per lease, with the team's bots and the task built | `observe`, `act`, `window`, `score` |
| any (`fake` by default) | [`FakeSandboxes`](../../guide/reference.md#fakesandboxes) (`rollout.testing`): records that honour their spec. A spec's process is launched with the lease's environment; writes stay in its scratch and off its mounts, within the scratch's size; only allowed hosts are reached | `describe`, `write`, `fetch`, and any given |

## Over HTTP

`rollout.harness.remote.serve_pool(pool)` serves a pool, and `RemotePool(url)` is the pool it serves, wherever it
is; a binding names it with `PoolBinding(url=...)`.

| Route | Body | Answer |
|---|---|---|
| `GET /operations` | | `operations`, `deduplicates` |
| `GET /capacity` | | a `Capacity` |
| `POST /acquire` | `spec`, `key`, `environment` | a `Lease`; 503 with `"full": true` when the pool is full, 409 for a key it refuses (`LeaseRefused`), 410 for a key whose sandbox is gone (`SandboxLost`) |
| `POST /release` | `key` | |
| `POST /call` | `key`, `name`, `arguments`, `effect_id`, `arguments_digest` | a `ToolResult`; 410 for a key with no live sandbox (`SandboxLost`) |

A pool that raises answers 500 with `error`, and the client raises it. A 503 that does not say the pool is full (a
proxy whose pool is not up) is `PoolUnavailable`; asking a pool how full it is, or what it offers, takes 10 seconds at
most. `url` may end in a path, under which every
route is (`https://IP:PORT/v1/sandboxes/minecraft`); `RemotePool(url, client=...)` sends every request through the
client given (one with a client certificate, say), and `await describe()` asks it once what the pool offers.

`rollout pool FACTORY [--directory DIRECTORY] [--ledger WHERE] [--name NAME] [--host 127.0.0.1] [--port 8710]`
serves a `SandboxPool` over the provider `FACTORY` (`module:function`, called with the directory) returns, named
`NAME` (by default `KIND@HOST`). With `--ledger`, its leases are kept beside that ledger, it refuses a key whose
claim has lapsed, its keeper ends leases with their claims, and it beats as `pools/NAME` (the pool's `name`).
Without, it cannot tell
whether a claim holds: it admits every key, keeps its leases in the process, and a lease ends only when released (a
run's own release, when it ends). A pool that serves training runs is served with `--ledger`.

`rollout pool --kind KIND --cluster [PATH or NAME]` serves the pool the cluster config's `[sandboxes.KIND]` describes:
its provider made with `--directory` (by default `[scratch]/sandboxes/KIND`), its `size` and its settings, named `KIND`,
its leases beside the cluster's ledger. Where the section also says `url`, runs reach the pool there and make none of
their own ([the cluster config](../../guide/cluster.md#every-section)); the chart runs it in a pod of its own for each
kind ([Where sandboxes run](../../research/sandbox-placement.md)).

## On a run's pods

A kind whose `[sandboxes.KIND]` says `on_pods` is served from the RunPod host pods a run leases, on the CPUs and memory
their engine and trainer leave ([sandboxes on a host pod](../../deploy/providers.md#sandboxes-on-a-host-pod)). Any kind
can be: a pod's image holds no environment's code, and what it serves is data its lease gives it.

| Part | Where | Does |
|---|---|---|
| The source (`rollout_train.pods.sources`) | Made by the run's driver | The provider (`module:name`) and its settings; its code, as projects' zips in the pods' blob store, packed as an imported environment is (a cluster's own kind: the provider's project, the platform's projects it depends on and `rollout`; a published version: its zip, read from the store versions are published to, and the platform's projects its dependencies name); pins of everything else, at the platform's versions; the Python (3.13); what a pool is sized by. A distribution the platform holds neither as a project nor installed is refused: nothing is ever fetched by a name alone. Given to each pod in its lease's settings (`sandboxes`, by kind) |
| The sandbox host (`rollout_train.pods.sandboxes`) | Each host pod | Follows its lease: makes each kind's Python environment with uv (the projects editable, the pins with nothing resolved), once per digest of its source, kept on the volume; runs each kind's pool in a process and as a user of its own, on a Unix socket only the host reaches; passes `/KIND/...` on to it; tells it which run it serves, and whether to take new keys |
| The pool's process (`rollout.harness.pool_server`) | Each kind on the pod, in its own Python, as its own user, given no secret | A `SandboxPool` over the provider, for one run at a time, its leases kept on the volume |
| `PodPools` (`rollout_train.pods.pools`) | The run's driver | The run's pods' pools of a kind as one pool, bound as a local pool (`PoolBinding(local=KIND)`): the runner and its programs see a pool like any other |

| `PodPools` | Does |
|---|---|
| Which pods | At each look (`LeasedPools`), the pods whose leases the run holds with an `https` address, of the providers that serve the kind, each a `RemotePool` at `ADDRESS/v1/sandboxes/KIND` reached over mutual TLS with the gateway's certificate, the pod's identity (`spiffe://rollout/pod/NAME`) checked in the handshake; given up on in 5 seconds where it does not connect. A pod whose beat is fresh is live |
| `acquire` | A key with a lease: from where its lease is, and only there. A new key: on a live pod with room, the most first (every pod asked at once, outside any lock; the acquires on their way to a pod, or sent since it was asked, order the pods and exclude none), the next while each answers full, is not taking new keys, or does not answer, then the pool at the section's `url` (the cluster's own, where it says one); else `NoCapacity`. Where it goes is kept before the pod is asked, so an acquire whose answer is lost is released on its pod, at once or at the next sweep |
| `release`, `call` | On the pod (or the cluster's pool) that holds the key's lease. A release that fails is kept and tried again at each sweep |
| `capacity` | The live pods' pools' summed, and the cluster's pool's: the runner asks one pool for room, as before |
| A lost pod | A lease whose pod the run no longer holds (released, deleted, taken by another run) is lost: its key gets `SandboxLost`, and its episode is played again. A pod that only misses beats takes no new lease and keeps answering for its own |
| Where each lease is | Kept in the run's directory (`pods/KIND.json`): a driver started again sends the leases of the runs it adopts to where they are |
| No pod serves it | Where the section has no `url`, the driver says in the run's beat and on its launch that it waits for a pod that serves the kind, and ends the run failed once none has for 30 minutes (`SandboxesUnserved`); a run would otherwise wait for ever, claiming no episode. A `url` is not required: a run on one pod may rather fail than play its worlds in the cluster |

**Claims.** The pod's ledger token reads the pod's own lease and nothing of the run's claims, so claims are checked
where the platform's token is, in the run's driver: `PodPools` refuses a key whose claim has lapsed (`admits`, as a pool
beside the ledger does) and releases its lease on its pod, and a keeper (`keep`) sweeps it like any pool of the run's,
releasing on its pod each lease whose claim it found lapsed at two looks running. The pod follows only its lease:

- a kind's process admits a key only while it is of the run the lease names (the key's first part is the run's id, or
  one of its evals', `RUN-...`), and refuses others with `LeaseRefused`; the host reads the lease again for an acquire
  of another run's key first;
- when the lease names another run, each process forgets the other runs' leases; when its `renewed` is older than 5
  minutes (the driver renews it every 30 seconds, but may stall), each takes no new keys and goes on serving those it
  holds; when it names no run, is idle, or `renewed` is older than 30 minutes (the driver is gone), each marks every
  lease lost;
- a lost lease's key gets `SandboxLost` until its run releases it, so its episode is played again, never in a fresh
  sandbox under the same key;
- its leases are kept on the pod's volume, so a process started again answers `SandboxLost` for the sandboxes it lost;
- an episode that fails because its sandbox is lost is played again as a new attempt, whether or not its runner adopted
  it ([rollouts](../rollout-train/rollouts.md#a-runner)).

A kind whose specs name model slots is not served from pods: a harness inside reaches its model at the run's gateway,
on the driver's loopback interface, which a pod cannot reach.

## In training: a lease ends with its claim

An [episode runner](../rollout-train/rollouts.md#a-runner) starts each episode's run with the claim's key as its
lease, so the run's sandboxes are leased under `RUN/GROUP/EPISODE/ATTEMPT/NAME`: each attempt gets sandboxes of its
own.

- **Capacity.** A runner claims an episode only while the pools of the sandboxes its program declares have room for
  them, asking each pool's `capacity()` once a look, and counting what it claims as it goes. Runners that share a
  pool can still claim more than it holds at once; their runs then wait for room ([the runner](#the-runner)). A
  runner says in each beat how full its pools are (`pools`: each one's `size`, `leased` and `free`).
- **Leases beside the ledger.** A pool a run's driver opens, or one served with `--ledger`, keeps its leases beside
  the ledger, as ordinary state changed in place: `sandboxes.json` beside a ledger of files (`FileLeases`), the
  `sandboxes` table of a database ledger's database (`DatabaseLeases`); `leases_of(ledger)` finds them.
- **A lapsed claim gets no sandbox.** A claim holds as the scheduler says
  ([claims](../rollout-train/rollouts.md#what-runners-write)): it lapses when a newer attempt of its episode is
  claimed, when its runner takes its fence anew without adopting it, notes the attempt cut short, or stops beating for
  90 seconds, and its lease ends with the episode's record too. Such a pool admits a key only while its
  claim holds (`admits(ledger, presence)`), so a run recovered after its claim lapsed is refused its sandboxes, and
  the lease it had is released at once. A key of a run the ledger does not know (a run started by hand) is admitted.
- **Expiry.** A pool's keeper (`keep(pool, ledger, presence)`) sweeps every 15 seconds, and releases a lease whose
  claim it found lapsed at two sweeps running (a runner started again adopts its claims a moment after it takes its
  fence anew), deleting its sandbox. A runner that dies leaves its sandboxes to their pools, which delete them within
  two minutes. A lease of a run the ledger does not know ends only when it is released.
- **The keeper ends the claim before it ends the lease** (`ending`). Just before releasing, it reads the claim again:
  one adopted since it was found lapsed holds again, and keeps its lease. One that is still its episode's latest
  attempt is ended in the ledger first: the keeper takes the
  [episode's fence](../rollout-train/rollouts.md#each-episodes-fence), which refuses whatever the claim's runner would
  still record of the attempt, and notes the attempt cut short under it (`RELEASED`). A runner that was only paused,
  and resumes to find its sandbox gone, therefore cannot record the failure that follows; the episode is played
  again. If another takes the episode's fence between the keeper's take and its note (a runner adopting the claim,
  or claiming the episode anew), the note is refused and the lease waits for the next sweep.
- **One keeper per pool name.** A keeper takes its pool's fence (`pools/NAME`) when it starts, and stops sweeping once
  another process takes it: a run's directory started twice on one machine has one keeper deleting the sandboxes no
  lease names, not two. Leases of a pool name that is never opened again (a host renamed, a run's directory moved)
  are swept by no keeper, and neither are their sandboxes where those outlive the process.
- **A runner started again** takes its fence anew: the claims it held lapse, the keepers end their leases and
  delete their sandboxes, and the episodes are played again as new attempts
  ([rollouts](../rollout-train/rollouts.md#a-runner-started-again)).
- **A pool started again** under its name finds its leases beside the ledger; those whose sandboxes are gone are
  marked lost, and sandboxes no lease names are deleted.
