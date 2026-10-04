# Sandboxes

Code: `rollout.harness.sandboxes`, `rollout_train.sandboxes` · See [tools](../../guide/tools.md),
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
| A keeper (training) | releases of leases whose claim no longer holds | the ledger's claims and the runners' beats |

Where each part runs is the deployment's business: a pool in the runner's process, or on a machine of its own served
over HTTP, is the same pool to everything else.

## Declaring sandboxes

A program declares its sandboxes in `Program.sandboxes()`, by name; a task declares them in its `sandboxes` class
attribute, as it declares `imports`. Each is a [`SandboxSpec`](../../guide/reference.md#sandboxspec), and the
program's parameters may decide it (the Minecraft episode's world is made from its task, seeds and team). A sandbox
for each slot is one entry per slot:

```py
def sandboxes(self) -> Mapping[str, SandboxSpec]:
    return {f"box-{slot}": SandboxSpec(kind="code", slots=(slot,)) for slot in ("agent-1", "agent-2")}
```

| `SandboxSpec` field | Says |
|---|---|
| `kind` | the kind of sandbox (`minecraft`): the binding names the pool that serves each kind |
| `parameters` | what the pool makes it from: a task and its seeds, an image |
| `slots` | the model slots a harness inside it samples: each slot's address is put in its environment ([below](#harnesses-inside-a-sandbox)) |
| `process` | a process it runs from its start (`Process`: `command`, `environment`, `directory`), launched by the pool with the lease's environment added to its own |
| `mounts` | files it sees, read-only (`Mount`: `source`, as the pool finds it, and `target`, inside): an environment version's files, its virtual environment |
| `scratch` | a directory it may write, empty at the start, and the most it may hold (`Scratch`: `path`, `mib`). Without one it writes nowhere |
| `network` | what it may reach besides its connection to the runner (`Network`: `allow`, a list of hosts): nothing by default |
| `limits` | `SandboxLimits`: `cpus`, `memory_mib`, `processes`, and `seconds` of wall time, past which its lease ends and the pool deletes it |

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

A provider enforces what its spec says as far as it can; the pool ends a lease past its wall time at its next sweep.

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
a run never carries a handle. An operation's `retry_class` says what a durable runner may do after a crash, as for
[imported tools](../../guide/tools.md#after-a-crash): `PURE` and `IDEMPOTENT` operations are performed again,
`SIDE_EFFECTING` and `UNKNOWN` ones are guarded unless the pool deduplicates. The run records its leases, once
acquired, in a [`sandboxes.acquired`](contracts/run-events.md#lifecycle) event.

## Harnesses inside a sandbox

A harness that brings its own loop (a coding agent in a container, an environment's worker) needs no wrapper program
to reach its model. For each slot a spec names, the runner puts the slot's
[address](../rollout-train/harness-endpoint.md) in the sandbox's environment: `OPENAI_BASE_URL`, `OPENAI_API_KEY` and
`OPENAI_MODEL`, each suffixed with the slot's name in capitals (`OPENAI_API_KEY_AGENT_1`), and unsuffixed too when
the spec names one slot. What the harness samples there is recorded for the slot like any other sample. A key names
the run's session of its slot only, and stops working once the recorder forgets the run, which an episode runner has
it do as the episode ends. A slot whose model is not served over HTTP has no address, and the run fails.

## The runner

| | |
|---|---|
| Binding | `RunBinding.pools` maps each kind to a [`PoolBinding`](../../guide/reference.md#poolbinding): `local`, a pool registered with the runner (`pools={"name": pool}`), or `url`, a pool served over HTTP. `bind` and `binding_for` bind each kind to the pool registered under the kind's name unless `pools` says otherwise |
| Key | each sandbox is acquired under `{lease}/{name}`. `Runner.start(..., lease=...)` gives the lease, by default the run's id; an [episode runner](../rollout-train/rollouts.md#a-runner) gives its claim's key, `RUN/GROUP/EPISODE/ATTEMPT` |
| When | before the program's `main`, in the order they are declared; each slot's address is taken first |
| A full pool | the runner tries again every second, for up to `ACQUIRE_SECONDS` (300); then the run fails, releasing what it had |
| Release | when the program ends, however it ends (completed, failed, cancelled), each lease the run acquired or began to. A release that fails is left to the lease's end |
| Durable runs | a run acquires its sandboxes each time it is executed, recovered after a crash or woken after eviction, under the same lease, and so gets the same sandboxes back while their leases hold. A run that is unloaded keeps them. A run refused its sandboxes on resuming (`LeaseRefused`, `SandboxLost`) fails |

## Pools

A [`Pool`](../../guide/reference.md#pool) hands out sandboxes of one kind under leases.

| Member | Does |
|---|---|
| `await acquire(spec, key, environment)` | the lease of `key`: the one there is, or a new sandbox given `environment`. Raises `NoCapacity` when the pool is full, `LeaseRefused` for a key that may hold no lease now (its claim lapsed), and `SandboxLost` for a key whose sandbox is gone |
| `await release(key)` | ends the lease and deletes its sandbox; nothing when there is no such lease |
| `await capacity()` | a `Capacity`: `size`, `leased`, and `free` |
| `operations()`, `deduplicates` | what can be done to its sandboxes, and whether it performs each `effect_id` at most once |
| `await call(key, name, arguments, *, effect_id, arguments_digest)` | an operation on the sandbox leased under `key` |

[`SandboxPool(provider, name=..., leases=..., admits=...)`](../../guide/reference.md#sandboxpool) is a pool over a
provider:

- **One sandbox per key.** Acquires of a key, at once or after a crash, get one sandbox: its handle is derived from
  the key, and the provider makes a handle once.
- **At most `provider.size`** leases at once, counting those being made.
- **Its leases are kept** in `leases` (in memory by default) under its `name`, the provider's kind unless it is
  given one: pools that share a table need names of their own. A key leased from another pool is refused.
- **`admits(key)`**, when given, says whether a key may hold a lease now. A key it refuses raises `LeaseRefused`,
  and the lease the key had is released: nothing is made for it, even for a moment. Beside a ledger it says whether
  the key's claim holds ([below](#in-training-a-lease-ends-with-its-claim)); without it every key is admitted.
- **A sandbox is never made again under a key whose lease it was.** A lease whose sandbox is gone (it ended with
  the pool's process, say) is lost, and its key gets `SandboxLost`: a new sandbox would not be the one its run was
  playing in.
- **`sweep(ended)`** releases the leases `ended` says have ended and those past their wall time, marks lost those
  whose sandbox is gone, and deletes the sandboxes no lease names. A lost lease holds no room.
- **`close()`** releases every lease it holds and closes the provider; `close(release=False)` leaves the leases, for
  the runs a durable runner resumes to acquire again.

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
| `POST /acquire` | `spec`, `key`, `environment` | a `Lease`; 503 when the pool is full, 409 for a key it refuses (`LeaseRefused`), 410 for a key whose sandbox is gone (`SandboxLost`) |
| `POST /release` | `key` | |
| `POST /call` | `key`, `name`, `arguments`, `effect_id`, `arguments_digest` | a `ToolResult` |

A pool that raises answers 500 with `error`, and the client raises it.

`rollout pool FACTORY [--directory DIRECTORY] [--ledger WHERE] [--name NAME] [--host 127.0.0.1] [--port 8710]`
serves a `SandboxPool` over the provider `FACTORY` (`module:function`, called with the directory) returns, named
`NAME` (by default `KIND@HOST`). With `--ledger`, its leases are kept beside that ledger, it refuses a key whose
claim has lapsed, its keeper ends leases with their claims, and it beats as `pools/NAME`. Without, it cannot tell
whether a claim holds: it admits every key, keeps its leases in the process, and a lease ends only when released (a
run's own release, when it ends). A pool that serves training runs is served with `--ledger`.

## In training: a lease ends with its claim

An [episode runner](../rollout-train/rollouts.md#a-runner) starts each episode's run with the claim's key as its
lease, so the run's sandboxes are leased under `RUN/GROUP/EPISODE/ATTEMPT/NAME`: an attempt played again (a durable
run resumed) gets the same sandboxes; a new attempt gets new ones.

- **Capacity.** A runner claims an episode only while the pools of the sandboxes its program declares have room for
  them, asking each pool's `capacity()` once a look, and counting what it claims as it goes. Runners that share a
  pool can still claim more than it holds at once; their runs then wait for room ([the runner](#the-runner)). A
  runner says in each beat how full its pools are (`pools`: each one's `size`, `leased` and `free`).
- **Leases beside the ledger.** A pool opened by a profile keeps its leases beside the ledger, as ordinary state
  changed in place: `sandboxes.json` beside a ledger of files (`FileLeases`), the `sandboxes` table of a database
  ledger's database (`DatabaseLeases`); `leases_of(ledger)` finds them.
- **A lapsed claim gets no sandbox.** A claim holds as the scheduler says
  ([claims](../rollout-train/rollouts.md#what-runners-write)): it lapses when its runner takes its fence anew without
  adopting it, notes the attempt cut short, or stops beating for 90 seconds, and its lease ends with the episode's
  record too. A pool opened by a profile admits a key only while its claim holds (`admits(ledger, presence)`), so a
  run recovered after its claim lapsed is refused its sandboxes, and the lease it had is released at once. A key of
  a run the ledger does not know (a run started by hand) is admitted.
- **Expiry.** A pool's keeper (`keep(pool, ledger, presence)`) sweeps every 15 seconds, and releases a lease whose
  claim it found lapsed at two sweeps running (a runner started again adopts its claims a moment after it takes its
  fence anew), deleting its sandbox. A runner that dies leaves its sandboxes to their pools, which delete them within
  two minutes. A lease of a run the ledger does not know ends only when it is released.
- **A runner started again** over a durable runner adopts the runs of its claims that held until it stopped
  ([rollouts](../rollout-train/rollouts.md#a-runner-started-again)); they acquire their sandboxes under the same keys
  and get them back. A pool opened by a profile with a durable runner leaves its leases when it closes. A sandbox
  that outlived the process (a pool on a machine of its own) is the run's again; one that ended with it (a Paper
  server in the process) is lost, and the run is cut short and played again as a new attempt.
- **A pool started again** under its name finds its leases beside the ledger; those whose sandboxes are gone are
  marked lost, and sandboxes no lease names are deleted.
