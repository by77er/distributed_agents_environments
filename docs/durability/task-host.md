# Task host

Status: **Proposed** · Layer: durability (optional) · See [ADR-0020](../decisions/0020-deterministic-event-loop.md), [ADR-0021](../decisions/0021-trust-tiers.md)

The task host runs program, task and agent code for the `DurableRunner`. It is a pure function of the inputs the
pump feeds it: given the same completions in the same order it makes the same effect requests. It holds no
credentials and has no network.

## HarnessHost protocol

```python
class HarnessHost(Protocol):                           # spoken over a local socket; JSON only
    async def start(self, specification: RunSpecification, state: StateReference | None) -> HostDecision: ...
    async def step(self, completions: list[EffectCompletion]) -> HostDecision: ...
    async def export_state(self) -> StateReference: ...  # at resumable points: WaitFor, generation hand-over
    async def close(self) -> None: ...

@dataclass(frozen=True)
class HostDecision:
    effects: list[EffectRequest]        # new requests, in deterministic order (ordinals assigned by the host)
    events: list[RunEvent]              # observations, rewards, outputs, messages to project
    outcome: RunOutcome | None = None   # terminal
    suspend: WaitFor | None = None      # end the activation and wait
    hand_over: HandOver | None = None   # start a new generation from exported state
```

## Deterministic event loop

The host runs code on a custom `asyncio` event loop (the design Temporal's Python SDK uses), so ordinary
`asyncio.gather`, `create_task`, `Lock`, `Queue` and `wait_for` work, and third-party agent frameworks run unchanged.

- Ready callbacks run in a deterministic order.
- Awaiting a `run` operation or a handle method creates an effect future resolved from `step` inputs.
- `loop.time()` maps to `run.now()`; timers become `timer.sleep` effects.
- Real I/O (`sock_*`, `create_connection`, `run_in_executor`, subprocesses) raises `NonDeterminismError`.
- `time.time`, `uuid.uuid4`, `os.urandom` and `random` are patched to run-scoped deterministic sources; threads are
  rejected.

## State and generations

- At resumable points (a `WaitFor`, a generation hand-over) the host exports task and agent state. Long-lived state
  (e.g. a conversation carried across activations) is exported as versioned records; pickle is allowed only for
  state that never outlives its `code_reference`, and is unpickled only inside the sandbox with a restricted
  unpickler.
- History is not part of exported state; it is rebuilt from run events.

## Isolation

The sandbox, how hosts are pooled and where they run depend on who writes the code
([trust tiers](../platform/trust-tiers.md)). In every tier the host has no network, no credentials, and only the
local socket to the pump.
