"""Sandboxes: a program declares them, the runner acquires them under a key before the program starts and releases
them when it ends, and the program reaches them by name; a pool hands out one sandbox per key, as many as it holds."""

import asyncio
from collections.abc import Mapping

import httpx
import pytest
from pydantic import JsonValue

from rollout.contracts import ModelAddress, RunEventType, ToolCall, ToolResult
from rollout.harness import (
    DirectModel,
    Lease,
    LeaseRefused,
    MemoryLeases,
    ModelBinding,
    ModelSlot,
    Mount,
    Network,
    NoCapacity,
    Observation,
    Pool,
    PoolBinding,
    Process,
    Program,
    ProgramReference,
    RunBinding,
    RunContext,
    RunSpecification,
    RunStatus,
    SandboxLimits,
    SandboxLost,
    SandboxPool,
    SandboxSpec,
    Scratch,
    Task,
    agent_program,
    bind,
    register,
    tool,
)
from rollout.harness.remote import RemotePool, serve_pool
from rollout.harness.sandboxes import harness_environment
from rollout.local import LocalRunner
from rollout.testing import FakeSandboxes, ScriptedModelEndpoint, payload, tool_call_reply

BOX = SandboxSpec(kind="fake", parameters={"image": "alpine"})


async def test_the_same_key_gives_the_same_sandbox_and_another_key_another() -> None:
    sandboxes = FakeSandboxes(size=2)
    pool = SandboxPool(sandboxes)
    first = await pool.acquire(BOX, "run/1/1/1/box")
    again = await pool.acquire(BOX, "run/1/1/1/box")  # a retry, or a durable run replayed
    other = await pool.acquire(BOX, "run/1/1/2/box")  # a new attempt
    assert again == first and other.handle != first.handle and len(sandboxes.made) == 2
    assert first.addresses == {"fake": f"fake://{first.handle}"} and first.pool == "fake" and first.kind == "fake"
    assert (await pool.capacity()).free == 0
    await pool.release(first.key)
    await pool.release(first.key)  # (nothing the second time)
    assert sandboxes.deleted == [first.handle] and (await pool.capacity()).free == 1


async def test_a_full_pool_refuses_until_a_sandbox_is_released() -> None:
    pool = SandboxPool(FakeSandboxes(size=1))
    held = await pool.acquire(BOX, "a/box")
    with pytest.raises(NoCapacity):
        await pool.acquire(BOX, "b/box")
    assert await pool.acquire(BOX, "a/box") == held  # what it holds, it still hands out
    await pool.release("a/box")
    assert (await pool.acquire(BOX, "b/box")).key == "b/box"


async def test_a_pool_makes_one_kind_and_two_acquires_of_a_key_at_once_make_one_sandbox() -> None:
    sandboxes = FakeSandboxes(size=4)
    pool = SandboxPool(sandboxes)
    with pytest.raises(ValueError, match="makes fake sandboxes"):
        await pool.acquire(SandboxSpec(kind="minecraft"), "a/world")
    leases = await asyncio.gather(*(pool.acquire(BOX, "a/box") for _ in range(3)))
    assert len({lease.handle for lease in leases}) == 1 and len(sandboxes.made) == 1


async def test_a_sweep_releases_what_has_ended_and_deletes_what_no_lease_names() -> None:
    sandboxes = FakeSandboxes()
    pool = SandboxPool(sandboxes)
    ending = await pool.acquire(BOX, "ending/box")
    staying = await pool.acquire(BOX, "staying/box")
    await sandboxes.create("s-left-behind", BOX, {})  # a sandbox no lease names (made by a process that died)
    gone = await pool.sweep(lambda lease: lease.key.startswith("ending/"))
    assert gone == ["ending/box"] and set(sandboxes.sandboxes) == {staying.handle}
    assert set(sandboxes.deleted) == {ending.handle, "s-left-behind"}

    again = FakeSandboxes()
    restarted = SandboxPool(again, leases=pool.leases)  # the pool started again: its sandboxes are gone
    assert [lease.key for lease in await restarted.held()] == ["staying/box"]
    assert await restarted.sweep() == ["staying/box"]  # marked lost: its key cannot have it back
    (lost,) = await restarted.held()
    assert lost.lost and (await restarted.capacity()).leased == 0
    for _ in range(2):  # (a new one would not be the one its run was playing in: never, until its run releases it)
        with pytest.raises(SandboxLost):
            await restarted.acquire(BOX, "staying/box")
    assert [lease.lost for lease in await restarted.held()] == [True] and again.made == []
    await restarted.release("staying/box")
    assert await restarted.held() == [] and (await restarted.acquire(BOX, "staying/box")).key == "staying/box"


async def test_a_pool_refuses_a_key_it_does_not_admit_and_releases_what_the_key_held() -> None:
    sandboxes = FakeSandboxes()
    lapsed: set[str] = set()

    async def admitted(key: str) -> bool:
        return key not in lapsed

    pool = SandboxPool(sandboxes, admits=admitted)
    lease = await pool.acquire(BOX, "run/1/1/1/box")
    lapsed.add(lease.key)  # its claim lapsed (its runner was replaced, say)
    with pytest.raises(LeaseRefused):
        await pool.acquire(BOX, lease.key)
    assert sandboxes.deleted == [lease.handle] and await pool.held() == []
    with pytest.raises(LeaseRefused):  # nor is it made again, even for a moment
        await pool.acquire(BOX, lease.key)
    assert sandboxes.made == [lease.handle]

    transport = httpx.ASGITransport(app=serve_pool(pool))
    remote = RemotePool("http://pool", client=httpx.AsyncClient(transport=transport, base_url="http://pool"))
    with pytest.raises(LeaseRefused):
        await remote.acquire(BOX, lease.key)


async def test_a_pool_over_http_is_the_same_pool() -> None:
    sandboxes = FakeSandboxes(size=1)
    transport = httpx.ASGITransport(app=serve_pool(SandboxPool(sandboxes)))
    client = httpx.AsyncClient(transport=transport, base_url="http://pool")
    remote = RemotePool("http://pool", client=client, operations=sandboxes.operations())
    lease = await remote.acquire(BOX, "a/box", {"GREETING": "hello"})
    assert await remote.acquire(BOX, "a/box") == lease and lease.environment == {"GREETING": "hello"}
    with pytest.raises(NoCapacity):
        await remote.acquire(BOX, "b/box")
    assert (await remote.capacity()).free == 0
    described = await remote.call(lease.key, "describe", {}, effect_id="e", arguments_digest="d")
    assert described.structured == {"handle": lease.handle, "parameters": {"image": "alpine"},
                                    "environment": {"GREETING": "hello"}, "process": None}  # fmt: skip
    await remote.release(lease.key)
    assert sandboxes.deleted == [lease.handle] and (await remote.capacity()).free == 1


class Addressed(ScriptedModelEndpoint):
    """A scripted endpoint that also says where a harness reaches each session."""

    def address(self, session_id: str) -> ModelAddress:
        return ModelAddress(base_url="http://models/v1", api_key=f"key-{session_id}", model="scripted")


class Builds(Program):
    """Declares a box for a coding agent that plays the `coder` slot, and one for each of two players; reports what
    it was given and what its operations answered."""

    def model_slots(self) -> Mapping[str, ModelSlot]:
        return {"coder": ModelSlot(), "one": ModelSlot(), "two": ModelSlot()}

    def sandboxes(self) -> Mapping[str, SandboxSpec]:
        return {
            "box": SandboxSpec(kind="fake", parameters={"image": "alpine"}, slots=("coder",)),
            "shared": SandboxSpec(kind="fake", slots=("one", "two")),
        }

    async def main(self, run: RunContext) -> None:
        box = run.sandbox("box")
        described = await box.call("describe")
        with pytest.raises(KeyError):
            run.sandbox("undeclared")
        given: JsonValue = {"box": dict(box.environment), "shared": dict(run.sandbox("shared").environment)}
        await run.emit("given", {"environment": given, "described": described.structured, "key": box.lease.key})


def runner_with(pool: Pool) -> tuple[LocalRunner, RunSpecification]:
    endpoint = Addressed([])
    runner = LocalRunner(providers={"scripted": lambda model: endpoint}, pools={"boxes": pool})
    direct = ModelBinding(direct=DirectModel(provider="scripted", model="s"))
    binding = RunBinding(
        models=dict.fromkeys(("coder", "one", "two"), direct), pools={"fake": PoolBinding(local="boxes")}
    )
    return runner, RunSpecification(program=ProgramReference(program=register(Builds)), binding=binding)


async def test_a_program_reaches_its_sandboxes_by_name_and_a_harness_inside_is_given_its_slots_model() -> None:
    sandboxes = FakeSandboxes()
    runner, specification = runner_with(SandboxPool(sandboxes))
    handle = await runner.start(specification, run_id="run_1", lease="train/3/2/1")
    assert (await handle.result()).status is RunStatus.COMPLETED
    events = handle.recorded_events()
    (given,) = [payload(event)["payload"] for event in events if event.type is RunEventType.OUTPUT_EMITTED]
    assert isinstance(given, dict) and given["key"] == "train/3/2/1/box"  # under the run's lease and its name
    environment: dict[str, dict[str, str]] = given["environment"]  # type: ignore[assignment]
    assert environment["box"] == {
        "OPENAI_BASE_URL": "http://models/v1", "OPENAI_API_KEY": "key-run_1/coder", "OPENAI_MODEL": "scripted",
        "OPENAI_BASE_URL_CODER": "http://models/v1", "OPENAI_API_KEY_CODER": "key-run_1/coder",
        "OPENAI_MODEL_CODER": "scripted",
        "ANTHROPIC_BASE_URL": "http://models", "ANTHROPIC_AUTH_TOKEN": "key-run_1/coder", "ANTHROPIC_MODEL": "scripted",
        "ANTHROPIC_BASE_URL_CODER": "http://models", "ANTHROPIC_AUTH_TOKEN_CODER": "key-run_1/coder",
        "ANTHROPIC_MODEL_CODER": "scripted",
    }  # fmt: skip
    shared = environment["shared"]
    assert shared["OPENAI_API_KEY_ONE"] == shared["ANTHROPIC_AUTH_TOKEN_ONE"] == "key-run_1/one"
    assert shared["OPENAI_API_KEY_TWO"] == shared["ANTHROPIC_AUTH_TOKEN_TWO"] == "key-run_1/two"
    assert "OPENAI_API_KEY" not in shared and "ANTHROPIC_AUTH_TOKEN" not in shared
    described: dict[str, JsonValue] = given["described"]  # type: ignore[assignment]
    assert described["environment"] == environment["box"]  # what the provider made it with
    (acquired,) = [payload(event)["sandboxes"] for event in events if event.type is RunEventType.SANDBOXES_ACQUIRED]
    assert isinstance(acquired, dict) and sorted(acquired) == ["box", "shared"]
    (operation,) = [
        payload(event)["payload"]
        for event in events
        if event.type is RunEventType.EFFECT_REQUESTED and payload(event)["kind"] == "tool.call"
    ]
    assert operation == {"sandbox": "box", "tool": "describe", "arguments": {}}  # a recorded effect, by name
    assert sandboxes.sandboxes == {} and len(sandboxes.deleted) == 2  # released when the program ended


class Breaks(Builds):
    async def main(self, run: RunContext) -> None:
        raise RuntimeError("the program broke")


class Waits(Builds):
    async def main(self, run: RunContext) -> None:
        await asyncio.Event().wait()


@pytest.mark.parametrize("program", [Breaks, Waits])
async def test_a_runs_sandboxes_are_released_however_it_ends(program: type[Program]) -> None:
    sandboxes = FakeSandboxes()
    runner, specification = runner_with(SandboxPool(sandboxes))
    specification = specification.model_copy(update={"program": ProgramReference(program=register(program))})
    handle = await runner.start(specification)
    if program is Waits:
        await asyncio.sleep(0.05)
        await runner.cancel(handle.run_id, reason="enough")
    outcome = await handle.result()
    assert outcome.status is (RunStatus.FAILED if program is Breaks else RunStatus.CANCELLED)
    assert sandboxes.sandboxes == {} and len(sandboxes.made) == len(sandboxes.deleted) == 2


async def test_a_run_that_cannot_get_its_sandboxes_fails_and_holds_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from rollout.harness import sandboxes as module

    monkeypatch.setattr(module, "ACQUIRE_SECONDS", 0.0)
    sandboxes = FakeSandboxes(size=1)  # room for the box, none for the shared one
    runner, specification = runner_with(SandboxPool(sandboxes))
    handle = await runner.start(specification)
    outcome = await handle.result()
    assert outcome.status is RunStatus.FAILED and "NoCapacity" in str(outcome.detail)
    assert sandboxes.sandboxes == {}  # what it did get, it released


def test_a_binding_serves_each_kind_from_the_pool_of_its_name_unless_told_otherwise() -> None:
    reference = ProgramReference(program=register(Builds))
    assert bind(reference, "policy").pools == {"fake": PoolBinding(local="fake")}
    elsewhere = PoolBinding(url="http://boxes:8710")
    assert bind(reference, "policy", pools={"fake": elsewhere}).pools == {"fake": elsewhere}


WORKER = SandboxSpec(
    kind="fake",
    process=Process(command=["python", "-m", "worker"], environment={"WORKER": "1"}, directory="/env"),
    mounts=[
        Mount(source="environments/minecraft-team/1.4.0", target="/env"),
        Mount(source="venvs/a1b2", target="/venv"),
    ],
    scratch=Scratch(path="/scratch", mib=1),
    network=Network(allow=["pypi.org"]),
    limits=SandboxLimits(cpus=2, memory_mib=2048, processes=64, seconds=3600),
    slots=("coder",),
)


class Works(Builds):
    """Runs an environment's worker in a sandbox, as it would a world: the worker is given its slot's model, may
    write only its scratch directory and reach only the hosts its network allows."""

    def sandboxes(self) -> Mapping[str, SandboxSpec]:
        return {"worker": WORKER}

    async def main(self, run: RunContext) -> None:
        worker = run.sandbox("worker")
        asked: list[tuple[str, dict[str, JsonValue]]] = [
            ("describe", {}),
            ("write", {"path": "/scratch/cache", "bytes": 1000}),
            ("write", {"path": "/env/pyproject.toml", "bytes": 10}),
            ("write", {"path": "/scratch/big", "bytes": 2**20}),
            ("write", {"path": "/tmp/elsewhere", "bytes": 10}),
            ("fetch", {"host": "pypi.org"}),
            ("fetch", {"host": "example.com"}),
        ]
        answered: list[JsonValue] = []
        for operation, arguments in asked:
            result = await worker.call(operation, arguments)
            answered.append("refused" if result.is_error else result.structured)
        await run.emit(
            "worked", {"answered": answered, "addresses": dict(worker.addresses), "seconds": worker.lease.seconds}
        )


async def test_a_worker_for_an_environment_is_a_sandbox_like_a_world() -> None:
    sandboxes = FakeSandboxes()
    runner, specification = runner_with(SandboxPool(sandboxes))
    specification = specification.model_copy(update={"program": ProgramReference(program=register(Works))})
    handle = await runner.start(specification, run_id="run_2")
    assert (await handle.result()).status is RunStatus.COMPLETED
    events = handle.recorded_events()
    (worked,) = [payload(event)["payload"] for event in events if event.type is RunEventType.OUTPUT_EMITTED]
    assert isinstance(worked, dict)
    described, *rest = worked["answered"]  # type: ignore[misc]
    process: dict[str, JsonValue] = described["process"]  # type: ignore[index]
    assert process["command"] == ["python", "-m", "worker"]
    assert process["environment"] == {  # its own variables, and its slot's model, scoped to the run
        "WORKER": "1", "OPENAI_BASE_URL": "http://models/v1", "OPENAI_API_KEY": "key-run_2/coder",
        "OPENAI_MODEL": "scripted", "OPENAI_BASE_URL_CODER": "http://models/v1",
        "OPENAI_API_KEY_CODER": "key-run_2/coder", "OPENAI_MODEL_CODER": "scripted",
        "ANTHROPIC_BASE_URL": "http://models", "ANTHROPIC_AUTH_TOKEN": "key-run_2/coder",
        "ANTHROPIC_MODEL": "scripted", "ANTHROPIC_BASE_URL_CODER": "http://models",
        "ANTHROPIC_AUTH_TOKEN_CODER": "key-run_2/coder", "ANTHROPIC_MODEL_CODER": "scripted",
    }  # fmt: skip
    assert rest == [{"written": 1000}, "refused", "refused", "refused", {"reached": "pypi.org"}, "refused"]
    addresses: dict[str, str] = worked["addresses"]  # type: ignore[assignment]
    assert addresses["process"].endswith("/process")  # how the runner reaches the worker
    assert worked["seconds"] == 3600  # its time limit
    assert sandboxes.sandboxes == {}


async def test_a_sandbox_past_its_wall_time_is_deleted_by_the_next_sweep() -> None:
    sandboxes = FakeSandboxes()
    pool = SandboxPool(sandboxes)
    brief = await pool.acquire(WORKER.model_copy(update={"limits": SandboxLimits(seconds=0)}), "brief/worker")
    lasting = await pool.acquire(WORKER, "lasting/worker")
    assert brief.seconds == 0 and lasting.seconds == 3600
    assert await pool.sweep() == ["brief/worker"] and list(sandboxes.sandboxes) == [lasting.handle]


async def test_a_pool_started_again_counts_the_time_its_leases_have_lasted() -> None:
    from rollout.harness.sandboxes import MemoryLeases

    sandboxes, leases = FakeSandboxes(), MemoryLeases()
    minute = WORKER.model_copy(update={"limits": SandboxLimits(seconds=60)})
    pool = SandboxPool(sandboxes, leases=leases)
    old, young = await pool.acquire(minute, "old/worker"), await pool.acquire(minute, "young/worker")
    leases.leases[old.key] = old.model_copy(update={"at": old.at - 61})  # made a minute and more ago
    again = SandboxPool(sandboxes, leases=leases)  # the pool's process started again: its clock starts anew
    assert await again.sweep() == ["old/worker"] and list(sandboxes.sandboxes) == [young.handle]


async def test_a_lost_key_gets_sandbox_lost_at_every_acquire_until_its_run_releases_it() -> None:
    sandboxes, leases = FakeSandboxes(), MemoryLeases()
    pool = SandboxPool(sandboxes, leases=leases)
    lost = await pool.acquire(BOX, "run/1/1/1/box")
    await pool.lose(lost.key)  # (its pool ended it while the episode played)
    found = await pool.acquire(BOX, "run/1/2/1/box")
    again = FakeSandboxes()
    restarted = SandboxPool(again, leases=leases)  # (started again: no sweep has looked, the acquire finds it gone)
    for asked, key in ((pool, lost.key), (restarted, found.key)):
        for _ in range(2):
            with pytest.raises(SandboxLost):
                await asked.acquire(BOX, key)
        assert leases.leases[key].lost  # (kept, marked lost: never forgotten until its run releases it)
    assert sandboxes.made == [lost.handle, found.handle] and again.made == []
    await restarted.release(found.key)  # (its run ended, and released it)
    assert found.key not in leases.leases and (await restarted.acquire(BOX, found.key)).handle in again.sandboxes


async def test_two_first_loads_at_once_keep_a_lease_made_in_between() -> None:
    class Slow(MemoryLeases):
        looks = 0

        async def all(self) -> list[Lease]:
            Slow.looks += 1
            found = list(self.leases.values())  # (read now, answered late the first time)
            await asyncio.sleep(0.1 if Slow.looks == 1 else 0)
            return found

    pool = SandboxPool(FakeSandboxes(size=1), leases=Slow())
    looking = asyncio.ensure_future(pool.held())  # (a keeper's first sweep, say)
    await asyncio.sleep(0)
    lease = await pool.acquire(BOX, "run/1/1/1/box")  # (a run starts meanwhile)
    await looking
    assert await pool.held() == [lease] and (await pool.capacity()).free == 0
    await pool.call(lease.key, "describe", {}, effect_id="e", arguments_digest="d")


async def test_a_pools_table_keeps_no_key_of_a_slot_and_an_acquire_that_finds_a_lease_there_is_given_them() -> None:
    address = ModelAddress(base_url="http://models/v1", api_key="key-of-the-run", model="scripted")
    given = harness_environment(["coder"], lambda slot: address)
    sandboxes, leases = FakeSandboxes(), MemoryLeases()
    lease = await SandboxPool(sandboxes, leases=leases).acquire(BOX, "run/1/1/1/box", given)
    assert lease.environment == given  # (what runs inside it is given the keys)
    (kept,) = leases.leases.values()
    assert "key-of-the-run" not in kept.model_dump_json() and kept.environment["OPENAI_MODEL_CODER"] == "scripted"
    again = SandboxPool(sandboxes, leases=leases)  # (started again: its sandbox is still there)
    assert await again.acquire(BOX, lease.key, given) == lease and len(sandboxes.made) == 1
    await again.lose(lease.key)
    assert "key-of-the-run" not in leases.leases[lease.key].model_dump_json()


async def test_a_runs_sandboxes_acquired_event_records_no_key_of_its_slots() -> None:
    runner, specification = runner_with(SandboxPool(FakeSandboxes()))
    handle = await runner.start(specification, run_id="run_3")
    assert (await handle.result()).status is RunStatus.COMPLETED
    (acquired,) = [event for event in handle.recorded_events() if event.type is RunEventType.SANDBOXES_ACQUIRED]
    leases = payload(acquired)["sandboxes"]
    assert isinstance(leases, dict) and sorted(leases) == ["box", "shared"]
    assert all(isinstance(lease, dict) and "environment" not in lease for lease in leases.values())
    assert "key-run_3" not in acquired.model_dump_json()


class Pokes(Task):
    """Two tool calls at once: one reaches its box, which its pool loses, while the other waits. Scored 1 whatever
    happens."""

    sandboxes = {"box": BOX}
    cancelled: list[str] = []

    async def start(self, run: RunContext) -> Observation:
        return Observation("go")

    @tool
    async def poke(self, run: RunContext) -> str:
        """Reach the box."""
        await asyncio.sleep(0.01)
        return str((await run.sandbox("box").call("describe")).structured)

    @tool
    async def wait(self, run: RunContext) -> str:
        """Wait a while."""
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            Pokes.cancelled.append(run.run_id)
            raise
        return "waited"

    async def score(self, run: RunContext) -> float:
        return 1.0


class Losing(SandboxPool):
    """Loses a sandbox as an operation reaches it (its process was started again, say)."""

    async def call(
        self, key: str, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        await self.lose(key)
        return await super().call(key, name, arguments, effect_id=effect_id, arguments_digest=arguments_digest)


async def test_a_tool_whose_sandbox_is_lost_fails_the_run_to_be_played_again_and_ends_the_other_calls() -> None:
    pool = Losing(FakeSandboxes())
    calls = [ToolCall(call_id="c1", name="poke", arguments={}), ToolCall(call_id="c2", name="wait", arguments={})]
    endpoint = ScriptedModelEndpoint([tool_call_reply(*calls), "done"])
    runner = LocalRunner(providers={"scripted": lambda model: endpoint}, pools={"boxes": pool})
    binding = RunBinding(
        models={"policy": ModelBinding(direct=DirectModel(provider="scripted", model="s"))},
        pools={"fake": PoolBinding(local="boxes")},
    )
    handle = await runner.start(RunSpecification(program=agent_program(Pokes), binding=binding))
    outcome = await handle.result()
    assert outcome.status is RunStatus.FAILED and SandboxLost.__name__ in str(outcome.detail)  # (what is played again)
    types = [event.type for event in handle.recorded_events()]
    assert RunEventType.REWARD_ASSIGNED not in types and types[-1] is RunEventType.RUN_FAILED
    assert handle.run_id in Pokes.cancelled and await pool.held() == []  # (its run released the lost lease)


class Slow(FakeSandboxes):
    async def delete(self, handle: str) -> None:
        await asyncio.sleep(0.05)
        await super().delete(handle)


async def test_a_second_cancel_does_not_cut_a_runs_release_of_its_sandboxes_short() -> None:
    sandboxes = Slow()
    runner, specification = runner_with(SandboxPool(sandboxes))
    specification = specification.model_copy(update={"program": ProgramReference(program=register(Waits))})
    handle = await runner.start(specification)
    await asyncio.sleep(0.02)
    first = asyncio.ensure_future(runner.cancel(handle.run_id, reason="lapsed"))
    await asyncio.sleep(0.02)  # (it is releasing its first sandbox now)
    await runner.cancel(handle.run_id, reason="closing")  # (at shutdown, say)
    await first
    assert (await handle.result()).status is RunStatus.CANCELLED
    assert sandboxes.sandboxes == {} and len(sandboxes.deleted) == 2
