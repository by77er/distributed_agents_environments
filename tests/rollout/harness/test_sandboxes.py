"""Sandboxes: a program declares them, the runner acquires them under a key before the program starts and releases
them when it ends, and the program reaches them by name; a pool hands out one sandbox per key, as many as it holds."""

import asyncio
from collections.abc import Mapping

import httpx
import pytest
from pydantic import JsonValue

from rollout.contracts import ModelAddress, ModelEndpoint, RunEventType
from rollout.harness import (
    DirectModel,
    ModelBinding,
    ModelSlot,
    Mount,
    Network,
    NoCapacity,
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
    SandboxPool,
    SandboxSpec,
    Scratch,
    bind,
    register,
)
from rollout.harness.remote import RemotePool, serve_pool
from rollout.local import LocalRunner
from rollout.testing import FakeSandboxes, ScriptedModelEndpoint, payload

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

    restarted = SandboxPool(FakeSandboxes(), leases=pool.leases)  # the pool started again: its sandboxes are gone
    assert [lease.key for lease in await restarted.held()] == ["staying/box"]
    assert await restarted.sweep() == ["staying/box"] and await restarted.held() == []


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

    def address(self, session_id: str, *, through: ModelEndpoint | None = None) -> ModelAddress:
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
    }  # fmt: skip
    assert environment["shared"]["OPENAI_API_KEY_ONE"] == "key-run_1/one"
    assert (
        environment["shared"]["OPENAI_API_KEY_TWO"] == "key-run_1/two" and "OPENAI_API_KEY" not in environment["shared"]
    )
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
        await run.emit("worked", {"answered": answered, "addresses": dict(worker.addresses), "ends": worker.lease.ends})


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
    }  # fmt: skip
    assert rest == [{"written": 1000}, "refused", "refused", "refused", {"reached": "pypi.org"}, "refused"]
    addresses: dict[str, str] = worked["addresses"]  # type: ignore[assignment]
    assert addresses["process"].endswith("/process")  # how the runner reaches the worker
    assert isinstance(worked["ends"], float)  # its wall time
    assert sandboxes.sandboxes == {}


async def test_a_sandbox_past_its_wall_time_is_deleted_by_the_next_sweep() -> None:
    sandboxes = FakeSandboxes()
    pool = SandboxPool(sandboxes)
    brief = await pool.acquire(WORKER.model_copy(update={"limits": SandboxLimits(seconds=0)}), "brief/worker")
    lasting = await pool.acquire(WORKER, "lasting/worker")
    assert brief.ends is not None and lasting.ends is not None and lasting.ends > brief.ends
    assert await pool.sweep() == ["brief/worker"] and list(sandboxes.sandboxes) == [lasting.handle]
