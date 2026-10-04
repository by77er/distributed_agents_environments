"""Hooks: a runner tells them of every event it records and every sample its models make, with the content."""

from pathlib import Path

from rollout.contracts import (
    AddressableEndpoint,
    ModelAddress,
    RunEvent,
    RunEventType,
)
from rollout.harness import (
    DirectModel,
    ModelBinding,
    ModelSample,
    ModelSlot,
    Program,
    ProgramReference,
    RunBinding,
    RunContext,
    RunHooks,
    RunSpecification,
    RunStatus,
    register,
)
from rollout.local import LocalRunner
from rollout.testing import LedgerEndpoint
from tests.rollout.harness.support import Miner, specification


class Watching(RunHooks):
    def __init__(self) -> None:
        self.events: list[RunEvent] = []
        self.samples: list[ModelSample] = []

    def on_event(self, event: RunEvent) -> None:
        self.events.append(event)

    def on_sample(self, sample: ModelSample) -> None:
        self.samples.append(sample)


class Broken(RunHooks):
    def on_event(self, event: RunEvent) -> None:
        raise RuntimeError("a hook's own trouble")

    def on_sample(self, sample: ModelSample) -> None:
        raise RuntimeError("a hook's own trouble")


async def test_hooks_see_every_event_and_every_sample_with_its_content() -> None:
    watching = Watching()
    runner = LocalRunner(providers={"scripted": lambda model: Miner()}, hooks=[Broken(), watching])
    handle = await runner.start(specification(), labels={"group": "g1"})
    outcome = await handle.result()
    assert outcome.status is RunStatus.COMPLETED  # a hook that raises does not fail the run
    assert watching.events == handle.recorded_events()  # each one, in order, as it was recorded
    assert (
        watching.events[0].type is RunEventType.RUN_CREATED and watching.events[-1].type is RunEventType.RUN_COMPLETED
    )
    (sample,) = watching.samples
    assert (sample.run_id, sample.slot) == (handle.run_id, "ada")
    assert [message.text for message in sample.request.context.append] == ["You mine.", "You see a wall."]
    assert [tool.name for tool in sample.request.tools] == ["mine"]
    assert sample.result.message.tool_calls[0].arguments == {"x": 3}
    assert sample.seconds >= 0


class ServedMiner(Miner):
    """A `Miner` that is also reached over HTTP; remembers the sessions it was asked the address of."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def address(self, session_id: str) -> ModelAddress:
        self.asked.append(session_id)
        return ModelAddress(base_url="http://models", api_key=session_id, model="miner")


class Addressed(Program):
    def model_slots(self) -> dict[str, ModelSlot]:
        return {"ada": ModelSlot()}

    async def main(self, run: RunContext) -> None:
        await run.emit("address", run.models["ada"].address().api_key)


async def test_a_harness_is_given_its_endpoints_address_through_hooks_and_wrappers(tmp_path: Path) -> None:
    served = ServedMiner()
    binding = RunBinding(models={"ada": ModelBinding(direct=DirectModel(provider="scripted", model="miner"))})
    addressed = RunSpecification(program=ProgramReference(program=register(Addressed)), binding=binding)
    ledger = LedgerEndpoint(served, tmp_path / "ledger.jsonl")
    for endpoint, hooks in ((served, []), (served, [Watching()]), (ledger, [Watching()])):
        handle = await LocalRunner(providers={"scripted": lambda model, e=endpoint: e}, hooks=hooks).start(addressed)
        assert (await handle.result()).status is RunStatus.COMPLETED
    assert len(served.asked) == 3  # (the endpoint's own address, whatever wraps it: hooks, or a ledger)

    unserved = await LocalRunner(providers={"scripted": lambda model: Miner()}, hooks=[Watching()]).start(addressed)
    outcome = await unserved.result()
    assert outcome.status is RunStatus.FAILED and "not served over HTTP" in str(outcome.detail)
    assert isinstance(served, AddressableEndpoint) and not isinstance(Miner(), AddressableEndpoint)


async def test_a_runner_without_hooks_is_unchanged() -> None:
    runner = LocalRunner(providers={"scripted": lambda model: Miner()})
    handle = await runner.start(specification())
    assert (await handle.result()).status is RunStatus.COMPLETED
