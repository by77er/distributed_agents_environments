from collections.abc import Mapping

from rollout.contracts import EffectKind, RunEventType
from rollout.harness import (
    EnvironmentSpecification,
    ExecutionResult,
    ModelSlot,
    Program,
    ProgramReference,
    RunBinding,
    RunContext,
    RunSpecification,
    RunStatus,
    register,
)
from rollout.local import LocalRunner
from rollout.testing import payload


class MemoryEnvironments:
    """An in-memory environment service: files in dictionaries, commands echoed."""

    def __init__(self) -> None:
        self.files: dict[str, dict[str, bytes]] = {}
        self.destroyed: list[str] = []

    async def create(self, environment_id: str, specification: EnvironmentSpecification) -> None:
        self.files.setdefault(environment_id, {})

    async def execute(
        self, environment_id: str, command: str, *, timeout_seconds: float, cwd: str | None, effect_id: str = ""
    ) -> ExecutionResult:
        return ExecutionResult(exit_code=0, output=f"ran: {command}\n")

    async def put(self, environment_id: str, path: str, data: bytes) -> None:
        self.files[environment_id][path] = data

    async def get(self, environment_id: str, path: str) -> bytes:
        return self.files[environment_id][path]

    async def destroy(self, environment_id: str) -> None:
        self.destroyed.append(environment_id)


class Build(Program):
    def model_slots(self) -> Mapping[str, ModelSlot]:
        return {}

    async def main(self, run: RunContext) -> None:
        assert run.environments is not None
        environment = await run.environments.create(EnvironmentSpecification(image="alpine"))
        await environment.put("main.c", "int main() {}")
        result = await environment.execute("cc main.c")
        await run.emit("result", {"output": result.output, "source": (await environment.get("main.c")).decode()})


async def test_environments_are_effects_and_are_released_when_the_run_ends() -> None:
    service = MemoryEnvironments()
    runner = LocalRunner(environments=service)
    specification = RunSpecification(program=ProgramReference(program=register(Build)), binding=RunBinding(models={}))
    handle = await runner.start(specification)
    assert (await handle.result()).status is RunStatus.COMPLETED

    events = handle.recorded_events()
    kinds = [payload(e)["kind"] for e in events if e.type is RunEventType.EFFECT_REQUESTED]
    lifecycle, call, emit = EffectKind.ENVIRONMENT_LIFECYCLE, EffectKind.ENVIRONMENT_CALL, EffectKind.OUTPUT_EMIT
    assert kinds == [lifecycle.value, call.value, call.value, call.value, emit.value]
    (output,) = [payload(e)["payload"] for e in events if e.type is RunEventType.OUTPUT_EMITTED]
    assert output == {"output": "ran: cc main.c\n", "source": "int main() {}"}
    (environment_id,) = service.files
    assert environment_id.startswith("e_")
    assert service.destroyed == [environment_id]  # released when the run ended
