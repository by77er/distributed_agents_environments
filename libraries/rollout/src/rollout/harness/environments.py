"""Environments: computers that task code creates and acts on (docs/guide/tasks.md#environments).

Task code sees `run.environments` and `Environment` handles. Every operation is an effect, performed by an
`EnvironmentService`: the backend a runner is given. Creation derives the environment's id from its `effect_id`, so a
retried creation finds the same environment.
"""

import hashlib
from typing import Protocol

from pydantic import JsonValue

from rollout.contracts import ContractModel, EffectKind, FrozenSequence
from rollout.harness.model import Effects


class EnvironmentSpecification(ContractModel):
    image: str = "alpine"
    """A base image the backend knows, e.g. `alpine` (latest) or `alpine:3.24.2`."""
    setup: FrozenSequence[str] = ()
    """Shell commands run once at creation: a recipe for identical start states."""


class ExecutionResult(ContractModel):
    exit_code: int | None
    """None when the command timed out."""
    output: str
    """Standard output and standard error, interleaved."""
    truncated: bool = False
    """`output` is only the end of the output (see `full_output_path`)."""
    timed_out: bool = False
    full_output_path: str | None = None
    """When truncated: where, inside the environment, the full output was saved."""


class EnvironmentService(Protocol):
    """An environment backend. Every method must be safe to repeat with the same arguments, except `execute`."""

    async def create(self, environment_id: str, specification: EnvironmentSpecification) -> None:
        """Create the environment, or do nothing if it already exists."""
        ...

    async def execute(
        self, environment_id: str, command: str, *, timeout_seconds: float, cwd: str | None, effect_id: str = ""
    ) -> ExecutionResult:
        """Run a command. `effect_id` identifies the attempt (it is the same on every re-execution)."""
        ...

    async def put(self, environment_id: str, path: str, data: bytes) -> None: ...
    async def get(self, environment_id: str, path: str) -> bytes: ...
    async def destroy(self, environment_id: str) -> None:
        """Destroy the environment, or do nothing if it is already gone."""
        ...


class Environment:
    """A handle to one environment. Its methods are effects."""

    def __init__(self, environment_id: str, service: EnvironmentService, effects: Effects) -> None:
        self.environment_id = environment_id
        self._service = service
        self._effects = effects

    async def execute(self, command: str, *, timeout_seconds: float = 120.0, cwd: str | None = None) -> ExecutionResult:
        """Run a shell command."""
        arguments: dict[str, JsonValue] = {
            "environment_id": self.environment_id,
            "operation": "execute",
            "command": command,
            "timeout_seconds": timeout_seconds,
            "cwd": cwd,
        }

        async def execute(effect_id: str, arguments_digest: str) -> ExecutionResult:
            return await self._service.execute(
                self.environment_id, command, timeout_seconds=timeout_seconds, cwd=cwd, effect_id=effect_id
            )

        return await self._effects.perform(
            EffectKind.ENVIRONMENT_CALL,
            arguments,
            execute,
            completion=lambda result: result.model_dump(mode="json"),
        )

    async def put(self, path: str, data: bytes | str) -> None:
        """Write a file (relative paths are under the working directory)."""
        content = data.encode() if isinstance(data, str) else data
        arguments: dict[str, JsonValue] = {
            "environment_id": self.environment_id,
            "operation": "put",
            "path": path,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }

        async def execute(effect_id: str, arguments_digest: str) -> None:
            await self._service.put(self.environment_id, path, content)

        await self._effects.perform(EffectKind.ENVIRONMENT_CALL, arguments, execute, completion=lambda _: None)

    async def get(self, path: str) -> bytes:
        arguments: dict[str, JsonValue] = {"environment_id": self.environment_id, "operation": "get", "path": path}

        async def execute(effect_id: str, arguments_digest: str) -> bytes:
            return await self._service.get(self.environment_id, path)

        return await self._effects.perform(
            EffectKind.ENVIRONMENT_CALL, arguments, execute, completion=lambda data: {"size": len(data)}
        )

    async def destroy(self) -> None:
        arguments: dict[str, JsonValue] = {"environment_id": self.environment_id, "operation": "destroy"}

        async def execute(effect_id: str, arguments_digest: str) -> None:
            await self._service.destroy(self.environment_id)

        await self._effects.perform(EffectKind.ENVIRONMENT_LIFECYCLE, arguments, execute, completion=lambda _: None)


class Environments:
    """`run.environments`: creates environments the run owns."""

    def __init__(self, service: EnvironmentService, effects: Effects) -> None:
        self._service = service
        self._effects = effects
        self.owned: list[str] = []
        """Environments this run created; runners destroy any that remain when the run ends."""

    async def create(self, specification: EnvironmentSpecification | None = None) -> Environment:
        specification = specification or EnvironmentSpecification()

        async def execute(effect_id: str, arguments_digest: str) -> str:
            environment_id = "e_" + hashlib.sha256(effect_id.encode()).hexdigest()[:24]
            await self._service.create(environment_id, specification)
            return environment_id

        environment_id = await self._effects.perform(
            EffectKind.ENVIRONMENT_LIFECYCLE,
            {"operation": "create", "specification": specification.model_dump(mode="json")},
            execute,
            completion=lambda identifier: {"environment_id": identifier},
        )
        self.owned.append(environment_id)
        return self.attach(environment_id)

    def attach(self, environment_id: str) -> Environment:
        """A handle to an existing environment."""
        return Environment(environment_id, self._service, self._effects)

    async def release_all(self) -> None:
        """Destroy every environment the run created, directly (not as effects); for runners when a run ends."""
        for environment_id in self.owned:
            await self._service.destroy(environment_id)
