"""Imported tools: tool sets outside task code, bound per run (docs/guide/tools.md#imported-tools).

Calling an imported tool is a `tool.call` effect. Its `effect_id` and arguments digest reach the tool set, so a tool
set that deduplicates performs each call at most once, however often it is asked.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol, Self

from pydantic import JsonValue, model_validator

from rollout.contracts import ContractModel, EffectKind, ToolResult, ToolSpecification
from rollout.harness.model import Effects


class ToolSet(Protocol):
    """A provider of tools: in process, or a client of an MCP server, an HTTP service or another agent."""

    def specifications(self) -> Sequence[ToolSpecification]: ...

    async def call(
        self, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        """Perform one call. Tool-level errors are results with `is_error`; exceptions are platform failures."""
        ...


class DeduplicatingToolSet(ToolSet, Protocol):
    """A tool set that says whether it performs each `effect_id` at most once (a `deduplicates = True` attribute, on
    a class): its side-effecting tools are then safe to call again (docs/libraries/rollout/contracts/effects.md)."""

    @property
    def deduplicates(self) -> bool: ...


def deduplicates(receiver: object) -> bool:
    """Whether a tool set, a sandbox pool or a sandbox provider performs each `effect_id` at most once: what its
    `deduplicates` says, or False if it says nothing."""
    return bool(getattr(receiver, "deduplicates", False))


class ToolBinding(ContractModel):
    """How an import is served. Exactly one kind is set."""

    local: str | None = None
    """The name of a tool set registered with the runner, in process."""
    url: str | None = None
    """A tool set served over HTTP (`rollout.harness.remote`): an environment's own infrastructure, wherever it
    runs."""

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.local is None) == (self.url is None):
            raise ValueError("a tool binding is exactly one of `local` or `url`")
        return self


class Tools:
    """The imported tools of a run (`run.tools`)."""

    def __init__(self, tool_sets: Mapping[str, ToolSet], effects: Effects) -> None:
        self._effects = effects
        self._owners: dict[str, ToolSet] = {}
        self._specifications: list[ToolSpecification] = []
        for import_name, tool_set in tool_sets.items():
            for specification in tool_set.specifications():
                if specification.name in self._owners:
                    raise ValueError(f"tool {specification.name!r} of import {import_name!r} is already imported")
                self._owners[specification.name] = tool_set
                self._specifications.append(specification)

    def __contains__(self, name: object) -> bool:
        return name in self._owners

    def specifications(self) -> list[ToolSpecification]:
        return list(self._specifications)

    async def call(self, name: str, arguments: Mapping[str, JsonValue]) -> ToolResult:
        """Call an imported tool as a `tool.call` effect."""
        tool_set = self._owners[name]

        async def execute(effect_id: str, arguments_digest: str) -> ToolResult:
            return await tool_set.call(name, arguments, effect_id=effect_id, arguments_digest=arguments_digest)

        return await self._effects.perform(
            EffectKind.TOOL_CALL,
            {"tool": name, "arguments": dict(arguments)},
            execute,
            completion=lambda result: result.model_dump(mode="json", exclude_none=True),
        )
