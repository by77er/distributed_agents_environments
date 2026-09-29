"""Runners, run specifications and deployments (docs/core/harness/README.md#runner)."""

import importlib
from collections.abc import AsyncIterator, Mapping
from enum import StrEnum
from typing import Any, Protocol

from pydantic import JsonValue

from rollout.core.contracts import ContractModel, FrozenSequence, RunEvent, RunFailureClass
from rollout.core.harness.agent import Agent
from rollout.core.harness.conversations import Address, ConversationKey, DeliveryPolicy, Envelope, Priority
from rollout.core.harness.program import AgentProgram, Program
from rollout.core.harness.task import Task


class SamplingParameters(ContractModel):
    """Configured on bindings, never by task or agent code."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    max_output_tokens: int | None = None
    stop: FrozenSequence[str] = ()
    seed: int | None = None
    reasoning_effort: str | None = None
    """For providers with reasoning controls, e.g. `low`, `medium`, `high`."""


class DirectModel(ContractModel):
    """A model served by a provider's API through a direct adapter; nothing is recorded."""

    provider: str
    """The key of an endpoint factory registered with the runner, e.g. `codex`."""
    model: str
    sampling: SamplingParameters = SamplingParameters()


class RecordedModel(ContractModel):
    """A channel served through the recorder (M1)."""

    channel: str
    sampling: SamplingParameters = SamplingParameters()


class ModelBinding(ContractModel):
    """Exactly one of `direct` or `recorded`."""

    direct: DirectModel | None = None
    recorded: RecordedModel | None = None


class ProgramReference(ContractModel):
    """What a run executes, by name, so a runner in another process can re-create it."""

    program: str
    """`module:QualifiedName` of a `Program` class."""
    parameters: JsonValue = None
    code_reference: str | None = None
    """`{package}@{content_hash}`; pins durable runs to the code they started with."""


class RunBinding(ContractModel):
    models: Mapping[str, ModelBinding]
    """Model slot → how it is served."""
    delivery: DeliveryPolicy = DeliveryPolicy()


class RunSpecification(ContractModel):
    program: ProgramReference
    binding: RunBinding


class Deployment(ContractModel):
    """A named, addressable agent: conversations addressed to it start runs of its specification."""

    name: str
    """`{namespace}/{name}`, e.g. `acme/support-bot`."""
    specification: RunSpecification


class RunStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunOutcome(ContractModel):
    status: RunStatus
    failure_class: RunFailureClass | None = None
    detail: str | None = None


class RunHandle(Protocol):
    @property
    def run_id(self) -> str: ...
    async def result(self) -> RunOutcome: ...
    def events(self, *, from_seq: int = 0) -> AsyncIterator[RunEvent]:
        """Every event from `from_seq`, then new ones as they are recorded, until the run ends."""
        ...


class Runner(Protocol):
    async def start(
        self,
        specification: RunSpecification,
        *,
        run_id: str | None = None,
        conversation: ConversationKey | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> RunHandle: ...

    async def send(
        self,
        to: Address,
        envelope: Envelope,
        *,
        priority: Priority = Priority.NORMAL,
        idempotency_key: str | None = None,
    ) -> None:
        """Deliver a message; a message to a conversation starts its run when none is live."""
        ...

    async def cancel(self, run_id: str, *, reason: str) -> None: ...


# Program references -----------------------------------------------------------------------------------------------

_LOCAL_CLASSES: dict[str, type] = {}


def qualified_name(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def register(cls: type) -> str:
    """Make a class resolvable by name in this process, even if it cannot be imported (e.g. defined in a script)."""
    name = qualified_name(cls)
    _LOCAL_CLASSES[name] = cls
    return name


def resolve(name: str) -> type:
    """The class a `module:QualifiedName` names: registered in this process, or imported."""
    if name in _LOCAL_CLASSES:
        return _LOCAL_CLASSES[name]
    module_name, _, qualified = name.partition(":")
    value: Any = importlib.import_module(module_name)
    for part in qualified.split("."):
        value = getattr(value, part)
    if not isinstance(value, type):
        raise TypeError(f"{name} is not a class")
    return value


def agent_program(
    task: type[Task],
    agent: type[Agent] = Agent,
    *,
    task_parameters: JsonValue = None,
    agent_configuration: JsonValue = None,
) -> ProgramReference:
    """A reference to the task loop for `task` and `agent`."""
    return ProgramReference(
        program=qualified_name(AgentProgram),
        parameters={
            "task": register(task),
            "agent": register(agent),
            "task_parameters": task_parameters,
            "agent_configuration": agent_configuration,
        },
    )


def instantiate(reference: ProgramReference) -> Program:
    """Create the program a reference names."""
    cls = resolve(reference.program)
    if cls is AgentProgram:
        parameters = reference.parameters
        if not isinstance(parameters, dict):
            raise TypeError("an AgentProgram reference needs parameters {task, agent, ...}")
        task_class, agent_class = resolve(str(parameters["task"])), resolve(str(parameters["agent"]))
        task_parameters, configuration = parameters.get("task_parameters"), parameters.get("agent_configuration")
        task_instance = task_class() if task_parameters is None else task_class(task_parameters)
        agent_instance = agent_class() if configuration is None else agent_class(configuration)
        return AgentProgram(task_instance, agent_instance)
    program = cls() if reference.parameters is None else cls(reference.parameters)
    if not isinstance(program, Program):
        raise TypeError(f"{reference.program} is not a Program")
    return program
