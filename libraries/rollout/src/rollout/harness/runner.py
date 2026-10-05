"""Runners and run specifications (docs/libraries/rollout/README.md#runner)."""

from collections.abc import AsyncIterator, Mapping
from enum import StrEnum
from typing import Protocol, Self

from pydantic import Field, JsonValue, model_validator

from rollout.contracts import ContractModel, ModelEndpoint, RunEvent, RunFailureClass
from rollout.harness.agent import Agent
from rollout.harness.imports import ToolBinding
from rollout.harness.program import AgentProgram, Program
from rollout.harness.sandboxes import PoolBinding
from rollout.harness.task import Task
from rollout.names import named


class SamplingParameters(ContractModel):
    """Configured on bindings, never by task or agent code."""

    temperature: float = 1.0
    top_p: float = 1.0
    reasoning_effort: str | None = None
    """For providers with reasoning controls, e.g. `low`, `medium`, `high`."""
    thinking_tokens: int | None = None
    """For a recorded channel: tokens of thinking per turn, and of answer after it, in place of the channel's own (an
    eval's, say); none: the channel's."""
    answer_tokens: int | None = None


class DirectModel(ContractModel):
    """A model served by a provider's API through a direct adapter; nothing is recorded."""

    provider: str
    """The key of an endpoint factory registered with the runner, e.g. `codex`."""
    model: str
    sampling: SamplingParameters = SamplingParameters()


class RecordedModel(ContractModel):
    """A channel served through the gateway, which records every sample."""

    channel: str
    sampling: SamplingParameters = SamplingParameters()
    trained: bool = True
    """Whether its turns may be trained on: the slot's declaration (`ModelSlot.trained`). Every turn records it."""


class RecordedEndpoints(Protocol):
    """Serves recorded bindings, as runners see it: the gateway (`rollout_train.gateway.GatewayEndpoints`)."""

    def endpoint(self, binding: RecordedModel) -> ModelEndpoint: ...


class ModelBinding(ContractModel):
    """Exactly one of `direct` or `recorded`."""

    direct: DirectModel | None = None
    recorded: RecordedModel | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.direct is None) == (self.recorded is None):
            raise ValueError("a model binding is exactly one of `direct` or `recorded`")
        return self


class ProgramReference(ContractModel):
    """What a run executes, by name, so a runner in another process can re-create it."""

    program: str
    """`module:QualifiedName` of a `Program` class."""
    parameters: JsonValue = None


class RunBinding(ContractModel):
    models: Mapping[str, ModelBinding]
    """Model slot → how it is served."""
    imports: Mapping[str, ToolBinding] = Field(default_factory=dict[str, ToolBinding])
    """Import name → how the tool set is served."""
    pools: Mapping[str, PoolBinding] = Field(default_factory=dict[str, PoolBinding])
    """Sandbox kind → the pool its sandboxes are acquired from."""


class RunSpecification(ContractModel):
    program: ProgramReference
    binding: RunBinding


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
    @property
    def done(self) -> bool: ...
    @property
    def outcome(self) -> RunOutcome | None:
        """How the run ended; None while it is live."""
        ...

    async def result(self) -> RunOutcome:
        """Wait for the run to end."""
        ...

    def events(self, *, from_seq: int = 0) -> AsyncIterator[RunEvent]:
        """Every event from `from_seq`, then new ones as they are recorded, until the run ends."""
        ...

    def recorded_events(self) -> list[RunEvent]:
        """Every event recorded so far."""
        ...


class Runner(Protocol):
    async def launch(self) -> None:
        """Make the runner ready: call it once, before anything else that starts or reaches a run."""
        ...

    async def close(self) -> None:
        """Release what the runner holds; it cannot be used afterwards."""
        ...

    def run(self, run_id: str) -> RunHandle:
        """The handle of a run; `KeyError` if the runner does not know it."""
        ...

    async def start(
        self,
        specification: RunSpecification,
        *,
        run_id: str | None = None,
        labels: Mapping[str, str] | None = None,
        lease: str | None = None,
    ) -> RunHandle:
        """Start a run. Its sandboxes are acquired under `lease` and each one's name (by default the `run_id`): an
        episode's claim, say, so that they end with it."""
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
    value = named(name)
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


def bind(
    reference: ProgramReference,
    channel: str,
    *,
    slots: Mapping[str, str] | None = None,
    tools: Mapping[str, ToolBinding] | None = None,
    pools: Mapping[str, PoolBinding] | None = None,
) -> RunBinding:
    """A binding that serves each model slot of a program from the recorded channel `slots` names for it, else from
    `channel` (each recorded as trained or not, as the slot declares), each of its imports from the tool set registered
    under the import's own name (or as `tools` says), and each kind of sandbox it declares from the pool registered
    under the kind's name (or as `pools` says)."""
    program = instantiate(reference)
    models = {
        slot: ModelBinding(recorded=RecordedModel(channel=(slots or {}).get(slot, channel), trained=declared.trained))
        for slot, declared in program.model_slots().items()
    }
    imports = {name: (tools or {}).get(name) or ToolBinding(local=name) for name in program.imports()}
    kinds = {spec.kind for spec in program.sandboxes().values()}
    served = {kind: (pools or {}).get(kind) or PoolBinding(local=kind) for kind in sorted(kinds)}
    return RunBinding(models=models, imports=imports, pools=served)


def with_row(reference: ProgramReference, row: JsonValue) -> ProgramReference:
    """The same program for another row of parameters (for the task loop: the task's parameters)."""
    if reference.program == qualified_name(AgentProgram) and isinstance(reference.parameters, dict):
        return reference.model_copy(update={"parameters": {**reference.parameters, "task_parameters": row}})
    return reference.model_copy(update={"parameters": row})


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
