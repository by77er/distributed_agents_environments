"""Runners, run specifications and deployments (docs/libraries/rollout/README.md#runner)."""

import importlib
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from enum import StrEnum
from typing import Any, Protocol, Self

from pydantic import Field, JsonValue, model_validator

from rollout.contracts import ContractModel, ModelEndpoint, RunEvent, RunFailureClass, new_message_id
from rollout.harness.agent import Agent
from rollout.harness.conversations import (
    Address,
    ConversationKey,
    DeliveryMode,
    DeliveryPolicy,
    Envelope,
    Priority,
)
from rollout.harness.imports import ToolBinding
from rollout.harness.program import AgentProgram, Program
from rollout.harness.sandboxes import PoolBinding
from rollout.harness.task import Task


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
    """A channel served through the recorder."""

    channel: str
    sampling: SamplingParameters = SamplingParameters()


class RecordedEndpoints(Protocol):
    """Serves recorded bindings: the recorder (`rollout_train.recorder.Recorder`), as runners see it."""

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

    def deploy(self, deployment: Deployment) -> None:
        """Register or replace a deployment; a conversation's next run uses the current version."""
        ...

    def run(self, run_id: str) -> RunHandle:
        """The handle of a run; `KeyError` if the runner does not know it."""
        ...

    def conversation_of(self, run_id: str) -> ConversationKey | None:
        """The conversation a run serves, if any."""
        ...

    def conversation_runs(self, deployment: str, key: str) -> Sequence[RunHandle]:
        """The conversation's runs, oldest first."""
        ...

    async def start(
        self,
        specification: RunSpecification,
        *,
        run_id: str | None = None,
        conversation: ConversationKey | None = None,
        labels: Mapping[str, str] | None = None,
        lease: str | None = None,
    ) -> RunHandle:
        """Start a run. Its sandboxes are acquired under `lease` and each one's name (by default the `run_id`): an
        episode's claim, say, so that they end with it."""
        ...

    async def send(
        self,
        to: Address,
        envelope: Envelope,
        *,
        priority: Priority = Priority.NORMAL,
        idempotency_key: str | None = None,
        sender: str | None = None,
    ) -> str:
        """Deliver a message and return its `message_id`; a message to a conversation starts its run when none
        is live. A message sent again with the same `idempotency_key` is delivered once."""
        ...

    async def cancel(self, run_id: str, *, reason: str) -> None: ...


class RunNotLive(Exception):
    """A message was addressed to a run that has ended."""


class MessageRouter(ABC):
    """`Runner.send`, and the hand-over between a conversation's runs, over the transport a runner supplies: where
    its runs are, how a message reaches one, and where delivered messages are remembered.

    A message is claimed (remembered as delivered, under its `message_id`) only once it is delivered. A send that
    fails before that can be retried with the same idempotency key; a retry of a delivered message is dropped.
    """

    async def send(
        self,
        to: Address,
        envelope: Envelope,
        *,
        priority: Priority = Priority.NORMAL,
        idempotency_key: str | None = None,
        sender: str | None = None,
    ) -> str:
        if to.kind == "external":
            raise ValueError("a runner delivers to runs and conversations, not to external addresses")
        message_id = idempotency_key or new_message_id()
        envelope = envelope.model_copy(update={"message_id": message_id, "sender": sender})
        async with self._exclusive(to):
            if self._is_claimed(message_id):
                return message_id
            run_id = to.value if to.kind == "run" else await self._conversation_run(to.value, envelope.reply_to)
            policy = self._delivery_policy(run_id)
            if policy is None:
                raise RunNotLive(run_id)
            await self._deliver(run_id, envelope, policy.mode(priority, sender))
            self._claim(message_id, to)
        return message_id

    async def _hand_over(self, address: str, undelivered: Sequence[Envelope]) -> None:
        """Give the messages a conversation's finished run never consumed to its next run, starting it unless one is
        live. They are queued for it whatever the delivery policy says: a policy maps a sender's priority, which a
        hand-over does not have, and the run takes them when it waits, in the order they were sent."""
        if not undelivered:
            return
        async with self._exclusive(Address(kind="conversation", value=address)):
            run_id = await self._conversation_run(address, None)
            for envelope in undelivered:
                await self._deliver(run_id, envelope, DeliveryMode.QUEUE)

    # The transport

    @abstractmethod
    def _exclusive(self, to: Address) -> AbstractAsyncContextManager[None]:
        """Held while a message to `to` is checked, delivered and claimed, among everything that sends to it."""

    @abstractmethod
    def _is_claimed(self, message_id: str) -> bool: ...

    @abstractmethod
    def _claim(self, message_id: str, to: Address) -> None: ...

    @abstractmethod
    async def _conversation_run(self, address: str, reply_to: Address | None) -> str:
        """The `run_id` of the conversation's live run, started if none is live. `reply_to` is the origin of a
        conversation that starts with this message."""

    @abstractmethod
    def _delivery_policy(self, run_id: str) -> DeliveryPolicy | None:
        """The delivery policy of a live run; None if the run has ended or is unknown."""

    @abstractmethod
    async def _deliver(self, run_id: str, envelope: Envelope, mode: DeliveryMode) -> None: ...


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


def bind(
    reference: ProgramReference,
    channel: str,
    *,
    tools: Mapping[str, ToolBinding] | None = None,
    pools: Mapping[str, PoolBinding] | None = None,
) -> RunBinding:
    """A binding that serves every model slot of a program from one recorded channel, each of its imports from the
    tool set registered under the import's own name (or as `tools` says), and each kind of sandbox it declares from
    the pool registered under the kind's name (or as `pools` says)."""
    program = instantiate(reference)
    recorded = ModelBinding(recorded=RecordedModel(channel=channel))
    imports = {name: (tools or {}).get(name) or ToolBinding(local=name) for name in program.imports()}
    kinds = {spec.kind for spec in program.sandboxes().values()}
    served = {kind: (pools or {}).get(kind) or PoolBinding(local=kind) for kind in sorted(kinds)}
    return RunBinding(models=dict.fromkeys(program.model_slots(), recorded), imports=imports, pools=served)


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
