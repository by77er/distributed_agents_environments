"""An agent session: a long-lived conversation with its own environment, able to create and message other sessions.

The session is a conversation run keyed by its name. It creates an environment when it starts and destroys it when it
is stopped. Messages from other sessions and from the operator arrive as observations, labelled with their sender.
"""

from rollout.contracts import Message, Role, Text, ToolSpecification
from rollout.harness import (
    Agent,
    Envelope,
    EnvironmentSpecification,
    History,
    Observation,
    RunContext,
    Task,
    WaitFor,
)
from rollout_computers.tools import ComputerTools

OPERATOR = "operator"

SYSTEM_PROMPT = """You are "{name}", one of several independent agent sessions that work alongside each other. The \
time is {now}.

{computer} Use the shell, read_file, write_file, edit_file and read_image tools on it.

You can coordinate with other sessions:
- list_sessions, create_session (starts a new session with its own workspace) and send_message;
- a shared board of channels: post notes or tasks, read_board, claim_task before working on a task, resolve_task with \
the result, and subscribe to channels to be told about new posts.

Messages from other sessions arrive as "[message from NAME]" and board notices as "[board #CHANNEL]". The operator is \
the person managing all sessions; you can message them with send_message(to="operator"). When another session gives \
you work, report the result back to it when you are done. Keep your replies short."""

COMPUTERS = {
    "alpine": "You have your own Linux computer (Alpine Linux; you are root; install software with `apk add`; it has "
    "internet access). Your workspace is /workspace; your files persist until your session is stopped.",
    "host": "You work in your own workspace directory on the operator's machine (its path is in $WORKSPACE): commands "
    "run as the operator's user, with the machine's programs and internet access. This is not a sandbox. Keep your "
    "files in your workspace, and do not install software system-wide or change anything outside your workspace "
    "unless you are asked to. Your workspace is deleted when your session is stopped.",
}
"""What a session is told about its computer, by the environment image it runs on."""


def _as_observation_message(envelope: Envelope) -> Message:
    text = "".join(block.text for block in envelope.content if isinstance(block, Text))
    if envelope.sender and envelope.sender != OPERATOR and not text.startswith("[board"):
        text = f"[message from {envelope.sender}] {text}"
    return Message(role=Role.USER, content=[Text(text=text)])


class AgentSession(ComputerTools, Task):
    """The computer tools come from `ComputerTools`; the coordination tools are imported."""

    imports = ["sessions", "board"]

    def __init__(self, parameters: dict[str, str] | None = None) -> None:
        self.image = (parameters or {}).get("image", "alpine")
        if self.image not in COMPUTERS:
            raise ValueError(f"no description of the image {self.image!r} for sessions")
        self.environment_id: str | None = None

    async def setup(self, run: RunContext) -> None:
        if run.environments is None:
            raise RuntimeError("agent sessions need an environment backend")
        environment = await run.environments.create(EnvironmentSpecification(image=self.image))
        self.environment_id = environment.environment_id

    async def start(self, run: RunContext) -> WaitFor:
        return WaitFor("message")

    async def resume(self, run: RunContext, envelope: Envelope) -> Observation:
        return Observation(_as_observation_message(envelope))

    async def steer(self, run: RunContext, envelopes: list[Envelope], observation: Observation) -> Observation:
        messages = observation.messages + tuple(_as_observation_message(envelope) for envelope in envelopes)
        return Observation(messages, reward=observation.reward, end=observation.end, info=observation.info)

    async def respond(self, run: RunContext, reply: Message) -> Observation | WaitFor:
        if reply.tool_calls:
            return await self.run_tools(run, reply)
        if reply.text:
            await run.emit("reply", reply.text)
        return WaitFor("message")

    async def teardown(self, run: RunContext) -> None:
        if run.environments is not None and self.environment_id is not None:
            await run.environments.attach(self.environment_id).destroy()


class SessionAgent(Agent):
    """Configured with {"image": ...}, the same image as the session's task."""

    def __init__(self, configuration: dict[str, str] | None = None) -> None:
        super().__init__(configuration)
        self.image = (configuration or {}).get("image", "alpine")

    async def act(self, run: RunContext, history: History, tools: list[ToolSpecification]) -> Message:
        name = run.conversation.key if run.conversation else "session"
        now = run.now().isoformat(timespec="minutes")
        system = Message.system(SYSTEM_PROMPT.format(name=name, now=now, computer=COMPUTERS[self.image]))
        return await run.model.sample([system, *history.messages(run.context_hints)], tools=tools)
