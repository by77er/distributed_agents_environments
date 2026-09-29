"""An agent session: a long-lived conversation with its own environment, able to create and message other sessions.

The session is a conversation run keyed by its name. It creates an environment when it starts and destroys it when it
is stopped. Messages from other sessions and from the operator arrive as observations, labelled with their sender.
"""

from rollout.core.contracts import Message, OutcomeUnknown, Role, Text, ToolSpecification
from rollout.core.harness import (
    Agent,
    Envelope,
    Environment,
    EnvironmentSpecification,
    History,
    Observation,
    RunContext,
    Task,
    WaitFor,
    tool,
)

OPERATOR = "operator"

SYSTEM_PROMPT = """You are "{name}", one of several independent agent sessions that work alongside each other. The \
time is {now}.

You have your own Linux computer (Alpine Linux; you are root; install software with `apk add`; it has internet \
access). Use the shell, read_file and write_file tools on it. Your files persist until your session is stopped.

You can coordinate with other sessions:
- list_sessions, create_session (starts a new session with its own computer) and send_message;
- a shared board of channels: post notes or tasks, read_board, claim_task before working on a task, resolve_task with \
the result, and subscribe to channels to be told about new posts.

Messages from other sessions arrive as "[message from NAME]" and board notices as "[board #CHANNEL]". The operator is \
the person managing all sessions; you can message them with send_message(to="operator"). When another session gives \
you work, report the result back to it when you are done. Keep your replies short."""


def _as_observation_message(envelope: Envelope) -> Message:
    text = "".join(block.text for block in envelope.content if isinstance(block, Text))
    if envelope.sender and envelope.sender != OPERATOR and not text.startswith("[board"):
        text = f"[message from {envelope.sender}] {text}"
    return Message(role=Role.USER, content=[Text(text=text)])


class AgentSession(Task):
    imports = ["sessions", "board"]

    def __init__(self, parameters: dict[str, str] | None = None) -> None:
        self.image = (parameters or {}).get("image", "alpine")
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

    # The computer

    @tool
    async def shell(self, run: RunContext, command: str, timeout_seconds: float = 120) -> str:
        """Run a shell command on your computer, in /workspace unless it changes directory. Each call starts a fresh
        shell: nothing keeps running between calls. Returns the exit code and the combined output."""
        try:
            result = await self._environment(run).execute(command, timeout_seconds=min(timeout_seconds, 1800))
        except OutcomeUnknown:
            return (
                "The command may or may not have run: the system restarted while it was running. "
                "Check its effects before running it again."
            )
        if result.timed_out:
            return f"timed out after {timeout_seconds} seconds\n{result.output}"
        return f"exit {result.exit_code}\n{result.output}"

    @tool
    async def read_file(self, run: RunContext, path: str) -> str:
        """Read a text file from your computer (relative paths are under /workspace)."""
        return (await self._environment(run).get(path)).decode("utf-8", errors="replace")

    @tool
    async def write_file(self, run: RunContext, path: str, content: str) -> str:
        """Write a text file on your computer (relative paths are under /workspace), creating directories."""
        await self._environment(run).put(path, content)
        return f"Wrote {len(content.encode())} bytes to {path}."

    def _environment(self, run: RunContext) -> Environment:
        if run.environments is None or self.environment_id is None:
            raise RuntimeError("this session has no environment")
        return run.environments.attach(self.environment_id)


class SessionAgent(Agent):
    async def act(self, run: RunContext, history: History, tools: list[ToolSpecification]) -> Message:
        name = run.conversation.key if run.conversation else "session"
        system = Message.system(SYSTEM_PROMPT.format(name=name, now=run.now().isoformat(timespec="minutes")))
        return await run.model.sample([system, *history.messages(run.context_hints)], tools=tools)
