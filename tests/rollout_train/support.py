"""Helpers the tests of rollout-train share: trainers that train nothing (`Counting`, `Steps`); a runner playing what
runs ask for in a ledger (`here`, `runner`, `episode_runner`) and asking it for groups (`ask`, `ask_boxed`); hooks that
take notes (`Notes`, `Running`, `Seen`); profiles (`PROFILE`, `write`, `profiles`, `a_profile`) and a ledger runs share
(`a_ledger`); a launched process (`Process`); and what a launcher offers (`OFFERED`)."""

import asyncio
import contextlib
import functools
from collections.abc import AsyncGenerator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import JsonValue

from rollout.harness import (
    ModelBinding,
    PoolBinding,
    Program,
    ProgramReference,
    RecordedModel,
    RunBinding,
    RunContext,
    Runner,
    SandboxPool,
    SandboxSpec,
    agent_program,
    register,
)
from rollout.harness.blobs import Blobs, FileBlobStore
from rollout.local import LocalRunner
from rollout_train import (
    Budget,
    Checkpoint,
    Checkpoints,
    FileLedger,
    Files,
    Ledger,
    Step,
    StepFailed,
    Weighted,
)
from rollout_train import loop as loop_module
from rollout_train.evals import (
    Schedule,
    Suite,
)
from rollout_train.gateway import GatewayEndpoints
from rollout_train.profile import Profile
from rollout_train.record import GROUPS, scope, table
from rollout_train.rollouts import (
    EpisodeRunner,
    Hooks,
    Plan,
    episodes_of,
    plan,
    playing,
)
from rollout_train.stores import FILES
from rollout_train.testing import Policy, plain_channel, recording
from rollout_train.trainer import STATE, WEIGHTS
from tests.rollout_train.rollouts.games import Guess


@pytest.fixture(autouse=True)
def quickly(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loop looks for its groups' episodes in the ledger often (a run looks twice a second)."""
    monkeypatch.setattr(loop_module, "episodes_of", functools.partial(episodes_of, every=0.01))


class Counting:
    """A trainer that trains nothing: it writes down what it was given and leaves files as a trainer would."""

    budget = Budget(segments=3)
    weights = "lora"

    def __init__(self, fails: int = 0) -> None:
        self.batches: list[list[Weighted]] = []
        self.parents: list[str | None] = []
        """What each step started from: the text of its parent's weights."""
        self.fails = fails
        """Steps that fail before one succeeds."""

    async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Files | None, into: Path) -> Step:
        if self.fails:
            self.fails -= 1
            raise StepFailed("the trainer failed:\nout of memory")
        self.batches.append(list(batch))
        self.parents.append((parent.weights / "adapter.bin").read_text() if parent else None)
        (into / WEIGHTS).mkdir(parents=True)
        (into / WEIGHTS / "adapter.bin").write_text(f"weights after {len(self.batches)} steps")
        (into / STATE).mkdir()
        (into / STATE / "optimizer.bin").write_text(f"moments after {len(self.batches)} steps")
        return Step({"segments": float(len(batch))})


class Notes(Hooks):
    def __init__(self) -> None:
        self.kinds: list[str] = []

    def on_note(self, event: Mapping[str, JsonValue]) -> None:
        self.kinds.append(str(event["kind"]))


async def made_by(checkpoints: Checkpoints, run: str = "train") -> list[Checkpoint]:
    """The checkpoints a run made, oldest first."""
    return sorted((v for v in await checkpoints.all() if v.run == run), key=lambda checkpoint: checkpoint.depth)


def answering() -> Policy:
    return Policy(plain_channel(always=[("yes\n", "stop"), ("no\n", "stop")]))


@contextlib.asynccontextmanager
async def here(
    ledger: Ledger,
    recorder: Policy,
    blobs: Blobs,
    *,
    runner: Runner | None = None,
    hooks: Sequence[Hooks] = (),
    places: int = 6,
    name: str = "here",
) -> AsyncGenerator[EpisodeRunner]:
    """A runner that plays what runs ask for in `ledger`, while the block runs, recording through a gateway in this
    process over `recorder`'s channels."""
    recorded = recorder.recording(ledger, blobs)
    played = runner if runner is not None else LocalRunner(recorder=recorded)
    episodes = EpisodeRunner(name, ledger, played, recorded, blobs, places, hooks=hooks, every=0.01)
    async with playing(episodes):
        yield episodes


class Running(Hooks):
    """Counts the episodes running, and the groups they are of, as the runner tells of them."""

    def __init__(self) -> None:
        self.running: dict[str, int] = {}
        """Each episode running: its group."""
        self.most = 0
        self.groups_at_once = 0

    def on_note(self, event: Mapping[str, JsonValue]) -> None:
        if event["kind"] == "started":
            self.running[str(event["run_id"])] = cast(int, event["group"])
        elif event["kind"] == "ended":
            self.running.pop(str(event["run_id"]), None)
        self.most = max(self.most, len(self.running))
        self.groups_at_once = max(self.groups_at_once, len(set(self.running.values())))


PROFILE = """
directory = "{directory}"

[channels.policy]
model = "a-checkpoint"
renderer = "rollout_train.testing:plain_renderer"
engine = "rollout_train.testing:scripted_engine"
thinking_tokens = 64
engines = [{{ device = 0 }}, {{ device = 1 }}]

[channels.judge]
model = "another-checkpoint"
renderer = "rollout_train.testing:plain_renderer"
engine = "rollout_train.testing:scripted_engine"

[trainer]
kind = "tests.rollout_train.support:Steps"
channel = "policy"
colocated = true
bookmark = "best"
segment_tokens = 900
segments_per_step = 3
"""


class Steps:
    """A trainer that trains nothing: what a profile's `[trainer]` names."""

    def __init__(self, model: str, *, segment_tokens: int, segments_per_step: int) -> None:
        self.model = model
        self.budget = Budget(segment_tokens, segments_per_step)
        self.weights = "lora"

    async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Files | None, into: Path) -> Step:
        (into / WEIGHTS).mkdir(parents=True)
        (into / WEIGHTS / "adapter.bin").write_text(f"trained on {len(batch)} segments")
        return Step({"segments": float(len(batch))})


def write(tmp_path: Path, text: str = PROFILE) -> Path:
    path = tmp_path / "profile.toml"
    path.write_text(text.format(directory=tmp_path / "run"))
    return path


def profiles(tmp_path: Path) -> Path:
    directory = tmp_path / "profiles"
    directory.mkdir()
    (directory / "small.toml").write_text(PROFILE.format(directory=tmp_path / "run"))
    (directory / "not-a-profile.toml").write_text("nonsense = [")
    without = PROFILE.format(directory=tmp_path / "run").split("[trainer]")[0]
    (directory / "serving-only.toml").write_text(without)
    return directory


class Process:
    """A started `rollout train`, as the launcher sees it: it ends with `code` once told to, or on an interrupt."""

    def __init__(self, code: int) -> None:
        self.pid, self.code, self.signals = 4242, code, list[int]()
        self.done = asyncio.Event()

    def send_signal(self, number: int) -> None:
        self.signals.append(number)
        self.done.set()

    async def wait(self) -> int:
        await self.done.wait()
        return self.code


BOX = SandboxSpec(kind="fake")


BOX_GATES: dict[str, asyncio.Event] = {}


class Boxed(Program):
    """Waits at its gate, if it has one; then asks its box who it is, and says."""

    def __init__(self, parameters: Mapping[str, JsonValue]) -> None:
        self.gate = parameters.get("gate")

    def sandboxes(self) -> Mapping[str, SandboxSpec]:
        return {"box": BOX}

    async def main(self, run: RunContext) -> None:
        if self.gate:
            await BOX_GATES.setdefault(str(self.gate), asyncio.Event()).wait()
        described: Any = (await run.sandbox("box").call("describe")).structured
        run.reward(1.0)
        await run.emit("result", {"solved": True, "handle": described["handle"], "key": run.sandbox("box").lease.key})


BOXED_BINDING = RunBinding(
    models={"policy": ModelBinding(recorded=RecordedModel(channel="policy"))},
    pools={"fake": PoolBinding(local="boxes")},
)


async def ask_boxed(
    ledger: Ledger, groups: Mapping[int, tuple[JsonValue, int]], program: type[Program] = Boxed
) -> None:
    fence = await ledger.take(scope("train"))
    await plan(ledger, "train", Plan(ProgramReference(program=register(program)), BOXED_BINDING), fence)
    for number, (row, count) in groups.items():
        record: JsonValue = {"parameters": row, "episodes": count, "decided": float(number)}
        await ledger.append(table("train", GROUPS), str(number), record, fence)


def episode_runner(tmp_path: Path, pool: SandboxPool, url: str | None = None, **options: Any) -> EpisodeRunner:
    """An episode runner over the ledger and blobs in `tmp_path`, recording through a gateway in this process (served
    to harnesses at `url`, if given)."""
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    recorder = recording(plain_channel(always=[("yes\n", "stop")]), ledger=ledger, blobs=blobs, url=url)
    return EpisodeRunner(
        "here",
        ledger,
        LocalRunner(recorder=recorder, pools={"boxes": pool}),
        recorder,
        blobs,
        places=4,
        pools={"boxes": pool},
        every=0.02,
        **options,
    )


BINDING = RunBinding(models={"policy": ModelBinding(recorded=RecordedModel(channel="policy"))})


class Seen:
    def __init__(self) -> None:
        self.notes: list[Mapping[str, JsonValue]] = []

    def on_note(self, event: Mapping[str, JsonValue]) -> None:
        self.notes.append(event)


def runner(
    tmp_path: Path, *says: str, name: str = "here", places: int = 8, **options: Any
) -> tuple[EpisodeRunner, GatewayEndpoints, Seen]:
    """A runner over the ledger and blob store in `tmp_path`, whose policy says `says` in turn, for ever."""
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    channel = plain_channel(always=[(f"{word}\n", "stop") for word in says])
    recorder = recording(channel, ledger=ledger, blobs=blobs)
    seen = Seen()
    played = EpisodeRunner(
        name,
        ledger,
        LocalRunner(recorder=recorder),
        recorder,
        blobs,
        places,
        hooks=[seen],
        every=0.02,
        **options,
    )
    return played, recorder, seen


async def ask(ledger: Ledger, run: str, groups: Mapping[int, tuple[JsonValue, int]], task: type = Guess) -> None:
    """What a run's loop writes: how its episodes are played, and each group with how many episodes it wants."""
    fence = await ledger.take(scope(run))
    await plan(ledger, run, Plan(agent_program(task), BINDING), fence)
    for number, (row, count) in groups.items():
        record: JsonValue = {"parameters": row, "episodes": count, "decided": float(number)}
        await ledger.append(table(run, GROUPS), str(number), record, fence)


@contextlib.asynccontextmanager
async def served(*runners: EpisodeRunner) -> AsyncGenerator[None]:
    async with contextlib.AsyncExitStack() as stack:
        for each in runners:
            await stack.enter_async_context(playing(each))
        yield


ENVIRONMENT = "tests.rollout_train.rollouts.games:words"


def a_schedule(suite: Suite, every: int) -> Schedule:
    """Evals of `suite` every `every` steps, two episodes of each start, each eval the run `eval-STEP`."""

    async def run(step: int, part: int | None = None) -> str:
        return f"eval-{step}" if part is None else f"eval-{step}-{part}"

    return Schedule(suite, run, every=every, episodes=2)


def files(tmp_path: Path, name: str) -> Path:
    directory = tmp_path / "made" / name
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(name.encode())
    return directory


async def a_ledger(tmp_path: Path) -> tuple[str, dict[str, str]]:
    """A ledger and blob store runs share (as a profile says them), holding a merged checkpoint ("merged"), an
    adapter over the model ("plain") and one over the merged weights ("stacked"), each bookmarked by that name."""
    shared = f'ledger = "{tmp_path / "ledger"}"\nblobs = {{ kind = "{FILES}", directory = "{tmp_path / "blobs"}" }}\n'
    (tmp_path / "setup.toml").write_text(shared + PROFILE.format(directory=tmp_path / "setup"))
    async with Profile.load(tmp_path / "setup.toml").open() as setup:
        assert setup.registry is not None
        fence = await setup.ledger.take(scope("elsewhere"))
        add = setup.checkpoints.add
        plain = await add(fence, "pppp" * 4, weights=files(tmp_path, "p"), run="elsewhere", base="a-checkpoint")
        merged = await add(fence, "mmmm" * 4, weights=files(tmp_path, "m"), run=None, kind="full", parents=[plain.id])
        stacked = await add(fence, "ssss" * 4, weights=files(tmp_path, "s"), run="elsewhere", parents=[merged.id])
        made = {"plain": plain.id, "merged": merged.id, "stacked": stacked.id}
        for name, id in made.items():
            await setup.registry.bookmark(name, id)
    return shared, made


def a_profile(tmp_path: Path, shared: str, trainer: str, start: str) -> Path:
    text = PROFILE.replace("tests.rollout_train.support:Steps", f"tests.rollout_train.test_full_weights:{trainer}")
    path = tmp_path / f"{trainer}-{start}.toml"
    directory = tmp_path / f"{trainer}-{start}"
    path.write_text(shared + text.replace('bookmark = "best"', f'start = "{start}"').format(directory=directory))
    return path


OFFERED: dict[str, Any] = {
    "profile": "one-gpu",
    "path": "/profiles/one-gpu.toml",
    "model": "m",
    "settings": {"trainer.learning_rate": 5e-5, "episodes_at_once": 6, "trainer.start": None},
}
