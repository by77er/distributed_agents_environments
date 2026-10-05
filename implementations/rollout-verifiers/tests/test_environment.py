"""A verifiers environment is one of ours: its tasks are a row's starts and its eval data, and an episode is one
verifiers episode whose harness reaches the model through the gateway, which records what it samples."""

import asyncio
import importlib
import random
import shutil
import sys
import types
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("verifiers")
import uvicorn

from rollout.environment import Description
from rollout.harness import ProgramReference, RecordedModel, instantiate
from rollout.harness.blobs import FileBlobStore
from rollout_train.check import checked
from rollout_train.gateway import GatewayEndpoints, create_app
from rollout_train.ledger import FileLedger
from rollout_train.testing import admitted, gateway_endpoints, plain_channel
from rollout_verifiers import VerifiersEnvironment, VerifiersProgram, play

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="the null harness runs as a uv script")
vf: Any = importlib.import_module("verifiers.v1")  # (it ships no type information)

PORT = 8809
WORDS = {"train": ["apple", "river"], "test": ["candle"]}


class EchoData(vf.TaskData):
    answer: str


class EchoTask(vf.Task[EchoData]):
    @vf.reward(weight=1.0)
    async def exact(self, trace: Any) -> float:
        return float(trace.last_reply == self.data.answer)


class EchoConfig(vf.TasksetConfig):
    split: str = "train"


class EchoTaskset(vf.Taskset[EchoTask, EchoConfig]):
    def load(self) -> list[EchoTask]:
        return [
            EchoTask(EchoData(prompt=f"Say {word}.", answer=word), self.config.task)
            for word in WORDS[self.config.split]
        ]


module = types.ModuleType("rollout_verifiers_echo")
module.__dict__.update(EchoTaskset=EchoTaskset, __all__=["EchoTaskset"])
sys.modules["rollout_verifiers_echo"] = module  # verifiers imports a taskset by its id


@pytest.fixture
async def gateway(tmp_path: Path) -> AsyncIterator[GatewayEndpoints]:
    """A gateway served over HTTP, whose policy says `apple` whatever it is asked, with run `r_1` admitted."""
    channel = plain_channel(always=[("apple\n", "stop")])
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    served = gateway_endpoints(channel, ledger=ledger, blobs=blobs, url=f"http://127.0.0.1:{PORT}")
    await admitted(served, "r_1")
    assert served.gateway is not None
    app = create_app(served.gateway)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    while not server.started:  # noqa: ASYNC110 - uvicorn says it has started by this flag alone
        await asyncio.sleep(0.02)
    yield served
    server.should_exit = True
    await serving


def environment() -> VerifiersEnvironment:
    return VerifiersEnvironment("rollout-verifiers-echo", train={"split": "train"}, eval={"split": "test"})


def test_the_training_tasks_are_the_starts_of_its_row_and_the_eval_tasks_its_eval_data() -> None:
    echo = environment()
    (row,) = echo.rows()
    assert (row.key, row.title) == ("train", "rollout-verifiers-echo (split train): 2 tasks")
    starts: list[Any] = [echo.start(row, random.Random(seed)) for seed in range(8)]
    assert {start["task"]["answer"] for start in starts} == {"apple", "river"}
    assert starts[3] == echo.start(row, random.Random(3))  # a seed draws the same task every time
    assert starts[0]["environment"]["taskset"] == {"id": "rollout-verifiers-echo", "split": "train"}

    ((name, held),) = echo.evals().items()  # the eval split's tasks, in order, on a row training never plays
    (start,) = held
    assert (name, start.task, start.title, start.seed) == ("rollout-verifiers-echo-test", "eval", "task 0", 0)
    parameters: Any = start.parameters
    assert parameters["task"]["answer"] == "candle" and parameters["environment"]["taskset"]["split"] == "test"
    assert VerifiersEnvironment("rollout-verifiers-echo", train={"split": "train"}).evals() == {}

    assert echo.description == Description(rewards=(0.0, 1.0), solved=True, duration="turns")
    assert echo.version.startswith("rollout_verifiers_echo, verifiers 0.3.2")
    assert echo.version != VerifiersEnvironment("rollout-verifiers-echo", train={"split": "test"}).version
    findings = checked(echo)
    assert all(finding.passed for finding in findings), [str(finding) for finding in findings]

    program = instantiate(ProgramReference(program=echo.program.program, parameters=starts[0]))
    assert isinstance(program, VerifiersProgram) and program.task["answer"] == starts[0]["task"]["answer"]


async def test_an_episode_is_played_by_the_harness_through_the_gateway_and_scored_by_the_task(
    gateway: GatewayEndpoints,
) -> None:
    train = environment()
    (row,) = train.rows()
    address = gateway.endpoint(RecordedModel(channel="policy")).address("r_1/policy")
    outcomes: list[tuple[str, float]] = []
    for seed in range(8):
        start: Any = train.start(row, random.Random(seed))
        if start["task"]["answer"] not in [answer for answer, _ in outcomes]:
            reward, info = await play(start["environment"], start["task"], address)
            outcomes.append((start["task"]["answer"], reward))
            assert info["rewards"] == {"exact": reward} and info["turns"] == 1
    assert sorted(outcomes) == [("apple", 1.0), ("river", 0.0)]
    segments: Sequence[Any] = (await gateway.sessions("train", "r_1"))["policy"]
    said = [
        "".join(chr(token) for token in segment.tokens[span.start : span.end])
        for segment in segments
        for span in segment.spans
    ]
    assert said == ["apple\n", "apple\n"]  # each request was sampled, and recorded, by the gateway
    prompts = ["".join(chr(token) for token in segment.tokens) for segment in segments]
    assert sorted(prompts) == ["user: Say apple.\nassistant: apple\n", "user: Say river.\nassistant: apple\n"]


async def test_an_episode_verifiers_could_not_play_fails(tmp_path: Path) -> None:
    train = environment()
    start: Any = train.start(train.rows()[0], random.Random(0))
    channel = plain_channel(always=[("apple\n", "stop")])
    ledger, blobs = FileLedger(tmp_path / "ledger"), FileBlobStore(tmp_path / "blobs")
    unserved = gateway_endpoints(channel, ledger=ledger, blobs=blobs, url="http://127.0.0.1:9")
    await admitted(unserved, "r_1")
    address = unserved.endpoint(RecordedModel(channel="policy")).address("r_1/policy")  # nothing listens there
    with pytest.raises(RuntimeError, match="the verifiers episode failed"):
        await play(start["environment"], start["task"], address)
