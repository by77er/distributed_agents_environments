"""An environment that brings its own agent loop, and one whose tools live on another machine: both are episodes like
any other to whoever trains on them."""

from collections.abc import Awaitable, Callable, Mapping, Sequence

import httpx
import pytest
from pydantic import JsonValue

from rollout.contracts import ModelAddress, Text, ToolResult, ToolSpecification
from rollout.harness import (
    ModelSample,
    Program,
    ProgramReference,
    RunBinding,
    RunContext,
    RunHooks,
    ToolBinding,
    bind,
    register,
)
from rollout.harness.remote import RemoteToolSet, serve
from rollout.local import LocalRunner
from rollout_train.recorder import Recorder
from rollout_train.rollouts import Outcome, RolloutJobs
from rollout_train.testing import plain_channel

pytest.importorskip("starlette")
from rollout_train.recorder.compat import create_app

HARNESS: dict[str, Callable[[ModelAddress], Awaitable[str]]] = {}
"""The harness a test stands in for: given where the model is, it plays and returns what was said at the end."""


class Outsourced(Program):
    """A program with no loop of its own: a harness inside its environment plays, given only a base URL and a key;
    the program scores what comes back."""

    def __init__(self, parameters: Mapping[str, JsonValue]) -> None:
        self.word = str(parameters["word"])

    async def main(self, run: RunContext) -> None:
        said = await HARNESS["play"](run.model.address())
        run.reward(1.0 if said == self.word else 0.0)
        await run.emit("result", {"solved": said == self.word})


class Seen(RunHooks):
    def __init__(self) -> None:
        self.samples: list[ModelSample] = []

    def on_sample(self, sample: ModelSample) -> None:
        self.samples.append(sample)


async def test_a_harness_given_only_an_address_plays_an_episode_that_is_recorded_and_watched() -> None:
    recorder = Recorder({"policy": plain_channel(always=[("yes\n", "stop")])}, base_url="http://recorder/v1")
    app = create_app(recorder)

    async def play(address: ModelAddress) -> str:
        headers = {"Authorization": f"Bearer {address.api_key}"}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=address.base_url, headers=headers) as client:
            body = {"model": address.model, "messages": [{"role": "user", "content": "Say the word."}]}
            reply = (await client.post("/chat/completions", json=body)).json()
        return str(reply["choices"][0]["message"]["content"])

    HARNESS["play"] = play
    seen = Seen()
    program = ProgramReference(program=register(Outsourced))
    jobs = RolloutJobs(LocalRunner(recorder=recorder, hooks=[seen]), recorder)
    job = await jobs.start(program=program, binding=binding_of(program), in_flight=4)
    won, lost = await (await job.run({"word": "yes"})).episodes(), await (await job.run({"word": "no"})).episodes()
    assert (won[0].reward, lost[0].reward) == (1.0, 0.0) and won[0].info == {"solved": True}
    (segment,) = won[0].trajectories["policy"].segments  # what the harness sampled is the slot's trajectory
    assert "".join(chr(token) for token in segment.tokens) == "user: Say the word.\nassistant: yes\n"
    assert len(seen.samples) == 2 and seen.samples[0].run_id == won[0].run_id and seen.samples[0].slot == "policy"
    await jobs.close()

    unserved = Recorder({"policy": plain_channel(always=[("yes\n", "stop")])})  # no base URL: nothing to hand out
    alone = RolloutJobs(LocalRunner(recorder=unserved), unserved)
    job = await alone.start(program=program, binding=binding_of(program), in_flight=1)
    (failed,) = await (await job.run({"word": "yes"})).episodes()
    assert failed.outcome is Outcome.FAILED and "not served over HTTP" in str(failed.detail)
    await alone.close()


def binding_of(program: ProgramReference) -> RunBinding:
    return bind(program.model_copy(update={"parameters": {"word": "x"}}), "policy")


class Dice:
    """A tool set an environment keeps on a machine of its own."""

    def __init__(self) -> None:
        self.effects: list[str] = []

    def specifications(self) -> Sequence[ToolSpecification]:
        return [ToolSpecification(name="roll", description="Roll a die.")]

    async def call(
        self, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        if arguments.get("sides") == 0:
            raise ValueError("a die has sides")
        self.effects.append(effect_id)
        return ToolResult(content=[Text(text="4")], structured={"rolled": 4})


class Roller(Program):
    def imports(self) -> list[str]:
        return ["dice"]

    async def main(self, run: RunContext) -> None:
        rolled = await run.tools.call("roll", {"sides": 6})
        await run.emit("result", {"rolled": rolled.structured})


async def test_a_tool_set_served_over_http_is_imported_by_its_address() -> None:
    dice = Dice()
    transport = httpx.ASGITransport(app=serve(dice))
    client = httpx.AsyncClient(transport=transport, base_url="http://dice")
    remote = RemoteToolSet("http://dice", client=client, specifications=dice.specifications())
    assert [tool.name for tool in remote.specifications()] == ["roll"]
    result = await remote.call("roll", {"sides": 6}, effect_id="r_1:0:0", arguments_digest="d")
    assert result.structured == {"rolled": 4} and dice.effects == ["r_1:0:0"]  # the effect's id goes with the call
    with pytest.raises(RuntimeError, match="ValueError: a die has sides"):  # a failure there is a failure here
        await remote.call("roll", {"sides": 0}, effect_id="r_1:0:1", arguments_digest="d")
    assert (await client.get("/specifications")).json()["specifications"][0]["name"] == "roll"

    # A binding names it by address; the program calls `run.tools` as it would a tool set in its own process.
    binding = bind(ProgramReference(program=register(Roller)), "policy", tools={"dice": ToolBinding(url="http://dice")})
    assert binding.imports["dice"].url == "http://dice" and binding.imports["dice"].local is None
