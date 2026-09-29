"""The computer tools: paging, exact edits, queued writes, truncated output and images, end to end on a local runner."""

import io
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from rollout.core.contracts import Media, Message, Text, ToolCall, ToolResult, ToolResultBlock
from rollout.core.harness import (
    DirectModel,
    EnvironmentSpecification,
    FileBlobStore,
    ModelBinding,
    Observation,
    RunBinding,
    RunContext,
    RunSpecification,
    RunStatus,
    Task,
    agent_program,
)
from rollout.core.local import LocalRunner
from rollout.core.testing import ScriptedModelEndpoint, tool_call_reply
from rollout.durable import DurableRunner
from rollout.environments import LocalEnvironments
from rollout.environments.tools import ComputerTools, apply_edits, page_text, prepare_image


def png(width: int, height: int, image_format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "blue").save(buffer, image_format)
    return buffer.getvalue()


# Edits


def test_edits_are_matched_against_the_original_and_applied_together() -> None:
    original = "alpha\nbeta\ngamma\n"
    assert apply_edits(original, [("alpha", "ALPHA"), ("gamma", "beta")]) == "ALPHA\nbeta\nbeta\n"


@pytest.mark.parametrize(
    ("edits", "error"),
    [
        ([("delta", "x")], "was not found"),
        ([("a", "x")], "occurs 5 times"),
        ([("alpha\nbe", "x"), ("beta", "y")], "overlap"),
        ([("beta", "beta")], "change nothing"),
        ([("", "x")], "is empty"),
    ],
)
def test_ambiguous_or_missing_edits_are_refused(edits: list[tuple[str, str]], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        apply_edits("alpha\nbeta\ngamma\n", edits)


def test_edits_keep_windows_line_endings_and_a_byte_order_mark() -> None:
    original = "﻿first\r\nsecond\r\n"
    assert apply_edits(original, [("first\nsecond", "one\ntwo")]) == "﻿one\r\ntwo\r\n"


# Reading


def test_reading_pages_through_long_files() -> None:
    text = "".join(f"line {n}\n" for n in range(1, 2501))
    first = page_text(text, "log.txt", 1, None)
    assert first.splitlines()[-1] == "[showing lines 1-2000 of 2500; use offset=2001 to continue]"
    assert page_text(text, "log.txt", 2001, None).splitlines()[0] == "line 2001"
    assert (
        page_text(text, "log.txt", 10, 2)
        == "line 10\nline 11\n\n[showing lines 10-11 of 2500; use offset=12 to continue]"
    )
    assert page_text("x" * 60_000, "wide.txt", 1, None).startswith("[line 1 of wide.txt is longer than 50 KB")
    with pytest.raises(ValueError, match="beyond the end"):
        page_text(text, "log.txt", 2501, None)


# Images


def test_images_within_the_limits_are_sent_as_they_are() -> None:
    data = png(64, 32)
    assert prepare_image(data) == (data, "image/png", "PNG, 64x32")


def test_large_or_unusual_images_are_scaled_and_converted() -> None:
    image, media_type, description = prepare_image(png(4000, 1000, "BMP"))
    assert media_type == "image/png"
    assert description == "BMP, 4000x1000, scaled to 2000x500"
    assert Image.open(io.BytesIO(image)).size == (2000, 500)
    with pytest.raises(ValueError, match="not an image"):
        prepare_image(b"plain text")


# End to end


class Workbench(ComputerTools, Task):
    async def setup(self, run: RunContext) -> None:
        assert run.environments is not None
        environment = await run.environments.create(EnvironmentSpecification(image="host"))
        self.environment_id = environment.environment_id
        await environment.put("picture.png", png(3000, 3000))

    async def start(self, run: RunContext) -> Observation:
        return Observation(Message.user("Work on your computer."))


def call(tool: str, /, **arguments: object) -> ToolCall:
    return ToolCall(call_id=f"c-{tool}-{len(arguments)}-{sorted(arguments)}", name=tool, arguments=arguments)  # type: ignore[arg-type]


SCRIPT = [
    tool_call_reply(call("write_file", path="notes.txt", content="one\ntwo\nthree\n")),
    tool_call_reply(  # two edits of one file in one turn: they must not overwrite each other
        ToolCall(
            call_id="e1",
            name="edit_file",
            arguments={"path": "notes.txt", "edits": [{"old_text": "one", "new_text": "1"}]},
        ),
        ToolCall(
            call_id="e2",
            name="edit_file",
            arguments={"path": "notes.txt", "edits": [{"old_text": "three", "new_text": "3"}]},
        ),
    ),
    tool_call_reply(call("read_file", path="notes.txt")),
    tool_call_reply(call("shell", command="seq 1 3000; exit 2")),
    tool_call_reply(call("read_image", path="picture.png")),
    Message.assistant("done"),
]


def results(endpoint: ScriptedModelEndpoint) -> dict[str, ToolResult]:
    return {
        block.call_id: block.result
        for request in endpoint.requests
        for message in request.context.append
        for block in message.content
        if isinstance(block, ToolResultBlock)
    }


def text(result: ToolResult) -> str:
    return "".join(part.text for part in result.content if isinstance(part, Text))


@pytest.mark.parametrize("durable", [False, True], ids=["local", "durable"])
async def test_an_agent_works_on_a_computer_with_the_tools(tmp_path: Path, durable: bool) -> None:
    endpoint = ScriptedModelEndpoint(SCRIPT)
    blobs = FileBlobStore(tmp_path / "blobs")

    def factory(model: DirectModel) -> ScriptedModelEndpoint:
        return endpoint

    services: dict[str, Any] = {
        "providers": {"scripted": factory},
        "environments": LocalEnvironments(tmp_path / "environments"),
        "blobs": blobs,
    }
    runner = DurableRunner(tmp_path / "runs", **services) if durable else LocalRunner(**services)
    if isinstance(runner, DurableRunner):
        await runner.launch()
    try:
        binding = RunBinding(models={"policy": ModelBinding(direct=DirectModel(provider="scripted", model="s"))})
        handle = await runner.start(RunSpecification(program=agent_program(Workbench), binding=binding))
        assert (await handle.result()).status is RunStatus.COMPLETED
    finally:
        if isinstance(runner, DurableRunner):
            await runner.close()

    found = results(endpoint)
    assert text(found["e1"]) == text(found["e2"]) == "Edited notes.txt: 1 replacement."
    (read,) = [r for key, r in found.items() if key.startswith("c-read_file")]
    assert text(read) == "1\ntwo\n3"  # both edits landed

    (shell,) = [r for key, r in found.items() if key.startswith("c-shell")]
    first, second, *output = text(shell).splitlines()
    assert shell.is_error and first == "exit 2"
    assert second.startswith("[output truncated to its end; the full output is in ") and second.endswith(".log]")
    assert (len(output), output[0], output[-1]) == (2000, "1001", "3000")

    (image,) = [r for key, r in found.items() if key.startswith("c-read_image")]
    assert text(image) == "picture.png: PNG, 3000x3000, scaled to 2000x2000"
    (media,) = [part for part in image.content if isinstance(part, Media)]
    assert Image.open(io.BytesIO(await blobs.read(media.source))).size == (2000, 2000)
