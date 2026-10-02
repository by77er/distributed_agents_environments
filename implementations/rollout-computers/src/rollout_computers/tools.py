"""Tools for a task whose agent works on a computer: a shell, reading, writing and editing files, and viewing images.

Mix `ComputerTools` into a `Task` and set `environment_id` (usually in `setup`, from `run.environments.create`):

    class MyTask(ComputerTools, Task): ...

The tools follow the conventions of common coding agents: relative paths are under the environment's working
directory; long files are read in pages; long command output keeps its end and saves the rest to a file; edits are
exact, unique text replacements. Every operation is an environment effect, so the tools work under any runner. Writes
and edits to the same file run one at a time, even when the model calls several tools at once.
"""

import asyncio
import io
import itertools
import posixpath

from pydantic import BaseModel, Field

from rollout.contracts import Media, OutcomeUnknown, Text, ToolResult
from rollout.harness.context import RunContext
from rollout.harness.environments import Environment
from rollout.harness.tools import error_result, tool

__all__ = ["ComputerTools", "Replacement", "apply_edits", "page_text", "prepare_image"]

MAX_READ_LINES = 2000
MAX_READ_BYTES = 50 * 1024
MAX_TIMEOUT_SECONDS = 1800
MAX_IMAGE_SIDE = 2000
"""Larger images are scaled down to fit within this many pixels on each side."""
MAX_IMAGE_BYTES = 4_500_000
IMAGE_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "GIF": "image/gif", "WEBP": "image/webp"}
"""Formats sent as they are; others are converted to PNG."""


class Replacement(BaseModel):
    old_text: str = Field(description="Exact text to replace. It must occur exactly once in the original file.")
    new_text: str = Field(description="The text to put in its place.")


class ComputerTools:
    """`@tool` methods over the environment `environment_id`. Mix into a `Task`."""

    environment_id: str | None = None

    @tool
    async def shell(self, run: RunContext, command: str, timeout_seconds: float = 120) -> ToolResult:
        """Run a shell command in your working directory. Each call starts a fresh shell, and nothing it started
        keeps running afterwards. Returns the exit code and the combined standard output and error. Long output keeps
        its last 2000 lines or 50 KB, and the full output is saved to a file you can read."""
        timeout = min(max(timeout_seconds, 1), MAX_TIMEOUT_SECONDS)
        try:
            result = await self.computer(run).execute(command, timeout_seconds=timeout)
        except OutcomeUnknown:
            return error_result(
                "The command may or may not have run: the system restarted while it was running. "
                "Check its effects before running it again."
            )
        notes: list[str] = []
        if result.timed_out:
            notes.append(f"timed out after {timeout:g} seconds; the command was killed")
        else:
            notes.append(f"exit {result.exit_code}")
        if result.truncated:
            where = f"; the full output is in {result.full_output_path}" if result.full_output_path else ""
            notes.append(f"[output truncated to its end{where}]")
        text = "\n".join(notes) + "\n" + result.output
        return ToolResult(content=[Text(text=text)], is_error=result.timed_out or result.exit_code != 0)

    @tool
    async def read_file(self, run: RunContext, path: str, offset: int = 1, limit: int | None = None) -> str:
        """Read a text file (relative paths are under your working directory). Returns at most 2000 lines or 50 KB
        from line `offset` (1-based), and says how to continue when there is more. `limit` caps the number of lines.
        Use read_image for images."""
        data = await self.computer(run).get(path)
        if b"\0" in data[:8192]:
            return f"{path} is a binary file ({len(data)} bytes). Use read_image for images, or the shell."
        return page_text(data.decode("utf-8", errors="replace"), path, offset, limit)

    @tool
    async def write_file(self, run: RunContext, path: str, content: str) -> str:
        """Write a text file (relative paths are under your working directory), replacing it if it exists and
        creating parent directories."""
        async with self._file_lock(path):
            await self.computer(run).put(path, content)
        return f"Wrote {len(content.encode())} bytes to {path}."

    @tool
    async def edit_file(self, run: RunContext, path: str, edits: list[Replacement]) -> str:
        """Edit a text file by exact replacement. Each old_text must occur exactly once in the original file, and
        edits must not overlap: every edit is matched against the original, not against the result of the others.
        For several changes in one file, make one call with several edits. Keep each old_text short but unique."""
        if not edits:
            raise ValueError("edits must contain at least one replacement")
        async with self._file_lock(path):
            environment = self.computer(run)
            original = (await environment.get(path)).decode("utf-8")
            edited = apply_edits(original, [(edit.old_text, edit.new_text) for edit in edits], path)
            await environment.put(path, edited)
        return f"Edited {path}: {len(edits)} replacement{'s' if len(edits) != 1 else ''}."

    @tool
    async def read_image(self, run: RunContext, path: str) -> ToolResult:
        """Look at an image file (PNG, JPEG, GIF, WebP, BMP and other common formats; relative paths are under your
        working directory). Large images are scaled down."""
        if run.blobs is None:
            raise RuntimeError("this runner has no blob store, so images cannot be shown to the model")
        data = await self.computer(run).get(path)
        image, media_type, description = await asyncio.to_thread(prepare_image, data)
        reference = await run.blobs.put(image, media_type)
        return ToolResult(content=[Text(text=f"{path}: {description}"), Media(media_type=media_type, source=reference)])

    def computer(self, run: RunContext) -> Environment:
        if run.environments is None or self.environment_id is None:
            raise RuntimeError("this task has no environment")
        return run.environments.attach(self.environment_id)

    def _file_lock(self, path: str) -> asyncio.Lock:
        locks: dict[str, asyncio.Lock] = self.__dict__.setdefault("_file_locks", {})
        key = posixpath.normpath(path if path.startswith(("/", "~")) else f"./{path}")
        return locks.setdefault(key, asyncio.Lock())


def apply_edits(original: str, edits: list[tuple[str, str]], path: str = "the file") -> str:
    """Apply exact replacements, each matched once against `original`. Line endings and a byte-order mark are kept:
    matching ignores the difference between CRLF and LF."""
    bom = "﻿" if original.startswith("﻿") else ""
    text = original.removeprefix(bom)
    crlf = "\r\n" in text
    text = text.replace("\r\n", "\n")
    spans: list[tuple[int, int, str]] = []
    for index, (old, new) in enumerate(edits):
        old = old.replace("\r\n", "\n")
        label = f"edits[{index}].old_text" if len(edits) > 1 else "old_text"
        if not old:
            raise ValueError(f"{label} is empty")
        count = text.count(old)
        if count == 0:
            raise ValueError(f"{label} was not found in {path}; it must match the file exactly, including whitespace")
        if count > 1:
            raise ValueError(f"{label} occurs {count} times in {path}; include more surrounding text to make it unique")
        start = text.index(old)
        spans.append((start, start + len(old), new.replace("\r\n", "\n")))
    spans.sort()
    for (_, end, _), (start, _, _) in itertools.pairwise(spans):
        if start < end:
            raise ValueError("two edits overlap; merge them into one")
    for start, end, new in reversed(spans):
        text = text[:start] + new + text[end:]
    if text == original.removeprefix(bom).replace("\r\n", "\n"):
        raise ValueError("the edits change nothing")
    return bom + (text.replace("\n", "\r\n") if crlf else text)


def prepare_image(data: bytes) -> tuple[bytes, str, str]:
    """An image the model can read: common formats within the limits as they are, others converted to PNG and
    scaled down to fit. Returns the bytes, their media type and a description."""
    from PIL import Image  # an optional dependency, needed only for images

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as error:
        raise ValueError(f"not an image this tool can read ({error})") from error
    width, height = image.size
    description = f"{image.format or 'image'}, {width}x{height}"
    fits = max(width, height) <= MAX_IMAGE_SIDE and len(data) <= MAX_IMAGE_BYTES
    if fits and image.format in IMAGE_FORMATS:
        return data, IMAGE_FORMATS[image.format], description
    if max(width, height) > MAX_IMAGE_SIDE:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        description += f", scaled to {image.size[0]}x{image.size[1]}"
    for kind, options in (("PNG", {"optimize": True}), ("JPEG", {"quality": 85}), ("JPEG", {"quality": 60})):
        converted = image.convert("RGB") if kind == "JPEG" else image
        buffer = io.BytesIO()
        converted.save(buffer, kind, **options)
        if buffer.tell() <= MAX_IMAGE_BYTES:
            return buffer.getvalue(), IMAGE_FORMATS[kind], description
    raise ValueError(f"the image is too large to show even after scaling ({description})")


def page_text(text: str, path: str, offset: int, limit: int | None) -> str:
    """Up to `MAX_READ_LINES` lines or `MAX_READ_BYTES` bytes of `text` from line `offset`, with a hint to continue."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # a final newline ends the last line; it does not start another
    total = len(lines)
    if total == 0:
        return f"{path} is empty."
    start = max(offset, 1) - 1
    if start >= total:
        raise ValueError(f"offset {offset} is beyond the end of {path} ({total} lines)")
    end = total if limit is None else min(start + max(limit, 1), total)
    shown: list[str] = []
    size = 0
    for line in lines[start:end]:
        line_size = len(line.encode()) + 1
        if len(shown) == MAX_READ_LINES or size + line_size > MAX_READ_BYTES:
            break
        shown.append(line)
        size += line_size
    if not shown:
        return (
            f"[line {start + 1} of {path} is longer than {MAX_READ_BYTES // 1024} KB; read part of it with the shell, "
            f"e.g. sed -n '{start + 1}p' {path} | head -c {MAX_READ_BYTES}]"
        )
    last = start + len(shown)
    body = "\n".join(shown)
    if last < total:
        body += f"\n\n[showing lines {start + 1}-{last} of {total}; use offset={last + 1} to continue]"
    return body
