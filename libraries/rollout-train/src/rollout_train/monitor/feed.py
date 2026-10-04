"""The feed: what a monitor shows, written to disk by a runner's hooks as runs happen.

`RunFeed` is a `RunHooks`: it appends one JSON line per run event and per model sample to `<directory>/<run_id>.jsonl`.
A sample's line holds what the model was sent (every message, in a plain form), the tools it was offered, and its
reply with its reasoning. `FeedReader` turns a directory of such files into what the monitor's page asks for.
"""

import json
import os
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import IO, Any, cast

from pydantic import JsonValue

from rollout.contracts import Message, Reasoning, RunEvent, RunEventType, Text, ToolCall, ToolResultBlock
from rollout.harness.hooks import ModelSample, RunHooks
from rollout_train.rollouts.episodes import DEFAULT
from rollout_train.rollouts.episodes import rewards as assigned
from rollout_train.rollouts.scheduler import Hooks

ENDED = (RunEventType.RUN_COMPLETED, RunEventType.RUN_FAILED, RunEventType.RUN_CANCELLED)


def plain(message: Message) -> dict[str, JsonValue]:
    """A message as the page shows it: its text, its reasoning, the tools it called and the results it carries."""
    reasoning = "".join(block.text or "" for block in message.content if isinstance(block, Reasoning))
    calls: list[JsonValue] = [
        {"id": block.call_id, "name": block.name, "arguments": dict(block.arguments)}
        for block in message.content
        if isinstance(block, ToolCall)
    ]
    results: list[JsonValue] = [
        {
            "id": block.call_id,
            "text": "".join(part.text for part in block.result.content if isinstance(part, Text)),
            "error": block.result.is_error,
        }
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    return {
        "role": message.role.value,
        "text": message.text,
        "reasoning": reasoning,
        "calls": calls,
        "results": results,
    }


NOTES = "_notes"
"""The feed file of notes beside the runs': episodes as runners start and end them, published weights, and the loop's
results and steps."""


class RunFeed(RunHooks, Hooks):
    """Writes every run's events and samples under `directory`, one file per run, as they happen; and what a
    runner and the loop note, at the level they think at, in one file more.

    `keep` bounds the directory: when more runs than that have files, the oldest are deleted. A directory has one
    writer at a time: runs that an earlier writer left without an end (its process was stopped) are marked cancelled
    when the next one starts, so that a monitor does not show them running for ever.
    """

    def __init__(self, directory: Path, *, keep: int = 200) -> None:
        self.directory = directory
        self.keep = keep
        directory.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, IO[str]] = {}
        for path in directory.glob("*.jsonl"):
            if path.stem != NOTES and not _ended(path):
                written = path.stat()
                line = {"kind": "event", "seq": -1, "type": RunEventType.RUN_CANCELLED.value, "at": written.st_mtime}
                with path.open("a") as file:
                    file.write("\n" + json.dumps({**line, "payload": {"detail": "its writer stopped"}}) + "\n")
                os.utime(path, (written.st_atime, written.st_mtime))  # it is as old as its run, for pruning

    def on_event(self, event: RunEvent) -> None:
        line: dict[str, JsonValue] = {
            "kind": "event",
            "seq": event.seq,
            "type": event.type.value,
            "at": event.recorded_at.timestamp(),
            "payload": event.payload,
        }
        self._write(event.run_id, line)
        if event.type in ENDED:
            self._close(event.run_id)

    def on_note(self, event: Mapping[str, JsonValue]) -> None:
        self._write(NOTES, event)

    def on_sample(self, sample: ModelSample) -> None:
        line: dict[str, JsonValue] = {
            "kind": "sample",
            "slot": sample.slot,
            "effect_id": sample.request.effect_id,
            "at": time.time(),
            "seconds": round(sample.seconds, 2),
            "messages": [plain(message) for message in sample.request.context.append],
            "tools": [tool.name for tool in sample.request.tools],
            "reply": plain(sample.result.message),
            "finish_reason": sample.result.finish_reason.value,
        }
        self._write(sample.run_id, line)

    def close(self) -> None:
        for run_id in list(self._files):
            self._close(run_id)

    def _write(self, run_id: str, line: Mapping[str, JsonValue]) -> None:
        file = self._files.get(run_id)
        if file is None:
            file = self._files[run_id] = (self.directory / f"{run_id}.jsonl").open("a")
            self._prune()
        file.write(json.dumps(line, default=str) + "\n")
        file.flush()

    def _close(self, run_id: str) -> None:
        file = self._files.pop(run_id, None)
        if file is not None:
            file.close()

    def _prune(self) -> None:
        files = sorted(self.directory.glob("*.jsonl"), key=lambda path: path.stat().st_mtime)
        for path in files[: max(0, len(files) - self.keep)]:
            if path.stem not in self._files and path.stem != NOTES:
                path.unlink(missing_ok=True)


def _ended(path: Path) -> bool:
    """Whether a run's file closes with the event that ends a run (its last line; read from the file's tail)."""
    with path.open("rb") as file:
        file.seek(max(0, path.stat().st_size - 4096))
        tail = file.read().decode(errors="replace").strip().rsplit("\n", 1)[-1]
    try:
        line: Any = json.loads(tail)
    except ValueError:  # (a long line cut by the read, or one its writer never finished)
        return False
    return isinstance(line, dict) and cast(dict[str, Any], line).get("type") in {kind.value for kind in ENDED}


# Reading


class Appended:
    """A file of JSON lines, read as it grows. `more` gives the lines appended since it was last called; `after`
    reads lines again from the file, by their place in it, so that a reader need not keep what it has read (a run's
    samples hold every message the model was sent)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._end = 0
        self._starts: list[int] = []

    def __len__(self) -> int:
        return len(self._starts)

    def more(self) -> list[dict[str, Any]]:
        try:
            if self.path.stat().st_size <= self._end:
                return []
            with self.path.open("rb") as file:
                file.seek(self._end)
                data = file.read()
        except OSError:
            return []
        data = data[: data.rfind(b"\n") + 1]  # a line still being written is left for the next read
        read: list[dict[str, Any]] = []
        for at, line in _parsed(data):
            self._starts.append(self._end + at)
            read.append(line)
        self._end += len(data)
        return read

    def after(self, index: int) -> list[dict[str, Any]]:
        if index >= len(self._starts):
            return []
        with self.path.open("rb") as file:
            file.seek(self._starts[index])
            data = file.read(self._end - self._starts[index])
        return [line for _, line in _parsed(data)]


def _parsed(data: bytes) -> Iterator[tuple[int, dict[str, Any]]]:
    """The JSON lines in `data`, each with where it begins (a line its writer was stopped in the middle of is no
    line)."""
    at = 0
    for raw in data.splitlines(keepends=True):
        try:
            line: Any = json.loads(raw) if raw.strip() else None
        except ValueError:
            line = None
        if isinstance(line, dict):
            yield at, cast(dict[str, Any], line)
        at += len(raw)


class FeedReader:
    """Reads a feed directory incrementally: each call picks up what was appended since the last. Of a run it
    keeps a summary; the run's lines are read from its file when they are asked for. Its methods may be called from
    several threads at once."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._reading = threading.Lock()
        self._runs: dict[str, tuple[Appended, _Summary]] = {}

    def refresh(self) -> None:
        with self._reading:
            self._refresh()

    def _refresh(self) -> None:
        here: set[str] = set()
        for path in self.directory.glob("*.jsonl"):
            if path.stem == NOTES:
                continue
            here.add(path.stem)
            file, summary = self._runs.setdefault(path.stem, (Appended(path), _Summary()))
            for line in file.more():
                summary.add(line)
        for run_id in set(self._runs) - here:
            del self._runs[run_id]

    def runs(self) -> list[dict[str, Any]]:
        """Every run in the feed, newest first: its labels, state, rewards and how much it has done."""
        self.refresh()
        summaries = [summary.of(run_id) for run_id, (_, summary) in self._runs.items()]
        return sorted(summaries, key=lambda run: run["started"], reverse=True)

    def lines(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        """A run's lines from index `after` on."""
        self.refresh()
        with self._reading:
            return self._runs[run_id][0].after(after) if run_id in self._runs else []


class _Summary:
    """What the page lists of a run, kept up as its lines are read."""

    def __init__(self) -> None:
        self.labels: dict[str, Any] = {}
        self.state = "running"
        self.started = self.updated = 0.0
        self.slots: list[str] = []
        self.samples = self.lines = 0
        self.rewarding: list[tuple[str, Mapping[str, Any]]] = []

    def add(self, line: Mapping[str, Any]) -> None:
        self.lines += 1
        self.updated = max(self.updated, float(line.get("at", 0.0)))
        if line["kind"] == "sample":
            self.samples += 1
            if line["slot"] not in self.slots:
                self.slots.append(line["slot"])
            return
        if line.get("kind") != "event" or "type" not in line:
            return  # (a line that is neither: not of a run)
        payload: Any = line.get("payload") or {}
        match line["type"]:
            case RunEventType.RUN_CREATED:
                self.labels = dict(payload.get("labels") or {})
                self.started = float(line["at"])
            case RunEventType.RUN_COMPLETED:
                self.state = "completed"
            case RunEventType.RUN_FAILED:
                self.state = "failed"
            case RunEventType.RUN_CANCELLED:
                self.state = "cancelled"
            case RunEventType.REWARD_ASSIGNED:
                self.rewarding.append((str(line["type"]), payload))
            case RunEventType.OBSERVATION_RECORDED if payload.get("reward") is not None:
                self.rewarding.append((str(line["type"]), {"reward": payload["reward"]}))
            case _:
                pass

    def of(self, run_id: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "labels": self.labels,
            "state": self.state,
            "started": self.started,
            "updated": self.updated,
            "rewards": {slot: by_key.get(DEFAULT, 0.0) for slot, by_key in assigned(self.rewarding).items()},
            "slots": self.slots,
            "samples": self.samples,
            "lines": self.lines,
        }
