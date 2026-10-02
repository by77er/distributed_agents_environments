"""The feed: what a monitor shows, written to disk by a runner's hooks as runs happen.

`RunFeed` is a `RunHooks`: it appends one JSON line per run event and per model sample to `<directory>/<run_id>.jsonl`.
A sample's line holds what the model was sent (every message, in a plain form), the tools it was offered, and its
reply with its reasoning. `read` turns a directory of such files into what the monitor's page asks for.
"""

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any, cast

from pydantic import JsonValue

from rollout.core.contracts import Message, Reasoning, RunEvent, RunEventType, Text, ToolCall, ToolResultBlock
from rollout.core.harness.hooks import ModelSample, RunHooks

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


class RunFeed(RunHooks):
    """Writes every run's events and samples under `directory`, one file per run, as they happen.

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
            if not _ended(path):
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
            if path.stem not in self._files:
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


class FeedReader:
    """Reads a feed directory incrementally: each call picks up what was appended since the last."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._offsets: dict[str, int] = {}
        self._lines: dict[str, list[dict[str, Any]]] = {}

    def refresh(self) -> None:
        for path in self.directory.glob("*.jsonl"):
            run_id = path.stem
            offset = self._offsets.get(run_id, 0)
            if path.stat().st_size <= offset:
                continue
            with path.open("rb") as file:
                file.seek(offset)
                data = file.read()
            end = data.rfind(b"\n") + 1  # a line still being written is left for the next read
            lines = self._lines.setdefault(run_id, [])
            for raw in data[:end].splitlines():
                if raw.strip():
                    try:
                        lines.append(json.loads(raw))
                    except ValueError:  # a line its writer was stopped in the middle of
                        continue
            self._offsets[run_id] = offset + end
        for run_id in [run_id for run_id in self._lines if not (self.directory / f"{run_id}.jsonl").exists()]:
            del self._lines[run_id], self._offsets[run_id]

    def runs(self) -> list[dict[str, Any]]:
        """Every run in the feed, newest first: its labels, state, rewards and how much it has done."""
        self.refresh()
        summaries = [summary(run_id, lines) for run_id, lines in self._lines.items()]
        return sorted(summaries, key=lambda run: run["started"], reverse=True)

    def lines(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        """A run's lines from index `after` on."""
        self.refresh()
        return self._lines.get(run_id, [])[after:]


def summary(run_id: str, lines: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    state, started, updated = "running", 0.0, 0.0
    rewards: dict[str, float] = {}
    slots: list[str] = []
    samples = 0
    for line in lines:
        updated = max(updated, float(line.get("at", 0.0)))
        if line["kind"] == "sample":
            samples += 1
            if line["slot"] not in slots:
                slots.append(line["slot"])
            continue
        payload: Any = line.get("payload") or {}
        match line["type"]:
            case "run.created":
                labels = dict(payload.get("labels") or {})
                started = float(line["at"])
            case "reward.assigned":
                rewards[str(payload.get("slot"))] = float(payload.get("value", 0.0))
            case "run.completed":
                state = "completed"
            case "run.failed":
                state = "failed"
            case "run.cancelled":
                state = "cancelled"
            case _:
                pass
    return {
        "run_id": run_id,
        "labels": labels,
        "state": state,
        "started": started,
        "updated": updated,
        "rewards": rewards,
        "slots": slots,
        "samples": samples,
        "lines": len(lines),
    }
