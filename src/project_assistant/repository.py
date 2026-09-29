"""Read-only tools over one git repository, as an imported tool set.

Every result comes from the repository through a recorded `tool.call` effect, so a durable run replays it exactly.
Paths are confined to the repository; results are capped so one call cannot flood the context.
"""

import asyncio
import fnmatch
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import JsonValue

from rollout.core.contracts import RetryClass, Text, ToolAnnotations, ToolResult, ToolSpecification

MAX_RESULT_CHARACTERS = 20_000
MAX_FILE_BYTES = 1_000_000


def _schema(properties: dict[str, JsonValue], required: list[str]) -> dict[str, JsonValue]:
    return {
        "type": "object",
        "properties": properties,
        "required": list[JsonValue](required),
        "additionalProperties": False,
    }


READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True)

SPECIFICATIONS = [
    ToolSpecification(
        name="list_files",
        description="List files in the repository (tracked and untracked, excluding ignored files). "
        "Optionally filter with a glob such as 'src/**/*.py'.",
        input_schema=_schema(
            {"glob": {"type": "string", "description": "A glob relative to the repository root."}}, []
        ),
        annotations=READ_ONLY,
        retry_class=RetryClass.IDEMPOTENT,
    ),
    ToolSpecification(
        name="search",
        description="Search file contents with a regular expression. Returns matching lines as path:line: text.",
        input_schema=_schema(
            {
                "pattern": {"type": "string", "description": "A Python regular expression."},
                "glob": {"type": "string", "description": "Only search files matching this glob."},
                "ignore_case": {"type": "boolean"},
            },
            ["pattern"],
        ),
        annotations=READ_ONLY,
        retry_class=RetryClass.IDEMPOTENT,
    ),
    ToolSpecification(
        name="read_file",
        description="Read lines from a file, numbered. Reads at most 400 lines per call.",
        input_schema=_schema(
            {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "line_count": {"type": "integer", "minimum": 1, "maximum": 400},
            },
            ["path"],
        ),
        annotations=READ_ONLY,
        retry_class=RetryClass.IDEMPOTENT,
    ),
    ToolSpecification(
        name="git_log",
        description="Recent commits, newest first: hash, date, author and subject. "
        "Optionally only those touching a path.",
        input_schema=_schema(
            {"path": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, []
        ),
        annotations=READ_ONLY,
        retry_class=RetryClass.IDEMPOTENT,
    ),
]


class RepositoryTools:
    """Implements `ToolSet` for one repository."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        if not (self.root / ".git").exists():
            raise ValueError(f"{self.root} is not a git repository")

    def specifications(self) -> Sequence[ToolSpecification]:
        return SPECIFICATIONS

    async def call(
        self, name: str, arguments: Mapping[str, JsonValue], *, effect_id: str, arguments_digest: str
    ) -> ToolResult:
        try:
            if name == "list_files":
                text = await asyncio.to_thread(self._list_files, _string(arguments.get("glob")))
            elif name == "search":
                text = await asyncio.to_thread(
                    self._search,
                    str(arguments["pattern"]),
                    _string(arguments.get("glob")),
                    bool(arguments.get("ignore_case", False)),
                )
            elif name == "read_file":
                text = await asyncio.to_thread(
                    self._read_file,
                    str(arguments["path"]),
                    _integer(arguments.get("start_line"), 1),
                    min(_integer(arguments.get("line_count"), 200), 400),
                )
            elif name == "git_log":
                text = await asyncio.to_thread(
                    self._git_log, _string(arguments.get("path")), min(_integer(arguments.get("limit"), 10), 50)
                )
            else:
                return _error(f"unknown tool {name!r}")
        except (ValueError, KeyError, re.error, OSError) as error:
            return _error(f"{type(error).__name__}: {error}")
        truncated = len(text) > MAX_RESULT_CHARACTERS
        if truncated:
            text = text[:MAX_RESULT_CHARACTERS] + "\n… (truncated; narrow the request)"
        return ToolResult(content=[Text(text=text or "(no results)")], truncated=truncated)

    # Tools

    def _files(self) -> list[str]:
        output = self._git("ls-files", "--cached", "--others", "--exclude-standard")
        return sorted(line for line in output.splitlines() if line)

    def _list_files(self, glob: str | None) -> str:
        files = self._files()
        if glob:
            files = [path for path in files if fnmatch.fnmatch(path, glob)]
        return "\n".join(files)

    def _search(self, pattern: str, glob: str | None, ignore_case: bool) -> str:
        expression = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        matches: list[str] = []
        for path in self._files():
            if glob and not fnmatch.fnmatch(path, glob):
                continue
            file = self.root / path
            if not file.is_file() or file.stat().st_size > MAX_FILE_BYTES:
                continue
            try:
                lines = file.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue  # binary
            for number, line in enumerate(lines, start=1):
                if expression.search(line):
                    matches.append(f"{path}:{number}: {line.strip()[:300]}")
                    if len(matches) >= 200:
                        return "\n".join(matches) + "\n… (first 200 matches)"
        return "\n".join(matches)

    def _read_file(self, path: str, start_line: int, line_count: int) -> str:
        file = self._inside(path)
        if not file.is_file():
            raise ValueError(f"{path} is not a file")
        lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[start_line - 1 : start_line - 1 + line_count]
        numbered = [f"{start_line + index:>5}  {line}" for index, line in enumerate(selected)]
        footer = f"\n({len(lines)} lines in total)" if start_line - 1 + line_count < len(lines) else ""
        return "\n".join(numbered) + footer

    def _git_log(self, path: str | None, limit: int) -> str:
        arguments = ["log", f"-{limit}", "--date=short", "--format=%h %ad %an: %s"]
        if path:
            arguments += ["--", str(self._inside(path).relative_to(self.root))]
        return self._git(*arguments)

    # Helpers

    def _inside(self, path: str) -> Path:
        resolved = (self.root / path).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError(f"{path} is outside the repository")
        return resolved

    def _git(self, *arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(self.root), *arguments], capture_output=True, text=True, timeout=30, check=False
        )
        if completed.returncode != 0:
            raise ValueError(completed.stderr.strip() or f"git {arguments[0]} failed")
        return completed.stdout.strip()


def _string(value: JsonValue) -> str | None:
    return str(value) if value not in (None, "") else None


def _integer(value: JsonValue, default: int) -> int:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else default


def _error(text: str) -> ToolResult:
    return ToolResult(content=[Text(text=text)], is_error=True)
