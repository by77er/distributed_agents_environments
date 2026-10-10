"""The bridge to the mineflayer harness (environments/minecraft/harness): one Node process per server, one bot per
agent, JSON lines over its stdin and stdout."""

import asyncio
import contextlib
import itertools
import json
from pathlib import Path
from typing import Any, Self

from minecraft_team.paper import HARNESS, PAPER_VERSION, Installation

__all__ = ["HARNESS", "NODE_FLAGS", "Harness", "HarnessError"]

NODE_FLAGS = ("--max-semi-space-size=4",)
"""What Node is started with: a young generation of 4 MiB a half, not 16 (docs/research/minecraft-memory.md)."""


class HarnessError(RuntimeError):
    pass


class Harness:
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process
        self._ids = itertools.count(1)
        self._replies: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader = asyncio.create_task(self._read())

    @classmethod
    async def start(cls, *, log: Path | None = None, installation: Installation | None = None) -> Self:
        """Start the harness with its packages and its `node` from `installation` (`Installation.harness_packages`
        and `Installation.node`: installed or downloaded the first time); its standard error goes to `log`."""
        installed = installation or Installation()
        try:
            packages = await asyncio.to_thread(installed.harness_packages)
            node = await asyncio.to_thread(installed.node)
        except RuntimeError as error:
            raise HarnessError(str(error)) from error
        stderr = log.open("a") if log is not None else asyncio.subprocess.DEVNULL
        process = await asyncio.create_subprocess_exec(
            str(node / "node"), *NODE_FLAGS, str(HARNESS / "harness.js"), cwd=HARNESS,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=stderr,
            limit=16 * 1024 * 1024, env={**installed.node_environment(), "NODE_PATH": str(packages)},
        )  # fmt: skip
        assert process.stdout is not None
        async with asyncio.timeout(30):
            while not (line := await process.stdout.readline()).startswith(b'{"ready"'):
                if not line:
                    raise HarnessError("the harness exited while starting")
        return cls(process)

    async def connect(self, host: str, port: int, team: list[str], *, version: str = PAPER_VERSION) -> None:
        """Join the server at `host` and `port`, which runs Minecraft `version`, with one bot for each of `team`."""
        await self.request("connect", host=host, port=port, team=team, version=version, wait_seconds=90)

    async def observe(self, bot: str) -> dict[str, Any]:
        """What a bot sees, what it has heard, whether it died, and how its last action went. Asked again before
        the next `thaw`, it answers the same."""
        return await self.request("observe", bot=bot)

    async def act(self, bot: str, action: dict[str, Any]) -> bool:
        """Start an action; whether it started (an unknown action is reported in the next observation)."""
        return bool((await self.request("act", bot=bot, action=action))["started"])

    async def busy(self) -> tuple[list[str], list[str], list[str]]:
        """The bots that are still acting, those that have been hurt since the thaw, and those threatened: idle with
        a hostile mob close by, or with a creeper about to go off whatever they are doing."""
        status = await self.request("busy")
        return list(status["acting"]), list(status.get("hurt") or []), list(status.get("threatened") or [])

    async def unloaded(self) -> list[str]:
        """The bots that do not yet hold the chunks around them."""
        return list((await self.request("unloaded"))["unloaded"])

    async def freeze(self) -> None:
        await self.request("freeze")

    async def thaw(self) -> None:
        """Resume the bots' physics, before actions start. What observations have told of messages and deaths is
        dropped."""
        await self.request("thaw")

    async def close(self) -> None:
        if self._process.returncode is None:
            with contextlib.suppress(HarnessError, TimeoutError, ConnectionError):
                await self.request("quit", wait_seconds=5)
            try:
                async with asyncio.timeout(5):
                    await self._process.wait()
            except TimeoutError:
                self._process.kill()
                await self._process.wait()
        self._reader.cancel()

    async def request(self, op: str, *, wait_seconds: float = 30, **arguments: Any) -> dict[str, Any]:
        stdin = self._process.stdin
        if stdin is None or self._process.returncode is not None:
            raise HarnessError("the harness is not running")
        request_id = next(self._ids)
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._replies[request_id] = future
        stdin.write(json.dumps({"id": request_id, "op": op, **arguments}).encode() + b"\n")
        await stdin.drain()
        reply = await asyncio.wait_for(future, wait_seconds)
        if "error" in reply:
            raise HarnessError(f"{op}: {reply['error']}")
        return reply["result"]

    async def _read(self) -> None:
        assert self._process.stdout is not None
        while line := await self._process.stdout.readline():
            if not line.startswith(b"{"):
                continue
            reply = json.loads(line)
            future = self._replies.pop(reply.get("id", -1), None)
            if future is not None and not future.done():
                future.set_result(reply)
        for future in self._replies.values():
            if not future.done():
                future.set_exception(ConnectionError("the harness exited"))
