"""Paper servers for episodes: a verified server jar, the ground-truth plugin built from `plugin/`, a template server
per world seed configured from `config/`, and temporary servers started from a template.

Everything is cached under `~/.cache/rollout/minecraft` (not `/tmp`, which may be memory-backed):

- `paper/`: the Paper jar, checked against its published SHA-256;
- `bootstrap/`: one server started once, for the libraries and the patched jar every server shares;
- `jdk/`: a JDK, only to compile the plugin (a Java runtime is enough to run Paper);
- `plugin/`: the plugin jar, rebuilt when its sources change;
- `templates/seed-N-KEY/`: a configured server whose world was generated from seed N, the overworld around the
  origin included (`GENERATED_CHUNKS`): servers copied from it hold the same chunks, where servers that each
  generated their own differed in details (a tree here, two ores there). KEY covers config/, the Paper build and
  the generated area;
- `servers/`: temporary servers, copies of a template, deleted when stopped. A server ends with the process that
  started it, and what such a process left behind is removed by the next one (`sweep`).

Starting a server means accepting the Minecraft EULA (https://aka.ms/MinecraftEULA) for a local, offline server.
"""

import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import tarfile
import time
import urllib.request
import uuid
import zipfile
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from minecraft_team.control import Control, ControlError
from rollout.processes import end_with_parent

PAPER_VERSION = "1.21.11"
"""The Minecraft version the servers run; the harness's bots are told it when they connect."""
PAPER_BUILD = 132
CACHE = Path.home() / ".cache" / "rollout" / "minecraft"
ENVIRONMENT = Path(__file__).resolve().parents[1]
"""environments/minecraft: the plugin's sources, the server configuration and this package."""
PLUGIN_SOURCES = ENVIRONMENT / "plugin"
CONFIG = ENVIRONMENT / "config"
GENERATED_CHUNKS = 19
"""A template's overworld is generated this many chunks out from the origin (304 blocks): staged tasks are built
within 240 blocks of it and reach 40 further."""
SHARED = ("libraries", "versions", "cache")
"""Directories every server shares with the bootstrap server, so none downloads or patches anything."""


@dataclass
class Installation:
    root: Path = CACHE
    version: str = PAPER_VERSION
    build: int = PAPER_BUILD
    java: str = "java"

    # The server jar and what it needs

    def paper_jar(self) -> Path:
        jar = self.root / "paper" / f"paper-{self.version}-{self.build}.jar"
        if jar.exists():
            return jar
        jar.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://fill.papermc.io/v3/projects/paper/versions/{self.version}/builds/{self.build}"
        download = _json(url)["downloads"]["server:default"]
        data = _fetch(download["url"])
        if hashlib.sha256(data).hexdigest() != download["checksums"]["sha256"]:
            raise RuntimeError(f"the Paper jar from {download['url']} does not match its checksum")
        _write_atomically(jar, data)
        return jar

    @contextlib.contextmanager
    def _lock(self, name: str) -> Generator[None]:
        """Exclusive across threads and processes (a lock file under `root`)."""
        path = self.root / "locks" / f"{name}.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def bootstrap(self) -> Path:
        """A server started once, so the shared libraries and patched jar exist."""
        with self._lock("bootstrap"):
            return self._bootstrap()

    def _bootstrap(self) -> Path:
        directory = self.root / "bootstrap"
        if all((directory / name).is_dir() for name in SHARED) and any((directory / "versions").rglob("*.jar")):
            return directory
        directory.mkdir(parents=True, exist_ok=True)
        _write_properties(directory, {**server_properties(), "server-port": str(free_port())})
        (directory / "eula.txt").write_text("eula=true\n")
        process = subprocess.Popen(
            [self.java, "-Xmx1G", "-jar", str(self.paper_jar()), "--nogui"],
            cwd=directory,
            stdin=subprocess.PIPE,
            stdout=(directory / "bootstrap.log").open("w"),
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_line(directory / "bootstrap.log", "Done (", process, seconds=300)
        finally:
            _stop_process(process)
        return directory

    def jdk(self) -> Path:
        """A JDK's bin directory, downloaded if no `javac` is on the path."""
        if system := shutil.which("javac"):
            return Path(system).parent
        found = sorted((self.root.parent / "jdk").glob("jdk-21*/bin/javac"))
        if found:
            return found[-1].parent
        target = self.root.parent / "jdk"
        target.mkdir(parents=True, exist_ok=True)
        asset = _json(
            "https://api.adoptium.net/v3/assets/latest/21/hotspot?architecture=x64&image_type=jdk&os=linux&vendor=eclipse"
        )[0]["binary"]["package"]
        data = _fetch(asset["link"])
        if hashlib.sha256(data).hexdigest() != asset["checksum"]:
            raise RuntimeError("the JDK download does not match its checksum")
        archive = target / asset["name"]
        archive.write_bytes(data)
        with tarfile.open(archive) as tar:
            tar.extractall(target, filter="data")
        archive.unlink()
        return sorted(target.glob("jdk-21*/bin/javac"))[-1].parent

    def plugin_jar(self) -> Path:
        """The ground-truth plugin, compiled against this Paper's API; rebuilt when its sources change."""
        with self._lock("plugin"):
            return self._plugin_jar()

    def _plugin_jar(self) -> Path:
        sources = sorted((PLUGIN_SOURCES / "src").rglob("*.java")) + sorted((PLUGIN_SOURCES / "resources").rglob("*"))
        digest = hashlib.sha256(
            b"".join(path.read_bytes() for path in sources if path.is_file()) + self.version.encode()
        ).hexdigest()[:16]
        jar = self.root / "plugin" / f"rollout-ground-truth-{digest}.jar"
        if jar.exists():
            return jar
        libraries = sorted(str(path) for path in (self.bootstrap() / "libraries").rglob("*.jar"))
        classes = self.root / "plugin" / f"classes-{digest}"
        shutil.rmtree(classes, ignore_errors=True)
        classes.mkdir(parents=True)
        java_files = [str(path) for path in sources if path.suffix == ".java"]
        compiled = subprocess.run(
            [str(self.jdk() / "javac"), "--release", "21", "-proc:none", "-nowarn", "-cp", os.pathsep.join(libraries),
             "-d", str(classes), *java_files],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        if compiled.returncode != 0:
            raise RuntimeError(f"the plugin does not compile:\n{compiled.stdout}{compiled.stderr}")
        temporary = jar.with_suffix(".partial")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(classes.rglob("*.class")):
                archive.write(path, path.relative_to(classes).as_posix())
            for path in sorted((PLUGIN_SOURCES / "resources").rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(PLUGIN_SOURCES / "resources").as_posix())
        temporary.replace(jar)
        shutil.rmtree(classes)
        return jar

    # Templates

    def template(self, seed: int) -> Path:
        """A configured server whose world was generated from `seed`; generated once, then copied for each server.
        Safe to call from several threads or processes at once: one generates, the others wait."""
        with self._lock(f"template-{seed}"):
            return self._template(seed)

    def _template(self, seed: int) -> Path:
        key = f"{configuration_digest()}-{self.version}-{self.build}-g{GENERATED_CHUNKS}"
        directory = self.root / "templates" / f"seed-{seed}-{key}"
        if (directory / "ready").exists():
            return directory
        shutil.rmtree(directory, ignore_errors=True)
        bootstrap = self.bootstrap()
        directory.mkdir(parents=True)
        for name in SHARED:
            (directory / name).symlink_to(bootstrap / name)
        (directory / "eula.txt").write_text("eula=true\n")
        _write_properties(directory, {**server_properties(), "level-seed": str(seed), "server-port": str(free_port())})
        _configure(bootstrap, directory)
        (directory / "plugins").mkdir(exist_ok=True)
        shutil.copy2(self.plugin_jar(), directory / "plugins" / "rollout-ground-truth.jar")
        control = free_port()
        process = subprocess.Popen(  # (the plugin is there to generate the play area)
            [self.java, "-Xmx2G", f"-Drollout.control.port={control}", "-jar", str(self.paper_jar()), "--nogui"],
            cwd=directory,
            stdin=subprocess.PIPE,
            stdout=(directory / "generate.log").open("w"),
            stderr=subprocess.STDOUT,
            preexec_fn=end_with_parent,
        )
        try:
            _wait_for_line(directory / "generate.log", "Done (", process, seconds=600)
            asyncio.run(_generate(f"http://127.0.0.1:{control}"))  # (a template is made in a thread of its own)
        finally:
            _stop_process(process)
            release_port(control)
        for name in ("logs", "generate.log", "usercache.json", "plugins"):
            path = directory / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        for lock in directory.rglob("session.lock"):
            lock.unlink()
        (directory / "ready").write_text(json.dumps({"seed": seed, "version": self.version, "build": self.build}))
        return directory


@dataclass
class PaperServer:
    """A temporary server: a copy of a template that is deleted when it stops."""

    installation: Installation
    seed: int
    heap: str = "1536M"
    name: str = field(default_factory=lambda: f"s-{uuid.uuid4().hex[:10]}")
    port: int = 0
    control_port: int = 0
    """Chosen when the server starts (immediately before Java does)."""
    process: asyncio.subprocess.Process | None = None

    @property
    def directory(self) -> Path:
        return self.installation.root / "servers" / self.name

    @property
    def control_url(self) -> str:
        return f"http://127.0.0.1:{self.control_port}"

    async def start(self, *, seconds: float = 120, attempts: int = 2) -> None:
        """Copy the template and start Java; a start that fails is tried again with other ports. A start that is
        cancelled, or fails for good, leaves no process and no directory."""
        try:
            template = await asyncio.to_thread(self.installation.template, self.seed)
            plugin = await asyncio.to_thread(self.installation.plugin_jar)
            await asyncio.to_thread(_copy_template, template, self.directory)
            (self.directory / "plugins").mkdir(exist_ok=True)
            shutil.copy2(plugin, self.directory / "plugins" / "rollout-ground-truth.jar")
            (self.directory / "ops.json").write_text(json.dumps(operator_entries(operators()), indent=1))
            (self.directory / OWNER).write_text(str(os.getpid()))
            for attempt in range(1, attempts + 1):
                try:
                    await self._launch(seconds)
                    return
                except (RuntimeError, TimeoutError):
                    await self._terminate()
                    if attempt == attempts:
                        raise
        except BaseException:
            await self._terminate()
            await asyncio.to_thread(shutil.rmtree, self.directory, True)
            raise

    async def _launch(self, seconds: float) -> None:
        # IPv4 only: Java otherwise listens on an IPv6 socket with a mapped address (::ffff:127.0.0.1), which WSL does
        # not forward to Windows' localhost, so a client on the Windows side could not join to watch.
        self._release_ports()
        self.port, self.control_port = free_port(), free_port()
        _write_properties(self.directory, {"server-port": str(self.port)})
        log = (self.directory / "server.log").open("w")
        self.process = await asyncio.create_subprocess_exec(
            self.installation.java, f"-Xmx{self.heap}", f"-Drollout.control.port={self.control_port}",
            f"-Drollout.server.name={self.name}",
            "-XX:+UseG1GC", "-Djava.net.preferIPv4Stack=true",  # see below
            "-jar", str(self.installation.paper_jar()), "--nogui",
            cwd=self.directory, stdin=asyncio.subprocess.PIPE, stdout=log, stderr=asyncio.subprocess.STDOUT,
            preexec_fn=end_with_parent,
        )  # fmt: skip
        deadline = time.monotonic() + seconds
        async with httpx.AsyncClient(timeout=2) as client:
            control = Control(self.control_url, client=client)
            while time.monotonic() < deadline:
                if self.process.returncode is not None:
                    raise RuntimeError(f"the server exited while starting; its log ends:\n{self._log_tail()}")
                with contextlib.suppress(httpx.HTTPError, ValueError, ControlError):  # not up yet
                    health = await control.health()
                    if health.get("server") != self.name:  # another server answers on this port
                        raise RuntimeError(f"port {self.control_port} belongs to another server")
                    return
                await asyncio.sleep(0.25)
        raise TimeoutError(f"the server did not start in {seconds} seconds; its log ends:\n{self._log_tail()}")

    def _log_tail(self) -> str:
        return "\n".join((self.directory / "server.log").read_text(errors="replace").splitlines()[-15:])

    async def _terminate(self) -> None:
        process = self.process
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        self._release_ports()

    def _release_ports(self) -> None:
        for port in (self.port, self.control_port):
            release_port(port)
        self.port = self.control_port = 0

    async def stop(self, *, keep: bool = False) -> None:
        """Stop the server and delete its directory (unless `keep`, e.g. to inspect a failure)."""
        process = self.process
        if process is not None and process.returncode is None:
            if process.stdin is not None:
                with contextlib.suppress(ConnectionError):
                    process.stdin.write(b"stop\n")
                    await process.stdin.drain()
            try:
                async with asyncio.timeout(30):
                    await process.wait()
            except TimeoutError:
                process.kill()
                await process.wait()
        self._release_ports()
        if not keep:
            await asyncio.to_thread(shutil.rmtree, self.directory, True)


OWNER = "owner.pid"
"""In a server's directory: the process that started it."""
_handed_out: set[int] = set()


def sweep(installation: Installation) -> list[str]:
    """Remove the servers that processes which are gone left behind (their Java processes ended with them); returns
    their names. Called when a process that will start servers begins."""
    removed: list[str] = []
    for directory in (installation.root / "servers").glob("s-*"):
        try:
            owner = int((directory / OWNER).read_text())
        except (OSError, ValueError):  # no owner written: a copy still being made, unless it is old
            if time.time() - directory.stat().st_mtime < 600:
                continue
            owner = None
        if owner is not None:
            try:
                os.kill(owner, 0)
                continue  # its process is alive
            except ProcessLookupError:
                pass
            except PermissionError:
                continue
        shutil.rmtree(directory, ignore_errors=True)
        removed.append(directory.name)
    return removed


def release_port(port: int) -> None:
    _handed_out.discard(port)


def free_port() -> int:
    """A free port to listen on, outside the range the system gives outgoing connections (a port from that range can
    be taken by a client's connection between choosing it and listening on it), and not one this process has given
    to a server that has yet to listen on it (Java binds ten seconds after it starts)."""
    for _ in range(200):
        port = random.randint(20000, 29999)
        if port in _handed_out:
            continue
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            _handed_out.add(port)
            return port
    raise RuntimeError("no free port between 20000 and 29999")


def operators() -> list[str]:
    """config/operators.txt: the players made operators of every episode server."""
    path = CONFIG / "operators.txt"
    lines = path.read_text().splitlines() if path.exists() else []
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def offline_uuid(name: str) -> uuid.UUID:
    """The id an offline-mode server gives a player: Java's name-based UUID of "OfflinePlayer:" and the name."""
    digest = bytearray(hashlib.md5(f"OfflinePlayer:{name}".encode(), usedforsecurity=False).digest())
    digest[6] = digest[6] & 0x0F | 0x30  # version 3
    digest[8] = digest[8] & 0x3F | 0x80  # the RFC 4122 variant
    return uuid.UUID(bytes=bytes(digest))


def operator_entries(names: list[str]) -> list[dict[str, Any]]:
    """The server's ops.json for these players."""
    return [{"uuid": str(offline_uuid(name)), "name": name, "level": 4, "bypassesPlayerLimit": True} for name in names]


def configuration_digest() -> str:
    """A digest of config/: templates made with other settings are not reused. Operators are written when a server
    starts, not into templates."""
    files = sorted(path for path in CONFIG.rglob("*") if path.is_file() and path.name != "operators.txt")
    return hashlib.sha256(b"".join(path.name.encode() + path.read_bytes() for path in files)).hexdigest()[:10]


def server_properties() -> dict[str, str]:
    """config/server.properties, as a dictionary."""
    values: dict[str, str] = {}
    for line in (CONFIG / "server.properties").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


async def _generate(control_url: str) -> None:
    """Have the server whose control API is at `control_url` generate the area a template holds."""
    control = Control(control_url)
    try:
        await control.generate(GENERATED_CHUNKS)
    finally:
        await control.close()


def _configure(bootstrap: Path, directory: Path) -> None:
    """Paper's own configuration with config/'s overrides merged in (anti-xray, the end allowed, and how far off
    entities are tracked)."""
    (directory / "config").mkdir(exist_ok=True)
    for relative in ("config/paper-world-defaults.yml", "bukkit.yml", "spigot.yml"):
        defaults: dict[str, Any] = yaml.safe_load((bootstrap / relative).read_text())
        overrides: dict[str, Any] = yaml.safe_load((CONFIG / Path(relative).name).read_text())
        (directory / relative).write_text(yaml.safe_dump(merge_configuration(defaults, overrides), sort_keys=False))
    shutil.copy2(bootstrap / "config" / "paper-global.yml", directory / "config" / "paper-global.yml")


def merge_configuration(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Overrides win; a key ending in `-also` appends its list to the list without the suffix."""
    merged = dict(defaults)
    for key, value in overrides.items():
        if key.endswith("-also"):
            base = key.removesuffix("-also")
            merged[base] = [*merged.get(base, []), *(item for item in value if item not in merged.get(base, []))]
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_configuration(merged[key], value)  # pyright: ignore[reportUnknownArgumentType]
        else:
            merged[key] = value
    return merged


def _copy_template(template: Path, target: Path) -> None:
    def ignore(directory: str, names: list[str]) -> list[str]:
        return [name for name in names if Path(directory) == template and name in (*SHARED, "ready")]

    shutil.copytree(template, target, ignore=ignore)
    for name in SHARED:
        (target / name).symlink_to((template / name).resolve())


def _write_properties(directory: Path, values: dict[str, str]) -> None:
    path = directory / "server.properties"
    lines = path.read_text().splitlines() if path.exists() else []
    remaining = dict(values)
    updated: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0]
        if key in remaining:
            updated.append(f"{key}={remaining.pop(key)}")
        else:
            updated.append(line)
    updated.extend(f"{key}={value}" for key, value in remaining.items())
    path.write_text("\n".join(updated) + "\n")


def _wait_for_line(log: Path, text: str, process: subprocess.Popen[bytes], *, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if log.exists() and text in log.read_text(errors="replace"):
            return
        if process.poll() is not None:
            raise RuntimeError(f"the server exited; see {log}")
        time.sleep(0.5)
    raise TimeoutError(f"{text!r} did not appear in {log} within {seconds} seconds")


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(BrokenPipeError, OSError):
        assert process.stdin is not None
        process.stdin.write(b"stop\n")
        process.stdin.flush()
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGKILL)
        process.wait()


def _json(url: str) -> Any:
    return json.loads(_fetch(url))


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def _write_atomically(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(data)
    temporary.replace(path)
