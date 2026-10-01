"""Paper servers for episodes: a verified server jar, the ground-truth plugin built from `plugin/`, a template server
per world seed configured from `config/`, and temporary servers started from a template.

Everything is cached under `~/.cache/rollout/minecraft` (not `/tmp`, which may be memory-backed):

- `paper/`: the Paper jar, checked against its published SHA-256;
- `bootstrap/`: one server started once, for the libraries and the patched jar every server shares;
- `jdk/`: a JDK, only to compile the plugin (a Java runtime is enough to run Paper);
- `plugin/`: the plugin jar, rebuilt when its sources change;
- `templates/seed-N/`: a configured server whose world was generated from seed N;
- `servers/`: temporary servers, copies of a template, deleted when stopped.

Starting a server means accepting the Minecraft EULA (https://aka.ms/MinecraftEULA) for a local, offline server.
"""

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import tarfile
import time
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

PAPER_VERSION = "1.21.11"
PAPER_BUILD = 132
CACHE = Path.home() / ".cache" / "rollout" / "minecraft"
ENVIRONMENT = Path(__file__).resolve().parents[1]
"""environments/minecraft: the plugin's sources, the server configuration and this package."""
PLUGIN_SOURCES = ENVIRONMENT / "plugin"
CONFIG = ENVIRONMENT / "config"
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

    def bootstrap(self) -> Path:
        """A server started once, so the shared libraries and patched jar exist."""
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
        """A configured server whose world was generated from `seed`; generated once, then copied for each server."""
        directory = self.root / "templates" / f"seed-{seed}"
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
        process = subprocess.Popen(
            [self.java, "-Xmx2G", "-jar", str(self.paper_jar()), "--nogui"],
            cwd=directory,
            stdin=subprocess.PIPE,
            stdout=(directory / "generate.log").open("w"),
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_line(directory / "generate.log", "Done (", process, seconds=600)
        finally:
            _stop_process(process)
        for name in ("logs", "generate.log", "usercache.json"):
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
    port: int = field(default_factory=lambda: free_port())
    control_port: int = field(default_factory=lambda: free_port())
    process: asyncio.subprocess.Process | None = None

    @property
    def directory(self) -> Path:
        return self.installation.root / "servers" / self.name

    @property
    def control_url(self) -> str:
        return f"http://127.0.0.1:{self.control_port}"

    async def start(self, *, seconds: float = 180) -> None:
        template = await asyncio.to_thread(self.installation.template, self.seed)
        plugin = await asyncio.to_thread(self.installation.plugin_jar)
        await asyncio.to_thread(_copy_template, template, self.directory)
        _write_properties(self.directory, {"server-port": str(self.port)})
        (self.directory / "plugins").mkdir(exist_ok=True)
        shutil.copy2(plugin, self.directory / "plugins" / "rollout-ground-truth.jar")
        log = (self.directory / "server.log").open("w")
        self.process = await asyncio.create_subprocess_exec(
            self.installation.java, f"-Xmx{self.heap}", f"-Drollout.control.port={self.control_port}",
            "-XX:+UseG1GC", "-jar", str(self.installation.paper_jar()), "--nogui",
            cwd=self.directory, stdin=asyncio.subprocess.PIPE, stdout=log, stderr=asyncio.subprocess.STDOUT,
        )  # fmt: skip
        deadline = time.monotonic() + seconds
        async with httpx.AsyncClient(timeout=2) as client:
            while time.monotonic() < deadline:
                if self.process.returncode is not None:
                    raise RuntimeError(f"the server exited while starting; see {self.directory / 'server.log'}")
                with contextlib.suppress(httpx.HTTPError):
                    if (await client.get(f"{self.control_url}/health")).status_code == 200:
                        return
                await asyncio.sleep(0.25)
        await self.stop()
        raise TimeoutError(f"the server did not start in {seconds} seconds")

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
        if not keep:
            await asyncio.to_thread(shutil.rmtree, self.directory, True)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def server_properties() -> dict[str, str]:
    """config/server.properties, as a dictionary."""
    values: dict[str, str] = {}
    for line in (CONFIG / "server.properties").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def _configure(bootstrap: Path, directory: Path) -> None:
    """Paper's own configuration with config/'s overrides merged in (anti-xray, no end dimension)."""
    (directory / "config").mkdir(exist_ok=True)
    for relative in ("config/paper-world-defaults.yml", "bukkit.yml"):
        defaults: dict[str, Any] = yaml.safe_load((bootstrap / relative).read_text())
        overrides: dict[str, Any] = yaml.safe_load((CONFIG / Path(relative).name).read_text())
        (directory / relative).write_text(yaml.safe_dump(_merge(defaults, overrides), sort_keys=False))
    shutil.copy2(bootstrap / "config" / "paper-global.yml", directory / "config" / "paper-global.yml")


def _merge(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Overrides win; a key ending in `-also` appends its list to the list without the suffix."""
    merged = dict(defaults)
    for key, value in overrides.items():
        if key.endswith("-also"):
            base = key.removesuffix("-also")
            merged[base] = [*merged.get(base, []), *(item for item in value if item not in merged.get(base, []))]
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)  # pyright: ignore[reportUnknownArgumentType]
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
