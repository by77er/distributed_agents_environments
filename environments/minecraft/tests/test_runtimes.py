"""The runtimes Minecraft's worlds need: the machine's `java`, `node` and `npm` where they are on the path, else a JDK's
`java` and Node 22, downloaded once beside the cache and checked against their published checksums (nothing is
downloaded here: the downloads are stood in for)."""

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from minecraft_team import paper
from minecraft_team.paper import NODE, Installation

NAME = "node-v22.99.0-linux-x64"


def on_path(name: str) -> str | None:
    return f"/usr/bin/{name}"


def nowhere(name: str) -> str | None:
    return None


def x86(*arguments: object) -> str:
    return "x86_64"


def node_tarball() -> bytes:
    """A tarball laid out as Node's: its `bin/node`, and `bin/npm` linked to npm's script."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz") as tar:
        for name, data in ((f"{NAME}/bin/node", b"#!/bin/sh\n"), (f"{NAME}/lib/node_modules/npm/bin/npm-cli.js", b"")):
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode = len(data), 0o755
            tar.addfile(entry, io.BytesIO(data))
        link = tarfile.TarInfo(f"{NAME}/bin/npm")
        link.type, link.linkname = tarfile.SYMTYPE, "../lib/node_modules/npm/bin/npm-cli.js"
        tar.addfile(link)
    return buffer.getvalue()


def test_the_machines_runtimes_are_used_where_they_are_on_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paper.shutil, "which", on_path)

    def fetch(url: str) -> bytes:
        pytest.fail(f"downloaded {url}")

    monkeypatch.setattr(paper, "_fetch", fetch)
    installation = Installation(root=tmp_path / "minecraft")
    assert installation.java_executable() == "/usr/bin/java"
    assert installation.node() == Path("/usr/bin")


def test_without_java_on_the_path_paper_runs_on_the_downloaded_jdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paper.shutil, "which", nowhere)
    jdk = tmp_path / "jdk" / "jdk-21.0.9+10" / "bin"
    jdk.mkdir(parents=True)
    (jdk / "javac").write_text("")  # (a JDK downloaded before: found, not downloaded again)
    installation = Installation(root=tmp_path / "minecraft")
    assert installation.java_executable() == str(jdk / "java")


def test_without_node_on_the_path_node_22_is_downloaded_once_and_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tarball = node_tarball()
    sums = f"{'0' * 64}  {NAME.replace('x64', 'arm64')}.tar.xz\n{hashlib.sha256(tarball).hexdigest()}  {NAME}.tar.xz\n"
    fetched: list[str] = []

    def fetch(url: str) -> bytes:
        fetched.append(url)
        return sums.encode() if url.endswith("SHASUMS256.txt") else tarball

    monkeypatch.setattr(paper.shutil, "which", nowhere)
    monkeypatch.setattr(paper.platform, "machine", x86)
    monkeypatch.setattr(paper, "_fetch", fetch)
    installation = Installation(root=tmp_path / "minecraft")
    bin = installation.node()
    assert bin == tmp_path / "node" / NAME / "bin" and (bin / "node").is_file() and (bin / "npm").exists()
    assert fetched == [f"{NODE}/SHASUMS256.txt", f"{NODE}/{NAME}.tar.xz"]
    assert installation.node() == bin and len(fetched) == 2  # (once: found beside the cache after)
    assert installation.node_environment()["PATH"].startswith(f"{bin}:")


def test_a_node_download_that_does_not_match_its_checksum_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paper.shutil, "which", nowhere)
    monkeypatch.setattr(paper.platform, "machine", x86)
    sums = f"{'0' * 64}  {NAME}.tar.xz\n".encode()

    def fetch(url: str) -> bytes:
        return sums if url.endswith(".txt") else node_tarball()

    monkeypatch.setattr(paper, "_fetch", fetch)
    with pytest.raises(RuntimeError, match="does not match its checksum"):
        Installation(root=tmp_path / "minecraft").node()
    assert not list((tmp_path / "node").glob("node-*"))
