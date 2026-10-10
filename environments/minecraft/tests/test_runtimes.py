"""The runtimes Minecraft's worlds need: the machine's `java`, `node` and `npm` where they are on the path, else the JDK
and the Node pinned in `minecraft_team.paper`, for the machine's architecture, downloaded once beside the cache and
refused unless they match their pinned checksums (nothing is downloaded here: the downloads are stood in for)."""

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from minecraft_team import paper
from minecraft_team.paper import JDK, JDK_RELEASE, NODE, NODE_VERSION, Installation


def on_path(name: str) -> str | None:
    return f"/usr/bin/{name}"


def nowhere(name: str) -> str | None:
    return None


def tarball(top: str, files: dict[str, bytes], *, compression: str = "xz") -> bytes:
    """A tarball of one top directory holding `files` (each executable), and `bin/npm` linked to npm's script."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz" if compression == "gz" else "w:xz") as tar:
        for name, data in files.items():
            entry = tarfile.TarInfo(f"{top}/{name}")
            entry.size, entry.mode = len(data), 0o755
            tar.addfile(entry, io.BytesIO(data))
        if "lib/node_modules/npm/bin/npm-cli.js" in files:
            link = tarfile.TarInfo(f"{top}/bin/npm")
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


@pytest.mark.parametrize(("machine", "arch"), [("x86_64", "x64"), ("aarch64", "aarch64")])
def test_without_java_on_the_path_the_pinned_jdk_for_the_machine_is_downloaded_once_and_runs_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, machine: str, arch: str
) -> None:
    data = tarball("jdk-21.0.12.1+1", {"bin/javac": b"", "bin/java": b""}, compression="gz")
    fetched: list[str] = []

    def fetch(url: str) -> bytes:
        fetched.append(url)
        return data

    monkeypatch.setattr(paper.shutil, "which", nowhere)
    monkeypatch.setattr(paper.platform, "machine", lambda: machine)
    monkeypatch.setattr(paper, "JDK_SHA256", {arch: hashlib.sha256(data).hexdigest()})
    monkeypatch.setattr(paper, "_fetch", fetch)
    installation = Installation(root=tmp_path / "minecraft")
    assert installation.java_executable() == str(tmp_path / "jdk" / JDK_RELEASE / "bin" / "java")
    assert installation.jdk() == tmp_path / "jdk" / JDK_RELEASE / "bin" and len(fetched) == 1  # (once)
    assert fetched == [f"{JDK}/OpenJDK21U-jdk_{arch}_linux_hotspot_21.0.12.1_1.tar.gz"]
    assert not list((tmp_path / "jdk").glob(".partial-*"))  # (unpacked beside it, then renamed)


def test_without_node_on_the_path_the_pinned_node_is_downloaded_once_and_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = f"node-{NODE_VERSION}-linux-x64"
    data = tarball(name, {"bin/node": b"#!/bin/sh\n", "lib/node_modules/npm/bin/npm-cli.js": b""})
    fetched: list[str] = []

    def fetch(url: str) -> bytes:
        fetched.append(url)
        return data

    monkeypatch.setattr(paper.shutil, "which", nowhere)
    monkeypatch.setattr(paper.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(paper, "NODE_SHA256", {"x64": hashlib.sha256(data).hexdigest()})
    monkeypatch.setattr(paper, "_fetch", fetch)
    installation = Installation(root=tmp_path / "minecraft")
    bin = installation.node()
    assert bin == tmp_path / "node" / name / "bin" and (bin / "node").is_file() and (bin / "npm").exists()
    assert fetched == [f"{NODE}/{name}.tar.xz"]
    assert installation.node() == bin and len(fetched) == 1  # (once: found beside the cache after)
    assert installation.node_environment()["PATH"].startswith(f"{bin}:")


def test_a_download_that_does_not_match_its_pinned_checksum_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paper.shutil, "which", nowhere)
    monkeypatch.setattr(paper.platform, "machine", lambda: "x86_64")
    data = tarball(f"node-{NODE_VERSION}-linux-x64", {"bin/node": b""})

    def fetch(url: str) -> bytes:
        return data

    monkeypatch.setattr(paper, "_fetch", fetch)
    with pytest.raises(RuntimeError, match="does not match its checksum"):
        Installation(root=tmp_path / "minecraft").node()
    assert not list((tmp_path / "node").glob("node-*"))
