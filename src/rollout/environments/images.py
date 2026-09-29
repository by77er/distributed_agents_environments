"""Base images: Alpine's mini root filesystem, downloaded once, verified against its SHA-256 and cached."""

import asyncio
import hashlib
import re
from pathlib import Path

import httpx

ALPINE = "https://dl-cdn.alpinelinux.org/alpine"


class ImageStore:
    """Resolves image names (`alpine`, `alpine:3.24.2`) to verified, cached root filesystem tarballs."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._lock = asyncio.Lock()

    async def tarball(self, image: str) -> Path:
        name, _, version = image.partition(":")
        if name != "alpine":
            raise ValueError(f"unknown image {image!r}; the namespace backend provides alpine[:version]")
        async with self._lock, httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            if not version:
                version = await self._latest_alpine(client)
            path = self.directory / f"alpine-minirootfs-{version}-x86_64.tar.gz"
            if path.exists():
                return path
            branch = "v" + ".".join(version.split(".")[:2])
            url = f"{ALPINE}/{branch}/releases/x86_64/{path.name}"
            expected = (await client.get(f"{url}.sha256")).raise_for_status().text.split()[0]
            data = (await client.get(url)).raise_for_status().content
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"checksum mismatch for {url}")
            self.directory.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".partial")
            partial.write_bytes(data)
            partial.replace(path)
            return path

    async def _latest_alpine(self, client: httpx.AsyncClient) -> str:
        cached = self.directory / "alpine-latest"
        if cached.exists():
            return cached.read_text().strip()
        listing = (await client.get(f"{ALPINE}/latest-stable/releases/x86_64/latest-releases.yaml")).raise_for_status()
        match = re.search(r"alpine-minirootfs-(\d+\.\d+\.\d+)-x86_64\.tar\.gz", listing.text)
        if match is None:
            raise ValueError("could not find the latest Alpine mini root filesystem")
        self.directory.mkdir(parents=True, exist_ok=True)
        cached.write_text(match.group(1))  # pinned from now on, so environments stay identical
        return match.group(1)
