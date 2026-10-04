"""Copying stores of files (`rollout.harness.blobs.FileBlobStore`) into a bucket: blobs are named by their SHA-256, so
any number of such stores merge into one `S3BlobStore` without a clash, and every reference to one of their blobs
reads the same bytes from it.

    python -m rollout_s3.copying DIRECTORY... --to s3://BUCKET/PREFIX [--dry-run]

Each file is uploaded in parts, never read whole. A blob the bucket has (an object of its size) is skipped, so a copy
can be run again and copies only what is new; a file that is not the blob its name says is not copied, and is
reported. The connection comes from boto3's usual sources, the endpoint from `AWS_ENDPOINT_URL` (`S3BlobStore`).
"""

import argparse
import hashlib
import re
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from rollout_s3.store import S3BlobStore, _not_found  # pyright: ignore[reportPrivateUsage]

__all__ = ["Copied", "blobs_in", "copy_files"]

DIGEST = re.compile(r"[0-9a-f]{64}")
MEDIA_TYPE = "application/octet-stream"
"""What a blob copied from a store of files is stored as: such a store keeps no media types."""


@dataclass
class Copied:
    """What a copy did: the blobs it copied and their bytes, the blobs the bucket had, the files that were not their
    blob."""

    copied: int = 0
    bytes: int = 0
    present: int = 0
    corrupt: list[Path] = field(default_factory=list[Path])


def blobs_in(directory: Path) -> Iterator[tuple[str, Path]]:
    """Every blob of a store of files, as its digest and its file (`DIRECTORY/ab/abcdef…`)."""
    for shard in sorted(directory.iterdir()) if directory.is_dir() else []:
        if not shard.is_dir() or len(shard.name) != 2:
            continue
        for path in sorted(shard.iterdir()):
            if DIGEST.fullmatch(path.name) and path.name.startswith(shard.name) and path.is_file():
                yield path.name, path


def copy_files(directories: Sequence[Path], store: S3BlobStore, *, workers: int = 8, dry_run: bool = False) -> Copied:
    """Copy every blob of the stores of files in `directories` into `store`'s bucket, under its prefix. With `dry_run`,
    say what would be copied, and copy nothing."""
    every = {digest: path for directory in directories for digest, path in blobs_in(directory)}
    done = Copied()

    def copied(digest: str, path: Path) -> tuple[Literal["copied", "present", "corrupt"], int]:
        return _copied(store, digest, path, dry_run=dry_run)

    with ThreadPoolExecutor(workers) as pool:
        for path, (outcome, size) in zip(every.values(), pool.map(copied, every.keys(), every.values()), strict=True):
            if outcome == "copied":
                done.copied, done.bytes = done.copied + 1, done.bytes + size
            elif outcome == "present":
                done.present += 1
            else:
                done.corrupt.append(path)
    return done


def _copied(
    store: S3BlobStore, digest: str, path: Path, *, dry_run: bool
) -> tuple[Literal["copied", "present", "corrupt"], int]:
    from botocore.exceptions import ClientError

    key, size = store._key(digest), path.stat().st_size  # pyright: ignore[reportPrivateUsage]
    try:
        if int(store.client.head_object(Bucket=store.bucket, Key=key)["ContentLength"]) == size:
            return "present", size
    except ClientError as error:
        if not _not_found(error):
            raise
    with path.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != digest:
            return "corrupt", size
    if dry_run:
        return "copied", size
    extra = {"ContentType": MEDIA_TYPE, "Metadata": {"sha256": digest}}
    store.client.upload_file(str(path), store.bucket, key, ExtraArgs=extra)
    return "copied", size


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m rollout_s3.copying", description="copy stores of files into S3")
    parser.add_argument("directories", nargs="+", type=Path, help="stores of files")
    parser.add_argument("--to", required=True, help="s3://BUCKET/PREFIX")
    parser.add_argument("--dry-run", action="store_true", help="say what would be copied, and copy nothing")
    given = parser.parse_args(arguments)
    done = copy_files(given.directories, S3BlobStore.from_url(given.to), dry_run=given.dry_run)
    verb = "to copy" if given.dry_run else "copied"
    print(f"{done.copied} blobs {verb} ({done.bytes} bytes), {done.present} there already")
    for path in done.corrupt:
        print(f"not copied, not the blob its name says: {path}", file=sys.stderr)
    if done.corrupt:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
