"""The stores a cluster config names, opened on this node: the ledger (SQLite, or Postgres with its URL named, never
written) and the blob store (files, or S3 with its credentials from the store's own environment); and the commands
over a ledger, given the cluster's with `--cluster`."""

from pathlib import Path

import pytest

from rollout.harness.blobs import FileBlobStore
from rollout_train.cluster import Cluster, ClusterError, parsed
from rollout_train.database import DatabaseLedger
from rollout_train.stores import FILES, Stores, ledger_url


def cluster_toml(ledger: str, blobs: str = "") -> str:
    return f'name = "here"\n[ledger]\n{ledger}\n[blobs]\n{blobs}\n'


def cluster_of(text: str) -> Cluster:
    import tomllib

    return parsed(tomllib.loads(text))


async def test_a_ledger_of_sqlite_and_blobs_of_files(tmp_path: Path) -> None:
    cluster = cluster_of(cluster_toml(f'url = "sqlite:///{tmp_path}/ledger.db"', f'directory = "{tmp_path}/blobs"'))
    stores = Stores.open(cluster)
    assert isinstance(stores.ledger, DatabaseLedger) and isinstance(stores.blobs, FileBlobStore)
    assert stores.location == {"kind": FILES, "directory": str(tmp_path / "blobs")}  # (what a run's start records)
    fence = await stores.ledger.take("here")
    assert await stores.ledger.append("table", "key", {"said": 1}, fence)
    reference = await stores.blobs.put(b"bytes", "text/plain")
    assert await stores.blobs.read(reference) == b"bytes" and (tmp_path / "blobs").is_dir()
    assert await stores.registry.runs() == [] and await stores.presets.all() == []
    assert await stores.checkpoints.all() == []


def test_a_ledger_url_named_is_read_on_this_node_and_never_said(tmp_path: Path) -> None:
    cluster = cluster_of(cluster_toml('url_env = "LEDGER_URL"'))
    url = f"sqlite:///{tmp_path}/ledger.db"
    assert ledger_url(cluster, {"LEDGER_URL": url}) == url
    with pytest.raises(ClusterError, match=r"names its URL as \$LEDGER_URL, which is not set on this node"):
        ledger_url(cluster, {})
    with pytest.raises(ClusterError, match=r"the URL \$LEDGER_URL holds is neither") as refused:
        ledger_url(cluster, {"LEDGER_URL": "mysql://user:hunter2@db/rollout"})
    assert "hunter2" not in str(refused.value)
    (tmp_path / "url").write_text(url + "\n")
    assert ledger_url(cluster_of(cluster_toml(f'url_file = "{tmp_path}/url"')), {}) == url
    with pytest.raises(ClusterError, match="is a database's"):
        ledger_url(cluster_of(cluster_toml(f'url = "{tmp_path}/ledger"')))


async def test_a_ledger_on_postgres(postgres: str, tmp_path: Path) -> None:
    cluster = cluster_of(cluster_toml('url_env = "ROLLOUT_LEDGER_URL"', f'directory = "{tmp_path}/blobs"'))
    stores = Stores.open(cluster, {"ROLLOUT_LEDGER_URL": postgres})
    assert isinstance(stores.ledger, DatabaseLedger) and stores.ledger.database.shared
    fence = await stores.ledger.take("here")
    assert await stores.ledger.append("table", "key", {"said": 1}, fence)
    assert await Stores.open(cluster, {"ROLLOUT_LEDGER_URL": postgres}).ledger.read("table") == {"key": {"said": 1}}


async def test_blobs_on_s3(s3_bucket: str, tmp_path: Path) -> None:
    from rollout_s3 import S3BlobStore

    blobs = f'kind = "rollout_s3:S3BlobStore"\nbucket = "{s3_bucket}"\nprefix = "blobs/"'
    stores = Stores.open(cluster_of(cluster_toml(f'url = "sqlite:///{tmp_path}/ledger.db"', blobs)))
    assert isinstance(stores.blobs, S3BlobStore)
    assert stores.location == {"kind": "rollout_s3:S3BlobStore", "bucket": s3_bucket, "prefix": "blobs/"}
    reference = await stores.blobs.put(b"bytes", "text/plain")
    assert await stores.blobs.read(reference) == b"bytes"


def run(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, str]:
    """A command's exit status and what it printed (and exited saying)."""
    from rollout_train.cli import main

    monkeypatch.setattr("sys.argv", ["rollout", *arguments])
    status, message = 0, ""
    try:
        main()
    except SystemExit as exited:
        status, message = (exited.code, "") if isinstance(exited.code, int) else (1, str(exited.code or ""))
    return status, capsys.readouterr().out + message


def test_cluster_check_says_what_does_not_resolve_here(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "here.toml"
    config.write_text(cluster_toml('url_env = "CHECKED_LEDGER_URL"', f'directory = "{tmp_path}/blobs"'))
    monkeypatch.delenv("CHECKED_LEDGER_URL", raising=False)
    status, said = run(monkeypatch, capsys, "cluster", "check", "--cluster", str(config))
    assert status == 1 and f"{config}: the cluster here (Ray namespace rollout-here)" in said
    assert "ledger.url: $CHECKED_LEDGER_URL is not set on this node" in said
    monkeypatch.setenv("CHECKED_LEDGER_URL", "sqlite:///secret-path.db")
    status, said = run(monkeypatch, capsys, "cluster", "check", "--cluster", str(config))
    assert status == 0 and "everything it names resolves on this node" in said and "secret-path" not in said
    monkeypatch.setenv("ROLLOUT_CLUSTER", str(config))  # (--cluster alone, or not at all: found)
    assert run(monkeypatch, capsys, "cluster", "check")[0] == 0
    assert run(monkeypatch, capsys, "cluster", "check", "--cluster", str(tmp_path / "nowhere.toml"))[0] == 1


def test_a_command_over_a_ledger_takes_the_clusters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import asyncio

    from rollout_train.checkpoints import new_id

    config = tmp_path / "here.toml"
    config.write_text(cluster_toml(f'url = "sqlite:///{tmp_path}/ledger.db"', f'directory = "{tmp_path}/blobs"'))
    stores = Stores.open(cluster_of(config.read_text()))
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "adapter.bin").write_text("weights")

    async def made() -> str:
        fence = await stores.ledger.take("runs/train")
        added = await stores.checkpoints.add(fence, new_id(), weights=tmp_path / "weights", run="train", step=1)
        return added.id

    checkpoint = asyncio.run(made())
    status, said = run(monkeypatch, capsys, "bookmark", "first", checkpoint, "--cluster", str(config))
    assert status == 0 and said.strip() == f"first is {checkpoint}"
    status, said = run(monkeypatch, capsys, "checkpoints", "--cluster", str(config))
    assert status == 0 and checkpoint[:4] in said and "[first]" in said
    monkeypatch.setenv("ROLLOUT_CLUSTER", str(config))
    assert "[first]" in run(monkeypatch, capsys, "checkpoints", "--cluster")[1]
    status, said = run(monkeypatch, capsys, "checkpoints", "--cluster", "--ledger", str(tmp_path))
    assert status == 2  # (one or the other)
