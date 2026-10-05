"""A sandbox pool served elsewhere (the cluster config's `[sandboxes.KIND] url`): `rollout pool --kind KIND` serves the
pool its section describes, its leases beside the cluster's ledger; a run on the session's Ray reaches it there through
the claiming interface alone, makes no pool of its own and counts nothing of it in its demand, and each episode's
sandbox is leased under the episode's claim and released when the episode ends."""

import argparse
import asyncio
from pathlib import Path
from typing import Any

import httpx

from rollout.testing import until
from rollout_train.cli import _pool  # pyright: ignore[reportPrivateUsage]
from rollout_train.demand import demand
from rollout_train.jobs import Run, ran
from rollout_train.presence import presence_of
from rollout_train.record import ENDS, newest_record, table
from rollout_train.rollouts import Record
from rollout_train.rollouts.scheduler import EPISODES
from rollout_train.run_settings import RunSettings
from rollout_train.sandboxes import leases_of
from rollout_train.stores import Stores
from tests.local_ray import LocalRay, free_port
from tests.rollout_train.clusters import POLICY, WORDS, a_cluster
from tests.rollout_train.rollouts.games import BOXES

BOXED = "tests.rollout_train.rollouts.games:boxed"


async def test_a_run_acquires_its_sandboxes_from_a_pool_served_elsewhere_under_its_episodes_claims(
    tmp_path: Path, local_ray: LocalRay
) -> None:
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    pool = f'\n[sandboxes.fake]\nprovider = "tests.rollout_train.rollouts.games:boxes"\nsize = 2\nurl = "{url}"\n'
    cluster = a_cluster(tmp_path, more=f'{pool}\n[environments."{BOXED}"]\npython = "platform"\n')
    BOXES.clear()
    serving = asyncio.ensure_future(_pool(argparse.Namespace(
        kind="fake", cluster=str(tmp_path / "cluster.toml"), factory=None, directory=None, ledger=None, name=None,
        host="127.0.0.1", port=port,
    )))  # fmt: skip

    async def answers() -> bool:
        try:
            async with httpx.AsyncClient() as client:
                return (await client.get(f"{url}/capacity")).json() == {"size": 2, "leased": 0}
        except httpx.HTTPError:
            return False

    try:
        await until(answers, 30, every=0.1)
        (worlds,) = BOXES  # (made by the pool's process, from [sandboxes.fake]: none in the run's)
        assert worlds.size == 2
        stores = Stores.open(cluster)
        settings = RunSettings({**POLICY, "kind": "train", "environment": BOXED, "name": "boxed"})
        run = Run(cluster, stores, settings, await stores.registry.create("boxed"))
        await ran(run)
        assert run.pool_bindings["fake"].url == url and run.runner is not None and not run.runner.pools
        assert run.demand == demand(RunSettings({**settings.values, "environment": WORDS}), cluster)
        records = [Record.from_json(each) for each in (await stores.ledger.read(table(run.run.id, EPISODES))).values()]  # type: ignore[arg-type]
        assert records
        keys = {str(record.episode.info["key"]) for record in records}
        assert all(key.startswith(f"{run.run.id}/") and key.endswith("/box") for key in keys)
        assert len(keys) == len(set(worlds.made)) == len(records)  # (a sandbox each, made by the pool's process)
        assert set(worlds.deleted) == set(worlds.made) and worlds.sandboxes == {}  # (each released with its episode)
        leases = leases_of(stores.ledger)
        assert leases is not None and await leases.all() == []
        presence = presence_of(stores.ledger)
        assert presence is not None
        beats: dict[str, Any] = {beat.runner: beat for beat in await presence.beats()}
        assert "pools/fake" in beats  # (the pool's own keeper, beating where the monitor sees it)
        assert newest_record(await stores.ledger.read(table(run.run.id, ENDS)))["how"] == "finished"
    finally:
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
