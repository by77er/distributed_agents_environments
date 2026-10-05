"""An episode with a judge, as the monitor shows it: the judge's turns beside the policy's, each labelled by its slot,
the judge's slot said to be untrained, and the episode's reward the policy's alone."""

from pathlib import Path

from pydantic import JsonValue

from rollout.contracts import Message
from rollout.harness import RecordedModel
from rollout.harness.blobs import FileBlobStore
from rollout_train.gateway import Attempt
from rollout_train.layout import BLOBS, FEED, LEDGER
from rollout_train.ledger import FileLedger
from rollout_train.monitor import FeedReader, RunFeed, System
from rollout_train.record import GROUPS, STARTS, scope, table
from rollout_train.rollouts.episodes import Episode, Outcome, Trajectory, assemble, stored
from rollout_train.rollouts.scheduler import EPISODES
from rollout_train.testing import gateway_endpoints, plain_channel, sample_request


async def test_an_episodes_judge_turns_are_shown_by_slot_and_its_slot_untrained(tmp_path: Path) -> None:
    ledger, blobs = FileLedger(tmp_path / LEDGER), FileBlobStore(tmp_path / BLOBS)
    fence = await ledger.take(scope("train"))
    start: JsonValue = {"from": None, "host": "here", "started": 5.0, "directory": str(tmp_path)}
    await ledger.append(table("train", STARTS), str(fence.number), start, fence)
    await ledger.append(table("train", GROUPS), "1", {"task": "t", "decided": 5.0, "episodes": 1}, fence)
    RunFeed(tmp_path / FEED).close()
    endpoints = gateway_endpoints(
        plain_channel(always=[("an answer\n", "stop"), ("8\n", "stop")]), ledger=ledger, blobs=blobs
    )
    endpoints.admit("r_one", Attempt("train", await ledger.take("runs/train/episodes/1/1"), "1/1", 1))
    for slot, trained in (("policy", True), ("judge", False)):
        endpoint = endpoints.endpoint(RecordedModel(channel="policy", trained=trained))
        await endpoint.sample(sample_request([Message.user("Go.")], f"{slot}-0", session_id=f"r_one/{slot}"))
    segments = await endpoints.sessions("train", "r_one")
    assembled = assemble([], segments, run="train", group=1, number=1)
    trajectories = {
        "policy": Trajectory(segments["policy"], {"default": 0.75}),
        "judge": assembled.trajectories["judge"],
    }
    episode = Episode("train", 1, 1, "r_one", {"run": "train"}, Outcome.COMPLETED, trajectories=trajectories)
    record = await stored(episode, [], blobs)
    await ledger.append(table("train", EPISODES), "1/1", record.to_json(), fence)

    shown = await System(tmp_path, FeedReader(tmp_path / FEED)).episode("r_one")
    assert shown["ended"]["untrained"] == ["judge"] and shown["ended"]["reward"] == 0.75
    assert shown["ended"]["slots"] == ["judge", "policy"] and shown["ended"]["sampled"] > 0
    assert [(line["slot"], line["reply"]["text"]) for line in shown["lines"]] == [
        ("policy", "an answer"),
        ("judge", "8"),
    ]
