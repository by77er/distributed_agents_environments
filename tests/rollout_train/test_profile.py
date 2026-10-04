"""A profile, opened: engines, renderer, trainer and tool sets by name, with nothing on a GPU; and the profiles the
documentation and the Minecraft environment give, which must load."""

import re
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rollout.catalog import binding_for
from rollout_train import Budget, Checkpoint, Step, Weighted, train
from rollout_train import testing as support
from rollout_train.profile import Profile
from rollout_train.trainer import WEIGHTS
from tests.rollout_train.rollouts.games import words

ROOT = Path(__file__).resolve().parents[2]

PROFILE = """
directory = "{directory}"

[channels.policy]
model = "a-checkpoint"
renderer = "rollout_train.testing:plain_renderer"
engine = "rollout_train.testing:scripted_engine"
thinking_tokens = 64
engines = [{{ device = 0 }}, {{ device = 1 }}]

[channels.judge]
model = "another-checkpoint"
renderer = "rollout_train.testing:plain_renderer"
engine = "rollout_train.testing:scripted_engine"

[trainer]
kind = "tests.rollout_train.test_profile:Steps"
channel = "policy"
colocated = true
segment_tokens = 900
segments_per_step = 3
"""


class Steps:
    """A trainer that trains nothing: what a profile's `[trainer]` names."""

    def __init__(self, model: str, *, segment_tokens: int, segments_per_step: int) -> None:
        self.model = model
        self.budget = Budget(segment_tokens, segments_per_step)

    async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Checkpoint | None, into: Path) -> Step:
        (into / WEIGHTS).mkdir(parents=True)
        (into / WEIGHTS / "adapter.bin").write_text(f"trained on {len(batch)} segments")
        return Step({"segments": float(len(batch))})


def write(tmp_path: Path, text: str = PROFILE) -> Path:
    path = tmp_path / "profile.toml"
    path.write_text(text.format(directory=tmp_path / "run"))
    return path


async def test_an_open_profile_trains_with_what_it_names(tmp_path: Path) -> None:
    support.STARTED.clear()
    profile = Profile.load(write(tmp_path))
    assert profile.trainer is not None and profile.trainer.settings == {"segment_tokens": 900, "segments_per_step": 3}
    async with profile.open() as platform:
        policy, judge = platform.channels["policy"], platform.channels["judge"]
        assert len(policy.engines) == 2 and len(judge.engines) == 1  # one engine per entry, each told its own
        assert [engine.told[0] for engine in support.STARTED] == [
            "started a-checkpoint [('device', 0)]",
            "started a-checkpoint [('device', 1)]",
            "started another-checkpoint []",
        ]
        # The trainer's longest segment is the longest turn of the channel it trains, and of no other.
        assert policy.limits.sequence == 900 and policy.limits.thinking == 64 and judge.limits.sequence is None
        assert platform.trainer is not None and platform.trainer.budget == Budget(900, 3)
        binding = binding_for(words, "policy", platform.tool_bindings)
        assert platform.policy == "run"  # (the run directory's name, unless the profile names the policy)
        await train(
            words,
            platform.trainer,
            platform.policies,
            policy=platform.policy,
            channel="policy",
            directory=tmp_path / "run" / "versions",
            publish=platform.publish,
            run="run",
            groups=2,
            binding=binding,
        )  # fmt: skip  (the platform's runner plays the run named after its directory)
        versions = await platform.policies.versions("run")
        steps = await platform.ledger.read("runs/run/steps")
        played = await platform.ledger.read("runs/run/episodes")
        assert versions and [version.number for version in versions] == list(range(1, len(steps) + 1))
        assert policy.adapter == versions[-1].name and judge.version == 0  # served on the trained channel only
        assert "sleep" in support.STARTED[0].told and "sleep" in support.STARTED[2].told  # colocated: all of them
    assert all(engine.told[-1] == "close" for engine in support.STARTED)
    assert (tmp_path / "run" / "engine.json").exists() and (tmp_path / "run" / "feed" / "_notes.jsonl").exists()
    assert len(played) == 8 and any((tmp_path / "run" / "blobs").iterdir())  # every episode, and its trajectories


async def test_a_profile_that_cannot_start_stops_what_it_started(tmp_path: Path) -> None:
    support.STARTED.clear()
    broken = PROFILE.replace("{{ device = 1 }}", "{{ fails = true }}")
    with pytest.raises(RuntimeError, match="no such device"):
        async with Profile.load(write(tmp_path, broken)).open():
            pass
    (started,) = support.STARTED
    assert started.told[-1] == "close"


def test_a_key_a_profile_does_not_have_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="memory has no run_gib"):
        Profile.load(write(tmp_path, PROFILE + "\n[memory]\nrun_gib = 6\n"))
    with pytest.raises(ValueError, match=r"channels\.policy has no gpu_share"):
        Profile.load(write(tmp_path, PROFILE.replace("thinking_tokens = 64", "gpu_share = 0.5")))


def described() -> list[Any]:
    """Every profile the repository gives: the Minecraft environment's, and the ones the guide shows."""
    files = sorted((ROOT / "environments").glob("*/profiles/*.toml"))
    found = [pytest.param(path.read_text(), id=str(path.relative_to(ROOT))) for path in files]
    for page in sorted((ROOT / "docs" / "guide").glob("*.md")):
        blocks = re.findall(r"^```toml\n(.*?)^```$", page.read_text(), re.MULTILINE | re.DOTALL)
        found += [pytest.param(block, id=f"{page.name} #{index}") for index, block in enumerate(blocks, start=1)]
    return found


@pytest.mark.parametrize("text", described())
def test_the_profiles_the_repository_gives_load(text: str, tmp_path: Path) -> None:
    path = tmp_path / "profile.toml"
    path.write_text(text)
    profile = Profile.load(path)
    assert profile.channels and tomllib.loads(text)["directory"]
    for channel in profile.channels.values():
        assert ":" in channel.engine and ":" in channel.renderer
