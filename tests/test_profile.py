"""A profile, opened: engines, renderer, trainer and tool sets by name, with nothing on a GPU; and the profiles the
documentation and the Minecraft environment give, which must load."""

import re
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rollout.profile import Profile
from rollout.rollouts import binding_for
from rollout.training import Budget, Step, Weighted, iterations, train
from tests import support
from tests.rollouts.games import words

ROOT = Path(__file__).resolve().parent.parent

PROFILE = """
directory = "{directory}"

[channels.policy]
model = "a-checkpoint"
renderer = "tests.support:plain_renderer"
engine = "tests.support:scripted_engine"
thinking_tokens = 64
engines = [{{ device = 0 }}, {{ device = 1 }}]

[channels.judge]
model = "another-checkpoint"
renderer = "tests.support:plain_renderer"
engine = "tests.support:scripted_engine"

[trainer]
kind = "tests.test_profile:Steps"
channel = "policy"
colocated = true
sequence_tokens = 900
sequences_per_step = 3
"""


class Steps:
    """A trainer that trains nothing: what a profile's `[trainer]` names."""

    def __init__(self, model: str, directory: Path, *, sequence_tokens: int, sequences_per_step: int) -> None:
        self.model, self.directory = model, directory
        self.budget = Budget(sequence_tokens, sequences_per_step)
        self.latest: tuple[str, str] | None = None
        self.taken = 0

    async def step(self, batch: Sequence[Weighted], *, seed: int) -> Step:
        self.taken += 1
        self.latest = (f"step-{self.taken}", f"/adapters/step-{self.taken}")
        return Step(*self.latest, {"sequences": float(len(batch))})


def write(tmp_path: Path, text: str = PROFILE) -> Path:
    path = tmp_path / "profile.toml"
    path.write_text(text.format(directory=tmp_path / "run"))
    return path


async def test_an_open_profile_trains_with_what_it_names(tmp_path: Path) -> None:
    support.STARTED.clear()
    profile = Profile.load(write(tmp_path))
    assert profile.trainer is not None and profile.trainer.settings == {"sequence_tokens": 900, "sequences_per_step": 3}
    async with profile.open() as platform:
        policy, judge = platform.channels["policy"], platform.channels["judge"]
        assert len(policy.engines) == 2 and len(judge.engines) == 1  # one engine per entry, each told its own
        assert [engine.told[0] for engine in support.STARTED] == [
            "started a-checkpoint [('device', 0)]",
            "started a-checkpoint [('device', 1)]",
            "started another-checkpoint []",
        ]
        # The trainer's longest sequence is the longest turn of the channel it trains, and of no other.
        assert policy.limits.sequence == 900 and policy.limits.thinking == 64 and judge.limits.sequence is None
        assert platform.trainer is not None and platform.trainer.budget == Budget(900, 3)
        binding = binding_for(words, "policy", platform.tool_bindings)
        await train(platform.jobs, words, platform.trainer, platform.store, channel="policy", groups=2, binding=binding)
        trained = [line for line in iterations(platform.store) if line.update is not None]
        assert trained and policy.version == len(trained) and judge.version == 0  # published to the trained channel
        assert "sleep" in support.STARTED[0].told and "sleep" in support.STARTED[2].told  # colocated: all of them
    assert all(engine.told[-1] == "close" for engine in support.STARTED)
    assert (tmp_path / "run" / "engine.json").exists() and (tmp_path / "run" / "feed" / "_job.jsonl").exists()


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
