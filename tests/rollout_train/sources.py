"""Environments to import, in git repositories the tests make (`repository`): a tiny project (`TINY`) whose environment
says a word, laid out at its root or under `src/`, with a trainer that trains nothing for a run on it; and a version as
an import records one (`a_version`)."""

import hashlib
import subprocess
import textwrap
from collections.abc import Mapping
from pathlib import Path

from rollout_train.published import EnvironmentVersion

ENVIRONMENT = textwrap.dedent('''
    """Say the word: a tiny environment for the tests of importing environments."""

    import random
    from collections.abc import Sequence
    from pathlib import Path

    from rollout.environment import Description, Row, drawn
    from rollout.harness import End, Observation, RunContext, Task, agent_program
    from rollout_train.trainer import WEIGHTS, Budget, Files, Step, Weighted


    class Say(Task):
        async def start(self, run: RunContext) -> Observation:
            return Observation("Say the word.")

        async def respond(self, run: RunContext, reply) -> Observation:
            said = reply.text.strip() == self.parameters["word"]
            await run.emit("result", {"solved": said})
            return End(reward=1.0 if said else 0.0)


    class Words:
        program = agent_program(Say)
        version = "VERSION"
        description = Description()

        def rows(self):
            return [Row(f"say-{word}", f"say {word}", {"word": word}) for word in ("yes", "no")]

        def start(self, row: Row, rng: random.Random):
            return {**row.parameters, "seed": rng.randrange(1000)}

        def evals(self):
            return {"words-eval": drawn(self, seeds=[1, 2])}


    environment = Words()


    class Steps:
        """A trainer that trains nothing."""

        weights = "lora"

        def __init__(self, model: str, *, segment_tokens: int, segments_per_step: int) -> None:
            self.budget = Budget(segment_tokens, segments_per_step)

        async def step(self, batch: Sequence[Weighted], *, seed: int, parent: Files | None, into: Path) -> Step:
            (into / WEIGHTS).mkdir(parents=True)
            (into / WEIGHTS / "adapter.bin").write_text(f"trained on {len(batch)} segments")
            return Step({"segments": float(len(batch))})
''')

PYPROJECT = textwrap.dedent("""
    [project]
    name = "say-the-word"
    version = "0.1.0"
    requires-python = ">=3.13"
    dependencies = DEPENDENCIES

    [project.entry-points."rollout.environments"]
    words = "words:environment"
""")


def tiny(
    *, version: str = "1", dependencies: str = '["rollout", "rollout-train"]', src: bool = False
) -> dict[str, str]:
    """The tiny project's files, by path: its environment's `version`, its `dependencies` (TOML), its package at its
    root or under `src/`."""
    package = "src/words" if src else "words"
    return {
        "pyproject.toml": PYPROJECT.replace("DEPENDENCIES", dependencies),
        f"{package}/__init__.py": ENVIRONMENT.replace("VERSION", version),
        "README.md": "Say the word.\n",
    }


TINY = tiny()


def repository(directory: Path, files: Mapping[str, str], *, under: str = "", branch: str = "main") -> Path:
    """A git repository at `directory` holding `files` (under the subdirectory `under`), committed on `branch`."""
    directory.mkdir(parents=True, exist_ok=True)
    git(directory, "init", "--quiet", f"--initial-branch={branch}")
    commit(directory, files, under=under, message="the environment")
    return directory


def commit(directory: Path, files: Mapping[str, str], *, under: str = "", message: str = "a change") -> str:
    """Write `files` (under `under`) into the repository and commit them: the commit."""
    for path, text in files.items():
        target = directory / under / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    git(directory, "add", "--all")
    git(directory, "-c", "user.name=Tests", "-c", "user.email=tests@example.com", "commit", "--quiet", "-m", message)
    return git(directory, "rev-parse", "HEAD").strip()


def git(directory: Path, *arguments: str) -> str:
    return subprocess.run(["git", "-C", str(directory), *arguments], check=True, capture_output=True, text=True).stdout


def a_version(source: str = "one", *, name: str = "words", imported: float = 1.0) -> EnvironmentVersion:
    """A published version as an import records it, without importing anything: its id the hash of `source`."""
    id = hashlib.sha256(source.encode()).hexdigest()
    return EnvironmentVersion(
        name=name, version=id, source="https://example.com/words.git", ref=None, commit="c0ffee" * 6 + "c0ff",
        subdirectory="", entry_point="words:environment", blob={"uri": "s3://b/k", "sha256": id, "size": 3},
        runtime_env={"working_dir": f"s3://b/{id}.zip"}, dependencies=("rollout",),
        description={"version": "1", "rows": [{"key": "say-yes", "title": "say yes"}], "evals": {"words-eval": []}},
        check=[{"check": "rows", "passed": True, "said": "1 row"}], imported=imported,
    )  # fmt: skip
