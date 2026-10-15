"""The developer guide stays true: its examples run, and the API reference matches the code."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GUIDE = ROOT / "docs" / "guide"
PAGES = sorted(page for page in GUIDE.glob("*.md") if page.name != "reference.md")
RUNNABLE_BLOCK = re.compile(r"^```python\n(.*?)^```$", re.MULTILINE | re.DOTALL)
"""Only blocks tagged exactly `python` run; a fragment is tagged `py` (highlighted the same, not run)."""
EXTRAS = {"cluster.md": ("rollout_tinker",)}
"""What a page's examples need of the workspace's extras: without it, they are not run (a run on Tinker is checked
with its trainer's settings, which `rollout_tinker` declares)."""
REFERENCED = ("torch", "rollout_vllm", "rollout_lora", "rollout_qwen", "rollout_gemma", "rollout_tinker")
"""What the reference is generated from, beyond what `uv sync` installs: every implementation's public modules."""


def missing(modules: tuple[str, ...]) -> list[str]:
    return [module for module in modules if importlib.util.find_spec(module) is None]


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_guide_examples_run(page: Path) -> None:
    code = "\n".join(RUNNABLE_BLOCK.findall(page.read_text()))
    if not code:
        pytest.skip("no runnable examples")
    if absent := missing(EXTRAS.get(page.name, ())):
        pytest.skip(f"its examples need {', '.join(absent)}, which the workspace's extras install")
    exec(compile(code, str(page), "exec"), {"__name__": "__main__"})


def test_reference_is_current() -> None:
    if absent := missing(REFERENCED):
        pytest.skip(f"the reference is generated with {', '.join(absent)}, which the workspace's extras install")
    result = subprocess.run(
        [sys.executable, "scripts/generate_reference.py", "--check"], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
