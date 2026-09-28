"""The developer guide stays true: its examples run, and the API reference matches the code."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GUIDE = ROOT / "docs" / "guide"
PAGES = sorted(page for page in GUIDE.glob("*.md") if page.name != "reference.md")
RUNNABLE_BLOCK = re.compile(r"^```python\n(.*?)^```$", re.MULTILINE | re.DOTALL)
"""Only blocks tagged exactly `python` run; a fragment is tagged `python fragment`."""


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_guide_examples_run(page: Path) -> None:
    code = "\n".join(RUNNABLE_BLOCK.findall(page.read_text()))
    if not code:
        pytest.skip("no runnable examples")
    exec(compile(code, str(page), "exec"), {"__name__": "__main__"})


def test_reference_is_current() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/generate_reference.py", "--check"], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
