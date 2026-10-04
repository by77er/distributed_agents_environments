"""Channels over a scripted engine in Qwen's own token format (its tokenizer must be in the local cache)."""

from typing import Any, cast

import pytest

from rollout_qwen import qwen3, qwen35
from rollout_train.inference import Channel, Limits
from rollout_train.recorder.renderers import Tokenizer
from rollout_train.testing import ScriptedEngine

MODEL = "Qwen/Qwen3.5-9B"


def qwen_tokenizer() -> Tokenizer:
    """The tokenizer the renderers are tested against (skips the test if it is not in the local cache)."""
    transformers = pytest.importorskip("transformers")
    try:
        loaded: Any = transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)  # pyright: ignore[reportUnknownMemberType]
    except OSError:
        pytest.skip(f"the {MODEL} tokenizer is not in the local cache")
    return cast(Tokenizer, loaded)


BUDGETS = {"thinking": 1024, "answer": 400}
"""The budgets the scripts are written for, unless a test gives its own: a thought, then its answer (two phases)."""


def channel(engine: ScriptedEngine, *, renderer: str = "qwen3.5", **limits: Any) -> Channel:
    family = {"qwen3.5": qwen35, "qwen3": qwen3}[renderer]
    return Channel("policy", [engine], family(engine.tokenizer), Limits(**(BUDGETS | limits)))
