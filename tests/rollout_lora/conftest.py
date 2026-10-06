# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""The trainer's tests need torch, which comes with the workspace's `gpu` extra: without it they are not collected.
`tiny` is the model the processes' tests train on the CPU."""

import importlib.util
from typing import Any

import pytest

collect_ignore_glob = [] if importlib.util.find_spec("torch") else ["test_*.py"]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A random Qwen3 of two layers, saved as a model's directory."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=False,
                         max_position_embeddings=256)  # fmt: skip
    torch.manual_seed(0)
    directory = tmp_path_factory.mktemp("tiny")
    model: Any = Qwen3ForCausalLM(config)
    model.to(torch.bfloat16).save_pretrained(directory)
    return str(directory)
