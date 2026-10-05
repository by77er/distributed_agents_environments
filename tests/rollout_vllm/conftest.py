"""The vLLM engine's tests need vLLM and torch, which come with the workspace's `gpu` extra: without them they are not
collected."""

import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("vllm") and importlib.util.find_spec("torch") else ["test_*.py"]
