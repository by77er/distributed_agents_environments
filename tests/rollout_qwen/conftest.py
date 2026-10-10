"""Qwen's renderers come with the workspace's `gpu` and `tinker` extras: without them these tests are not collected."""

import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("rollout_qwen") else ["test_*.py"]
