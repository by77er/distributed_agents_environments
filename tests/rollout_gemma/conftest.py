"""Gemma's renderer comes with the workspace's `gemma` extra: without it these tests are not collected."""

import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("rollout_gemma") else ["test_*.py"]
