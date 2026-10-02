"""The trainer's tests need torch, which comes with the workspace's `gpu` extra: without it they are not collected."""

import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("torch") else ["test_*.py"]
