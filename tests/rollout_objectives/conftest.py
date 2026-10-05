"""The objectives' tests need torch, which comes with the workspace's `gpu` and `tinker` extras: without it they are not
collected."""

import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("torch") else ["test_*.py"]
