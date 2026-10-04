"""The Tinker trainer's and engine's tests need Tinker's SDK, which comes with the workspace's `tinker` extra: without
it they are not collected."""

import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("tinker") else ["test_*.py"]
