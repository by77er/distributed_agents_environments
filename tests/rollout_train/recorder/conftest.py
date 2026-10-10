"""`test_recorder.py` records through Qwen's renderers, which come with the workspace's `gpu` and `tinker` extras:
without them it is not collected."""

import importlib.util

collect_ignore = [] if importlib.util.find_spec("rollout_qwen") else ["test_recorder.py"]
