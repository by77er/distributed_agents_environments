"""What a process sets before it talks to a Ray cluster.

- Token authentication: a cluster started where `~/.ray/auth_token` exists checks every caller's token, and Ray's
  clients send it only when told to (`RAY_AUTH_MODE=token`).
- Workers run in the cluster's own environment: Ray is told not to start them through `uv run`, which would build
  each an environment of its own from the project, without its extras.
"""

import os
from pathlib import Path


def prepare() -> None:
    """Set what this process needs to talk to a Ray cluster, unless it is set already."""
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    if (Path.home() / ".ray" / "auth_token").exists():
        os.environ.setdefault("RAY_AUTH_MODE", "token")
