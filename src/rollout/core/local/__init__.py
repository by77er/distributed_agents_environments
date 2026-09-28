"""In-process implementations for the local profile: nothing persists, a process crash loses in-flight runs."""

from rollout.core.local.context import LocalRunContext, RewardAssignment

__all__ = ["LocalRunContext", "RewardAssignment"]
