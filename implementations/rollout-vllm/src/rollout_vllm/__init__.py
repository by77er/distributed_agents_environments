"""vLLM as an engine: an implementation of `rollout_train.inference.Engine`.

A profile names it for a channel (`engine = "rollout_vllm:VllmEngine"`); each entry of the channel's `engines` is
one replica's options.
"""

from rollout_vllm.engine import VllmEngine

__all__ = ["VllmEngine"]
