"""vLLM as an engine: an implementation of `rollout_train.inference.Engine`.

A `vllm` inference provider of the cluster config runs it in each engine host (`rollout_train.inference.hosts`),
made with the model and that model's `options` (`[inference.NAME.models."MODEL"] options`).
"""

from rollout_vllm.engine import VllmEngine

__all__ = ["VllmEngine"]
