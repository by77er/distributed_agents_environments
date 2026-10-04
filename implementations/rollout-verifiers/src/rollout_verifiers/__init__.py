"""Prime Intellect's verifiers environments, played through the gateway (docs/implementations/rollout-verifiers.md).

`VerifiersEnvironment` wraps any verifiers taskset as an environment; `rollout_verifiers.environments` holds the ones
this package gives by name (`rollout_verifiers.environments:gsm8k`).
"""

from rollout_verifiers.environment import VerifiersEnvironment, VerifiersProgram, play

__all__ = ["VerifiersEnvironment", "VerifiersProgram", "play"]
