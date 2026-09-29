"""Environment backends: services that give runs computers (docs/environments/README.md).

`rollout.core` defines what task code sees (`run.environments`, `Environment`); a backend here implements the
`EnvironmentService` a runner is given. The namespace backend is the first; microVMs come later behind the same
protocol.
"""

from rollout.environments.images import ImageStore
from rollout.environments.namespaces import NamespaceEnvironments

__all__ = ["ImageStore", "NamespaceEnvironments"]
