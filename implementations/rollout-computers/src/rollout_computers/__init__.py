"""Environment backends: services that give runs computers (docs/environments/README.md).

`rollout.core` defines what task code sees (`run.environments`, `Environment`); a backend here implements the
`EnvironmentService` a runner is given. The namespace backend isolates processes and files; the local backend
runs commands on the host itself, for uses where a sandbox matters less than convenience. MicroVMs come later behind
the same protocol.
"""

from rollout_computers.images import ImageStore
from rollout_computers.local import LocalEnvironments
from rollout_computers.namespaces import NamespaceEnvironments

__all__ = ["ImageStore", "LocalEnvironments", "NamespaceEnvironments"]
