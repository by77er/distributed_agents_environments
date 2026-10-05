"""Who may do what through the ledger service: the platform's token, and pods' tokens.

The service is given one secret, the platform's token. A request that carries it may do everything: the platform's own
roles (runs' drivers, runners, the monitor, the gateway, the reaper) hold it. A pod's token is made from that secret
(`pod_token`): it names the pod and the run it serves, and is signed with the secret, so the service checks it with
nothing but the secret and keeps no list of tokens. A pod's token may:

- read the run's serving records and starts (`runs/RUN/serving`, `runs/RUN/starts`), and those checkpoints (in
  `checkpoints` and `checkpoints/released`) that the run made, that its serving records name, that its channels are
  fixed on, and the checkpoints each of those was trained over;
- write the pod's own beat (`presence.beat` under the pod's name);
- read the pod's own lease (`pods.get` of its name), which says which run holds the pod now, and the token for that
  run (a pod taken by another run reads its new token there).

While a lease names the pod, its token is honoured for reads only if the lease names the token's run: a pod released by
one run and taken by another reads nothing more of the first. Everything else is refused (`Forbidden`).
"""

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass

__all__ = ["PLATFORM", "Forbidden", "Scope", "pod_token", "scope_of"]

PREFIX = "rlp1"
"""What a pod's token begins with."""


class Forbidden(Exception):
    """The token may not do what was asked."""


@dataclass(frozen=True)
class Scope:
    """Whose a token is: the platform's (`pod` None), or a pod's, for the run it serves."""

    pod: str | None = None
    run: str | None = None

    @property
    def platform(self) -> bool:
        return self.pod is None

    @property
    def key(self) -> str:
        """Who it is, as the service keeps answers to retried requests by."""
        return "platform" if self.pod is None else f"pod/{self.pod}/{self.run}"


PLATFORM = Scope()


def _signature(secret: str, claims: bytes) -> str:
    digest = hmac.new(secret.encode(), b"pod-token:" + claims, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def pod_token(secret: str, pod: str, run: str) -> str:
    """A token for pod `pod` serving run `run`, signed with the platform's token `secret`."""
    claims = json.dumps({"pod": pod, "run": run}, separators=(",", ":"), sort_keys=True).encode()
    encoded = base64.urlsafe_b64encode(claims).decode().rstrip("=")
    return f"{PREFIX}.{encoded}.{_signature(secret, claims)}"


def scope_of(token: str | None, secret: str) -> Scope | None:
    """Whose `token` is, checked against the platform's token `secret`; None for a token that is neither the platform's
    nor a pod's signed with it."""
    if not token or not secret:
        return None
    if hmac.compare_digest(token.encode(), secret.encode()):
        return PLATFORM
    prefix, _, rest = token.partition(".")
    encoded, _, signature = rest.partition(".")
    if prefix != PREFIX or not encoded or not signature:
        return None
    try:
        claims = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except ValueError:
        return None
    if not hmac.compare_digest(signature.encode(), _signature(secret, claims).encode()):
        return None
    said = json.loads(claims)
    pod, run = said.get("pod"), said.get("run")
    if not isinstance(pod, str) or not isinstance(run, str) or not pod or not run:
        return None
    return Scope(pod, run)
