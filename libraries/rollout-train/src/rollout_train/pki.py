"""The certificates the platform itself holds, from the cluster's step-ca: what its gateway and runs' drivers reach pods
with, and what they mint pods' one-time tokens with.

`rollout pki publish` (the chart's CronJob beside its step-ca, every six hours and once at each install) reads step-ca's
configuration and root from its volume (`--ca-directory`), opens its JWK provisioner's key with the provisioner's
password (`STEP_CA_PASSWORD`), and writes two Secrets through the API server:

- `step-ca` (`--secret`): `root_ca.crt`, the cluster's root, and `provisioner.jwk`, the provisioner's key, which a run's
  driver signs its pods' one-time tokens with (the cluster config's `step_ca` names both files where they are mounted);
- `gateway-tls` (`--tls-secret`): `tls.crt`, `tls.key` and `ca.crt`, a certificate for `spiffe://rollout/gateway` good
  for 24 hours (the provisioner's default), made anew each time, which the gateway and runs' drivers present to pods
  (the cluster config's `[tls]` names the three files where they are mounted). A Secret mounted as a volume is updated
  in place, so each process reads the newest without starting again.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from rollout_train.pods.identity import GATEWAY_IDENTITY

__all__ = ["provisioner_key", "publish"]


def provisioner_key(
    configuration: Mapping[str, Any], password: str, name: str | None = None
) -> tuple[str, dict[str, Any]]:
    """The name and private key (a JWK) of step-ca's JWK provisioner `name` (by default its first), from its
    configuration (`ca.json`) and the provisioner's password. Raises `ValueError` where there is none, or the password
    does not open its key."""
    from rollout_runpod import decrypted_key

    listed: Any = cast(dict[str, Any], configuration.get("authority") or {}).get("provisioners") or []
    for each in cast(list[dict[str, Any]], listed):
        if each.get("type") == "JWK" and (name is None or each.get("name") == name) and each.get("encryptedKey"):
            return str(each["name"]), decrypted_key(str(each["encryptedKey"]), password)
    raise ValueError(f"step-ca has no JWK provisioner {name or ''} with an encrypted key".replace("  ", " "))


async def publish(
    directory: Path,
    password: str,
    url: str,
    namespace: str,
    *,
    secret: str = "step-ca",
    tls_secret: str = "gateway-tls",
    provisioner: str | None = None,
    api: Any = None,
) -> str:
    """Write the root and the provisioner's key (`secret`), and a new gateway certificate (`tls_secret`), from the
    step-ca whose state is in `directory` and which answers at `url`; what it did, in words."""
    from rollout_runpod import StepCa
    from rollout_train.submitting import KubernetesApi

    configuration = json.loads((directory / "config" / "ca.json").read_text())
    root = (directory / "certs" / "root_ca.crt").read_bytes()
    name, key = provisioner_key(configuration, password, provisioner)
    authority = StepCa(url, provisioner=name, key=key, root=root)
    try:
        chain, private = await authority.certificate(GATEWAY_IDENTITY)
    finally:
        await authority.aclose()
    given = api if api is not None else KubernetesApi()
    await given.put_secret(namespace, secret, {"root_ca.crt": root, "provisioner.jwk": json.dumps(key).encode()})
    await given.put_secret(namespace, tls_secret, {"tls.crt": chain, "tls.key": private, "ca.crt": root})
    return f"published the root and {name}'s key ({secret}) and a certificate for {GATEWAY_IDENTITY} ({tls_secret})"
