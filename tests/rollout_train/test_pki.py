"""The certificates the platform holds: from a step-ca's state (its configuration and root) and its provisioner's
password, the root and the provisioner's key are published, and a certificate for the gateway's identity is made and
published with its key, as Secrets (here, to a stand-in of the API server)."""

import json
from pathlib import Path
from typing import Any

import pytest
from cryptography import x509

from rollout_train.pki import provisioner_key, publish
from rollout_train.pods import GATEWAY_IDENTITY
from tests.rollout_runpod.test_certificates import PROVISIONER, FakeStepCa, encrypted
from tests.rollout_runpod.test_certificates import provisioner_key as made_key
from tests.rollout_train.pods.authority import Authority, served_tls, server_context


class Secrets:
    """The API server, as far as Secrets go."""

    def __init__(self) -> None:
        self.written: dict[str, dict[str, bytes]] = {}

    async def put_secret(self, namespace: str, name: str, data: dict[str, bytes]) -> None:
        self.written[f"{namespace}/{name}"] = dict(data)


async def test_the_root_the_provisioners_key_and_the_gateways_certificate_are_published(tmp_path: Path) -> None:
    authority = Authority(tmp_path / "ca")
    jwk, public = made_key()
    directory = tmp_path / "step"
    (directory / "config").mkdir(parents=True)
    (directory / "certs").mkdir()
    (directory / "certs" / "root_ca.crt").write_bytes(authority.pem)
    configuration: dict[str, Any] = {"authority": {"provisioners": [
        {"type": "ACME", "name": "acme"},
        {"type": "JWK", "name": PROVISIONER, "key": {"kid": jwk["kid"]}, "encryptedKey": encrypted(jwk, "password")},
    ]}}  # fmt: skip
    (directory / "config" / "ca.json").write_text(json.dumps(configuration))
    assert provisioner_key(configuration, "password") == (PROVISIONER, jwk)
    with pytest.raises(ValueError, match="no JWK provisioner other"):
        provisioner_key(configuration, "password", "other")
    fake = FakeStepCa(authority, public, jwk["kid"])
    tls = authority.issue("step-ca", None, ips=["127.0.0.1"])
    secrets = Secrets()
    async with served_tls(fake.app(), server_context(authority, tls, client=None)) as url:
        fake.url = url
        said = await publish(directory, "password", url, "rollout", api=secrets)
    assert "spiffe://rollout/gateway" in said
    root = secrets.written["rollout/step-ca"]
    assert root["root_ca.crt"] == authority.pem and json.loads(root["provisioner.jwk"]) == jwk
    gateway = secrets.written["rollout/gateway-tls"]
    issued = x509.load_pem_x509_certificates(gateway["tls.crt"])[0]
    sans = issued.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert sans.get_values_for_type(x509.UniformResourceIdentifier) == [GATEWAY_IDENTITY]
    assert gateway["ca.crt"] == authority.pem and gateway["tls.key"].startswith(b"-----BEGIN PRIVATE KEY")
