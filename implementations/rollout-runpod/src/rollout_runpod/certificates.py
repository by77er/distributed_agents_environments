"""The cluster's certificate authority, step-ca, as a launcher uses it: one-time tokens for pods' first certificates,
and revoking a pod's certificate.

A pod gets its first certificate with a one-time token the launcher mints before it asks for the pod: a JWT signed by
one of step-ca's JWK provisioners, naming the pod's identity (`spiffe://rollout/pod/NAME`) as its subject and its only
SAN, good for 15 minutes, and once (step-ca keeps the token's id). The pod makes its key itself, sends a certificate
request with the token (`step ca certificate`), and gets a certificate for that identity and no other, good for 24
hours. From then on the pod renews it itself, over mutual TLS with the certificate it has (`step ca renew --daemon`),
at about two thirds of its life.

step-ca renews any certificate it has not revoked, so the launcher's account of which pods are its own reaches the CA
as revocations: when a pod is stopped or deleted, or is no longer one the launcher counts as its own, the launcher
revokes the certificate its beats name (`revoke`). Revocation in step-ca is passive: the certificate is not renewed, and
lapses within a day. A pod that is gone cannot renew anyway, so its certificate lapses all the same.

The token is what `step ca token` makes: a JWT (`ES256`, the provisioner's key id as `kid`) whose claims are the
provisioner's name (`iss`), the CA's sign or revoke endpoint (`aud`), the subject (`sub`), the SANs (`sans`), the root
certificate's SHA-256 fingerprint (`sha`), a random id (`jti`) and its times. The provisioner's private key is a JWK
(EC P-256), the decrypted form of the `encryptedKey` in step-ca's configuration
(`step crypto jwe decrypt < encrypted.json > provisioner.jwk`), kept as a secret file of the launcher's.
"""

import base64
import hashlib
import json
import secrets
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import httpx
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils

TOKEN_LIFETIME = 15 * 60
"""Seconds a one-time token is good for."""


def fingerprint(root: bytes) -> str:
    """The SHA-256 fingerprint of a certificate (PEM), in lowercase hexadecimal: what step-ca's clients trust a root
    by."""
    certificate = x509.load_pem_x509_certificate(root)
    return hashlib.sha256(certificate.public_bytes(serialization.Encoding.DER)).hexdigest()


class StepCa:
    """step-ca at `url`, whose root certificate is `root` (PEM), with the JWK provisioner `provisioner` whose private
    key is `key` (a JWK, EC P-256)."""

    def __init__(
        self,
        url: str,
        *,
        provisioner: str,
        key: Mapping[str, Any],
        root: bytes,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if key.get("kty") != "EC" or key.get("crv") != "P-256" or "d" not in key:
            raise ValueError("the provisioner's key must be a private EC P-256 JWK")
        self.url = url.rstrip("/")
        self.provisioner = provisioner
        self.root = root
        self.fingerprint = fingerprint(root)
        self._key = ec.derive_private_key(int.from_bytes(_unpadded(str(key["d"]))), ec.SECP256R1())
        self._kid = str(key.get("kid") or _thumbprint(key))
        self._http = client
        self._owned = client is None

    def __repr__(self) -> str:
        return f"StepCa(url={self.url!r}, provisioner={self.provisioner!r})"

    @classmethod
    def from_files(cls, url: str, *, provisioner: str, key: Path, root: Path) -> "StepCa":
        """With the provisioner's key and the root certificate read from files."""
        return cls(url, provisioner=provisioner, key=json.loads(key.read_text()), root=root.read_bytes())

    def token(
        self, subject: str, sans: Sequence[str] | None = None, *, audience: str = "sign",
        lifetime: float = TOKEN_LIFETIME, now: float | None = None,
    ) -> str:  # fmt: skip
        """A one-time token for a certificate of `subject` (with `sans`, by default the subject alone), good for
        `lifetime` seconds; with `audience = "revoke"`, for revoking the certificate whose serial `subject` is."""
        issued = int(time.time() if now is None else now)
        claims: dict[str, Any] = {
            "iss": self.provisioner, "aud": f"{self.url}/1.0/{audience}", "sub": subject, "iat": issued,
            "nbf": issued, "exp": issued + int(lifetime), "jti": secrets.token_hex(32), "sha": self.fingerprint,
        }  # fmt: skip
        if audience == "sign":
            claims["sans"] = list(sans if sans is not None else [subject])
        header = {"alg": "ES256", "kid": self._kid, "typ": "JWT"}
        signing = f"{_encoded(json.dumps(header).encode())}.{_encoded(json.dumps(claims).encode())}"
        r, s = utils.decode_dss_signature(self._key.sign(signing.encode(), ec.ECDSA(hashes.SHA256())))
        return f"{signing}.{_encoded(r.to_bytes(32) + s.to_bytes(32))}"

    def pod_token(self, identity: str, *, lifetime: float = TOKEN_LIFETIME) -> str:
        """The one-time token a pod gets its first certificate with: its identity as the subject and the only SAN."""
        return self.token(identity, [identity], lifetime=lifetime)

    async def revoke(self, serial: str, *, reason: str = "") -> None:
        """Revoke the certificate whose serial is `serial` (decimal), so that it is not renewed (passive revocation)."""
        body = {"serial": serial, "ott": self.token(serial, audience="revoke"), "passive": True, "reason": reason}
        response = await self._client().post(f"{self.url}/1.0/revoke", json=body)
        if response.status_code >= 400:
            raise RuntimeError(f"step-ca refused to revoke {serial}: {response.status_code} {_message(response)}")

    async def aclose(self) -> None:
        if self._owned and self._http is not None:
            await self._http.aclose()

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            import ssl

            context = ssl.create_default_context(cadata=self.root.decode())
            self._http = httpx.AsyncClient(timeout=30.0, verify=context)
        return self._http


def _encoded(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unpadded(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _thumbprint(key: Mapping[str, Any]) -> str:
    """A JWK's thumbprint (RFC 7638): the key id step gives a provisioner's key."""
    members = {name: key[name] for name in ("crv", "kty", "x", "y")}
    return _encoded(hashlib.sha256(json.dumps(members, separators=(",", ":"), sort_keys=True).encode()).digest())


def _message(response: httpx.Response) -> str:
    try:
        said: Any = response.json()
    except ValueError:
        return response.text[:300]
    return str(cast(dict[str, Any], said).get("message") if isinstance(said, dict) else said)[:300]
