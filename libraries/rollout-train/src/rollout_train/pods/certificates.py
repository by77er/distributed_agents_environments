"""The cluster's certificate authority, step-ca, for the pods a cluster rents: one-time tokens for pods' first
certificates, and revoking a pod's certificate.

A pod gets its first certificate with a one-time token minted before the pod is asked for: a JWT signed by one of
step-ca's JWK provisioners, naming the pod's identity (`spiffe://rollout/pod/NAME`) as its subject and its only SAN,
good for 15 minutes, and once (step-ca keeps the token's id). The pod makes its key itself, sends a certificate request
with the token (`step ca certificate`), and gets a certificate for that identity and no other, good for 24 hours. From
then on the pod renews it itself, over mutual TLS with the certificate it has (`step ca renew --daemon`), at about two
thirds of its life.

step-ca renews any certificate it has not revoked, so which pods are the cluster's own reaches the CA as revocations:
when a pod is stopped or deleted, or is no longer counted as the cluster's, the certificate its beats name is revoked
(`revoke`). Revocation in step-ca is passive: the certificate is not renewed, and lapses within a day. A pod that is
gone cannot renew anyway, so its certificate lapses all the same.

A certificate for a client of the platform's own (the gateway's, `spiffe://rollout/gateway`) is issued here the same
way (`certificate`): a key made here, a request for the identity, a one-time token. `decrypted_key` reads a JWK
provisioner's private key from the `encryptedKey` step-ca keeps in its configuration, with the password it was made
with (what `step crypto jwe decrypt` does).

The token is what `step ca token` makes: a JWT (`ES256`, the provisioner's key id as `kid`) whose claims are the
provisioner's name (`iss`), the CA's sign or revoke endpoint (`aud`), the subject (`sub`), the SANs (`sans`), the root
certificate's SHA-256 fingerprint (`sha`; not in a pod's token, whose step then checks step-ca's TLS by the roots it is
given rather than that root alone), a random id (`jti`) and its times. The provisioner's private key is a JWK
(EC P-256), the decrypted form of the `encryptedKey` in step-ca's configuration
(`step crypto jwe decrypt < encrypted.json > provisioner.jwk`), kept as a secret file.
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
    key is `key` (a JWK, EC P-256). Its own TLS is checked by the root; with `system`, by the system's roots (behind a
    proxy that ends TLS with a public certificate, such as a Cloudflare Tunnel)."""

    def __init__(
        self,
        url: str,
        *,
        provisioner: str,
        key: Mapping[str, Any],
        root: bytes,
        client: httpx.AsyncClient | None = None,
        system: bool = False,
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
        self.system = system

    def __repr__(self) -> str:
        return f"StepCa(url={self.url!r}, provisioner={self.provisioner!r})"

    @classmethod
    def from_files(cls, url: str, *, provisioner: str, key: Path, root: Path, system: bool = False) -> "StepCa":
        """With the provisioner's key and the root certificate read from files."""
        return cls(url, provisioner=provisioner, key=json.loads(key.read_text()), root=root.read_bytes(), system=system)

    def token(
        self, subject: str, sans: Sequence[str] | None = None, *, audience: str = "sign",
        lifetime: float = TOKEN_LIFETIME, now: float | None = None, pinned: bool = True,
    ) -> str:  # fmt: skip
        """A one-time token for a certificate of `subject` (with `sans`, by default the subject alone), good for
        `lifetime` seconds; with `audience = "revoke"`, for revoking the certificate whose serial `subject` is.
        `pinned` names the root's fingerprint in it (`sha`): step then trusts only that root for step-ca's own TLS,
        whatever `--root` says, which fails behind a proxy that ends TLS with a public certificate."""
        issued = int(time.time() if now is None else now)
        claims: dict[str, Any] = {
            "iss": self.provisioner, "aud": f"{self.url}/1.0/{audience}", "sub": subject, "iat": issued,
            "nbf": issued, "exp": issued + int(lifetime), "jti": secrets.token_hex(32),
        }  # fmt: skip
        if pinned:
            claims["sha"] = self.fingerprint
        if audience == "sign":
            claims["sans"] = list(sans if sans is not None else [subject])
        header = {"alg": "ES256", "kid": self._kid, "typ": "JWT"}
        signing = f"{_encoded(json.dumps(header).encode())}.{_encoded(json.dumps(claims).encode())}"
        r, s = utils.decode_dss_signature(self._key.sign(signing.encode(), ec.ECDSA(hashes.SHA256())))
        return f"{signing}.{_encoded(r.to_bytes(32) + s.to_bytes(32))}"

    def pod_token(self, identity: str, *, lifetime: float = TOKEN_LIFETIME) -> str:
        """The one-time token a pod gets its first certificate with: its identity as the subject and the only SAN. It
        names no root (`pinned` false): the pod is given the cluster's root and checks step-ca's TLS by its `--root`."""
        return self.token(identity, [identity], lifetime=lifetime, pinned=False)

    async def certificate(self, identity: str, *, lifetime: float | None = None) -> tuple[bytes, bytes]:
        """A certificate for `identity` (its subject and its only URI SAN), from a key made here: the certificate with
        its chain, and the key (both PEM). `lifetime` asks for fewer seconds than the provisioner's default."""
        from cryptography.x509.oid import NameOID

        key = ec.generate_private_key(ec.SECP256R1())
        request = (
            x509.CertificateSigningRequestBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, identity)]))
            .add_extension(x509.SubjectAlternativeName([x509.UniformResourceIdentifier(identity)]), critical=False)
            .sign(key, hashes.SHA256())
        )
        body: dict[str, Any] = {"csr": request.public_bytes(serialization.Encoding.PEM).decode(),
                                "ott": self.token(identity, [identity])}  # fmt: skip
        if lifetime is not None:
            body["notAfter"] = f"{int(lifetime)}s"
        response = await self._client().post(f"{self.url}/1.0/sign", json=body)
        if response.status_code >= 400:
            raise RuntimeError(f"step-ca refused a certificate for {identity}: {response.status_code} "
                               f"{_message(response)}")  # fmt: skip
        said: Any = response.json()
        chain = cast(list[str], said.get("certChain") or [said["crt"], said.get("ca") or ""])
        pem = "".join(each if each.endswith("\n") else f"{each}\n" for each in chain if each)
        private = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption())  # fmt: skip
        return pem.encode(), private

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

            context = ssl.create_default_context(cadata=None if self.system else self.root.decode())
            self._http = httpx.AsyncClient(timeout=30.0, verify=context)
        return self._http


def decrypted_key(encrypted: str, password: str) -> dict[str, Any]:
    """A JWK provisioner's private key from its `encryptedKey` (a JWE, compact, encrypted with a password: PBES2 key
    wrapping and AES-GCM content, as step makes it) and the password. Raises `ValueError` for a wrong password or a
    JWE it does not read."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.keywrap import InvalidUnwrap, aes_key_unwrap

    try:
        protected, wrapped, iv, ciphertext, tag = encrypted.strip().split(".")
    except ValueError:
        raise ValueError("an encrypted key is a JWE in five parts") from None
    header: Any = json.loads(_unpadded(protected))
    sizes = {"PBES2-HS256+A128KW": (hashes.SHA256(), 16), "PBES2-HS384+A192KW": (hashes.SHA384(), 24),
             "PBES2-HS512+A256KW": (hashes.SHA512(), 32)}  # fmt: skip
    if header.get("alg") not in sizes or not str(header.get("enc", "")).endswith("GCM"):
        raise ValueError(f"an encrypted key is PBES2 and AES-GCM, not {header.get('alg')} and {header.get('enc')}")
    digest, length = sizes[str(header["alg"])]
    salt = str(header["alg"]).encode() + b"\x00" + _unpadded(str(header["p2s"]))
    derived = PBKDF2HMAC(digest, length, salt, int(header["p2c"])).derive(password.encode())
    try:
        content = aes_key_unwrap(derived, _unpadded(wrapped))
        plain = AESGCM(content).decrypt(_unpadded(iv), _unpadded(ciphertext) + _unpadded(tag), protected.encode())
    except (InvalidUnwrap, ValueError, KeyError, TypeError, InvalidTag) as error:
        raise ValueError(f"the password does not open the key ({type(error).__name__})") from None
    return cast(dict[str, Any], json.loads(plain))


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
