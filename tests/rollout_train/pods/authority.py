"""A certificate authority for tests, and servers over TLS: what step-ca and Envoy give a deployment, made here with
`cryptography` and uvicorn so that mutual TLS is tested with no network, no Docker and no step-ca.

`Authority` signs leaf certificates whose one URI SAN is an identity (`spiffe://rollout/pod/NAME`,
`spiffe://rollout/gateway`), for server and client use alike, as step-ca does; `served_tls` serves an app with any
`ssl.SSLContext` (one that asks for a client certificate, and one whose identity it requires)."""

import asyncio
import contextlib
import datetime
import ipaddress
import ssl
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from rollout_train.inference.remote import requiring


@dataclass(frozen=True)
class Issued:
    """A certificate and its key, as files."""

    certificate: Path
    key: Path


def _pem(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )


class Authority:
    """A root certificate authority whose files are under `directory`."""

    def __init__(self, directory: Path, name: str = "rollout test root") -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.key = ec.generate_private_key(ec.SECP256R1())
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
        now = datetime.datetime.now(datetime.UTC)
        self.certificate = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(self.key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(self.key.public_key()), critical=False)
            .sign(self.key, hashes.SHA256())
        )
        self.root = directory / "root.crt"
        self.root.write_bytes(self.pem)

    @property
    def pem(self) -> bytes:
        return self.certificate.public_bytes(serialization.Encoding.PEM)

    def signed(
        self,
        public: Any,
        subject: str,
        identities: Sequence[str],
        *,
        lifetime: datetime.timedelta = datetime.timedelta(hours=24),
        ips: Sequence[str] = (),
    ) -> x509.Certificate:
        """A leaf certificate for `public`, its SANs the URIs `identities` (and the addresses `ips`)."""
        now = datetime.datetime.now(datetime.UTC)
        names: list[x509.GeneralName] = [x509.UniformResourceIdentifier(each) for each in identities]
        names += [x509.IPAddress(ipaddress.ip_address(each)) for each in ips]
        return (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)]))
            .issuer_name(self.certificate.subject)
            .public_key(public)
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1))
            .not_valid_after(now + lifetime)
            .add_extension(x509.SubjectAlternativeName(names), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(public), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(self.key.public_key()), critical=False)
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]),
                critical=False,
            )
            .sign(self.key, hashes.SHA256())
        )

    def issue(self, name: str, identity: str | None, *, ips: Sequence[str] = ()) -> Issued:
        """A certificate and key for `identity` (none: a certificate with no URI), written as `name.crt`, `name.key`."""
        key = ec.generate_private_key(ec.SECP256R1())
        certificate = self.signed(key.public_key(), name, [identity] if identity else [], ips=ips)
        issued = Issued(self.directory / f"{name}.crt", self.directory / f"{name}.key")
        issued.certificate.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
        issued.key.write_bytes(_pem(key))
        return issued


def server_context(authority: Authority, issued: Issued, *, client: str | None) -> ssl.SSLContext:
    """What a pod's proxy does: serve `issued`, and with `client`, take only a client certificate from `authority` that
    carries that identity."""
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH, cafile=str(authority.root))
    context.load_cert_chain(str(issued.certificate), str(issued.key))
    if client is not None:
        context.verify_mode = ssl.CERT_REQUIRED
        requiring(context, client)
    return context


@contextlib.asynccontextmanager
async def served_tls(app: Any, context: ssl.SSLContext | None = None) -> AsyncGenerator[str]:
    """`app` served on 127.0.0.1, on a free port, over TLS with `context` (plain HTTP with none): its base URL."""
    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="critical", lifespan="off")
    config.load()
    config.ssl = context
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
        if task.done():
            task.result()
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"{'https' if context else 'http'}://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task
