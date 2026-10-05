"""step-ca as a launcher uses it, against a fake of step-ca served here over TLS: a one-time token gets one pod its
first certificate (for its identity, once, within 15 minutes), and that certificate is what the gateway reaches the pod
by; a revoked certificate's serial is sent for passive revocation. The fake checks a token as step-ca's JWK provisioner
does: its signature by the provisioner's key, its issuer, audience, times, root fingerprint, and that its id is new."""

import base64
import datetime
import json
import time
from pathlib import Path
from typing import Any

import pytest
from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.x509.oid import NameOID
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from rollout_runpod import StepCa, fingerprint
from rollout_train.inference import Connection, RemoteEngine
from rollout_train.inference.remote import Unreachable
from rollout_train.pods import GATEWAY_IDENTITY, pod_identity
from tests.rollout_train.machines import MODEL, Saying, fake_vllm
from tests.rollout_train.pods.authority import Authority, Issued, served_tls, server_context

PROVISIONER = "launcher"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def provisioner_key() -> tuple[dict[str, Any], ec.EllipticCurvePublicKey]:
    """A JWK provisioner's private key, as `step crypto jwe decrypt` gives it, and its public key."""
    key = ec.generate_private_key(ec.SECP256R1())
    numbers = key.private_numbers()
    jwk = {"kty": "EC", "crv": "P-256", "use": "sig", "alg": "ES256", "kid": "provisioner-kid",
           "x": _b64(numbers.public_numbers.x.to_bytes(32)), "y": _b64(numbers.public_numbers.y.to_bytes(32)),
           "d": _b64(numbers.private_value.to_bytes(32))}  # fmt: skip
    return jwk, key.public_key()


class FakeStepCa:
    """`/1.0/sign` and `/1.0/revoke` as step-ca answers them for a JWK provisioner."""

    def __init__(self, authority: Authority, public: ec.EllipticCurvePublicKey, kid: str) -> None:
        self.authority = authority
        self.public = public
        self.kid = kid
        self.url = ""
        self.used: set[str] = set()
        self.revoked: list[dict[str, Any]] = []

    def claims(self, token: str, audience: str) -> dict[str, Any]:
        header_part, claims_part, signature_part = token.split(".")
        header, claims = json.loads(_unb64(header_part)), json.loads(_unb64(claims_part))
        signature = _unb64(signature_part)
        der = utils.encode_dss_signature(int.from_bytes(signature[:32]), int.from_bytes(signature[32:]))
        try:
            self.public.verify(der, f"{header_part}.{claims_part}".encode(), ec.ECDSA(hashes.SHA256()))
        except InvalidSignature:
            raise PermissionError("the token's signature is not the provisioner's") from None
        now = time.time()
        checks = {
            "algorithm": header == {"alg": "ES256", "kid": self.kid, "typ": "JWT"},
            "issuer": claims["iss"] == PROVISIONER,
            "audience": claims["aud"] == f"{self.url}/1.0/{audience}",
            "times": claims["nbf"] <= now + 60 and now < claims["exp"] and claims["exp"] - claims["iat"] <= 15 * 60,
            "root": claims.get("sha", fingerprint(self.authority.pem))
            == fingerprint(self.authority.pem),  # (as step-ca: optional)
            "once": claims["jti"] not in self.used,
        }
        if failed := [name for name, passed in checks.items() if not passed]:
            raise PermissionError(f"the token fails: {', '.join(failed)}")
        self.used.add(claims["jti"])
        return claims

    def app(self) -> Starlette:
        async def sign(request: Request) -> JSONResponse:
            body = await request.json()
            try:
                claims = self.claims(body["ott"], "sign")
            except PermissionError as error:
                return JSONResponse({"status": 401, "message": str(error)}, status_code=401)
            csr = x509.load_pem_x509_csr(body["csr"].encode())
            common = csr.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
            uris = csr.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            asked = uris.get_values_for_type(x509.UniformResourceIdentifier)
            if common != claims["sub"] or asked != claims["sans"] or not csr.is_signature_valid:
                return JSONResponse({"status": 403, "message": "the request is not for the token's names"}, 403)
            certificate = self.authority.signed(csr.public_key(), claims["sub"], asked)
            crt = certificate.public_bytes(serialization.Encoding.PEM).decode()
            return JSONResponse({"crt": crt, "ca": self.authority.pem.decode(), "certChain": [crt]}, status_code=201)

        async def revoke(request: Request) -> JSONResponse:
            body = await request.json()
            try:
                claims = self.claims(body["ott"], "revoke")
            except PermissionError as error:
                return JSONResponse({"status": 401, "message": str(error)}, status_code=401)
            if claims["sub"] != body["serial"] or body.get("passive") is not True:
                return JSONResponse({"status": 400, "message": "the token is not for this serial"}, status_code=400)
            self.revoked.append(body)
            return JSONResponse({"status": "ok"})

        return Starlette(routes=[Route("/1.0/sign", sign, methods=["POST"]),
                                 Route("/1.0/revoke", revoke, methods=["POST"])])  # fmt: skip


def certificate_request(identity: str) -> tuple[ec.EllipticCurvePrivateKey, str]:
    """What `step ca certificate IDENTITY` sends from the pod: a key made there, and a request for the identity."""
    key = ec.generate_private_key(ec.SECP256R1())
    request = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, identity)]))
        .add_extension(x509.SubjectAlternativeName([x509.UniformResourceIdentifier(identity)]), critical=False)
        .sign(key, hashes.SHA256())
    )
    return key, request.public_bytes(serialization.Encoding.PEM).decode()


async def test_a_one_time_token_gets_one_pod_its_certificate_once(tmp_path: Path) -> None:
    authority = Authority(tmp_path / "ca")
    jwk, public = provisioner_key()
    fake = FakeStepCa(authority, public, jwk["kid"])
    tls = authority.issue("step-ca", None, ips=["127.0.0.1"])
    async with served_tls(fake.app(), server_context(authority, tls, client=None)) as url:
        fake.url = url
        (tmp_path / "provisioner.jwk").write_text(json.dumps(jwk))
        ca = StepCa.from_files(url, provisioner=PROVISIONER, key=tmp_path / "provisioner.jwk", root=authority.root)
        identity = pod_identity("inference-1")
        token = ca.pod_token(identity)
        http = ca._client()  # pyright: ignore[reportPrivateUsage]  (verifies step-ca by the root, as the pod does)
        key, csr = certificate_request(identity)
        answer = await http.post(f"{url}/1.0/sign", json={"csr": csr, "ott": token})
        assert answer.status_code == 201
        certificate = x509.load_pem_x509_certificate(answer.json()["crt"].encode())
        sans = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        assert sans.get_values_for_type(x509.UniformResourceIdentifier) == [identity]
        life = certificate.not_valid_after_utc - certificate.not_valid_before_utc
        assert datetime.timedelta(hours=24) <= life <= datetime.timedelta(hours=24, minutes=2)
        again = await http.post(f"{url}/1.0/sign", json={"csr": csr, "ott": token})
        assert again.status_code == 401 and "once" in again.json()["message"]  # (a token is taken once)
        _, other = certificate_request(pod_identity("inference-2"))
        stolen = await http.post(f"{url}/1.0/sign", json={"csr": other, "ott": ca.pod_token(identity)})
        assert stolen.status_code == 403  # (a token names one identity)
        late = ca.token(identity, now=time.time() - 16 * 60)
        assert (await http.post(f"{url}/1.0/sign", json={"csr": csr, "ott": late})).status_code == 401
        long = ca.token(identity, lifetime=3600)
        assert (await http.post(f"{url}/1.0/sign", json={"csr": csr, "ott": long})).status_code == 401
        await ca.aclose()

    # The certificate the pod got is what the gateway reaches it by.
    pod = Issued(tmp_path / "pod.crt", tmp_path / "pod.key")
    pod.certificate.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    pod.key.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                          serialization.NoEncryption()))  # fmt: skip
    gateway = authority.issue("gateway", GATEWAY_IDENTITY)
    async with served_tls(fake_vllm(Saying()), server_context(authority, pod, client=GATEWAY_IDENTITY)) as address:
        reaching = {"ca": str(authority.root), "certificate": str(gateway.certificate), "key": str(gateway.key)}
        engine = RemoteEngine(MODEL, address=address, connection=Connection(identity=identity, **reaching))
        assert set(await engine.models()) == {MODEL}
        engine.close()
        impostor = RemoteEngine(
            MODEL, address=address, connection=Connection(identity=pod_identity("inference-2"), **reaching)
        )
        with pytest.raises(Unreachable):
            await impostor.models()
        impostor.close()


async def test_a_pod_s_certificate_is_revoked_by_its_serial(tmp_path: Path) -> None:
    authority = Authority(tmp_path / "ca")
    jwk, public = provisioner_key()
    del jwk["kid"]  # (a key without its id: the id is its thumbprint, as step gives it)
    fake = FakeStepCa(authority, public, "")
    tls = authority.issue("step-ca", None, ips=["127.0.0.1"])
    async with served_tls(fake.app(), server_context(authority, tls, client=None)) as url:
        fake.url = url
        ca = StepCa(url, provisioner=PROVISIONER, key=jwk, root=authority.pem)
        fake.kid = ca._kid  # pyright: ignore[reportPrivateUsage]
        assert len(fake.kid) == 43  # (a SHA-256, base64url without padding)
        await ca.revoke("1234567890", reason="the pod was deleted")
        assert fake.revoked == [{"serial": "1234567890", "ott": fake.revoked[0]["ott"], "passive": True,
                                 "reason": "the pod was deleted"}]  # fmt: skip
        fake.public = ec.generate_private_key(ec.SECP256R1()).public_key()  # (another provisioner's key)
        with pytest.raises(RuntimeError, match="refused to revoke 42: 401"):
            await ca.revoke("42")
        await ca.aclose()


def test_a_provisioner_s_key_must_be_a_private_p256_key() -> None:
    jwk, _ = provisioner_key()
    for wrong in ({**jwk, "crv": "P-384"}, {key: value for key, value in jwk.items() if key != "d"}, {"kty": "RSA"}):
        with pytest.raises(ValueError, match="P-256"):
            StepCa("https://ca", provisioner=PROVISIONER, key=wrong, root=b"")


def encrypted(key: dict[str, Any], password: str) -> str:
    """A JWK encrypted with a password as step encrypts a provisioner's key (PBES2-HS256+A128KW, A256GCM)."""
    import os

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.keywrap import aes_key_wrap

    salt, iterations = os.urandom(16), 1000
    header = {"alg": "PBES2-HS256+A128KW", "enc": "A256GCM", "p2s": _b64(salt), "p2c": iterations, "cty": "jwk+json"}
    protected = _b64(json.dumps(header).encode())
    derived = PBKDF2HMAC(hashes.SHA256(), 16, b"PBES2-HS256+A128KW\x00" + salt, iterations).derive(password.encode())
    content, iv = os.urandom(32), os.urandom(12)
    sealed = AESGCM(content).encrypt(iv, json.dumps(key).encode(), protected.encode())
    return ".".join([protected, _b64(aes_key_wrap(derived, content)), _b64(iv), _b64(sealed[:-16]), _b64(sealed[-16:])])


def test_a_provisioner_s_key_is_read_from_step_ca_s_configuration_with_its_password() -> None:
    from rollout_runpod import decrypted_key

    jwk, _ = provisioner_key()
    said = encrypted(jwk, "the CA's password")
    assert decrypted_key(said, "the CA's password") == jwk
    with pytest.raises(ValueError, match="does not open"):
        decrypted_key(said, "another password")
    with pytest.raises(ValueError, match="five parts"):
        decrypted_key("not.a.jwe", "the CA's password")


async def test_the_gateway_s_certificate_is_issued_for_its_identity(tmp_path: Path) -> None:
    authority = Authority(tmp_path / "ca")
    jwk, public = provisioner_key()
    fake = FakeStepCa(authority, public, jwk["kid"])
    tls = authority.issue("step-ca", None, ips=["127.0.0.1"])
    async with served_tls(fake.app(), server_context(authority, tls, client=None)) as url:
        fake.url = url
        ca = StepCa(url, provisioner=PROVISIONER, key=jwk, root=authority.pem)
        chain, key = await ca.certificate(GATEWAY_IDENTITY)
        issued = x509.load_pem_x509_certificates(chain)[0]
        sans = issued.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        assert sans.get_values_for_type(x509.UniformResourceIdentifier) == [GATEWAY_IDENTITY]
        assert serialization.load_pem_private_key(key, None).public_key() == issued.public_key()
        await ca.aclose()


def test_a_pods_token_names_no_root_so_step_checks_the_cas_tls_by_the_roots_it_is_given(tmp_path: Path) -> None:
    authority = Authority(tmp_path / "ca")
    jwk, _ = provisioner_key()
    (tmp_path / "provisioner.jwk").write_text(json.dumps(jwk))
    ca = StepCa.from_files(
        "https://ca.example.com", provisioner=PROVISIONER, key=tmp_path / "provisioner.jwk", root=authority.root
    )

    def claims(token: str) -> dict[str, Any]:
        return json.loads(_unb64(token.split(".")[1]))

    # (a token naming the root makes step trust only that root for step-ca's own TLS, which fails behind a tunnel)
    assert "sha" not in claims(ca.pod_token("spiffe://rollout/pod/p"))
    assert claims(ca.token("spiffe://rollout/gateway"))["sha"] == fingerprint(authority.pem)
