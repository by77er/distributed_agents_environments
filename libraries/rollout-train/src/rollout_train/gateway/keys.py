"""Signed keys: a harness's API key that says, by itself, which session it samples for.

Whoever plays an attempt (an episode runner, a sandbox's lease) mints a key per model slot: a `Grant` naming the run,
the episode and attempt, the program's run and the slot, the channel, how to sample, the fence its turns are recorded
under, and when it expires, signed with HMAC-SHA256. The gateway verifies a key with nothing but the secret: no lookup,
no session kept anywhere.

    rk1.KID.PAYLOAD.SIGNATURE

`PAYLOAD` is the grant as JSON, base64url-encoded without padding; `SIGNATURE` is the HMAC of `rk1.KID.PAYLOAD` under
the secret named `KID`, base64url-encoded the same way. A `Keyring` holds the secrets by id: the first signs, every one
verifies, so a secret is rotated by putting a new one first and removing the old once the keys it signed have expired.
Secrets come from the environment (`ROLLOUT_GATEWAY_KEYS`: `KID:SECRET` pairs separated by commas) or a file
(`ROLLOUT_GATEWAY_KEYS_FILE`: one `KID SECRET` per line), never from the ledger.
"""

import base64
import hashlib
import hmac
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rollout_train.ledger import Fence

PREFIX = "rk1"
"""What every key begins with: the format's version."""
KEYS, KEYS_FILE = "ROLLOUT_GATEWAY_KEYS", "ROLLOUT_GATEWAY_KEYS_FILE"
"""The environment variables a keyring is read from."""
SHORTEST_SECRET = 32
"""Bytes a secret has at least."""


class KeyRefused(Exception):
    """A key that is malformed, signed by no secret of the keyring, forged, or expired."""


@dataclass(frozen=True)
class Grant:
    """What a key lets its holder do: sample for one model slot of one attempt, until it expires."""

    run: str
    """The run whose tables the turns go under (a training run, an eval's run)."""
    run_id: str
    """The program's run that plays the attempt: the session is `{run_id}/{slot}`."""
    slot: str
    channel: str
    fence: Fence
    """What its turns are appended under: a turn whose fence was taken again since is refused."""
    expires: float
    """Seconds since the epoch."""
    episode: str = ""
    """`GROUP/EPISODE`, for an attempt of an episode a run asked for."""
    attempt: int = 0
    temperature: float = 1.0
    top_p: float = 1.0

    @property
    def session_id(self) -> str:
        return f"{self.run_id}/{self.slot}"

    def to_json(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "run_id": self.run_id,
            "slot": self.slot,
            "channel": self.channel,
            "fence": [self.fence.scope, self.fence.number],
            "expires": self.expires,
            "episode": self.episode,
            "attempt": self.attempt,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Grant":
        scope, number = data["fence"]
        return cls(
            run=str(data["run"]),
            run_id=str(data["run_id"]),
            slot=str(data["slot"]),
            channel=str(data["channel"]),
            fence=Fence(str(scope), int(number)),
            expires=float(data["expires"]),
            episode=str(data.get("episode", "")),
            attempt=int(data.get("attempt", 0)),
            temperature=float(data.get("temperature", 1.0)),
            top_p=float(data.get("top_p", 1.0)),
        )


@dataclass(frozen=True)
class Keyring:
    """Secrets by id: `signing` signs, every one verifies."""

    secrets: Mapping[str, bytes]
    signing: str
    leeway: float = 30.0
    """Seconds a key is still taken after it expires, for clocks that differ."""

    def __post_init__(self) -> None:
        if self.signing not in self.secrets:
            raise ValueError(f"the keyring has no secret {self.signing!r} to sign with")
        for name, secret in self.secrets.items():
            if not name or "." in name or len(secret) < SHORTEST_SECRET:
                raise ValueError(f"secret {name!r}: an id without dots, and {SHORTEST_SECRET} bytes at least")

    @classmethod
    def parse(cls, pairs: list[tuple[str, str]]) -> "Keyring":
        """Secrets as (id, secret) pairs, the signing one first."""
        if not pairs:
            raise ValueError("a keyring needs a secret")
        return cls({name: secret.encode() for name, secret in pairs}, pairs[0][0])

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "Keyring":
        """The keyring `ROLLOUT_GATEWAY_KEYS` or `ROLLOUT_GATEWAY_KEYS_FILE` says."""
        environment = os.environ if environment is None else environment
        if given := environment.get(KEYS):
            return cls.parse([_pair(each, ":") for each in given.split(",") if each.strip()])
        if path := environment.get(KEYS_FILE):
            return cls.load(Path(path).expanduser())
        raise KeyError(f"no gateway secrets: set {KEYS} (KID:SECRET,...) or {KEYS_FILE}")

    @classmethod
    def load(cls, path: Path) -> "Keyring":
        """The keyring in a file: one `KID SECRET` per line, the signing one first (`#` begins a comment)."""
        lines = [line.split("#", 1)[0].strip() for line in path.read_text().splitlines()]
        return cls.parse([_pair(line, None) for line in lines if line])

    def mint(self, grant: Grant) -> str:
        """A key for `grant`, signed with the signing secret."""
        payload = _encoded(json.dumps(grant.to_json(), separators=(",", ":"), sort_keys=True).encode())
        signed = f"{PREFIX}.{self.signing}.{payload}"
        return f"{signed}.{_encoded(self._signature(self.signing, signed))}"

    def verify(self, key: str, now: float | None = None) -> Grant:
        """The grant a key carries. Raises `KeyRefused` unless a secret of the keyring signed it and it has not
        expired."""
        parts = key.split(".")
        if len(parts) != 4 or parts[0] != PREFIX:
            raise KeyRefused("not a gateway key")
        _, name, payload, signature = parts
        if name not in self.secrets:
            raise KeyRefused(f"signed with a secret this gateway does not have ({name})")
        try:
            given = _decoded(signature)
        except ValueError:
            raise KeyRefused("the key's signature is not base64url") from None
        if not hmac.compare_digest(given, self._signature(name, f"{PREFIX}.{name}.{payload}")):
            raise KeyRefused("the key's signature does not match")
        try:
            grant = Grant.from_json(json.loads(_decoded(payload)))
        except (ValueError, KeyError, TypeError) as error:
            raise KeyRefused(f"the key's grant cannot be read: {error}") from None
        if grant.expires + self.leeway < (time.time() if now is None else now):
            raise KeyRefused("the key has expired")
        return grant

    def _signature(self, name: str, signed: str) -> bytes:
        return hmac.new(self.secrets[name], signed.encode(), hashlib.sha256).digest()


def granted(grant: Grant, lifetime: float, now: float | None = None) -> Grant:
    """`grant`, expiring `lifetime` seconds from now."""
    return replace(grant, expires=round((time.time() if now is None else now) + lifetime, 1))


def _pair(text: str, separator: str | None) -> tuple[str, str]:
    """An id and a secret, split at `separator` (None: at whitespace)."""
    parts = text.strip().split(separator, 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError("a secret is given as its id and the secret")
    return parts[0].strip(), parts[1].strip()


def _encoded(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _decoded(text: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (ValueError, TypeError) as error:
        raise ValueError(str(error)) from None
