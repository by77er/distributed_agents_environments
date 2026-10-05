"""Who may use a monitor: whoever holds its token, from a page served under one of its names.

A monitor has one token (`ROLLOUT_MONITOR_TOKEN`, or the cluster config's `[monitor] token_env` or `token_file`; made
up when it starts where none is set, and printed with its sign-in link). Every request under `/api/` carries it, as a
bearer token (`Authorization: Bearer TOKEN`: a program, another monitor) or as the cookie a browser is given when it
signs in: `/login?token=TOKEN`, opened once, sets the cookie and goes on to the page; `POST /login` with
`{"token": TOKEN}` does the same from the page's own sign-in form. The cookie is `HttpOnly` and `SameSite=Strict`, so
the page's scripts never read it and no other site's page sends it; it is `Secure` when the page is served over https.
The page itself and its scripts (`/`, `/assets/…`) hold nothing and need no token.

Besides the token (`guarded`):

- **Its names.** A request whose `Host` is none of the names the monitor is served under is refused (400):
  `localhost`, `127.0.0.1`, `[::1]` and this machine's name (`HOSTS`), and those it is given (`--allow-host`: its
  Service's names in a cluster, an ingress's host). A page of another site whose name was pointed at this machine (DNS
  rebinding) reaches nothing.
- **What changes something is JSON.** A request that is not a GET or a HEAD is refused (415) unless it says
  `Content-Type: application/json`, which a page of another site can send only after the browser has asked the monitor
  whether it may (CORS's preflight, which the monitor never allows). And where it says where it comes from (`Origin`),
  that must be the monitor itself, by the name it was reached at (403 otherwise).
"""

import contextlib
import hmac
import json
import secrets
import socket
from collections.abc import Iterable, Mapping
from http.cookies import SimpleCookie
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

__all__ = ["HOSTS", "TOKEN_ENV", "cookie_name", "guarded", "hosts_of", "new_token"]

TOKEN_ENV = "ROLLOUT_MONITOR_TOKEN"
"""The environment variable a monitor's token is read from, unless the cluster config names another."""
HOSTS = ("localhost", "127.0.0.1", "[::1]")
"""The names every monitor answers under, beside this machine's own and those it is given."""
COOKIE = "rollout-monitor"
"""What the sign-in cookie is called (with the port the page was reached at, where it names one: `cookie_name`)."""
LOGIN = "/login"
SAFE = ("GET", "HEAD")
"""Methods that change nothing."""
MONTH = 30 * 24 * 3600
"""Seconds a browser keeps the sign-in cookie."""


def new_token() -> str:
    """A token for a monitor that was given none."""
    return secrets.token_urlsafe(32)


def hosts_of(given: Iterable[str] = ()) -> list[str]:
    """The names a monitor answers under: `HOSTS`, this machine's name (and its fully qualified one), and `given`."""
    names = [*HOSTS, socket.gethostname(), socket.getfqdn(), *given]
    return list(dict.fromkeys(name.lower() for name in names if name))


def cookie_name(host: str) -> str:
    """The sign-in cookie's name for a page reached at `host` (`HOST[:PORT]`): browsers keep cookies by name and not by
    port, so two monitors on one machine keep two cookies."""
    port = host.rpartition(":")[2] if not host.endswith("]") and ":" in host else ""
    return f"{COOKIE}-{port}" if port.isdigit() else COOKIE


def guarded(token: str, hosts: Iterable[str]) -> list[Middleware]:
    """What serves an app only under `hosts` (each a name, or `*.DOMAIN`), its `/api/` only for whoever holds `token`,
    and `/login` (`?token=`, or `POST {"token"}`) setting the cookie that holds it: middleware, outermost first."""
    allowed = [name.lower() for name in hosts]
    return [
        Middleware(TrustedHostMiddleware, allowed_hosts=allowed, www_redirect=False),
        Middleware(_Guard, token=token),
    ]


class _Guard:
    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.token = token

    def _holds(self, said: str | None) -> bool:
        return said is not None and hmac.compare_digest(said.encode(), self.token.encode())

    def _signed_in(self, headers: Mapping[str, str]) -> bool:
        """Whether a request carries the token: as a bearer token, or in the sign-in cookie."""
        return self._holds(_bearer(headers)) or self._holds(_cookie(headers, cookie_name(headers.get("host", ""))))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = _headers(scope)
        method, path = scope["method"], _path(scope)
        if method not in SAFE:
            refused = _unsafe_refusal(headers)
            if refused is not None:
                await _answer(send, *refused)
                return
        if path == LOGIN:
            await self._login(scope, receive, send, headers)
            return
        if (path == "/api" or path.startswith("/api/")) and not self._signed_in(headers):
            await _answer(send, 401, {"error": "sign in: open the monitor's sign-in link, /login?token=TOKEN"})
            return
        await self.app(scope, receive, send)

    async def _login(self, scope: Scope, receive: Receive, send: Send, headers: Mapping[str, str]) -> None:
        if scope["method"] == "GET":
            said = parse_qs(scope.get("query_string", b"").decode()).get("token", [None])[0]
        elif scope["method"] == "POST":
            body: Any = None
            with contextlib.suppress(ValueError):
                body = json.loads(await _body(receive) or b"null")
            given: Any = cast(dict[str, Any], body).get("token") if isinstance(body, dict) else None
            said = str(given) if given else None
        else:
            await _answer(send, 405, {"error": 'sign in with GET /login?token=TOKEN or POST /login {"token"}'})
            return
        if not self._holds(said):
            await _answer(send, 401, {"error": "that is not this monitor's token"})
            return
        cookie = _cookie_header(cookie_name(headers.get("host", "")), self.token, scope.get("scheme") == "https")
        if scope["method"] == "POST":
            await _answer(send, 200, {"signed_in": True}, [(b"set-cookie", cookie)])
            return
        # (on to the page, by a path relative to /login: wherever the monitor is served from, behind a proxy's path)
        extra = [(b"set-cookie", cookie), (b"location", b"./"), (b"cache-control", b"no-store")]
        await _answer(send, 303, {"signed_in": True}, extra)


def _unsafe_refusal(headers: Mapping[str, str]) -> tuple[int, dict[str, str]] | None:
    """Why a request that may change something is refused, if it is: not JSON, or from another origin."""
    media = headers.get("content-type", "").split(";")[0].strip().lower()
    if media != "application/json":
        return 415, {"error": "a request that changes something says Content-Type: application/json"}
    origin = headers.get("origin")
    if origin is not None:
        reached = headers.get("host", "").lower()
        if origin == "null" or urlsplit(origin).netloc.lower() != reached:
            return 403, {"error": f"a request from {origin} is not the monitor's own page"}
    return None


def _headers(scope: Scope) -> dict[str, str]:
    return {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in scope.get("headers", [])}


def _path(scope: Scope) -> str:
    root = scope.get("root_path", "")
    path: str = scope.get("path", "")
    return path[len(root) :] if root and path.startswith(root) else path


def _bearer(headers: Mapping[str, str]) -> str | None:
    kind, _, said = headers.get("authorization", "").partition(" ")
    return said.strip() if kind.lower() == "bearer" and said.strip() else None


def _cookie(headers: Mapping[str, str], name: str) -> str | None:
    jar: SimpleCookie = SimpleCookie()
    try:
        jar.load(headers.get("cookie", ""))
    except Exception:  # (a header it cannot read holds no token)
        return None
    found = jar.get(name)
    return found.value if found is not None else None


def _cookie_header(name: str, token: str, secure: bool) -> bytes:
    parts = [f"{name}={token}", "Path=/", f"Max-Age={MONTH}", "HttpOnly", "SameSite=Strict"]
    return "; ".join([*parts, *(["Secure"] if secure else [])]).encode("latin-1")


async def _body(receive: Receive) -> bytes:
    chunks: list[bytes] = []
    while True:
        message: Message = await receive()
        chunks.append(message.get("body", b""))
        if not message.get("more_body"):
            return b"".join(chunks)


async def _answer(send: Send, status: int, said: Mapping[str, Any], extra: Iterable[tuple[bytes, bytes]] = ()) -> None:
    body = json.dumps(said).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), *extra]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
