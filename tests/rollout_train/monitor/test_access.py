"""Who may use a monitor: its `/api/` answers only a request with its token (a bearer token, or the cookie signing in
sets); it answers only under its own names; what changes something must say JSON and, where it says its origin, come
from the monitor's own page. A page of another site cannot import an environment or launch a run through a browser,
whether it posts plain text, posts from its own origin, or reaches the monitor by a name of its own (DNS rebinding)."""

from pathlib import Path

import httpx
import pytest

from rollout_train.monitor.access import cookie_name, hosts_of
from rollout_train.monitor.app import create_app
from tests.rollout_train.support import signed_in

TOKEN = "a-token-for-the-test"


def client(tmp_path: Path, host: str = "localhost:8765", **options: object) -> httpx.AsyncClient:
    app = create_app(tmp_path, beat=0.0, token=TOKEN, **options)  # pyright: ignore[reportArgumentType]
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=f"http://{host}")


async def test_the_api_answers_only_with_the_monitors_token(tmp_path: Path) -> None:
    async with client(tmp_path) as anyone:
        refused = await anyone.get("/api/system")
        assert refused.status_code == 401 and "/login?token=" in refused.json()["error"]
        assert (await anyone.get("/api/stream")).status_code == 401
        assert (await anyone.get("/api/system", headers={"Authorization": "Bearer wrong"})).status_code == 401
        assert (await anyone.get("/api/system", headers={"Authorization": f"Bearer {TOKEN}"})).status_code == 200
        assert (await anyone.get("/favicon.svg")).status_code in (200, 404)  # (the page holds nothing: no token)
        assert (await anyone.get("/")).status_code in (200, 503)


async def test_signing_in_sets_a_cookie_the_page_then_reaches_the_api_with(tmp_path: Path) -> None:
    async with client(tmp_path) as browser:
        assert (await browser.get("/login", params={"token": "wrong"})).status_code == 401
        signed = await browser.get("/login", params={"token": TOKEN})
        assert signed.status_code == 303 and signed.headers["location"] == "./"
        cookie = signed.headers["set-cookie"]
        assert cookie.startswith(f"{cookie_name('localhost:8765')}={TOKEN};")
        assert {"HttpOnly", "SameSite=Strict", "Path=/"} <= {part.strip() for part in cookie.split(";")}
        assert "Secure" not in cookie  # (served over http here)
        assert (await browser.get("/api/system")).status_code == 200  # (the client keeps the cookie, as a browser does)
    async with client(tmp_path) as form:  # (the page's own sign-in form)
        posted = await form.post("/login", json={"token": TOKEN})
        assert posted.status_code == 200 and posted.json() == {"signed_in": True}
        assert (await form.get("/api/system")).status_code == 200
        assert (await form.post("/login", json={"token": "wrong"})).status_code == 401


def test_two_monitors_on_one_machine_keep_two_cookies() -> None:
    assert cookie_name("localhost:8765") == "rollout-monitor-8765" != cookie_name("localhost:8766")
    assert cookie_name("monitor.example.com") == "rollout-monitor" and cookie_name("[::1]") == "rollout-monitor"
    assert cookie_name("[::1]:8765") == "rollout-monitor-8765"


async def test_a_page_of_another_site_cannot_import_or_launch(tmp_path: Path) -> None:
    async with signed_in(create_app(tmp_path, beat=0.0, token=TOKEN)) as browser:
        asked = {"url": "https://attacker.example/env.git"}
        plain = await browser.post(
            "/api/environments/import", content=b'{"url": "x"}', headers={"Content-Type": "text/plain"}
        )
        assert plain.status_code == 415  # (a no-cors POST of text/plain, which a page of any site may send)
        for path in ("/api/environments/import", "/api/launches", "/api/launches/x/stop", "/api/rename"):
            elsewhere = await browser.post(path, json=asked, headers={"Origin": "https://attacker.example"})
            assert elsewhere.status_code == 403, path
        assert (await browser.post("/api/launches", json={}, headers={"Origin": "null"})).status_code == 403
        assert (await browser.delete("/api/bookmarks/x", headers={"Content-Type": ""})).status_code == 415
        own = await browser.post("/api/launches", json={"kind": "train"}, headers={"Origin": "http://localhost"})
        assert own.status_code == 409  # (its own page: heard, and refused for a monitor with no cluster config)


@pytest.mark.parametrize("host", ["attacker.example", "attacker.example:8765", "10.0.0.5:8765"])
async def test_a_name_the_monitor_is_not_served_under_reaches_nothing(tmp_path: Path, host: str) -> None:
    async with client(tmp_path, host) as rebound:
        for path in ("/", "/api/system", "/login"):
            answer = await rebound.get(path, headers={"Authorization": f"Bearer {TOKEN}"}, params={"token": TOKEN})
            assert answer.status_code == 400, path


async def test_the_names_it_is_given_and_its_own_are_served(tmp_path: Path) -> None:
    import socket

    names = ["monitor-main.rollout.svc", "*.example.com"]
    for host in ("127.0.0.1:8765", "[::1]:8765", socket.gethostname(), "monitor-main.rollout.svc:8765",
                 "monitor.example.com"):  # fmt: skip
        async with client(tmp_path, host, hosts=names) as reached:
            assert (await reached.get("/api/system", headers={"Authorization": f"Bearer {TOKEN}"})).status_code == 200
    assert {"localhost", "127.0.0.1", "[::1]", socket.gethostname().lower()} <= set(hosts_of())


async def test_a_monitor_given_no_token_makes_one_up(tmp_path: Path) -> None:
    first, second = create_app(tmp_path, beat=0.0), create_app(tmp_path, beat=0.0)
    assert len(first.state.token) >= 32 and first.state.token != second.state.token
    async with signed_in(first) as reaches:
        assert (await reaches.get("/api/system")).status_code == 200
