"""A client for the ground-truth plugin's control API (environments/minecraft/plugin): ticks, episodes and ground
truth. Only the environment service and the episode program use it; agents never see it."""

from typing import Any

import httpx


class ControlError(RuntimeError):
    pass


class Control:
    def __init__(self, url: str, *, client: httpx.AsyncClient | None = None) -> None:
        self.url = url
        self._client = client or httpx.AsyncClient(timeout=60)

    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/health")

    async def state(self) -> dict[str, Any]:
        """Every player's position, health, food, inventory and diamonds, and `team_diamonds`: the reward."""
        return await self._request("GET", "/state")

    async def team_diamonds(self) -> int:
        return int((await self.state())["team_diamonds"])

    async def freeze(self) -> dict[str, Any]:
        return await self._request("POST", "/tick", {"action": "freeze"})

    async def unfreeze(self) -> dict[str, Any]:
        return await self._request("POST", "/tick", {"action": "unfreeze"})

    async def step(self, ticks: int) -> dict[str, Any]:
        """Run `ticks` game ticks of the frozen game; returns once they have run."""
        return await self._request("POST", "/tick", {"action": "step", "ticks": ticks})

    async def ticks(self) -> dict[str, Any]:
        return await self._request("GET", "/tick")

    async def episode(self, setup: dict[str, Any]) -> dict[str, Any]:
        """Set up the team and the world: `team`, `spawn`, `kit`, `gamemode`, `difficulty`, `time`, `gamerules`."""
        return await self._request("POST", "/episode", setup)

    async def ores(self, x: int, y: int, z: int, *, radius: int = 32, exposed: bool = False) -> list[dict[str, Any]]:
        """Diamond ores near a point (ground truth, for choosing where episodes start)."""
        query = {"x": x, "y": y, "z": z, "radius": radius, "exposed": str(exposed).lower()}
        return (await self._request("GET", "/ores", params=query))["ores"]

    async def events(self, after: int = -1) -> list[dict[str, Any]]:
        return (await self._request("GET", "/events", params={"after": after}))["events"]

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None, *, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        response = await self._client.request(method, f"{self.url}{path}", json=body, params=params)
        payload: dict[str, Any] = response.json()
        if response.status_code != 200:
            raise ControlError(f"{method} {path}: {payload.get('error', response.text)}")
        return payload
