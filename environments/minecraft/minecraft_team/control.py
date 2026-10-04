"""A client for the ground-truth plugin's control API (environments/minecraft/plugin): ticks, episodes and ground
truth. The worlds use it (`minecraft_team.worlds`, to build tasks, run game time and score) and so do the servers
(`minecraft_team.paper`, to start and to generate templates); agents never see it."""

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
        """Ground truth, which tasks are scored from: every player's position, health, food, inventory and diamonds;
        the team's diamonds, the advancements it earned and what it got hold of since the baseline; and whether the
        dragon is dead and how much it was hurt."""
        return await self._request("GET", "/state")

    async def freeze(self) -> dict[str, Any]:
        return await self._request("POST", "/tick", {"action": "freeze"})

    async def step(self, ticks: int) -> dict[str, Any]:
        """Run `ticks` game ticks of the frozen game; returns once they have run."""
        return await self._request("POST", "/tick", {"action": "step", "ticks": ticks})

    async def run(self, ticks: int) -> dict[str, Any]:
        """Start `ticks` game ticks of the frozen game and return at once; `stop` ends them."""
        return await self._request("POST", "/tick", {"action": "run", "ticks": ticks})

    async def stop(self) -> int:
        """End a `run`; returns the game ticks that ran since it started."""
        return int((await self._request("POST", "/tick", {"action": "stop"}))["stepped"])

    async def ticks(self) -> dict[str, Any]:
        return await self._request("GET", "/tick")

    async def episode(self, setup: dict[str, Any]) -> dict[str, Any]:
        """Set up the team and the world: `team`, `spawn`, `placements` (where each agent stands, and its kit),
        `gamemode`, `difficulty`, `time`, `gamerules`."""
        return await self._request("POST", "/episode", setup)

    async def ores(self, x: int, y: int, z: int, *, radius: int = 32) -> list[dict[str, Any]]:
        """Diamond ores near a point (ground truth, for choosing where episodes start)."""
        query = {"x": x, "y": y, "z": z, "radius": radius}
        return (await self._request("GET", "/ores", params=query))["ores"]

    # Setup: building tasks from ground truth

    async def carve(
        self,
        x: int,
        y: int,
        z: int,
        *,
        width: int,
        height: int,
        depth: int,
        light: bool = True,
        floor: str = "stone",
        world: str = "world",
    ) -> None:
        """Empty a box (its corner at x, y, z) and give it a floor; light it with invisible light blocks."""
        box = {"x": x, "y": y, "z": z, "width": width, "height": height, "depth": depth, "light": light, "floor": floor}
        await self._request("POST", "/setup/carve", {**box, "world": world})

    async def drop_items(self, x: int, y: int, z: int, items: list[dict[str, Any]]) -> int:
        return int((await self._request("POST", "/setup/items", {"x": x, "y": y, "z": z, "items": items}))["dropped"])

    async def chest(self, x: int, y: int, z: int, items: list[dict[str, Any]]) -> None:
        await self._request("POST", "/setup/chest", {"x": x, "y": y, "z": z, "items": items})

    async def set_block(self, x: int, y: int, z: int, block: str) -> None:
        await self._request("POST", "/setup/block", {"x": x, "y": y, "z": z, "block": block})

    async def standing_spots(
        self, x: int, y: int, z: int, *, radius: int = 16, limit: int = 64, world: str = "world"
    ) -> list[dict[str, int]]:
        """Places to stand near a point (air at the feet and head, a solid floor, no lava near), nearest first."""
        query = {"x": x, "y": y, "z": z, "radius": radius, "limit": limit, "world": world}
        return (await self._request("GET", "/setup/stand", params=query))["spots"]

    async def surface(self, x: int, z: int, world: str = "world") -> int:
        return int((await self._request("GET", "/setup/surface", params={"x": x, "z": z, "world": world}))["y"])

    async def locate(
        self, structure: str, *, world: str, x: int = 0, z: int = 0, radius: int = 100
    ) -> dict[str, int] | None:
        """The nearest `fortress`, `stronghold` or `end_city`; None if there is none within `radius` chunks."""
        query = {"structure": structure, "world": world, "x": x, "z": z, "radius": radius}
        found = await self._request("GET", "/setup/locate", params=query)
        return {"x": int(found["x"]), "y": int(found["y"]), "z": int(found["z"])} if "x" in found else None

    async def find_blocks(
        self, block: str, x: int, y: int, z: int, *, world: str = "world", radius: int = 32, limit: int = 64
    ) -> list[dict[str, int]]:
        query = {"block": block, "x": x, "y": y, "z": z, "world": world, "radius": radius, "limit": limit}
        return (await self._request("GET", "/setup/blocks", params=query))["blocks"]

    async def spawn(self, entity: str, x: int, y: int, z: int, *, world: str = "world", ai: bool = True) -> int:
        """Spawn a creature; returns its entity id. Without `ai` it stands where it is put."""
        body = {"entity": entity, "x": x, "y": y, "z": z, "world": world, "ai": ai}
        return int((await self._request("POST", "/setup/spawn", body))["id"])

    async def set_food(self, name: str, food: int) -> None:
        """A player's hunger (20 is full)."""
        await self._request("POST", "/setup/food", {"name": name, "food": food})

    async def generate(self, radius: int, *, seconds: float = 900) -> None:
        """Generate the overworld's chunks within `radius` chunks of the origin and save them (for a server
        template); this takes minutes, so the request waits up to `seconds`."""
        await self._request("POST", "/setup/generate", {"radius": radius}, seconds=seconds)

    async def baseline(self) -> dict[str, Any]:
        """Remember each team member's advancements now: `state` then reports only newer ones."""
        return await self._request("POST", "/baseline", {})

    async def events(self, after: int = -1) -> list[dict[str, Any]]:
        return (await self._request("GET", "/events", params={"after": after}))["events"]

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        params: dict[str, Any] | None = None,
        seconds: float | None = None,
    ) -> dict[str, Any]:
        """One request to the plugin; `seconds` is how long to wait for the answer (the client's own limit if None)."""
        timeout = httpx.USE_CLIENT_DEFAULT if seconds is None else seconds
        response = await self._client.request(method, f"{self.url}{path}", json=body, params=params, timeout=timeout)
        payload: dict[str, Any] = response.json()
        if response.status_code != 200:
            raise ControlError(f"{method} {path}: {payload.get('error', response.text)}")
        return payload
