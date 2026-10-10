"""Minecraft horizons: objectives with no ceiling, each raced against a budget of game time the agents see.

- `objectives`: what an objective counts (items held, or advancements), and the reward `log(1 + amount)`.
- `tasks`: each objective from each setting it suits, at each budget of the ladder (`LADDER`).
- `prompts`: the system prompt (what counts, how long the game lasts) and the clock atop every observation.
- `episode`: `HorizonEpisode`, the team's turns until the budget is spent.
- `worlds`: `HorizonWorlds`, sandboxes of the kind `minecraft-horizons`.
- `environment`: `Horizons`, the rows, starts and held-out eval data.

The server, the bots, the ground-truth plugin and the starts' builders are the team package's (`minecraft_team`).
"""
