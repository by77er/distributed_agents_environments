"""A swarm of four agents in Minecraft, trained with RL to hold as many diamonds as possible together.

- `paper`: Paper servers from templates, with the ground-truth plugin (minecraft/plugin).
- `control`: the plugin's control API: tick freezing and stepping, episode setup, ground truth.
- `service`: an HTTP service that creates and destroys temporary servers and reports rewards.
- `harness`: the bridge to the mineflayer bots (minecraft/harness), with filtered observations.
- `episode`: the lockstep episode program; `curriculum`: where episodes start.
"""
