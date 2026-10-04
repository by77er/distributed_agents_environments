"""Four agents in a shared Minecraft world, as an environment: what an episode is, and what there is to train on.

- `environment`: the tasks as rows, and how a start is drawn
  (`rollout train PROFILE minecraft_team.environment:environment`).
- `episode`: the episode program: four model slots, lockstep turns, one shared reward.
- `prompts`: what agents read and call: the system prompt, observations as text, the actions as tools.
- `limits`: the limits the prompts state and the harness keeps (`limits.json`, which both read).
- `tasks`: the task catalog, kits, scoring, and how each task is built in a live world.
- `worlds`: temporary worlds as the tool set `minecraft` (in this process, or served from another machine).
- `paper`, `control`, `harness`: Paper servers from templates, the ground-truth plugin's control API, and the bridge
  to the mineflayer bots.
"""
