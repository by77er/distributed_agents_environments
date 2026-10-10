"""One to four agents in a shared Minecraft world, as an environment: what an episode is, and what there is to train
on.

- `environment`: the tasks as rows, and how a start is drawn
  (`rollout train minecraft_team.environment:environment --preset minecraft-one-gpu`).
- `episode`: the episode program: a model slot for each agent, lockstep turns, one shared reward, one world.
- `prompts`: what agents read and call: the system prompt, observations as text, the actions as tools.
- `limits`: the limits the prompts state and the harness keeps (`limits.json`, which both read).
- `tasks`: the task catalog, kits, scoring, and how each task is built in a live world.
- `worlds`: temporary worlds as sandboxes of the kind `minecraft`, and the operations on them (in a pool in this
  process, or served from another machine).
- `paper`, `control`, `harness`: Paper servers from templates, the ground-truth plugin's control API, and the bridge
  to the mineflayer bots.
- `datasets`: `worked`, a turn filter that keeps the turns whose action came back ok.
- `cli`: `minecraft-team server`, a temporary server to look at.
"""
