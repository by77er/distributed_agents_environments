# Example environments

Three environments built on `rollout` alone, each trained with the platform: they show what an environment holds
(rows, a program, eval data, sandboxes, slots that are not trained) and the run settings and presets that train it.
This section is for whoever writes an environment and wants a worked one to read.

**Read first:** [Write a task](../guide/tasks.md). **Next:** [Minecraft team](minecraft-team.md).

| Environment | What it shows |
|---|---|
| [Minecraft team](minecraft-team.md) | One to four agents in a Minecraft world, each world a sandbox from a pool; a curriculum of 100 rows; the presets `minecraft-one-gpu` and `minecraft-tinker` |
| [Minecraft horizons](minecraft-horizons.md) | Objectives with no ceiling (iron, food, advancements…) and speedruns to the dragon, whole or in segments, each raced against a budget of game time the agents see, from 5 to 640 minutes; the team package's server, bots and builders; a sandbox kind of its own |
| [Gridworld](gridworld.md) | Two to four agents share out the plates of a grid level over chat; no sandbox; the preset `gridworld-qwen3-0.6b` |
| [Judging](judging.md) | Open-ended answers scored by a judge against a versioned rubric: a reward from a slot that is not trained, bound to a channel of its own by the run's settings |
