"""Two to four agents on a grid level who must share out its plates, as an environment: what an episode is, and what
there is to train on.

- `environment`: the rows, and how a start is drawn (`rollout env check gridworld.environment:environment`).
- `episode`: the episode program: a model slot for each agent, turns all at once, one shared reward.
- `scoring`: the reward: half for solving, half for progress through the level's stages.
- `curriculum`: rows unlocked by success or progress, drawn most where outcomes are mixed.
- `level`: levels drawn from a layout, a number of agents and a seed, and the check that one can be solved.
- `game`: the rules, a turn at a time.
- `prompts`: what agents read and call: the system prompt, observations as text, the actions as tools.
- `scripted`: a scripted team that solves every row from observations alone.
"""
