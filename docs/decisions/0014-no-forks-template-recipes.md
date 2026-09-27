# 0014 — No forks: setup cost is paid by content-addressed template builds

Status: **Accepted** · Date: 2026-09-27

## Context

Earlier revisions forked a prepared task and its environments into the N samples of a training group
(`group_setup`, `InitialState`, memory forks, ownership transfer). The requirement cut through every layer: a task
hook with determinism rules, a harness run mode and identifier remapping, a driver capability, placement for
copy-on-write sharing, and rollout-controller orchestration. All of it bought one optimization (set up once instead
of N times) and one convenience (identical start states).

## Decision

- Remove forking from the task, harness, rollout and driver contracts.
- Environment preparation is a declarative **`Template` recipe** — base OCI image, files, build commands — that the
  Environment Manager hashes, builds once per distinct hash (concurrent requests wait for one build), and keeps as
  a golden snapshot. Every environment created from the recipe restores that snapshot.
- A recipe that is only a `base` is a prebuilt image, so complicated environments can be baked by any external
  pipeline and referenced directly.
- `snapshot` / `hibernate` remain on environment handles for long-lived workspaces; that is an environment
  durability feature, unrelated to training.

## Consequences

- Setup cost and identical start states come from the environment layer alone; task code just creates an
  environment from a recipe.
- Same-host restores of one snapshot still share memory pages, so the density benefit of forks remains.
- Setup that must be Python logic repeats per run, or moves into the dataset build.
- Branching partway through an episode (tree search, Monte Carlo value estimates from intermediate states) is not
  supported; it will be added as an explicit feature only when an algorithm needs it.

## Alternatives considered

- **Group forks** (previous design): see Context.
- **Memoized runtime setup** (snapshot after the first run's `setup`, keyed by row): no fork in the interface, but
  still imposes determinism rules on Python setup code and task-state pickling across runs.
