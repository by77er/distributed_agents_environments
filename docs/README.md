# Documentation

`rollout` runs agents in environments, and trains them. Start with the [developer guide](guide/README.md) to build
with it, or with [three ways in](guide/perspectives.md) to see it from where you stand: building an environment,
designing training, or deploying.

```
docs/
├── guide/                  how to build with it; every example runs in the test suite
│   ├── perspectives.md     the three surfaces: environments, training, deployment
│   ├── getting-started.md  tasks.md  tools.md  agents.md  conversations.md  content.md  models.md
│   ├── runs-and-events.md  testing.md  deploying.md
│   └── reference.md        every public name, generated from the source
├── architecture/
│   ├── overview.md         layers, protocols and their implementations, who sees what, a turn under each runner
│   └── glossary.md
├── contracts/              types that cross layers: content, identifiers, effects, run events, the model endpoint
├── core/
│   ├── harness/            the loop, programs and runners; determinism rules; hooks; memory
│   ├── recorder/           token-exact recording, epochs, engines, the endpoint for harnesses
│   ├── rollouts/           rollout jobs: rows in, episodes out, weights published
│   ├── trajectories/       the episode and its assembly
│   ├── training.md         the loop, the group algorithm, the curriculum, the trainer
│   └── monitor.md          a live page over a job and its runs
├── inference/              channels: engines, limits, publishing weights
├── durability/             runs that survive their process: the durable runner, eviction, several runners
├── environments/           computers for tasks: the protocol, the backends, the tools
├── products/               what is built on it: the project assistant, agent sessions, the Minecraft swarm
└── development/            local services for development (Postgres, an S3-compatible store)
```

## Conventions

- **Pages describe the code as it is.** A page's status line names the code it describes.
- **Define once.** Every type and every fact has one defining page; others link to it. Types that cross layers are
  in `contracts/`.
- **Full type names.** `Observation`, not `Obs`.
