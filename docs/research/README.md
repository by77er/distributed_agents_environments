# Design notes

Designs, proposals, measurements and records of spikes. These pages are for people who change the platform and want
the reasoning behind it. Unlike the rest of the documentation, a design note may describe what is not built yet:
each one says at its top whether it is **built**, **in progress** or **proposed**, and marks its parts that differ.

**Read first:** [the architecture](../architecture/overview.md). For what the code does now, the guide and library
pages are the reference; a design note never overrides them.

## Runtime and deployment

- [Runtime design](runtime-design.md), **in progress**: one cluster config, run settings and presets, providers and
  the bridges between their formats, every role on Ray, one gateway, and runs that claim their own resources.
- [RunPod pods as providers](runpod-providers.md), **built**: GPU pods on RunPod leased by runs for their channels,
  their training steps, or both on one GPU, with mutual TLS (mTLS) and step-ca certificates; warm for the next run,
  and reaped when no run holds them.
- [Ledger guarantees](ledger-guarantees.md), **built**: what the ledger promises its writers and readers (fences,
  claims, retries), each promise with the test that holds it. Its last section is what the HTTP ledger service keeps.
- [Minecraft memory](minecraft-memory.md), **built**: the memory of the Minecraft team's worlds measured on real
  servers and episodes, the Java and Node settings chosen from it, and why several worlds do not share one server.
- [Where sandboxes run](sandbox-placement.md), **built**: sandboxes as a system of their own, reached only through
  the claiming interface; the options for where a pool's worlds run, and the choice: a pool served from a pod of its
  own on Kubernetes, which a run's demand does not count.
- [Where a run runs](run-placement.md), **proposed**: a run's driver, gateway, runners, environment and sandbox pools
  on the GPU pod it leases, in a Ray cluster of its own there, with the cluster admitting, watching and storing; the
  internet path measured on a live run, the options, phases and what each costs.
- [Cleanup inventory](cleanup-inventory.md), **in progress**: what to remove and what to factor out, ranked, and the
  order of the removal commits.
- [Scaling models and topologies](scaling-models-and-topologies.md), **trainer on several GPUs built, the rest
  proposed**: many more model families and sizes (MoE, hybrid attention, vision, up to a trillion parameters), tensor,
  pipeline and expert parallelism for engines and trainers, gangs across nodes, how weights move from trainer to
  engines, and a planner that chooses GPU types and counts from a model, a workload, a time target and a budget.

## Training

- [Objectives design](objectives-design.md), **in progress**: objectives as families and components, the
  literature's losses as presets, distillation from a teacher's logprobs, and LLM judges. The composable objective,
  the preference family and the distillation family are built; judges are proposed.
- [The checkpoint graph](policy-dag.md), **proposed**: the graph of checkpoints with what trains and serves them,
  distillation, shared trainers and their queues. The monitor's view of the graph that exists is built.
- [Curricula](curricula.md), **proposed**: building training curricula and frozen evaluation suites from a run's data.
  `Curriculum`, suites and evals are built.
- [Minecraft rewards](minecraft-rewards.md), **built**: the noise and coarseness of curriculum-9's rewards, measured,
  and the reward from 0 to 1 designed from them: half for solving, half for progress along the task's path, no speed
  by default; replayed over the run's episodes.
- [SFT datasets](sft-datasets.md), **built**: datasets made by rejection sampling from a run's episodes, the rules
  measured on one ledger, and the one recommended.
- [Pretraining](pretraining.md), **proposed**: pretraining from nothing and continued pretraining as a run kind
  with a trainer and a data source and no engines: tokenized corpora and mixtures in the blob store, a deterministic
  data source, a contract for trainers that run for days, the options (the platform's own trainer, torchtitan
  wrapped, or neither), and what runs from 125M to 9B parameters cost on one node.

## Other systems

- [Thinking Machines' API](thinking-machines.md), **built**: training and sampling through Tinker as a trainer and an
  engine; a few parts marked proposed.
- [Prime Intellect's verifiers](prime-compat.md), **built**, with parts marked proposed: verifiers environments run
  here, what maps, a spike on a Hub environment, and exporting ours.
