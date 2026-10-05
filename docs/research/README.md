# Design notes

Designs, proposals, measurements and records of spikes. These pages are for people who change the platform and want
the reasoning behind it. Unlike the rest of the documentation, a design note may describe what is not built yet:
each one says at its top whether it is **built**, **in progress** or **proposed**, and marks its parts that differ.

**Read first:** [the architecture](../architecture/overview.md). For what the code does now, the guide and library
pages are the reference; a design note never overrides them.

## Runtime and deployment

- [Runtime design](runtime-design.md), **in progress**: one cluster config, run settings and presets, providers and
  the bridges between their formats, every role on Ray, one gateway, and runs that claim their own resources.
- [RunPod pods as providers](runpod-providers.md), **in progress**: GPU pods on RunPod serving a channel or taking
  training steps, with mutual TLS (mTLS) and step-ca certificates. The images and the code on the pods are built;
  the provider kinds are not wired yet.
- [Ledger guarantees](ledger-guarantees.md), **built**: what the ledger promises its writers and readers (fences,
  claims, retries), each promise with the test that holds it. Its last section proposes an HTTP ledger service.
- [Minecraft memory](minecraft-memory.md), **built**: the memory of the Minecraft team's worlds measured on real
  servers and episodes, the Java and Node settings chosen from it, and why several worlds do not share one server.
- [Cleanup inventory](cleanup-inventory.md), **in progress**: what to remove and what to factor out, ranked, and the
  order of the removal commits.
- [Scaling models and topologies](scaling-models-and-topologies.md), **proposed**: many more model families and
  sizes (MoE, hybrid attention, vision, up to a trillion parameters), tensor, pipeline and expert parallelism for
  engines and trainers, gangs across nodes, how weights move from trainer to engines, and a planner that chooses GPU
  types and counts from a model, a workload, a time target and a budget.

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

## Other systems

- [Thinking Machines' API](thinking-machines.md), **built**: training and sampling through Tinker as a trainer and an
  engine; a few parts marked proposed.
- [Prime Intellect's verifiers](prime-compat.md), **built**, with parts marked proposed: verifiers environments run
  here, what maps, a spike on a Hub environment, and exporting ours.
