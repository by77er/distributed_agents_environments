# Extend the platform

This section is for people who add backends: an inference provider, a trainer, an objective, a model family's
renderer, a blob store or a kind of environment. Each backend implements one interface a library defines, and the
pages here describe the ones that exist.

**Read first:** [the architecture](../architecture/overview.md), for the layers and protocols.
**Next:** the page of the backend closest to yours, below.

## How a backend plugs in

- Libraries define interfaces; implementations are packages of their own that implement one each. The
  [protocols and their implementations](../architecture/overview.md#protocols-and-their-implementations) table lists
  every interface and what implements it.
- A cluster names its backends by kind or as `module:name`, so the libraries import none of them. The
  [cluster config](../guide/cluster.md) lists the kinds of inference provider and trainer, what each declares it can
  do, and the bridges between checkpoint formats.
- Types that cross layers are defined once, under [Contracts](../libraries/rollout/contracts/README.md).
- `tests/test_layers.py` checks what may depend on what: the libraries require no implementation, and an environment
  imports `rollout` and nothing above it.

## Backends that exist

| To add | Interface | Read |
|---|---|---|
| An inference engine | `Engine` (`rollout_train.inference`) | [vLLM engine](../implementations/rollout-vllm.md), [Tinker trainer and engine](../implementations/rollout-tinker.md), [channels and engines](../libraries/rollout-train/channels.md) |
| A trainer | `Trainer` (`rollout_train`) | [LoRA trainer](../implementations/rollout-lora.md), [Tinker trainer and engine](../implementations/rollout-tinker.md), [the trainer protocol](../libraries/rollout-train/training.md#the-trainer) |
| An objective or a loss | `rollout_train.objectives` | [objectives](../libraries/rollout-train/training.md#objectives), [objectives in torch](../implementations/rollout-objectives.md), [objectives design](../research/objectives-design.md) |
| A model family's token format | `Renderer` (`rollout_train.recorder`) | [Qwen renderers](../implementations/rollout-qwen.md), [Gemma renderers](../implementations/rollout-gemma.md), [renderers](../libraries/rollout-train/recorder.md#renderers) |
| A hosted model API for a model slot | `ModelEndpoint` (`rollout.contracts`) | [use a hosted model](../guide/models.md), [model endpoint](../libraries/rollout/contracts/model-endpoint.md) |
| A blob store | `Blobs` (`rollout.harness`) | [messages, files and digests](../guide/content.md#media-and-blobs), [the stores](../guide/cluster.md#the-stores) |
| A checkpoint format bridge | a bridge's task (`module:name`) | [bridges](../libraries/rollout-train/checkpoints.md#bridges), [bridges in the cluster config](../guide/cluster.md#bridges) |
| Environments from another framework | `Environment` (`rollout.environment`) | [verifiers environments](../implementations/rollout-verifiers.md) |
| GPU pods elsewhere as providers | provider kinds `runpod-inference`, `runpod-trainer`, `runpod-host`; leases (`rollout_train.pods.leases`) | [GPU pods on RunPod](../deploy/providers.md#gpu-pods-on-runpod), [RunPod pods as providers](../research/runpod-providers.md) |

## Read in this order

1. [vLLM engine](../implementations/rollout-vllm.md): options, adapters by name, sleep and wake, measurements.
2. [LoRA trainer](../implementations/rollout-lora.md): settings, its processes, the memory bound.
3. [Objectives in torch](../implementations/rollout-objectives.md): an objective's loss from its components, the step,
   metrics.
4. [Tinker trainer and engine](../implementations/rollout-tinker.md): training and sampling at Thinking Machines.
5. [Qwen renderers](../implementations/rollout-qwen.md) and [Gemma renderers](../implementations/rollout-gemma.md):
   model families' token formats.
6. [verifiers environments](../implementations/rollout-verifiers.md): Prime Intellect's environments as environments
   here.
7. [Contracts](../libraries/rollout/contracts/README.md): [identifiers](../libraries/rollout/contracts/identifiers.md),
   [canonical content](../libraries/rollout/contracts/canonical-content.md),
   [run events](../libraries/rollout/contracts/run-events.md), [effects](../libraries/rollout/contracts/effects.md),
   and the [model endpoint](../libraries/rollout/contracts/model-endpoint.md).
