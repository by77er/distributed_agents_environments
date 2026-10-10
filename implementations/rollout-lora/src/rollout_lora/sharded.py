# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# (torch.distributed's FSDP2 and checkpoint APIs are partly untyped.)
"""A policy sharded over the GPUs of one machine with PyTorch's FSDP2, and the files a sharded step leaves.

Each process (one per GPU: `rollout_lora.workers`) loads the policy onto the CPU and shards it, the processes taking
turns (`in_turn`) so that the machine holds one unsharded copy at a time. Each decoder layer is a unit of its own
(`fully_shard`), and the rest of the model (the output layer, the last norm, embeddings held on the GPU) is its root's,
the policy's `Scorer`. A unit's weights are gathered from every GPU's shard when it computes, and its gradients reduced
to the shards in float32, added up rather than averaged (`summed`: the step divides each item's loss by its
minibatch's units, `rollout_objectives.step`). Activations are checkpointed as on one GPU.

- **An adapter** (`shard_adapter`): the frozen model is sharded with the adapter. Each decoder layer's adapter layers
  are a unit of their own, inside the layer's, gathered for each pass, kept in float32 and multiplied in the
  activations' dtype as on one GPU. Where each GPU holds the whole frozen model (`LoraSettings.whole_base`), the
  frozen layers are gathered once and kept, and the adapter's gradients are kept in each process until a minibatch's
  last pass, which reduces them once (`gradient_sync`); else each layer's frozen weights are gathered as it computes,
  and each pass's gradients reduced. The output layer is frozen, and kept once gathered. The frozen model is held in
  bfloat16, so gathering it casts nothing.
- **Every weight** (`shard_full`, on one GPU too, so that one GPU computes as several do): the weights, their
  gradients and Adam's moments in float32, sharded, each unit's weights gathered in bfloat16 (`MixedPrecisionPolicy`:
  the forward and backward passes in bfloat16, the residual stream and norms too); a frozen reference, where one is
  asked for, sharded beside it. Each pass's gradients are reduced as it ends: kept until a minibatch's last, the whole
  model's unsharded gradients would be in every process. On the CPU (gloo, for tests) full weights compute in float32.

Rank 0 writes a step's files from tensors every process gathers in the same order, one process the same as several:

- an adapter, as the serving copy engines load, in PEFT's layout in bfloat16 (`write_adapter`, which gives rank 0 the
  float32 adapter it gathered, a copy in host memory); and its optimizer's state, gathered into host memory in the
  layout of one process's state (`gathered_optimizer`, `read_optimizer`): trainers on one GPU and on several go on from
  each other's files;
- full weights, as the serving copy engines load (`write_serving_copy`): bfloat16 safetensors in files of at most
  4 GB, an index, the model's configuration and tokenizer; and, when a step writes its full state, the float32
  weights and the optimizer's state with PyTorch's distributed checkpoint (`write_state`, under
  `rollout_lora.workers.SHARDS`, on disk or in the process's memory: every process writes its shards, a file for each
  tensor's, and however many processes read them take their own shares, `read_state`).
"""

import copy
import json
import os
import shutil
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from rollout_lora.full import FullPolicy
from rollout_lora.layers import LoraLinear, adapter_tensors, host_copy, save_adapter
from rollout_lora.models import COPIED, local
from rollout_lora.policy import Policy, layers_of
from rollout_objectives.ranks import Ranks

__all__ = [
    "MEMORY",
    "TIMEOUT",
    "gathered_optimizer",
    "gradient_sync",
    "in_turn",
    "joined",
    "read_optimizer",
    "read_state",
    "shard_adapter",
    "shard_full",
    "state_bytes",
    "summed",
    "whole",
    "write_adapter",
    "write_serving_copy",
    "write_state",
]

TIMEOUT = timedelta(hours=1)
"""How long a process waits for the others at a collective: as long as one takes to load its share of a large model,
or rank 0 to write a step's files."""
SERVING_FILE_BYTES = 4 * 2**30
"""The most a file of a serving copy holds."""
MEMORY = "memory://"
"""Where `write_state` writes into the process's own memory (an in-memory filesystem's paths begin so)."""


def joined(device: str) -> tuple[Ranks, torch.device, Any]:
    """This process among those torchrun started (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`): its process group (NCCL on the
    GPUs and gloo on the CPU, or gloo alone with `device = "cpu"`), the GPU it computes on, and their device mesh."""
    import torch.distributed as distributed
    from torch.distributed.device_mesh import init_device_mesh

    rank, size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if device == "cuda":
        found = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(found)
        # (NCCL set up at once on several GPUs; one never calls it, and spends none of the GPU's memory on it)
        distributed.init_process_group("cuda:nccl,cpu:gloo", timeout=TIMEOUT, device_id=found if size > 1 else None)
    else:
        found = torch.device("cpu")
        distributed.init_process_group("gloo", timeout=TIMEOUT)
    mesh = init_device_mesh(found.type, (size,))
    return Ranks(rank, size, distributed.new_group(backend="gloo", timeout=TIMEOUT)), found, mesh


def in_turn[T](make: Callable[[], T], ranks: Ranks) -> T:
    """`make()` in each process in turn, the others waiting: a policy loaded onto the CPU and sharded onto the GPU, one
    process at a time, holds one unsharded copy of the model in the machine's memory at once."""
    import torch.distributed as distributed

    made: T | None = None
    for turn in range(ranks.size):
        if turn == ranks.rank:
            made = make()
        if ranks.shared:
            distributed.barrier(group=ranks.group)
    return cast(T, made)


def _precision(device: torch.device | None = None) -> Any:
    """Weights gathered in bfloat16 and gradients reduced in float32 (on the CPU, where `device` says it, nothing
    cast); without a device, nothing cast, on any (an adapter's units: float32 as on one GPU)."""
    from torch.distributed.fsdp import MixedPrecisionPolicy

    if device is None or device.type != "cuda":
        return MixedPrecisionPolicy()
    return MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)


def _frozen() -> Any:
    """An adapter's frozen units, on every device: gathered in bfloat16, as they are held (nothing is reduced)."""
    from torch.distributed.fsdp import MixedPrecisionPolicy

    return MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)


def summed(root: nn.Module) -> None:
    """Have every sharded unit under `root` add its gradients up across processes, not average them."""
    from torch.distributed.fsdp import FSDPModule

    for module in root.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)
            module.set_force_sum_reduction_for_comms(True)  # (a sum, which gloo takes too)


def shard_adapter(policy: Policy, mesh: Any, device: torch.device, *, whole_base: bool) -> None:
    """Shard an adapter's policy (loaded onto the CPU) onto `device`: the frozen model whole on each GPU, or a share;
    each layer's adapter layers a unit of their own, in float32."""
    from torch.distributed.fsdp import FSDPModule, fully_shard

    frozen = _frozen()
    for layer in layers_of(policy.model):
        adapters: list[nn.Module] = [part for each in layer.modules() if isinstance(each, LoraLinear)
                                     for part in (each.lora_A, each.lora_B)]  # fmt: skip
        fully_shard(adapters, mesh=mesh, mp_policy=_precision())  # (one unit, gathered once its first runs)
        if whole_base:
            fully_shard(layer, mesh=mesh, mp_policy=frozen, reshard_after_forward=False)
            cast(FSDPModule, layer).set_reshard_after_backward(False, recurse=False)
        else:
            fully_shard(layer, mesh=mesh, mp_policy=frozen)
    fully_shard(policy.scorer, mesh=mesh, mp_policy=frozen)
    cast(FSDPModule, policy.scorer).set_reshard_after_backward(False, recurse=False)  # (frozen: kept once gathered)
    summed(policy.scorer)
    if whole_base and mesh.size() > 1:  # (only the adapter's units communicate: their reductions once a minibatch)
        policy.gradient_sync = lambda last: gradient_sync(policy, last)


def gradient_sync(policy: Policy, last: bool) -> None:
    """Before one of a minibatch's gradient passes: have it reduce the adapter's gradients across processes (`last`:
    what each process kept of the passes before, and its own), or keep them in each process, in float32 (the
    others)."""
    from torch.distributed.fsdp import FSDPModule

    cast(FSDPModule, policy.scorer).set_requires_gradient_sync(last)


def shard_full(policy: FullPolicy, mesh: Any, device: torch.device) -> None:
    """Shard a full-weight policy (loaded onto the CPU) onto `device`, and its frozen reference if it holds one."""
    from torch.distributed.fsdp import fully_shard

    precision = _precision(device)
    for layer in layers_of(policy.model):
        fully_shard(layer, mesh=mesh, mp_policy=precision)
    fully_shard(policy.scorer, mesh=mesh, mp_policy=precision)
    summed(policy.scorer)
    if policy.frozen is not None and policy.frozen_scorer is not None:
        for layer in layers_of(policy.frozen):
            fully_shard(layer, mesh=mesh, mp_policy=precision)
        fully_shard(policy.frozen_scorer, mesh=mesh, mp_policy=precision)


def whole(tensor: torch.Tensor) -> torch.Tensor:
    """A tensor gathered from its shards (every process calls it, in the same order), or as it is if it is not
    sharded."""
    from torch.distributed.tensor import DTensor

    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def write_adapter(policy: Policy, directory: Path, ranks: Ranks) -> dict[str, torch.Tensor] | None:
    """The adapter gathered, and written by rank 0 in PEFT's layout in bfloat16, what engines load; rank 0's float32
    adapter, a copy in host memory (none in the others)."""
    tensors = adapter_tensors(policy.model, whole)
    if ranks.rank != 0:
        return None
    served = {name: each.to(torch.bfloat16) for name, each in tensors.items()}
    save_adapter(policy.model, directory, base_model=policy.checkpoint, rank=policy.rank, alpha=policy.alpha,
                 tensors=served)  # fmt: skip
    return tensors


def gathered_optimizer(optimizer: torch.optim.Optimizer, ranks: Ranks) -> dict[str, Any] | None:
    """The optimizer's state gathered, as one GPU's step has it (its `state_dict`, each parameter's by its index), a
    copy in rank 0's host memory (none in the others): what the steps after do does not change it."""
    said = optimizer.state_dict()
    state: dict[int, dict[str, Any]] = {}
    for index in sorted(said["state"]):
        state[index] = {}
        for key, value in said["state"][index].items():
            gathered = whole(value).detach() if isinstance(value, torch.Tensor) else value  # (every process gathers)
            if ranks.rank == 0:
                state[index][key] = host_copy(gathered) if isinstance(gathered, torch.Tensor) else gathered
    if ranks.rank != 0:
        return None
    return {"state": state, "param_groups": copy.deepcopy(said["param_groups"])}


def read_optimizer(optimizer: torch.optim.Optimizer, path: Path, parameters: Sequence[nn.Parameter]) -> None:
    """An optimizer's state as one GPU's step wrote it (`write_optimizer`'s layout), each tensor sharded as its
    parameter is."""
    from torch.distributed.tensor import DTensor, distribute_tensor

    saved = torch.load(path, map_location="cpu", weights_only=True)
    state: dict[int, dict[str, Any]] = {}
    for index, values in saved["state"].items():
        parameter = parameters[int(index)]
        state[int(index)] = {}
        for key, value in values.items():
            if isinstance(parameter, DTensor) and isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                value = distribute_tensor(
                    value.to(parameter.device), parameter.device_mesh, parameter.placements, src_data_rank=None
                )
            state[int(index)][key] = value
    optimizer.load_state_dict({"state": state, "param_groups": saved["param_groups"]})


def write_serving_copy(policy: FullPolicy, directory: Path, ranks: Ranks, *, source: str) -> None:
    """Full weights gathered in bfloat16, and written by rank 0 as engines load them: safetensors files of at most
    `SERVING_FILE_BYTES`, their index, and the model's configuration (saying `bfloat16`) and the tokenizer and other
    files of `source` (the model trained)."""
    from safetensors.torch import save_file

    model = policy.model
    named = [(name, each) for name, each in model.named_parameters()]  # (tied weights once)
    persistent = set(model.state_dict())
    named += [(name, each) for name, each in model.named_buffers() if name in persistent]
    files: list[tuple[str, list[str]]] = []
    current: dict[str, torch.Tensor] = {}
    held = 0

    def flush() -> None:
        nonlocal current, held
        name = f"model-{len(files) + 1:05d}.safetensors"
        save_file(current, str(directory / name), metadata={"format": "pt"})
        files.append((name, list(current)))
        current, held = {}, 0

    if ranks.rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
    for name, tensor in named:
        value = tensor.detach()
        if value.is_floating_point():
            value = value.to(torch.bfloat16)  # (cast before it is gathered: half the bytes)
        gathered = whole(value)
        if ranks.rank != 0:
            continue
        current[name] = gathered.cpu().contiguous()
        held += current[name].numel() * current[name].element_size()
        if held >= SERVING_FILE_BYTES:
            flush()
    if ranks.rank != 0:
        return
    if current:
        flush()
    weight_map: dict[str, str] = {}
    for name, keys in files:
        final = name.replace(".safetensors", f"-of-{len(files):05d}.safetensors")
        os.replace(directory / name, directory / final)
        weight_map |= dict.fromkeys(keys, final)
    total = sum(each.numel() * 2 for _, each in named)
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    _described(directory, source, model)


def _described(directory: Path, source: str, model: nn.Module) -> None:
    """The model's configuration (saying `bfloat16`, as the serving copy is) and its other files, beside its
    weights."""
    said: dict[str, Any] = cast(Any, model).config.to_dict()
    said |= {key: "bfloat16" for key in ("torch_dtype", "dtype") if key in said}
    (directory / "config.json").write_text(json.dumps(said, indent=2))
    found = local(source)
    for name in COPIED:
        if (found / name).exists() and not (directory / name).exists():
            shutil.copy2(found / name, directory / name)


def state_bytes(model: nn.Module) -> int:
    """How many bytes this process's shares of a full-weight trainer's full state take: its trained weights and Adam's
    two moments of each, in float32."""
    from torch.distributed.tensor import DTensor

    trained = [each for each in model.parameters() if each.requires_grad]
    return sum(3 * 4 * (each.to_local() if isinstance(each, DTensor) else each).numel() for each in trained)


def write_state(model: nn.Module, optimizer: torch.optim.Optimizer, where: Path | str, ranks: Ranks) -> None:
    """The trainer's full state (the model's trained weights and the optimizer's state), every process writing its
    shards, a file for each tensor's (so no file is larger than a share of one tensor): into a directory, or, where
    `where` begins with `MEMORY`, into the process's own memory (each its own files, rank 0 the checkpoint's metadata
    too). Every process calls it at once (the processes agree on the files as they write)."""
    from torch.distributed.checkpoint._fsspec_filesystem import FsspecWriter  # (into memory, through fsspec's)
    from torch.distributed.checkpoint.filesystem import FileSystemWriter
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
    from torch.distributed.checkpoint.state_dict_saver import save

    options = StateDictOptions(ignore_frozen_params=True)
    model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
    if str(where).startswith(MEMORY):
        writer: Any = FsspecWriter(str(where), single_file_per_rank=False, sync_files=False)
    else:
        writer = FileSystemWriter(str(where), single_file_per_rank=False)
    save({"model": model_state, "optimizer": optimizer_state}, storage_writer=writer, process_group=ranks.group)


def read_state(model: nn.Module, optimizer: torch.optim.Optimizer, directory: Path, ranks: Ranks) -> None:
    """A full state `write_state` wrote, by however many processes, into the model and the optimizer."""
    from torch.distributed.checkpoint.filesystem import FileSystemReader
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
    from torch.distributed.checkpoint.state_dict_loader import load

    options = StateDictOptions(ignore_frozen_params=True)
    model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
    load({"model": model_state, "optimizer": optimizer_state}, storage_reader=FileSystemReader(str(directory)),
         process_group=ranks.group)  # fmt: skip
    set_state_dict(model, optimizer, model_state_dict=model_state, optim_state_dict=optimizer_state, options=options)
