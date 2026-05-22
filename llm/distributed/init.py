"""Bootstrap for distributed training.

Sets up the three parallelism axes:

    World = TP × PP × DP

* **Tensor Parallel (TP)**: shards matmuls within a layer (Megatron-LM style).
* **Pipeline Parallel (PP)**: shards consecutive layers across stages.
* **Data Parallel (DP)**: classical replica-level parallelism, with ZeRO
  optimizer-state sharding layered on top.

We expect the launcher (``torchrun`` / ``deepspeed``) to set the standard
``RANK``/``WORLD_SIZE``/``LOCAL_RANK`` env vars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import torch.distributed as dist


@dataclass
class DistributedConfig:
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int | None = None  # inferred from world size
    backend: str = "nccl"
    init_method: str = "env://"
    seed: int = 1234


@dataclass
class DistributedContext:
    world_size: int
    rank: int
    local_rank: int
    device: torch.device
    tp_size: int
    pp_size: int
    dp_size: int
    tp_rank: int
    pp_rank: int
    dp_rank: int
    tp_group: dist.ProcessGroup | None = field(default=None, repr=False)
    pp_group: dist.ProcessGroup | None = field(default=None, repr=False)
    dp_group: dist.ProcessGroup | None = field(default=None, repr=False)


_CTX: DistributedContext | None = None


def init_distributed(cfg: DistributedConfig) -> DistributedContext:
    """Initialize NCCL process groups for TP, PP and DP."""
    global _CTX

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    elif "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        dist.init_process_group(backend=cfg.backend, init_method=cfg.init_method)
    else:
        # Single-process fallback so the rest of the codebase doesn't need to special-case.
        world_size = 1
        rank = 0

    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))

    if cfg.dp_size is None:
        dp_size = world_size // (cfg.tp_size * cfg.pp_size)
    else:
        dp_size = cfg.dp_size

    if cfg.tp_size * cfg.pp_size * dp_size != world_size:
        raise ValueError(
            f"TP*PP*DP ({cfg.tp_size}*{cfg.pp_size}*{dp_size}) must equal world_size ({world_size})"
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    tp_rank, pp_rank, dp_rank, tp_group, pp_group, dp_group = (0, 0, 0, None, None, None)
    if world_size > 1 and dist.is_initialized():
        tp_group, pp_group, dp_group = _build_groups(world_size, cfg.tp_size, cfg.pp_size, dp_size)
        tp_rank = rank % cfg.tp_size
        pp_rank = (rank // cfg.tp_size) % cfg.pp_size
        dp_rank = rank // (cfg.tp_size * cfg.pp_size)

    torch.manual_seed(cfg.seed + dp_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed + dp_rank)

    _CTX = DistributedContext(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        device=device,
        tp_size=cfg.tp_size,
        pp_size=cfg.pp_size,
        dp_size=dp_size,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
        dp_rank=dp_rank,
        tp_group=tp_group,
        pp_group=pp_group,
        dp_group=dp_group,
    )
    return _CTX


def _build_groups(world: int, tp: int, pp: int, dp: int):
    """Compute Megatron-style group decomposition.

    Layout: rank = dp_rank * (pp*tp) + pp_rank * tp + tp_rank.
    """
    tp_groups, pp_groups, dp_groups = [], [], []
    # TP groups: contiguous along innermost axis.
    for d in range(dp):
        for p in range(pp):
            ranks = [d * pp * tp + p * tp + t for t in range(tp)]
            tp_groups.append(dist.new_group(ranks=ranks))
    # PP groups: across pp axis for each (dp, tp).
    for d in range(dp):
        for t in range(tp):
            ranks = [d * pp * tp + p * tp + t for p in range(pp)]
            pp_groups.append(dist.new_group(ranks=ranks))
    # DP groups: across dp axis for each (pp, tp).
    for p in range(pp):
        for t in range(tp):
            ranks = [d * pp * tp + p * tp + t for d in range(dp)]
            dp_groups.append(dist.new_group(ranks=ranks))

    my_rank = dist.get_rank()

    def find(groups: list, my: int):
        for g in groups:
            if my in dist.get_process_group_ranks(g):
                return g
        return None

    return find(tp_groups, my_rank), find(pp_groups, my_rank), find(dp_groups, my_rank)


def get_dist_context() -> DistributedContext:
    if _CTX is None:
        raise RuntimeError("Distributed context not initialized. Call init_distributed first.")
    return _CTX


def is_main_process() -> bool:
    if _CTX is None:
        return True
    return _CTX.rank == 0
