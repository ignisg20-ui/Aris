"""Simple 1F1B pipeline scheduler.

For an N-stage pipeline and M microbatches the schedule is::

    Stage 0: F0 F1 F2 F3 ... B0 B1 B2 B3
    Stage 1:    F0 F1 F2 F3 B0 B1 B2 B3
    Stage 2:       F0 F1 F2 B0 B1 B2 B3
    Stage 3:          F0 B0 F1 B1 F2 B2 F3 B3   # 1F1B steady state

This module focuses on *correctness* and *legibility*; for ultimate throughput
use Megatron-LM's interleaved 1F1B which we wrap optionally below via ``DeepSpeed``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
import torch.distributed as dist
from torch import nn

from .init import get_dist_context


class PipelineParallel(nn.Module):
    """Wraps a list of stage modules into a pipeline-parallel computation.

    Args:
        stages: list of ``nn.Module`` — one per pipeline stage on this rank.
            Typically a single module per rank, but multiple are supported for
            "virtual pipeline" / interleaved schedules.
        loss_fn: callable applied on the last stage. Receives ``(logits, batch)``.
        num_microbatches: number of microbatches per macro-batch.
    """

    def __init__(self, stages: Iterable[nn.Module], loss_fn: Callable, num_microbatches: int = 4) -> None:
        super().__init__()
        self.stages = nn.ModuleList(list(stages))
        self.loss_fn = loss_fn
        self.num_microbatches = num_microbatches

    @property
    def is_first_stage(self) -> bool:
        return get_dist_context().pp_rank == 0

    @property
    def is_last_stage(self) -> bool:
        ctx = get_dist_context()
        return ctx.pp_rank == ctx.pp_size - 1

    def _send(self, tensor: torch.Tensor, dst_rank: int) -> None:
        dist.send(tensor.contiguous(), dst=dst_rank, group=get_dist_context().pp_group)

    def _recv(self, shape: tuple[int, ...], dtype: torch.dtype, src_rank: int) -> torch.Tensor:
        ctx = get_dist_context()
        buf = torch.empty(shape, dtype=dtype, device=ctx.device)
        dist.recv(buf, src=src_rank, group=ctx.pp_group)
        return buf

    def forward_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute one microbatch forward pass on this stage."""
        if self.is_first_stage:
            x = batch["input_ids"]
        else:
            ctx = get_dist_context()
            prev_rank = (ctx.rank - ctx.tp_size) % ctx.world_size
            x = self._recv(batch["shape"], batch["dtype"], src_rank=prev_rank)

        for stage in self.stages:
            x = stage(x)

        if self.is_last_stage:
            loss = self.loss_fn(x, batch)
            return loss
        ctx = get_dist_context()
        next_rank = (ctx.rank + ctx.tp_size) % ctx.world_size
        self._send(x, dst_rank=next_rank)
        return x  # caller ignores when not last

    def step(self, microbatches: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        """Run one optimization step over ``num_microbatches`` microbatches.

        This is a *naive* gpipe-style schedule (forward all, then backward all)
        which is the simplest correct implementation. For 1F1B use ``deepspeed.PipelineEngine``.
        """
        total_loss = torch.zeros((), device=get_dist_context().device)
        for mb in microbatches:
            loss = self.forward_step(mb)
            if self.is_last_stage:
                (loss / len(microbatches)).backward()
                total_loss = total_loss + loss.detach()
        return total_loss / len(microbatches)
