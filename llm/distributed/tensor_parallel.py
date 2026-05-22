"""Megatron-style tensor-parallel primitives.

A *column-parallel* linear shards the output dimension across TP ranks::

    Y = X · A      where  A = [A_1 | A_2 | ... | A_t]
    Y_local = X · A_i

A *row-parallel* linear shards the input dimension::

    Y = X · A      where  A = [A_1 ; A_2 ; ... ; A_t]^T
    Y_local = X_i · A_i, then all-reduce across ranks.

When ``gather_output=False`` for column-parallel, the next layer is expected to be
row-parallel so the activation stays sharded — no extra communication. This is the
standard pattern used for QKV → O and Gate/Up → Down inside transformer blocks.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .init import get_dist_context


def _all_reduce(x: torch.Tensor, group) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return x
    dist.all_reduce(x, group=group)
    return x


def _all_gather(x: torch.Tensor, group, dim: int = -1) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return x
    world = dist.get_world_size(group)
    buf = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(buf, x.contiguous(), group=group)
    return torch.cat(buf, dim=dim)


class _ReduceFromTPRegion(torch.autograd.Function):
    """All-reduce in forward, identity in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:  # type: ignore[override]
        ctx.group = group
        return _all_reduce(x.clone(), group)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        return grad, None


class _CopyToTPRegion(torch.autograd.Function):
    """Identity in forward, all-reduce in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:  # type: ignore[override]
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        return _all_reduce(grad.clone(), ctx.group), None


class ColumnParallelLinear(nn.Module):
    """Linear layer with output dimension sharded across TP ranks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        gather_output: bool = True,
        init_method=nn.init.xavier_uniform_,
    ) -> None:
        super().__init__()
        ctx = get_dist_context()
        if out_features % ctx.tp_size != 0:
            raise ValueError(f"out_features ({out_features}) not divisible by tp_size ({ctx.tp_size})")
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_partition = out_features // ctx.tp_size
        self.gather_output = gather_output
        self.tp_group = ctx.tp_group

        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)
        init_method(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _CopyToTPRegion.apply(x, self.tp_group)
        out = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            out = _all_gather(out, self.tp_group, dim=-1)
        return out


class RowParallelLinear(nn.Module):
    """Linear layer with input dimension sharded across TP ranks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        input_is_parallel: bool = True,
        init_method=nn.init.xavier_uniform_,
    ) -> None:
        super().__init__()
        ctx = get_dist_context()
        if in_features % ctx.tp_size != 0:
            raise ValueError(f"in_features ({in_features}) not divisible by tp_size ({ctx.tp_size})")
        self.in_features = in_features
        self.out_features = out_features
        self.in_features_per_partition = in_features // ctx.tp_size
        self.input_is_parallel = input_is_parallel
        self.tp_group = ctx.tp_group

        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        init_method(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            # Scatter from gathered activation: take our shard along the last dim.
            ctx = get_dist_context()
            x = x.chunk(ctx.tp_size, dim=-1)[ctx.tp_rank]
        out = F.linear(x, self.weight)
        out = _ReduceFromTPRegion.apply(out, self.tp_group)
        if self.bias is not None:
            out = out + self.bias
        return out


class VocabParallelEmbedding(nn.Module):
    """Embedding with the vocabulary dimension sharded across TP ranks.

    Tokens outside the local shard return zero; an all-reduce sums contributions
    from every rank.
    """

    def __init__(self, vocab_size: int, hidden_size: int, init_std: float = 0.02) -> None:
        super().__init__()
        ctx = get_dist_context()
        # Pad vocab so it divides evenly.
        pad = (-vocab_size) % ctx.tp_size
        padded = vocab_size + pad
        self.vocab_size = vocab_size
        self.padded_vocab_size = padded
        self.vocab_per_partition = padded // ctx.tp_size
        self.vocab_start = ctx.tp_rank * self.vocab_per_partition
        self.vocab_end = self.vocab_start + self.vocab_per_partition
        self.tp_group = ctx.tp_group

        self.weight = nn.Parameter(torch.empty(self.vocab_per_partition, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        local_ids = (input_ids - self.vocab_start) * mask
        out = F.embedding(local_ids, self.weight)
        out = out * mask.unsqueeze(-1)
        return _ReduceFromTPRegion.apply(out, self.tp_group)
