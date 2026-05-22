"""ZeRO optimizer-state sharding.

This is a minimal ZeRO-1 implementation: optimizer states are sharded across DP
ranks; each rank only updates the parameters it owns and an all-gather restores
the full parameter set after the step.

For ZeRO-2 / ZeRO-3 (gradient & parameter sharding), prefer the
:func:`build_deepspeed_engine` helper which delegates to DeepSpeed.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch import nn

from .init import get_dist_context


class ZeROOptimizer:
    """Wraps a regular optimizer and shards its state across DP ranks."""

    def __init__(self, params: Iterable[nn.Parameter], optimizer_cls=torch.optim.AdamW, **opt_kwargs) -> None:
        ctx = get_dist_context()
        params = list(params)
        self.dp_group = ctx.dp_group
        self.dp_size = ctx.dp_size
        self.dp_rank = ctx.dp_rank
        self.params = params

        # Partition parameters round-robin by index → simple, balanced for transformer blocks.
        self._shards: list[list[nn.Parameter]] = [[] for _ in range(self.dp_size)]
        for i, p in enumerate(params):
            self._shards[i % self.dp_size].append(p)
        owned = self._shards[self.dp_rank]
        self.optimizer = optimizer_cls(owned, **opt_kwargs)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        if self.dp_group is not None and self.dp_size > 1:
            # Reduce-scatter would be more efficient; we use all-reduce + take-shard for clarity.
            for p in self.params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, group=self.dp_group)
                    p.grad.div_(self.dp_size)

        self.optimizer.step()

        # Re-broadcast updated parameters to peers.
        if self.dp_group is not None and self.dp_size > 1:
            for rank_id, shard in enumerate(self._shards):
                src = dist.get_global_rank(self.dp_group, rank_id) if hasattr(dist, "get_global_rank") else rank_id
                for p in shard:
                    dist.broadcast(p.data, src=src, group=self.dp_group)

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict(), "dp_rank": self.dp_rank, "dp_size": self.dp_size}

    def load_state_dict(self, state: dict) -> None:
        if state.get("dp_size") != self.dp_size:
            raise RuntimeError(
                f"checkpoint dp_size={state.get('dp_size')} != runtime dp_size={self.dp_size}; "
                "use the conversion utility before resuming."
            )
        self.optimizer.load_state_dict(state["optimizer"])


def build_deepspeed_engine(model: nn.Module, config: dict):
    """Optional: hand control to DeepSpeed for ZeRO-2/3 + offloading."""
    import deepspeed  # type: ignore

    engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
    return engine, optimizer
