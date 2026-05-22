"""Optimizer & LR scheduler factories."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn


def _split_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Standard split: weight decay only on dim>=2 parameters (no decay on norms / biases / embeddings)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or name.endswith(".bias") or "norm" in name.lower() or "embed" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.1,
    fused: bool = True,
) -> torch.optim.Optimizer:
    groups = _split_param_groups(model, weight_decay)
    kw = {"lr": lr, "betas": betas, "eps": eps}
    if fused and torch.cuda.is_available():
        try:
            return torch.optim.AdamW(groups, fused=True, **kw)
        except TypeError:  # older torch without fused
            pass
    return torch.optim.AdamW(groups, **kw)


class WarmupCosineSchedule(torch.optim.lr_scheduler.LRScheduler):
    """Linear warmup → cosine decay to ``min_lr_ratio * lr``."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = step / self.warmup_steps
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos
        return [base_lr * scale for base_lr in self.base_lrs]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
) -> WarmupCosineSchedule:
    return WarmupCosineSchedule(optimizer, warmup_steps, total_steps, min_lr_ratio=min_lr_ratio)


def clip_grad_norm(params: Iterable[nn.Parameter], max_norm: float) -> torch.Tensor:
    return torch.nn.utils.clip_grad_norm_([p for p in params if p.grad is not None], max_norm=max_norm)
