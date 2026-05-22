"""Main training loop.

Features:
* Gradient accumulation.
* Optional gradient checkpointing toggle.
* BF16 autocast.
* ZeRO-1 optimizer sharding (via :class:`ZeROOptimizer`) or single-device fallback.
* Tensor-parallel and pipeline-parallel awareness (handled by the model/distributed module).
* Curriculum learning (sequence length / domain mixing).
* Robust checkpointing & resume.
* Prometheus metrics.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..distributed.init import DistributedConfig, init_distributed, is_main_process
from ..distributed.precision import autocast_dtype, mixed_precision
from ..distributed.zero import ZeROOptimizer
from ..model.aris import ArisForCausalLM
from ..model.config import ArisConfig
from ..monitoring.metrics import MetricRegistry
from .checkpoint import CheckpointManager, CheckpointMetadata
from .curriculum import CurriculumScheduler
from .optimizer import build_optimizer, build_scheduler, clip_grad_norm

logger = logging.getLogger("aris.train")


@dataclass
class TrainingConfig:
    total_steps: int = 100_000
    warmup_steps: int = 2000
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 1000
    save_every: int = 1000
    dtype: str = "bf16"
    checkpoint_dir: str = "checkpoints"
    keep_last_n: int = 5
    use_zero: bool = True
    seed: int = 1234

    distributed: DistributedConfig = field(default_factory=DistributedConfig)


class Trainer:
    """Drives pretraining for any ``ArisForCausalLM`` model."""

    def __init__(
        self,
        model_config: ArisConfig,
        train_config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        curriculum: CurriculumScheduler | None = None,
    ) -> None:
        self.model_config = model_config
        self.train_config = train_config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.curriculum = curriculum

        self.ctx = init_distributed(train_config.distributed)
        torch.manual_seed(train_config.seed + self.ctx.dp_rank)

        self.model = ArisForCausalLM(model_config).to(self.ctx.device, dtype=autocast_dtype(train_config.dtype))
        if model_config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        base_optimizer = build_optimizer(
            self.model,
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
            betas=train_config.betas,
        )
        if train_config.use_zero and self.ctx.dp_size > 1:
            self.optimizer = ZeROOptimizer(
                self.model.parameters(),
                optimizer_cls=type(base_optimizer),
                lr=train_config.lr,
                weight_decay=train_config.weight_decay,
                betas=train_config.betas,
            )
            self.scheduler_target = self.optimizer.optimizer
        else:
            self.optimizer = base_optimizer
            self.scheduler_target = base_optimizer
        self.scheduler = build_scheduler(
            self.scheduler_target,
            total_steps=train_config.total_steps,
            warmup_steps=train_config.warmup_steps,
            min_lr_ratio=train_config.min_lr_ratio,
        )

        self.checkpointer = CheckpointManager(train_config.checkpoint_dir, keep_last_n=train_config.keep_last_n)
        self.metrics = MetricRegistry(namespace="aris_train")
        self.global_step = 0
        self.tokens_seen = 0
        self._start_time = time.time()

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def fit(self) -> None:
        cfg = self.train_config
        accum = cfg.gradient_accumulation_steps
        device = self.ctx.device

        self.model.train()
        data_iter = iter(self.train_dataloader)
        while self.global_step < cfg.total_steps:
            self.optimizer.zero_grad(set_to_none=True)
            loss_accum = torch.zeros((), device=device, dtype=torch.float32)
            for _micro_step in range(accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with mixed_precision(autocast_dtype(cfg.dtype)):
                    out = self.model(**batch)
                    loss = out.loss / accum
                loss.backward()
                loss_accum = loss_accum + loss.detach()

            grad_norm = clip_grad_norm(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1
            tokens_step = cfg.micro_batch_size * accum * batch["input_ids"].shape[-1] * self.ctx.dp_size
            self.tokens_seen += tokens_step
            if self.curriculum is not None:
                self.curriculum.step(tokens_step)

            if is_main_process() and self.global_step % cfg.log_every == 0:
                self._log_step(loss_accum.item(), grad_norm.item())
            if self.eval_dataloader is not None and self.global_step % cfg.eval_every == 0:
                self.evaluate()
            if self.global_step % cfg.save_every == 0:
                self.save_checkpoint()

        self.save_checkpoint()

    @torch.no_grad()
    def evaluate(self) -> float:
        assert self.eval_dataloader is not None
        self.model.eval()
        total, n = 0.0, 0
        for batch in self.eval_dataloader:
            batch = {k: v.to(self.ctx.device) for k, v in batch.items()}
            with mixed_precision(autocast_dtype(self.train_config.dtype)):
                out = self.model(**batch)
            total += out.loss.item()
            n += 1
            if n >= 50:
                break
        self.model.train()
        avg = total / max(1, n)
        if is_main_process():
            ppl = math.exp(min(20.0, avg))
            logger.info("eval step=%d loss=%.4f ppl=%.2f", self.global_step, avg, ppl)
            self.metrics.gauge("eval_loss").set(avg)
            self.metrics.gauge("eval_ppl").set(ppl)
        return avg

    def save_checkpoint(self) -> Path:
        meta = CheckpointMetadata(
            step=self.global_step,
            epoch=0,
            tokens_seen=self.tokens_seen,
            config_hash="",
            aris_version="0.1.0",
            world_size=self.ctx.world_size,
        )
        return self.checkpointer.save(self.global_step, self.model, self.optimizer, self.scheduler, meta)

    def load_checkpoint(self, path: str | Path) -> None:
        meta = self.checkpointer.load(path, self.model, self.optimizer, self.scheduler)
        self.global_step = meta.step
        self.tokens_seen = meta.tokens_seen
        logger.info("Resumed from step=%d tokens=%d", self.global_step, self.tokens_seen)

    # ------------------------------------------------------------------ #
    def _log_step(self, loss: float, grad_norm: float) -> None:
        elapsed = max(1e-6, time.time() - self._start_time)
        tps = self.tokens_seen / elapsed
        logger.info(
            "step=%d loss=%.4f grad_norm=%.3f lr=%.2e tok/s=%.0f",
            self.global_step,
            loss,
            grad_norm,
            self.scheduler.get_last_lr()[0],
            tps,
        )
        self.metrics.gauge("train_loss").set(loss)
        self.metrics.gauge("grad_norm").set(grad_norm)
        self.metrics.gauge("tokens_per_second").set(tps)
        self.metrics.counter("tokens_total").inc(self.tokens_seen)
