"""Sharded checkpoint manager.

Layout::

    checkpoints/step_000123/
        model_rank_0000.safetensors
        model_rank_0001.safetensors
        ...
        optimizer_rank_0000.pt
        ...
        rng_state_rank_0000.pt
        metadata.json

Only rank 0 writes ``metadata.json`` and atomic ``latest`` symlink. Other ranks
write their own model+optimizer shards. Resume is symmetric: each rank loads its
own shard, and DP groups all-gather the consistent global step.

For full ZeRO-3 or DeepSpeed engines we delegate save/load to those libraries.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from ..distributed.init import get_dist_context, is_main_process


@dataclass
class CheckpointMetadata:
    step: int
    epoch: int
    tokens_seen: int
    config_hash: str
    aris_version: str
    world_size: int


class CheckpointManager:
    def __init__(self, root: str | Path, keep_last_n: int = 3) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        metadata: CheckpointMetadata,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        ctx = None
        try:
            ctx = get_dist_context()
            rank = ctx.rank
            world_size = ctx.world_size
        except RuntimeError:
            rank, world_size = 0, 1

        step_dir = self.root / f"step_{step:08d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Save model shard. Prefer safetensors when possible.
        model_path = step_dir / f"model_rank_{rank:04d}.pt"
        torch.save({"model": model.state_dict()}, model_path)

        opt_path = step_dir / f"optimizer_rank_{rank:04d}.pt"
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
            },
            opt_path,
        )

        rng_path = step_dir / f"rng_state_rank_{rank:04d}.pt"
        torch.save(
            {
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            rng_path,
        )

        if is_main_process():
            meta = asdict(metadata)
            meta["world_size"] = world_size
            meta["extra"] = extra or {}
            (step_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
            self._update_latest_symlink(step_dir)
            self._gc()

        return step_dir

    def load(
        self,
        step_dir: str | Path,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> CheckpointMetadata:
        step_dir = Path(step_dir)
        try:
            ctx = get_dist_context()
            rank = ctx.rank
        except RuntimeError:
            rank = 0

        ckpt = torch.load(step_dir / f"model_rank_{rank:04d}.pt", map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)} (load on rank {rank})")

        if optimizer is not None:
            opt = torch.load(step_dir / f"optimizer_rank_{rank:04d}.pt", map_location="cpu")
            optimizer.load_state_dict(opt["optimizer"])
            if scheduler is not None and opt.get("scheduler") is not None:
                scheduler.load_state_dict(opt["scheduler"])

        rng = torch.load(step_dir / f"rng_state_rank_{rank:04d}.pt", map_location="cpu")
        torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

        meta = json.loads((step_dir / "metadata.json").read_text())
        meta.pop("extra", None)
        meta.pop("world_size", None)
        return CheckpointMetadata(**meta)

    # ------------------------------------------------------------------ #
    def find_latest(self) -> Path | None:
        latest = self.root / "latest"
        if latest.exists():
            return latest.resolve()
        candidates = sorted(self.root.glob("step_*"))
        return candidates[-1] if candidates else None

    def _update_latest_symlink(self, step_dir: Path) -> None:
        latest = self.root / "latest"
        if latest.is_symlink() or latest.exists():
            try:
                latest.unlink()
            except IsADirectoryError:
                shutil.rmtree(latest)
        try:
            os.symlink(step_dir.name, latest)
        except OSError:
            # Fallback on filesystems without symlink support: write a pointer file.
            latest.write_text(step_dir.name)

    def _gc(self) -> None:
        if self.keep_last_n <= 0:
            return
        steps = sorted([p for p in self.root.glob("step_*") if p.is_dir()])
        for old in steps[: -self.keep_last_n]:
            shutil.rmtree(old, ignore_errors=True)
