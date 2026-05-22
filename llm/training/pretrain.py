"""Pretraining entry point.

Usage::

    aris-train --config llm/configs/7b.yaml --data data/pretrain --out checkpoints/7b
    # or
    torchrun --nproc_per_node=8 -m llm.training.pretrain \\
        --config llm/configs/70b.yaml --data data/pretrain --tp 8

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..distributed.init import DistributedConfig
from ..model.config import ArisConfig
from .curriculum import CurriculumScheduler, LengthCurriculum
from .data import StreamingShardedDataset, build_dataloader
from .trainer import Trainer, TrainingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("aris-pretrain")
    p.add_argument("--config", required=True, help="model config YAML")
    p.add_argument("--data", required=True, help="directory containing .bin shards")
    p.add_argument("--out", required=True, help="checkpoint directory")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = ArisConfig.from_yaml(args.config)
    train_cfg = TrainingConfig(
        total_steps=args.steps,
        micro_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.accum,
        lr=args.lr,
        checkpoint_dir=args.out,
        distributed=DistributedConfig(tp_size=args.tp, pp_size=args.pp),
    )

    shards = sorted(str(p) for p in Path(args.data).glob("*.bin"))
    if not shards:
        raise SystemExit(f"No .bin shards found under {args.data}")
    dataset = StreamingShardedDataset(shards=shards, seq_len=args.seq_len)
    loader = build_dataloader(dataset, batch_size=args.micro_batch, num_workers=4)

    curriculum = CurriculumScheduler(length=LengthCurriculum(end_seq_len=args.seq_len))

    trainer = Trainer(model_cfg, train_cfg, train_dataloader=loader, curriculum=curriculum)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.fit()


if __name__ == "__main__":
    main()
