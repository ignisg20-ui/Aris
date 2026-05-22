"""Data loading utilities for pretraining.

We treat the corpus as a tokenized memory-mapped flat array of ``uint32`` ids.
``PackedDataset`` slices the array into fixed-length windows and yields
``(input_ids, labels)`` pairs. This is the same layout used by Megatron-LM /
nanotron / llm-foundry — it is simple, deterministic, and supports trivial
resume by checkpointing only the global step.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info


class PackedDataset(Dataset):
    """Random access over a tokenized memory-mapped corpus."""

    def __init__(self, path: str | Path, seq_len: int, dtype: str = "uint32") -> None:
        self.path = Path(path)
        self.seq_len = seq_len
        self.dtype = np.dtype(dtype)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._mmap: np.memmap | None = None
        nbytes = os.path.getsize(self.path)
        self._num_tokens = nbytes // self.dtype.itemsize
        # We need ``seq_len + 1`` tokens per sample (labels are inputs shifted by one).
        self._num_samples = (self._num_tokens - 1) // self.seq_len

    def _ensure_mmap(self) -> np.memmap:
        if self._mmap is None:
            self._mmap = np.memmap(self.path, dtype=self.dtype, mode="r")
        return self._mmap

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self._num_samples:
            raise IndexError(index)
        start = index * self.seq_len
        chunk = np.asarray(self._ensure_mmap()[start : start + self.seq_len + 1], dtype=np.int64)
        input_ids = torch.from_numpy(chunk[:-1].copy())
        labels = torch.from_numpy(chunk[1:].copy())
        return {"input_ids": input_ids, "labels": labels}


class StreamingShardedDataset(IterableDataset):
    """Iterate multiple ``.bin`` shards in order with deterministic sharding across DP ranks."""

    def __init__(
        self,
        shards: list[str | Path],
        seq_len: int,
        dp_rank: int = 0,
        dp_size: int = 1,
        dtype: str = "uint32",
        shuffle_within_shard: bool = True,
        seed: int = 0,
    ) -> None:
        self.shards = [Path(p) for p in shards]
        self.seq_len = seq_len
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.dtype = np.dtype(dtype)
        self.shuffle_within_shard = shuffle_within_shard
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        nw = worker.num_workers if worker else 1
        wid = worker.id if worker else 0

        for shard_idx, shard in enumerate(self.shards):
            # Shard the shards: each DP rank owns 1/dp_size, each worker 1/nw of that.
            if (shard_idx % self.dp_size) != self.dp_rank:
                continue
            mm = np.memmap(shard, dtype=self.dtype, mode="r")
            num_samples = (len(mm) - 1) // self.seq_len
            order = np.arange(num_samples)
            if self.shuffle_within_shard:
                rng = np.random.default_rng(self.seed + shard_idx)
                rng.shuffle(order)
            for i, sample_idx in enumerate(order):
                if (i % nw) != wid:
                    continue
                start = sample_idx * self.seq_len
                chunk = np.asarray(mm[start : start + self.seq_len + 1], dtype=np.int64)
                yield {
                    "input_ids": torch.from_numpy(chunk[:-1].copy()),
                    "labels": torch.from_numpy(chunk[1:].copy()),
                }


def build_dataloader(
    dataset: Dataset | IterableDataset,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = True,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle if isinstance(dataset, Dataset) else False,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def estimate_tokens(num_samples: int, seq_len: int) -> int:
    return num_samples * seq_len


def chinchilla_target_tokens(num_params: int) -> int:
    """Hoffmann et al. 2022: roughly 20 tokens per parameter."""
    return 20 * num_params


def warmup_steps(total_steps: int, ratio: float = 0.01) -> int:
    return max(100, math.ceil(total_steps * ratio))
