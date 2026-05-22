"""Supervised fine-tuning.

Standard recipe:
* Load a pretrained ``ArisForCausalLM`` checkpoint.
* Apply chat template to ``(messages, response)`` pairs.
* Compute cross-entropy loss on assistant tokens only (mask everything else).
* Train with low LR (1e-5 .. 5e-5) for a few epochs.

The dataset is expected to yield dicts with ``"messages": list[ChatMessage]`` where
the *last* message has role ``"assistant"`` and is the target.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..tokenizer.tokenizer import ArisTokenizer, ChatMessage


@dataclass
class SFTConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_steps: int = 50
    total_epochs: int = 3
    max_seq_len: int = 4096
    pack_sequences: bool = True
    label_loss_only_on_assistant: bool = True


class SFTDataset(Dataset):
    """JSONL dataset of ``{"messages": [...]}`` records.

    Loss-masking is computed eagerly when ``label_loss_only_on_assistant=True``
    by re-tokenizing prefix/target separately.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: ArisTokenizer,
        max_seq_len: int = 4096,
        label_loss_only_on_assistant: bool = True,
    ) -> None:
        self.records = [json.loads(line) for line in Path(jsonl_path).read_text().splitlines() if line.strip()]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.label_loss_only_on_assistant = label_loss_only_on_assistant

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        messages = [ChatMessage(**m) for m in record["messages"]]
        if messages[-1].role != "assistant":
            raise ValueError(f"Record {idx} must end with assistant message")

        # Build inputs in two parts: prefix (no loss) + assistant turn (loss).
        prefix_msgs = messages[:-1]
        prefix_ids = self.tokenizer.apply_chat_template(prefix_msgs, add_generation_prompt=True)
        full_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False)
        full_ids[len(prefix_ids) :]

        input_ids = full_ids[: self.max_seq_len]
        labels = list(input_ids)
        if self.label_loss_only_on_assistant:
            for i in range(min(len(prefix_ids), len(labels))):
                labels[i] = self.tokenizer.pad_token_id  # ignored by CE

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def packed_collate(batch: Iterable[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a batch to the longest sequence."""
    items = list(batch)
    max_len = max(it["input_ids"].shape[0] for it in items)
    input_ids = torch.full((len(items), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(items), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(items), max_len), dtype=torch.long)
    for i, it in enumerate(items):
        n = it["input_ids"].shape[0]
        input_ids[i, :n] = it["input_ids"]
        labels[i, :n] = it["labels"]
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class SFTTrainer:
    """Thin wrapper that reuses the pretrainer for SFT-specific defaults."""

    def __init__(self, base_trainer, sft_config: SFTConfig) -> None:
        self.base = base_trainer
        self.sft_config = sft_config
        # Override LR & weight decay on the underlying optimizer.
        for g in self.base.optimizer.param_groups if hasattr(self.base.optimizer, "param_groups") else self.base.optimizer.optimizer.param_groups:
            g["lr"] = sft_config.learning_rate
            g["weight_decay"] = sft_config.weight_decay

    def fit(self) -> None:
        self.base.fit()


def make_loader_iter(dataset: SFTDataset, batch_size: int, tokenizer: ArisTokenizer) -> Iterator:
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: packed_collate(b, pad_id=tokenizer.pad_token_id),
    )


def main() -> None:
    raise SystemExit("Run via `aris-finetune` once a checkpoint is available.")
