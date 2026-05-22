"""Reward model.

Architecture:
* Reuse the SFT-tuned ``ArisModel`` backbone.
* Replace the LM head with a single scalar projection.
* Score per-token; the reward for a sequence is the score at the final
  non-pad token (Bradley-Terry convention).

Training objective (Bradley-Terry over preferences):

    P(y_w ≻ y_l | x) = sigmoid(r(x, y_w) - r(x, y_l))

Loss:

    L_RM = -log sigmoid(r_w - r_l)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..model.aris import ArisModel
from ..model.config import ArisConfig


class RewardModel(nn.Module):
    def __init__(self, config: ArisConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = ArisModel(config)
        self.value_head = nn.Linear(config.hidden_size, 1, bias=False)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden, _ = self.backbone(input_ids, attention_mask=attention_mask)
        # Per-token reward, then index the last non-pad token per row.
        scores = self.value_head(hidden).squeeze(-1)  # (B, T)
        if attention_mask is None:
            return scores[:, -1]
        last_idx = attention_mask.long().sum(dim=-1) - 1
        return scores.gather(1, last_idx.unsqueeze(1)).squeeze(1)


@dataclass
class PreferenceBatch:
    chosen_input_ids: torch.Tensor
    chosen_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_mask: torch.Tensor


class RewardModelTrainer:
    """Bradley-Terry trainer."""

    def __init__(self, model: RewardModel, optimizer: torch.optim.Optimizer, margin: float = 0.0) -> None:
        self.model = model
        self.optimizer = optimizer
        self.margin = margin

    def step(self, batch: PreferenceBatch) -> dict[str, float]:
        r_w = self.model(batch.chosen_input_ids, batch.chosen_mask)
        r_l = self.model(batch.rejected_input_ids, batch.rejected_mask)
        diff = r_w - r_l - self.margin
        loss = -torch.nn.functional.logsigmoid(diff).mean()
        acc = (diff > 0).float().mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return {"rm_loss": loss.item(), "rm_acc": acc.item(), "r_w_mean": r_w.mean().item(), "r_l_mean": r_l.mean().item()}
