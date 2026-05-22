"""Learned harmlessness classifier.

Architecture: small encoder (typically a 0.1–1B Aris model) with a binary
classification head on the final ``[CLS]`` token. Trained on
``(prompt, response, harmful_flag)`` triples from the Anthropic HH-RLHF /
ToxicChat / BeaverTails datasets.

At inference time, the classifier runs *after* the rule-based refusal policy
and *before* the response is streamed to the user.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..model.aris import ArisModel
from ..model.config import ArisConfig


@dataclass
class HarmlessnessConfig:
    threshold: float = 0.5
    label_names: tuple[str, ...] = ("safe", "harmful")


class HarmlessnessClassifier(nn.Module):
    def __init__(self, model_config: ArisConfig, cfg: HarmlessnessConfig | None = None) -> None:
        super().__init__()
        self.config = cfg or HarmlessnessConfig()
        self.backbone = ArisModel(model_config)
        self.head = nn.Linear(model_config.hidden_size, len(self.config.label_names))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden, _ = self.backbone(input_ids, attention_mask=attention_mask)
        if attention_mask is None:
            cls = hidden[:, -1]
        else:
            last = attention_mask.long().sum(dim=-1) - 1
            cls = hidden.gather(1, last.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])).squeeze(1)
        return self.head(cls)

    @torch.no_grad()
    def score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.forward(input_ids, attention_mask)
        return F.softmax(logits, dim=-1)[:, 1]  # probability of "harmful"

    @torch.no_grad()
    def is_harmful(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.score(input_ids, attention_mask) > self.config.threshold
