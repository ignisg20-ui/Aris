"""Best-of-N rejection sampling.

Given a prompt:
1. Sample ``N`` candidate responses from the policy (with high temperature for
   diversity).
2. Score each with the reward model.
3. Keep the top-``K`` (typically K=1) responses.

This is both a *finetuning* strategy (collect a high-quality SFT dataset by
self-distillation) and an *inference* strategy (just take the best-of-N at
serving time).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .reward_model import RewardModel


@dataclass
class RejectionSampler:
    generate_fn: Callable[[str, int], list[str]]
    reward_model: RewardModel
    tokenizer_encode: Callable[[str], list[int]]
    device: torch.device

    def sample(self, prompt: str, n: int = 16, top_k: int = 1, min_reward: float | None = None) -> list[tuple[str, float]]:
        candidates = self.generate_fn(prompt, n)
        scores = self._score(prompt, candidates)
        pairs = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)
        if min_reward is not None:
            pairs = [p for p in pairs if p[1] >= min_reward]
        return pairs[:top_k]

    @torch.no_grad()
    def _score(self, prompt: str, responses: list[str]) -> list[float]:
        out: list[float] = []
        for response in responses:
            ids = torch.tensor([self.tokenizer_encode(prompt + response)], dtype=torch.long, device=self.device)
            mask = torch.ones_like(ids)
            score = self.reward_model(ids, attention_mask=mask).item()
            out.append(score)
        return out
