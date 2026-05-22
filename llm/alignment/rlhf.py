"""PPO-based RLHF.

Algorithm follows Christiano et al. 2017 / Ouyang et al. 2022:

1. Sample rollouts from policy π_θ.
2. Score them with the reward model r_φ.
3. Estimate advantages with GAE on a value head V_ψ.
4. Take a clipped PPO step on π_θ + V_ψ with a KL penalty against a frozen
   reference policy π_ref.

The total reward is

    R(x, y) = r_φ(x, y) - β · KL(π_θ || π_ref)

where β is annealed via an adaptive KL controller (Schulman et al. 2017).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..model.aris import ArisForCausalLM
from .reward_model import RewardModel


@dataclass
class PPOConfig:
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    entropy_coef: float = 0.0
    gamma: float = 1.0
    lam: float = 0.95
    target_kl: float = 6.0
    init_kl_coef: float = 0.05
    adap_kl_horizon: int = 10_000
    ppo_epochs: int = 4
    mini_batch_size: int = 2
    max_generation_len: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0


class AdaptiveKLController:
    """KL-coefficient controller from Ouyang et al. 2022 (Eq. 4)."""

    def __init__(self, init_value: float, target: float, horizon: int) -> None:
        self.value = init_value
        self.target = target
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        proportional_error = max(-0.2, min(0.2, current_kl / self.target - 1.0))
        self.value *= math.exp(proportional_error * n_steps / self.horizon)


class ValueHead(nn.Module):
    """Linear value head sharing the policy's backbone via hidden states."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.v = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.v.weight, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.v(hidden).squeeze(-1)


@dataclass
class PPOBatch:
    queries: torch.Tensor       # (B, Tq)
    responses: torch.Tensor     # (B, Tr)
    full: torch.Tensor          # (B, Tq+Tr)
    attention_mask: torch.Tensor
    response_mask: torch.Tensor # 1 on response tokens
    rewards: torch.Tensor       # scalar reward per sequence
    old_logprobs: torch.Tensor  # (B, Tr) from sampling policy
    ref_logprobs: torch.Tensor  # (B, Tr) from frozen reference
    values: torch.Tensor        # (B, Tr) from value head


class PPOTrainer:
    """Clipped PPO + KL penalty + adaptive KL controller."""

    def __init__(
        self,
        policy: ArisForCausalLM,
        ref_policy: ArisForCausalLM,
        reward_model: RewardModel,
        value_head: ValueHead,
        optimizer: torch.optim.Optimizer,
        config: PPOConfig,
    ) -> None:
        self.policy = policy
        self.ref_policy = ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False
        self.reward_model = reward_model.eval()
        for p in self.reward_model.parameters():
            p.requires_grad = False
        self.value_head = value_head
        self.optimizer = optimizer
        self.config = config
        self.kl_ctl = AdaptiveKLController(config.init_kl_coef, config.target_kl, config.adap_kl_horizon)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _gather_logprobs(self, model: ArisForCausalLM, full_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        out = model(full_ids)
        logits = out.logits[:, :-1, :]
        targets = full_ids[:, 1:]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        # Trim to response region (mask out prompt).
        return gathered * response_mask[:, 1:].float()

    def compute_advantages(self, values: torch.Tensor, rewards_per_token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """GAE-λ advantage estimation over a fixed-length response."""
        gae = 0.0
        advantages = torch.zeros_like(rewards_per_token)
        for t in reversed(range(rewards_per_token.shape[-1])):
            next_v = values[:, t + 1] if t + 1 < values.shape[-1] else torch.zeros_like(values[:, 0])
            delta = rewards_per_token[:, t] + self.config.gamma * next_v - values[:, t]
            gae = delta + self.config.gamma * self.config.lam * gae
            advantages[:, t] = gae
        returns = advantages + values[:, : advantages.shape[-1]]
        return advantages.detach(), returns.detach()

    # ------------------------------------------------------------------ #
    def step(self, batch: PPOBatch) -> dict[str, float]:
        cfg = self.config

        # Per-token shaped reward: scalar reward at last token; KL penalty distributed per-token.
        kl_per_token = batch.old_logprobs - batch.ref_logprobs
        rewards_per_token = -self.kl_ctl.value * kl_per_token
        last_idx = batch.response_mask.sum(dim=-1).long() - 1
        rewards_per_token[torch.arange(batch.full.shape[0]), last_idx] += batch.rewards

        advantages, returns = self.compute_advantages(batch.values, rewards_per_token)
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        stats: dict[str, float] = {}
        for _ in range(cfg.ppo_epochs):
            out = self.policy(batch.full)
            logits = out.logits[:, :-1, :]
            targets = batch.full[:, 1:]
            new_logprobs = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            new_logprobs = new_logprobs * batch.response_mask[:, 1:].float()
            ratio = (new_logprobs - batch.old_logprobs).exp()

            unclipped = ratio * adv_norm
            clipped = torch.clamp(ratio, 1 - cfg.cliprange, 1 + cfg.cliprange) * adv_norm
            pg_loss = -torch.min(unclipped, clipped).mean()

            # Value loss with clipping.
            new_values = self.value_head(out.hidden_states)[:, :-1] * batch.response_mask[:, 1:].float()
            vf_loss_unclipped = (new_values - returns) ** 2
            vf_clipped = batch.values + torch.clamp(new_values - batch.values, -cfg.cliprange_value, cfg.cliprange_value)
            vf_loss_clipped = (vf_clipped - returns) ** 2
            vf_loss = 0.5 * torch.maximum(vf_loss_unclipped, vf_loss_clipped).mean()

            loss = pg_loss + cfg.vf_coef * vf_loss
            if cfg.entropy_coef > 0:
                entropy = -(F.softmax(logits.float(), dim=-1) * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()
                loss = loss - cfg.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            stats = {
                "pg_loss": pg_loss.item(),
                "vf_loss": vf_loss.item(),
                "policy_kl": kl_per_token.mean().item(),
                "reward_mean": batch.rewards.mean().item(),
                "ratio_mean": ratio.mean().item(),
                "kl_coef": self.kl_ctl.value,
            }

        self.kl_ctl.update(stats["policy_kl"], n_steps=1)
        return stats
