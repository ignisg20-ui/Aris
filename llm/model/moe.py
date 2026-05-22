"""Sparse Mixture-of-Experts feed-forward layer.

For an input token :math:`x \\in \\mathbb{R}^h`:

1. Router scores ``s = softmax(W_r x)`` over ``E`` experts.
2. Top-``k`` experts are selected; gates are renormalized over the selected set.
3. Output is ``sum_i  g_i * Expert_i(x)``.

Auxiliary losses encourage uniform expert utilization:

* **Load-balance loss** (Shazeer 2017 / Switch Transformer):
    ``L_balance = E * sum_e (f_e * P_e)``
  where ``f_e`` is the fraction of tokens routed to expert ``e`` and ``P_e`` the mean
  router probability for that expert.
* **Router-z loss** (ST-MoE, Zoph et al. 2022):
    ``L_z = mean_b (logsumexp_e logits_{b,e})^2``
  This stabilizes logit magnitudes and dramatically improves training stability.

A *shared expert* (always activated, dense) can optionally be added on top of the
sparse routing; this is the design used in DeepSeek-MoE.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .activations import SwiGLU
from .config import ArisConfig


@dataclass
class MoEAuxLoss:
    """Diagnostic losses returned by the MoE layer."""

    load_balance: torch.Tensor
    router_z: torch.Tensor

    def total(self, balance_coef: float, z_coef: float) -> torch.Tensor:
        return balance_coef * self.load_balance + z_coef * self.router_z


class Top2Router(nn.Module):
    """Top-k router with auxiliary losses.

    The name is historical; ``k`` is configurable via ``num_experts_per_token``.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int, jitter: float = 0.0) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter = jitter
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        # Small init so early router decisions are near-uniform.
        nn.init.normal_(self.gate.weight, mean=0.0, std=hidden_size**-0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, MoEAuxLoss]:
        """
        Returns:
            top_weights: (N, k) renormalized gates.
            top_indices: (N, k) expert ids.
            aux:         load-balance + router-z losses (per call).
        """
        if self.training and self.jitter > 0:
            x = x * (1.0 + (torch.rand_like(x) * 2 - 1) * self.jitter)
        logits = self.gate(x.float())  # cast to fp32 for numerical stability
        probs = F.softmax(logits, dim=-1)
        top_weights, top_indices = probs.topk(self.top_k, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # ---- Auxiliary losses ------------------------------------------------
        # Tokens-per-expert fraction.
        one_hot = F.one_hot(top_indices, num_classes=self.num_experts).float()  # (N, k, E)
        tokens_per_expert = one_hot.sum(dim=(0, 1)) / (x.shape[0] * self.top_k)
        # Mean router probability per expert.
        mean_probs = probs.mean(dim=0)
        load_balance = self.num_experts * (tokens_per_expert * mean_probs).sum()
        # Router-z loss.
        router_z = (torch.logsumexp(logits, dim=-1) ** 2).mean()
        aux = MoEAuxLoss(load_balance=load_balance, router_z=router_z)

        return top_weights.to(x.dtype), top_indices, aux


class MoEFFN(nn.Module):
    """Sparse mixture-of-experts FFN block.

    Implementation note: we use a token-grouped dispatch (sort by expert, then batched
    matmul). This is simple, correct and scales well within a single device; for
    cross-device dispatch use the expert-parallel wrapper in :mod:`llm.distributed`.
    """

    def __init__(self, config: ArisConfig) -> None:
        super().__init__()
        moe = config.moe
        self.num_experts = moe.num_experts
        self.top_k = moe.num_experts_per_token
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.capacity_factor = moe.capacity_factor

        self.router = Top2Router(
            hidden_size=config.hidden_size,
            num_experts=moe.num_experts,
            top_k=moe.num_experts_per_token,
            jitter=moe.router_jitter,
        )
        # ModuleList of SwiGLU experts. This is the most memory-efficient layout when
        # you also want to *shard* experts across ranks (each rank owns a slice).
        self.experts = nn.ModuleList(
            [SwiGLU(config.hidden_size, config.intermediate_size) for _ in range(moe.num_experts)]
        )
        self.shared_expert: nn.Module | None
        if moe.shared_expert:
            self.shared_expert = SwiGLU(
                config.hidden_size, moe.shared_expert_intermediate_size or config.intermediate_size
            )
        else:
            self.shared_expert = None

        self.last_aux: MoEAuxLoss | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h = x.shape
        x_flat = x.reshape(-1, h)
        n_tokens = x_flat.shape[0]

        top_weights, top_indices, aux = self.router(x_flat)
        self.last_aux = aux

        # Dispatch: gather tokens per expert.
        flat_indices = top_indices.reshape(-1)  # (N*k,)
        flat_weights = top_weights.reshape(-1, 1)  # (N*k, 1)
        token_ids = torch.arange(n_tokens, device=x.device).repeat_interleave(self.top_k)  # (N*k,)

        out = torch.zeros_like(x_flat)
        # We keep this loop in Python because ``E`` is small (≤ 256 in practice) and
        # each call dispatches a single batched matmul.
        for expert_id in range(self.num_experts):
            mask = flat_indices == expert_id
            if not mask.any():
                continue
            tok_idx = token_ids[mask]
            weights = flat_weights[mask]
            expert_in = x_flat[tok_idx]
            expert_out = self.experts[expert_id](expert_in) * weights
            out.index_add_(0, tok_idx, expert_out.to(out.dtype))

        out = out.view(b, t, h)
        if self.shared_expert is not None:
            out = out + self.shared_expert(x)
        return out
