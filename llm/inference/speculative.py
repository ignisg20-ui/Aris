"""Speculative decoding.

Algorithm (Leviathan et al. 2023):
1. A *draft* model proposes ``γ`` candidate tokens from the current context.
2. The *target* model evaluates all γ proposals in a single forward pass.
3. For each draft token ``q_i`` with target probability ``p_i`` and draft
   probability ``q̃_i``, accept it with probability ``min(1, p_i / q̃_i)``. On
   first rejection, sample from ``max(0, p_i - q̃_i)`` (renormalized).
4. If all γ are accepted, sample one extra token from the target distribution.

This guarantees that the output distribution is *exactly* that of the target
model — speculative decoding is a wall-clock speedup, not an approximation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..model.aris import ArisForCausalLM
from ..model.kv_cache import KVCache
from .sampling import SamplingParams

logger = logging.getLogger("aris.spec")


@dataclass
class SpeculativeStats:
    drafted: int = 0
    accepted: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(1, self.drafted)


class SpeculativeDecoder:
    """Speculative decoding with a small draft model."""

    def __init__(
        self,
        target: ArisForCausalLM,
        draft: ArisForCausalLM,
        gamma: int = 4,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.target = target.eval()
        self.draft = draft.eval()
        self.gamma = gamma
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.stats = SpeculativeStats()

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, sampling: SamplingParams) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        device = self.device

        # Allocate caches independently for draft/target.
        def _alloc(model: ArisForCausalLM) -> KVCache:
            a = model.config.attention
            return KVCache.allocate(
                num_layers=model.config.num_layers,
                batch_size=input_ids.shape[0],
                num_kv_heads=a.num_kv_heads,
                head_dim=a.head_dim or (model.config.hidden_size // a.num_heads),
                max_seq=input_ids.shape[1] + sampling.max_new_tokens,
                device=device,
                dtype=self.dtype,
            )

        target_cache = _alloc(self.target)
        draft_cache = _alloc(self.draft)

        # Prefill both.
        with torch.autocast(device_type=device.type, dtype=self.dtype, enabled=device.type == "cuda"):
            _ = self.target(input_ids, kv_cache=target_cache)
            _ = self.draft(input_ids, kv_cache=draft_cache)
        output_ids = input_ids
        new_tokens = 0

        while new_tokens < sampling.max_new_tokens:
            gamma = min(self.gamma, sampling.max_new_tokens - new_tokens)
            drafts, draft_probs = self._draft_tokens(output_ids[:, -1:], draft_cache, gamma, sampling)
            target_probs = self._target_probs(output_ids[:, -1:], drafts, target_cache, sampling)

            n_accepted = 0
            for i in range(gamma):
                p_t = target_probs[i, drafts[0, i]]
                q_d = draft_probs[i, drafts[0, i]]
                acceptance = min(1.0, (p_t / max(q_d.item(), 1e-9)).item())
                self.stats.drafted += 1
                if torch.rand((), device=device).item() < acceptance:
                    n_accepted += 1
                    self.stats.accepted += 1
                else:
                    break

            accepted = drafts[:, :n_accepted]
            if n_accepted < gamma:
                # Resample from corrected residual distribution.
                p = target_probs[n_accepted]
                q = draft_probs[n_accepted]
                residual = torch.clamp(p - q, min=0.0)
                residual = residual / max(residual.sum().item(), 1e-9)
                extra = torch.multinomial(residual, num_samples=1).unsqueeze(0)
            else:
                # All accepted -> sample one bonus token from target dist after the chunk.
                bonus_probs = target_probs[-1]
                extra = torch.multinomial(bonus_probs, num_samples=1).unsqueeze(0)

            chunk = torch.cat([accepted, extra], dim=1)
            output_ids = torch.cat([output_ids, chunk], dim=1)
            new_tokens += chunk.shape[1]

            # Roll back draft cache to align with the actually-accepted sequence.
            draft_cache.seq_offset -= (gamma - chunk.shape[1])
            target_cache.seq_offset -= (gamma - chunk.shape[1])

            if (chunk == self.target.config.eos_token_id).any():
                break

        return output_ids

    # ------------------------------------------------------------------ #
    def _draft_tokens(
        self,
        last_token: torch.Tensor,
        cache: KVCache,
        gamma: int,
        sampling: SamplingParams,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, probs = [], []
        x = last_token
        for _ in range(gamma):
            with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
                out = self.draft(x, kv_cache=cache)
            logits = out.logits[:, -1, :].float() / max(sampling.temperature, 1e-5)
            p = F.softmax(logits, dim=-1)
            t = torch.multinomial(p, num_samples=1)
            tokens.append(t)
            probs.append(p.squeeze(0))
            x = t
        return torch.cat(tokens, dim=1), torch.stack(probs, dim=0)

    def _target_probs(
        self,
        last_token: torch.Tensor,
        drafts: torch.Tensor,
        cache: KVCache,
        sampling: SamplingParams,
    ) -> torch.Tensor:
        chunk = torch.cat([last_token, drafts[:, :-1]], dim=1)
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            out = self.target(chunk, kv_cache=cache)
        logits = out.logits[0].float() / max(sampling.temperature, 1e-5)
        return F.softmax(logits, dim=-1)
