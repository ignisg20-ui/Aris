"""Token sampling.

Supports:
* Greedy / temperature.
* Top-k.
* Top-p (nucleus).
* Min-p.
* Repetition penalty (CTRL-style).
* Frequency / presence penalties.

We do *all* sampling on the GPU, in fp32 logits for numerical stability.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    temperature: float = 0.7
    top_k: int = 0
    top_p: float = 0.95
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_new_tokens: int = 512
    stop_token_ids: tuple[int, ...] = ()
    seed: int | None = None


def _apply_repetition_penalty(
    logits: torch.Tensor,
    token_history: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    # Gather logits at history positions.
    gathered = torch.gather(logits, -1, token_history)
    pos = gathered > 0
    gathered = torch.where(pos, gathered / penalty, gathered * penalty)
    return logits.scatter(-1, token_history, gathered)


def _apply_frequency_presence(
    logits: torch.Tensor,
    token_history: torch.Tensor,
    freq_penalty: float,
    pres_penalty: float,
) -> torch.Tensor:
    if freq_penalty == 0.0 and pres_penalty == 0.0:
        return logits
    counts = torch.zeros_like(logits)
    counts.scatter_add_(-1, token_history, torch.ones_like(token_history, dtype=logits.dtype))
    presence = (counts > 0).to(logits.dtype)
    return logits - freq_penalty * counts - pres_penalty * presence


def _filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    topk_vals, _ = torch.topk(logits, k, dim=-1)
    threshold = topk_vals[..., -1, None]
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)


def _filter_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cum_probs = sorted_probs.cumsum(dim=-1)
    sorted_mask = cum_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask = torch.zeros_like(sorted_mask).scatter_(-1, sorted_idx, sorted_mask)
    return logits.masked_fill(mask, float("-inf"))


def _filter_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    if min_p <= 0.0:
        return logits
    probs = F.softmax(logits, dim=-1)
    threshold = min_p * probs.max(dim=-1, keepdim=True).values
    return logits.masked_fill(probs < threshold, float("-inf"))


def sample_token(
    logits: torch.Tensor,
    params: SamplingParams,
    token_history: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token per batch row from ``logits``.

    Args:
        logits: (B, V) fp32 logits.
        params: sampling configuration.
        token_history: (B, T) previously generated tokens for penalty terms.
    """
    logits = logits.float()
    if token_history is not None:
        if params.repetition_penalty != 1.0:
            logits = _apply_repetition_penalty(logits, token_history, params.repetition_penalty)
        logits = _apply_frequency_presence(logits, token_history, params.frequency_penalty, params.presence_penalty)

    if params.temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits / max(params.temperature, 1e-5)
    logits = _filter_top_k(logits, params.top_k)
    logits = _filter_top_p(logits, params.top_p)
    logits = _filter_min_p(logits, params.min_p)

    probs = F.softmax(logits, dim=-1)
    if probs.sum(dim=-1).min().item() < 1e-9:  # everything filtered out → fall back to argmax
        return logits.argmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
