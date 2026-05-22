"""Adversarial defenses.

This module implements two well-tested defenses:

* **Perplexity gate** — outliers in input perplexity (computed by a small
  reference LM) are strong signals of GCG-style adversarial suffixes (Zou et al.
  2023). Inputs with perplexity above ``ppl_threshold`` are sanitized or refused.
* **Smoothing** — paraphrase-based smoothing (Robey et al. 2023): generate
  multiple paraphrases of the input, run them all, and aggregate. Approximates
  randomized smoothing.

Both defenses are configurable and optional. They are intended to *raise the bar*
for known attack categories; they are not a substitute for RLHF-trained refusals
or the rule-based refusal layer.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

import torch


@dataclass
class AdversarialDefense:
    """Composable adversarial defense pipeline."""

    perplexity_fn: Callable[[str], float] | None = None
    ppl_threshold: float = 100.0
    paraphrase_fn: Callable[[str, int], list[str]] | None = None
    smoothing_samples: int = 5
    log: list[str] = field(default_factory=list)

    def perplexity_gate(self, text: str) -> bool:
        if self.perplexity_fn is None:
            return True
        ppl = self.perplexity_fn(text)
        if math.isnan(ppl) or ppl > self.ppl_threshold:
            self.log.append(f"high_ppl ppl={ppl:.2f}")
            return False
        return True

    def smooth(self, text: str, run_fn: Callable[[str], str]) -> str:
        """Run the model on several paraphrases and return the most common answer."""
        if self.paraphrase_fn is None:
            return run_fn(text)
        variants = self.paraphrase_fn(text, self.smoothing_samples)
        responses = [run_fn(v) for v in variants]
        return statistics.mode(responses)


def gcg_token_distribution_score(input_ids: torch.Tensor, vocab_size: int) -> float:
    """A simple sanity heuristic: GCG suffixes contain rare-byte token clusters.

    Returns the share of tokens above the 95th percentile of token-id (rare BPE
    pieces tend to land at high ids in many tokenizers).
    """
    cutoff = int(0.95 * vocab_size)
    rare = (input_ids >= cutoff).float().mean().item()
    return rare
