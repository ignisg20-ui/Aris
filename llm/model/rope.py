"""Rotary Positional Embeddings.

The construction below applies a 2-D rotation in every consecutive pair of feature
dimensions::

    R(θ_k, m) =  | cos(m·θ_k)  -sin(m·θ_k) |
                 | sin(m·θ_k)   cos(m·θ_k) |

with frequency schedule

    θ_k = base^(-2k/d),  k ∈ [0, d/2)

Applied to queries/keys this gives the inner product property
``<R(m) q, R(n) k> = <q, R(n-m) k>``, i.e. attention scores depend on relative
positions only. We support three position-interpolation schemes for length extension:

* ``linear`` — Chen et al. 2023 (position scaled by ``1/factor``).
* ``ntk``    — bloc97 (base rescaled by ``factor^(d/(d-2))``).
* ``yarn``   — Peng et al. 2023 (frequency-band weighted blend of linear & NTK).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RotaryCache:
    cos: torch.Tensor
    sin: torch.Tensor


class RotaryEmbedding(nn.Module):
    """Precomputes and caches cos/sin tables.

    Args:
        dim: rotated head dimension (typically ``head_dim`` for vanilla, or
            ``qk_rope_head_dim`` for MLA).
        max_position_embeddings: maximum context length supported by the cache.
        base: ``theta`` in the RoPE formula.
        scaling_type: ``none|linear|ntk|yarn``.
        scaling_factor: target_length / original_length.
        original_max_position: pretraining context length used by YaRN.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131_072,
        base: float = 1_000_000.0,
        scaling_type: str = "none",
        scaling_factor: float = 1.0,
        original_max_position: int = 8192,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        self.original_max_position = original_max_position
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow

        inv_freq = self._build_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_position_embeddings, device=device, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Frequency construction (scaling schemes)                            #
    # ------------------------------------------------------------------ #
    def _build_inv_freq(self, device: torch.device | None) -> torch.Tensor:
        d = self.dim
        if self.scaling_type == "ntk":
            base = self.base * (self.scaling_factor ** (d / (d - 2)))
        else:
            base = self.base
        inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32, device=device) / d))

        if self.scaling_type == "linear":
            inv_freq = inv_freq / self.scaling_factor
        elif self.scaling_type == "yarn":
            inv_freq = self._yarn_correction(inv_freq)
        return inv_freq

    def _yarn_correction(self, inv_freq: torch.Tensor) -> torch.Tensor:
        # See "YaRN: Efficient Context Window Extension" (Peng et al. 2023).
        low = self._find_correction_dim(self.yarn_beta_fast)
        high = self._find_correction_dim(self.yarn_beta_slow)
        low, high = max(low, 0), min(high, self.dim // 2 - 1)
        if low == high:
            high += 1
        ramp = torch.arange(self.dim // 2, dtype=torch.float32, device=inv_freq.device)
        ramp = (ramp - low) / (high - low)
        ramp = ramp.clamp(0.0, 1.0)
        # Frequencies below ``low`` keep their original value, above ``high`` are
        # interpolated by ``1/scaling_factor`` (linear), in between we blend.
        return inv_freq * (1.0 - ramp) + (inv_freq / self.scaling_factor) * ramp

    def _find_correction_dim(self, num_rotations: float) -> int:
        # Inverse of inv_freq formula solved for the dim where exactly
        # ``num_rotations`` cycles fit in ``original_max_position``.
        return int(
            self.dim
            * math.log(self.original_max_position / (num_rotations * 2 * math.pi))
            / (2 * math.log(self.base))
        )

    # ------------------------------------------------------------------ #
    # Caching                                                              #
    # ------------------------------------------------------------------ #
    def _build_cache(self, seq_len: int, device: torch.device | None, dtype: torch.dtype) -> None:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Duplicate frequencies to match (head_dim,) layout used by ``apply_rotary_emb``.
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        self._cached_len = seq_len

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> RotaryCache:
        if seq_len > self._cached_len or self.cos_cached.device != device:
            self._build_cache(seq_len, device=device, dtype=dtype)
        return RotaryCache(
            cos=self.cos_cached[:seq_len].to(dtype),
            sin=self.sin_cached[:seq_len].to(dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query / key tensors.

    Shapes:
        q, k:        (B, H, T, D) or (B, H, T, D_rot)  (only last dim is rotated)
        cos, sin:    (T_cache, D) or, after indexing,  (B, T, D)
        position_ids:(B, T) optional. When provided we gather rather than slice.
    """
    if position_ids is None:
        cos = cos[: q.shape[-2]].unsqueeze(0).unsqueeze(0)
        sin = sin[: q.shape[-2]].unsqueeze(0).unsqueeze(0)
    else:
        cos = cos[position_ids].unsqueeze(1)  # (B, 1, T, D)
        sin = sin[position_ids].unsqueeze(1)

    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)
