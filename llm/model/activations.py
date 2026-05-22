"""Activation functions used by Aris.

Aris uses **SwiGLU** in its position-wise feed-forward sub-layer:

    SwiGLU(x) = Swish(W_gate x) ⊙ (W_up x)
    FFN(x)    = W_down · SwiGLU(x)

This is the variant introduced in Noam Shazeer's "GLU Variants Improve Transformer"
(2020) and adopted by LLaMA, PaLM, DeepSeek, etc. Compared to ReLU/GELU it offers a
modest but reliable quality improvement at a ~50% FFN parameter overhead.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    """Gated SiLU feed-forward.

    Shapes:
        in:  (..., hidden)
        out: (..., hidden)
    Internally uses 3 projections totaling 3*hidden*intermediate parameters.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GeGLU(nn.Module):
    """GELU-gated variant used by some Gemini/PaLM models."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


def build_ffn(activation: str, hidden_size: int, intermediate_size: int) -> nn.Module:
    if activation == "swiglu":
        return SwiGLU(hidden_size, intermediate_size)
    if activation == "geglu":
        return GeGLU(hidden_size, intermediate_size)
    raise ValueError(f"unknown activation: {activation}")
