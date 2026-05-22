"""Root-Mean-Square LayerNorm.

Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019).

Given input x of shape (..., d):

    rms(x) = sqrt(mean(x ** 2) + eps)
    y      = (x / rms(x)) * g

where ``g`` is a learnable per-feature gain. The mean-subtraction step from LayerNorm
is dropped, which empirically preserves quality while reducing kernel work.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm with an optional bias-free formulation."""

    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # Always compute in fp32 for numerical stability, then cast back.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            out = out * self.weight.to(x.dtype)
        return out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"
