"""KV cache for incremental decoding.

The cache is a *per-layer* ring of contiguous tensors. For each layer we keep:

    K: (batch, num_kv_heads, max_seq, head_dim)
    V: (batch, num_kv_heads, max_seq, head_dim)

At decode time the attention module appends a single token's K/V at offset
``pos`` and reads the whole prefix in one shot. This is the most common layout used in
production (vLLM PagedAttention is a more advanced variant of the same idea).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class LayerCache:
    """Per-layer KV cache slab."""

    k: torch.Tensor  # (B, H_kv, S, D)
    v: torch.Tensor  # (B, H_kv, S, D)

    def append(self, key: torch.Tensor, value: torch.Tensor, start: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Append ``key``/``value`` at offset ``start`` and return the full prefix."""
        seq_len = key.shape[-2]
        end = start + seq_len
        if end > self.k.shape[-2]:
            raise RuntimeError(
                f"KV cache overflow: attempted to write to position {end} of {self.k.shape[-2]}"
            )
        self.k[:, :, start:end].copy_(key)
        self.v[:, :, start:end].copy_(value)
        return self.k[:, :, :end], self.v[:, :, :end]


@dataclass
class KVCache:
    """Container for the cache of every layer + a shared decode offset."""

    layers: list[LayerCache]
    seq_offset: int = 0
    max_seq: int = 0
    dtype: torch.dtype = field(default=torch.bfloat16)

    @classmethod
    def allocate(
        cls,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> KVCache:
        layers = [
            LayerCache(
                k=torch.zeros(batch_size, num_kv_heads, max_seq, head_dim, device=device, dtype=dtype),
                v=torch.zeros(batch_size, num_kv_heads, max_seq, head_dim, device=device, dtype=dtype),
            )
            for _ in range(num_layers)
        ]
        return cls(layers=layers, max_seq=max_seq, dtype=dtype)

    def advance(self, n: int) -> None:
        self.seq_offset += n

    def reset(self) -> None:
        self.seq_offset = 0
        for layer in self.layers:
            layer.k.zero_()
            layer.v.zero_()

    @property
    def memory_bytes(self) -> int:
        if not self.layers:
            return 0
        l0 = self.layers[0]
        per_layer = l0.k.numel() * l0.k.element_size() + l0.v.numel() * l0.v.element_size()
        return per_layer * len(self.layers)
