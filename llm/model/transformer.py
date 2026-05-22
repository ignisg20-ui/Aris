"""Decoder-only Transformer block.

Pre-norm residual architecture::

      x = x + Attn(RMSNorm(x))
      x = x + FFN(RMSNorm(x))      # FFN may be dense (SwiGLU) or sparse (MoE)

Residual stream optimization:
* All sub-layer outputs are added back into a single contiguous residual tensor.
* RMSNorm runs in fp32 internally and casts back to the residual dtype.
* When ``use_gradient_checkpointing`` is enabled we recompute the block in
  backward via ``torch.utils.checkpoint``; this trades ~30% extra compute for
  a >2× reduction in activation memory.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .activations import SwiGLU
from .attention import build_attention
from .config import ArisConfig
from .kv_cache import LayerCache
from .moe import MoEFFN


def _is_moe_layer(config: ArisConfig, layer_idx: int) -> bool:
    if not config.moe.is_sparse:
        return False
    return (layer_idx % config.moe.moe_layer_freq) == 0


class TransformerBlock(nn.Module):
    """Single transformer decoder layer."""

    def __init__(self, config: ArisConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        from .rms_norm import RMSNorm  # local import to avoid cycle

        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = build_attention(config, layer_idx)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn: nn.Module = (
            MoEFFN(config) if _is_moe_layer(config, layer_idx) else SwiGLU(config.hidden_size, config.intermediate_size)
        )
        self.is_moe = isinstance(self.ffn, MoEFFN)

        self.residual_dropout = nn.Dropout(config.residual_dropout) if config.residual_dropout > 0 else nn.Identity()

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        cache: LayerCache | None,
        cache_offset: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        attn_out = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            cache_offset=cache_offset,
        )
        hidden_states = residual + self.residual_dropout(attn_out)

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        ffn_out = self.ffn(hidden_states)
        hidden_states = residual + self.residual_dropout(ffn_out)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache: LayerCache | None = None,
        cache_offset: int = 0,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        if use_checkpoint and self.training:
            # `use_reentrant=False` is the recommended variant in PyTorch >=2.1.
            return checkpoint(
                self._forward_impl,
                hidden_states,
                attention_mask,
                position_ids,
                cache,
                cache_offset,
                use_reentrant=False,
            )
        return self._forward_impl(hidden_states, attention_mask, position_ids, cache, cache_offset)
