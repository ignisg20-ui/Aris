"""Attention modules.

Two implementations are provided:

* :class:`GroupedQueryAttention` — standard QKV attention with optional KV-head
  sharing (``num_kv_heads <= num_heads``). Uses PyTorch's fused SDPA kernel by
  default (which transparently dispatches to Flash-Attention 2 when available),
  or falls back to a math implementation.
* :class:`MultiHeadLatentAttention` — DeepSeek-V2 style attention with a *latent*
  low-rank compression of K/V (and optionally Q). Drastically reduces KV-cache
  footprint at long contexts.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import ArisConfig
from .kv_cache import LayerCache
from .rms_norm import RMSNorm
from .rope import RotaryEmbedding, apply_rotary_emb

try:  # pragma: no cover - optional dep
    from flash_attn import flash_attn_func  # type: ignore

    _HAS_FLASH = True
except ImportError:  # pragma: no cover
    _HAS_FLASH = False


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads (GQA)."""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention with RoPE, KV-cache and Flash-Attn dispatch.

    Args:
        config: top-level :class:`ArisConfig`.
        layer_idx: layer index, used for cache routing.
    """

    def __init__(self, config: ArisConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        a = config.attention

        self.hidden_size = config.hidden_size
        self.num_heads = a.num_heads
        self.num_kv_heads = a.num_kv_heads
        self.num_kv_groups = a.num_heads // a.num_kv_heads
        self.head_dim = a.head_dim or (config.hidden_size // a.num_heads)
        self.scaling = a.softmax_scale or (1.0 / math.sqrt(self.head_dim))
        self.dropout = a.attention_dropout
        self.attention_impl = a.attention_impl

        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.use_qk_norm = a.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)

        self.rotary = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope.rope_theta,
            scaling_type=config.rope.rope_scaling_type,
            scaling_factor=config.rope.rope_scaling_factor,
            original_max_position=config.rope.rope_original_max_position,
            yarn_beta_fast=config.rope.rope_yarn_beta_fast,
            yarn_beta_slow=config.rope.rope_yarn_beta_slow,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache: LayerCache | None = None,
        cache_offset: int = 0,
    ) -> torch.Tensor:
        b, t, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        kv_seq_len = t + cache_offset
        rot = self.rotary(kv_seq_len, hidden_states.device, hidden_states.dtype)
        if position_ids is None:
            position_ids = torch.arange(cache_offset, cache_offset + t, device=hidden_states.device).unsqueeze(0).expand(b, t)
        q, k = apply_rotary_emb(q, k, rot.cos, rot.sin, position_ids=position_ids)

        if cache is not None:
            k, v = cache.append(k, v, start=cache_offset)

        k = _repeat_kv(k, self.num_kv_groups)
        v = _repeat_kv(v, self.num_kv_groups)

        attn_out = self._compute_attention(q, k, v, attention_mask, is_causal=cache_offset == 0 and cache is None)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, self.num_heads * self.head_dim)
        return self.o_proj(attn_out)

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        if self.attention_impl == "flash" and _HAS_FLASH and q.is_cuda:
            # flash_attn expects (B, T, H, D)
            qf, kf, vf = (t.transpose(1, 2) for t in (q, k, v))
            out = flash_attn_func(qf, kf, vf, dropout_p=self.dropout if self.training else 0.0, causal=is_causal, softmax_scale=self.scaling)
            return out.transpose(1, 2)

        if self.attention_impl in {"sdpa", "flash"}:
            # SDPA: fused, dispatches to flash on supported hardware.
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal and mask is None,
                scale=self.scaling,
            )

        # Reference math implementation.
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if is_causal and mask is None:
            t_q, t_k = scores.shape[-2], scores.shape[-1]
            causal = torch.ones(t_q, t_k, device=scores.device, dtype=torch.bool).tril(diagonal=t_k - t_q)
            scores = scores.masked_fill(~causal, float("-inf"))
        if mask is not None:
            scores = scores + mask
        probs = F.softmax(scores.float(), dim=-1).to(q.dtype)
        if self.training and self.dropout > 0:
            probs = F.dropout(probs, p=self.dropout)
        return torch.matmul(probs, v)


class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention (DeepSeek-V2).

    Key idea: project K and V from a *shared low-rank latent* (``kv_lora_rank``),
    so the KV cache stores only the latent vector + the RoPE-rotated keys. This
    cuts long-context cache memory by 5-10× compared to GQA.

    Per-head layout::

        q_head = [q_nope ; q_rope]  shape D = nope_dim + rope_dim
        k_head = [k_nope ; k_rope]
        v_head shape D_v

    The non-positional half (nope) is reconstructed from latents while the
    positional half (rope) is rotated as usual.
    """

    def __init__(self, config: ArisConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        a = config.attention
        assert a.mla_q_lora_rank is not None and a.mla_kv_lora_rank is not None

        self.num_heads = a.num_heads
        self.qk_rope_head_dim = a.mla_qk_rope_head_dim
        self.qk_nope_head_dim = a.mla_qk_nope_head_dim
        self.v_head_dim = a.mla_v_head_dim
        self.q_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        self.q_lora_rank = a.mla_q_lora_rank
        self.kv_lora_rank = a.mla_kv_lora_rank
        self.scaling = 1.0 / math.sqrt(self.q_head_dim)
        self.dropout = a.attention_dropout

        h = config.hidden_size
        self.q_a_proj = nn.Linear(h, self.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(self.q_lora_rank, eps=config.norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(h, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_norm = RMSNorm(self.kv_lora_rank, eps=config.norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, h, bias=False)

        self.rotary = RotaryEmbedding(
            dim=self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope.rope_theta,
            scaling_type=config.rope.rope_scaling_type,
            scaling_factor=config.rope.rope_scaling_factor,
            original_max_position=config.rope.rope_original_max_position,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache: LayerCache | None = None,
        cache_offset: int = 0,
    ) -> torch.Tensor:
        b, t, _ = hidden_states.shape

        # ----- Q -----
        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(hidden_states)))
        q = q.view(b, t, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # ----- KV (compressed) -----
        kv_mixed = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_rope = torch.split(
            kv_mixed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_norm(compressed_kv)
        kv = (
            self.kv_b_proj(compressed_kv)
            .view(b, t, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )
        k_nope, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_rope = k_rope.view(b, t, 1, self.qk_rope_head_dim).transpose(1, 2)

        # ----- RoPE on rope-half only -----
        kv_seq_len = t + cache_offset
        rot = self.rotary(kv_seq_len, hidden_states.device, hidden_states.dtype)
        if position_ids is None:
            position_ids = torch.arange(cache_offset, cache_offset + t, device=hidden_states.device).unsqueeze(0).expand(b, t)
        q_rope, k_rope_rot = apply_rotary_emb(q_rope, k_rope, rot.cos, rot.sin, position_ids=position_ids)
        k_rope = k_rope_rot.expand(-1, self.num_heads, -1, -1)

        query = torch.cat([q_nope, q_rope], dim=-1)
        key = torch.cat([k_nope, k_rope], dim=-1)

        if cache is not None:
            # MLA's "true" cache stores only the latent; for simplicity we
            # cache the materialized k/v which is still correct numerically.
            key, value = cache.append(key, value, start=cache_offset)

        attn_out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=cache_offset == 0 and cache is None,
            scale=self.scaling,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, self.num_heads * self.v_head_dim)
        return self.o_proj(attn_out)


def build_attention(config: ArisConfig, layer_idx: int) -> nn.Module:
    a = config.attention
    if a.mla_q_lora_rank is not None and a.mla_kv_lora_rank is not None:
        return MultiHeadLatentAttention(config, layer_idx)
    return GroupedQueryAttention(config, layer_idx)
