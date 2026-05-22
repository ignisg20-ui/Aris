"""Configuration objects for Aris models.

Configs are dataclasses so they are easy to serialize/deserialize via YAML and trivially
hashable for caching. ``ArisConfig`` is the single source of truth for the model
architecture; every module in ``llm.model`` reads from it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

AttentionImpl = Literal["sdpa", "flash", "math"]
NormType = Literal["rmsnorm", "layernorm"]
ActivationType = Literal["swiglu", "geglu", "gelu", "silu"]
RouterType = Literal["topk", "expert_choice", "switch"]


@dataclass
class MoEConfig:
    """Sparse mixture-of-experts configuration.

    The model is *dense* when ``num_experts == 1``. Otherwise every ``moe_layer_freq``-th
    layer (counting from layer 0) is replaced with a sparse MoE FFN.
    """

    num_experts: int = 1
    num_experts_per_token: int = 2
    router_type: RouterType = "topk"
    capacity_factor: float = 1.25
    expert_dropout: float = 0.0
    router_jitter: float = 0.0
    router_z_loss_coef: float = 1e-3
    load_balance_loss_coef: float = 1e-2
    moe_layer_freq: int = 2
    shared_expert: bool = True
    shared_expert_intermediate_size: int | None = None

    @property
    def is_sparse(self) -> bool:
        return self.num_experts > 1


@dataclass
class AttentionConfig:
    """Attention sub-block configuration.

    * ``num_kv_heads < num_heads`` enables Grouped-Query Attention.
    * ``mla_q_lora_rank`` / ``mla_kv_lora_rank`` enable Multi-Head Latent Attention
      (DeepSeek-V2 style). Set both to ``None`` to disable MLA.
    """

    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int | None = None  # falls back to hidden_size // num_heads
    attention_impl: AttentionImpl = "sdpa"
    attention_dropout: float = 0.0
    use_alibi: bool = False
    use_qk_norm: bool = True
    softmax_scale: float | None = None

    # MLA (Multi-Head Latent Attention)
    mla_q_lora_rank: int | None = None
    mla_kv_lora_rank: int | None = None
    mla_qk_rope_head_dim: int = 64
    mla_qk_nope_head_dim: int = 128
    mla_v_head_dim: int = 128


@dataclass
class RoPEConfig:
    """Rotary Positional Embeddings configuration."""

    rope_theta: float = 1_000_000.0
    rope_scaling_type: Literal["none", "linear", "ntk", "yarn"] = "none"
    rope_scaling_factor: float = 1.0
    rope_original_max_position: int = 8192
    rope_yarn_beta_fast: float = 32.0
    rope_yarn_beta_slow: float = 1.0


@dataclass
class ArisConfig:
    """Top-level Aris model configuration.

    Presets (1B..200B) live in ``llm/configs/*.yaml`` and are loaded with
    :meth:`ArisConfig.from_yaml`.
    """

    # Core dimensions
    vocab_size: int = 128_256
    hidden_size: int = 4096
    intermediate_size: int = 14_336
    num_layers: int = 32
    max_position_embeddings: int = 131_072

    # Normalization / activation
    norm_type: NormType = "rmsnorm"
    norm_eps: float = 1e-5
    activation: ActivationType = "swiglu"

    # Embedding / output
    tie_word_embeddings: bool = False
    embedding_init_std: float = 0.02

    # Initialization
    initializer_range: float = 0.02
    init_method: Literal["normal", "scaled", "small_init"] = "scaled"

    # Sub-configs
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    rope: RoPEConfig = field(default_factory=RoPEConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)

    # Training / regularization
    dropout: float = 0.0
    residual_dropout: float = 0.0
    hidden_dropout: float = 0.0
    use_gradient_checkpointing: bool = False
    gradient_checkpointing_layers: int | None = None  # checkpoint first N layers; None=all

    # Precision
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Misc
    use_kv_cache: bool = True
    name: str = "aris"

    # ------------------------------------------------------------------ #
    # Validation                                                          #
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        a = self.attention
        if self.hidden_size % a.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({a.num_heads})"
            )
        if a.num_heads % a.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({a.num_heads}) must be divisible by num_kv_heads ({a.num_kv_heads})"
            )
        if a.head_dim is None:
            a.head_dim = self.hidden_size // a.num_heads
        if self.moe.is_sparse and self.moe.num_experts_per_token > self.moe.num_experts:
            raise ValueError("num_experts_per_token must be <= num_experts")
        if self.moe.shared_expert and self.moe.shared_expert_intermediate_size is None:
            self.moe.shared_expert_intermediate_size = self.intermediate_size

    # ------------------------------------------------------------------ #
    # Parameter accounting                                                #
    # ------------------------------------------------------------------ #
    def num_parameters(self) -> int:
        """Approximate total parameter count (dense + sparse experts)."""
        h = self.hidden_size
        i = self.intermediate_size
        L = self.num_layers
        v = self.vocab_size

        # Embeddings
        embed = v * h
        out = 0 if self.tie_word_embeddings else v * h

        # Attention (QKVO). If MLA, account for compressed projections.
        a = self.attention
        if a.mla_q_lora_rank is not None and a.mla_kv_lora_rank is not None:
            q_head = a.mla_qk_rope_head_dim + a.mla_qk_nope_head_dim
            attn = (
                h * a.mla_q_lora_rank
                + a.mla_q_lora_rank * a.num_heads * q_head
                + h * (a.mla_kv_lora_rank + a.mla_qk_rope_head_dim)
                + a.mla_kv_lora_rank * a.num_heads * (a.mla_qk_nope_head_dim + a.mla_v_head_dim)
                + a.num_heads * a.v_head_dim_safe() * h  # type: ignore[attr-defined]
            )
        else:
            kv_dim = a.num_kv_heads * (a.head_dim or 0)
            attn = h * h + 2 * h * kv_dim + h * h

        # FFN (dense). For SwiGLU, gate+up+down = 3*h*i.
        ffn_dense = 3 * h * i

        # MoE FFN per layer = (num_experts * ffn_dense) + shared expert + router.
        if self.moe.is_sparse:
            ffn_moe = self.moe.num_experts * 3 * h * i + h * self.moe.num_experts
            if self.moe.shared_expert:
                ffn_moe += 3 * h * (self.moe.shared_expert_intermediate_size or i)
            num_moe_layers = L // self.moe.moe_layer_freq
            ffn_total = num_moe_layers * ffn_moe + (L - num_moe_layers) * ffn_dense
        else:
            ffn_total = L * ffn_dense

        # Two RMSNorms per layer + 1 final norm
        norm = (2 * L + 1) * h

        return int(embed + out + L * attn + ffn_total + norm)

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArisConfig:
        data = dict(data)
        attn = AttentionConfig(**data.pop("attention", {}))
        rope = RoPEConfig(**data.pop("rope", {}))
        moe = MoEConfig(**data.pop("moe", {}))
        return cls(attention=attn, rope=rope, moe=moe, **data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ArisConfig:
        data = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(data)


# Tiny patch: AttentionConfig.v_head_dim_safe helper used by parameter accounting.
def _v_head_dim_safe(self: AttentionConfig) -> int:
    return self.mla_v_head_dim if self.mla_kv_lora_rank is not None else (self.head_dim or 0)


AttentionConfig.v_head_dim_safe = _v_head_dim_safe  # type: ignore[attr-defined]
