"""Common test fixtures and config helpers."""

from __future__ import annotations

import pytest
import torch

from llm.model.config import ArisConfig, AttentionConfig, MoEConfig, RoPEConfig


@pytest.fixture
def tiny_config() -> ArisConfig:
    return ArisConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        max_position_embeddings=256,
        attention=AttentionConfig(num_heads=4, num_kv_heads=2, attention_impl="sdpa", use_qk_norm=False),
        rope=RoPEConfig(rope_theta=10000.0),
        moe=MoEConfig(num_experts=1),
        tie_word_embeddings=True,
        dtype="fp32",
    )


@pytest.fixture
def moe_config() -> ArisConfig:
    return ArisConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        max_position_embeddings=256,
        attention=AttentionConfig(num_heads=4, num_kv_heads=2, attention_impl="sdpa", use_qk_norm=False),
        rope=RoPEConfig(rope_theta=10000.0),
        moe=MoEConfig(
            num_experts=4,
            num_experts_per_token=2,
            moe_layer_freq=1,
            shared_expert=False,
        ),
        tie_word_embeddings=True,
        dtype="fp32",
    )


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
