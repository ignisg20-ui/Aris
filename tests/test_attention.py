import torch

from llm.model.attention import GroupedQueryAttention


def test_gqa_forward_shape(tiny_config):
    attn = GroupedQueryAttention(tiny_config, layer_idx=0).eval()
    x = torch.randn(2, 8, tiny_config.hidden_size)
    out = attn(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_gqa_causal_property(tiny_config):
    """Modifying token at position t should not affect outputs at positions < t."""
    attn = GroupedQueryAttention(tiny_config, layer_idx=0).eval()
    x = torch.randn(1, 8, tiny_config.hidden_size)
    with torch.no_grad():
        out1 = attn(x)
        x_mod = x.clone()
        x_mod[0, -1] = torch.randn_like(x_mod[0, -1])
        out2 = attn(x_mod)
    # Outputs at positions 0..-2 should be unchanged.
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)
