import torch

from llm.model.rope import RotaryEmbedding, apply_rotary_emb


def test_rope_relative_invariance():
    rope = RotaryEmbedding(dim=16, max_position_embeddings=128, base=10000.0)
    q = torch.randn(1, 2, 4, 16)
    k = torch.randn(1, 2, 4, 16)
    cache = rope(8, q.device, q.dtype)

    pos1 = torch.arange(4).view(1, 4)
    pos2 = pos1 + 2  # shift by 2
    q1, k1 = apply_rotary_emb(q, k, cache.cos, cache.sin, position_ids=pos1)
    q2, k2 = apply_rotary_emb(q, k, cache.cos, cache.sin, position_ids=pos2)

    # Inner product depends only on relative offset → diagonal of QK^T preserved.
    s1 = (q1 * k1).sum(-1)
    s2 = (q2 * k2).sum(-1)
    assert torch.allclose(s1, s2, atol=1e-4)
